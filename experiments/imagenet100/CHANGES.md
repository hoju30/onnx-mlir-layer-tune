# Code Changes & Bug Fixes

---

## 2026-07-07 — GPT-2 eval now records results to TSV (like the CNN runner)

`run_gpt2_text_eval.py` (score mode) and `run_gpt2_text_eval_parallel.py` now
append one TSV row per run to `<model dir>/gpt2_text_eval_results.tsv` (override
with `--results-log PATH`, disable with `--results-log off`; `--tag` for a label).
Columns: ts_iso, tag, model, format, mode, num_tokens, num_predicted, ppl,
avg_nll, next_token_top1_acc, elapsed_sec, tokens_per_sec, omp_threads,
max_tokens, text_file. Format tag auto-derived from the .so/.onnx name
(nqdq-p8e1 / qdq-f32 / onnx-int8 ...). Runs accumulate so f32/int8/p8/p16/p32 are
comparable in one file. No compute change.

---

## 2026-07-07 — GPT-2 OOM at ~200 tokens = missing buffer deallocation (FIXED)

**Symptom**: `run_gpt2_text_eval.py --mode score` on any posit GPT-2 `.so`
(p8/p16/p32) gets force-killed around token 192–224. Not a compute bug.

**Diagnosis**: OOM. RSS grows ~152 MB **per token forward** (measured, OMP=1) and
exhausts the 31 GB RAM at ~200 tokens → OS kills python. 152 MB ≈ the lm_head
weight `[768×50257]` decoded to f32 (154 MB). Root cause: the generated
`main_graph` has **`memref.alloc`=311/399, `memref.dealloc`=0** — the hand-rolled
`onnx-mlir-opt` pipeline in `build_model11_sos.sh` (unlike the real onnx-mlir
driver) NEVER inserted buffer-deallocation passes, so every intermediate buffer
(incl. the lm_head f32 decode) leaked on every forward. CNNs never showed it:
each image is a separate `--jobs` process that exits after one forward.

**Fix** (`onnx-mlir/src/bash/build_model11_sos.sh`, stage-4 krnl→llvm): inserted
the driver's memory passes (mirrors `CompilerPasses.cpp addKrnlToLLVMPasses`)
between `--convert-krnl-to-affine` and `--convert-krnl-to-llvm`:
`--convert-vector-to-scf --lower-affine --lower-krnl-region --buffer-loop-hoisting
--buffer-deallocation-pipeline --optimize-allocation-liveness`.
The dealloc pipeline MUST run after affine is lowered and krnl.region removed
(it asserts otherwise). It correctly keeps returned buffers: on gpt2-mini it
turned alloc=311/dealloc=0 into dealloc=286 (311−286 = 25 = logits + 24 KV
outputs, which are not freed).

**Verified**: rebuilt `build_gpt2_nqdq_p8e1_dealloc`. RSS now FLAT ~230–290 MB
(peak 413 MB) across 300 tokens (old: 30 GB at 176 tok). ppl **bit-identical** to
the old leaky `.so` at 60 tok (both `14679.32594688268`, top1 `0.15254237288`).
Computation unchanged; only the leak is gone. All posit GPT-2 formats must be
rebuilt to run 512/1024-token evals. CNN builds get the (harmless, bit-identical)
deallocs too.

---

## 2026-07-01

### 加速：f32-math 路徑預解碼 A（消除逐-j 冗餘 decode）

- **評估**：(a) 直接對 posit decode/encode 做 SIMD 不可行（變長編碼）；(b) 預解碼 A + 向量化 → 可行、安全、bit-identical；(c) 跨 forward 快取「解碼後的 f32 權重」→ 收益最大(3~5x) 但需 conversion 層標記常數權重（安全區分 weight vs activation），是獨立大工程。先做 (b)。
- **根因**：`useF32Dot`（`POSIT_QOP_F32_MATH=on`）路徑中，對固定 (i) 隨 j 變化時，A 的每個元素被重複 decode N 次（lm_head M=1,N=50257：A[768] 被 decode 3800萬次）。
- **修法**：`gemm_kernel` 的 **batched matmul** 與 **2D gemm** 的 useF32Dot 分支，迴圈前**序列**預解碼 A（M*K，很小）成 f32 buffer，內層改用 buffer（B 仍 inline，M=1 無冗餘）。**踩雷**：一開始對這個小預解碼迴圈加了 `#pragma omp` → 每次 gemm fork/join 反而慢 6×（103s→678s）；改序列後正常。
- **效果（p16e1, POSIT_QOP_F32_MATH=on, 64 tok, OMP=24）**：103s → **66.4s（~1.55×）**，ppl=74.79103708440599 **完全不變（bit-identical）**。只影響 f32-math 路徑；預設 posit quire 路徑不變（ppl 86.41）。
- **接續**：已實作 (c) decoded-weight 快取（見下）。

### 加速 (c)：decoded-weight 快取（跨 forward 重用解碼後的 f32 權重）

- **問題**：f32-math 路徑剩餘主成本 = 每個 forward 重複 decode 常數權重 B（lm_head/線性層）。
- **修法（自包含，不改 MLIR）**：`posit_runtime.cpp` 加 `getDecodedF32<Fmt>` + 全域快取（`gDecodedF32Cache`，mutex）。keyed by data 指標；**用 raw posit words 的 content-hash（FNV-1a）驗證**——權重內容不變 → 命中 → 跳過 decode（雜湊 ~2 指令/元素 vs decode ~15）；activation 內容變 → 雜湊不符 → 重新 decode（**永不回傳過期資料，安全**）。查找在 omp 平行區外（gemm 呼叫序列化），單一 mutex 足夠。
- 套用：**2D gemm**（線性層權重）與 **batched matmul 的 sharedB**（lm_head 權重）。attention 的 B 是 activation → 不快取（`sharedB` gate）。預設開，`POSIT_WEIGHT_CACHE=0` 可關。
- **效果（p16e1, f32-math, 64 tok, OMP=24）**：66s → **35s**（再 ~1.9×；累計 103s→35s ≈ **3×**）。ppl=74.79103708440599 **完全不變（bit-identical）**。
- 記憶體：快取解碼後的 f32 權重（GPT-2 ~500MB）。只在 useF32Dot（f32-math）啟用；預設 posit quire 路徑與 CNN 的 ALPS metadata 路徑不受影響（不同分支）。CNN 的 conv2d 未套（另需 conv kernel 優化）；但 CNN 的 FC gemm 在 f32-math 下也會受惠。


## 2026-06-26

### ✅ 已修復：GPT-2 posit 全炸的真正主因 = batched matmul 未支援（非精度！）

**重要更正**：先前說「p8 ppl=50257 是 8-bit 精度崩潰」是**誤判**。p16 也一樣 50257、top1=0 → 與 bit-width 無關，是 posit gemm **不支援 batched matmul** 的 bug。f32(native) 正常、posit(p8/p16) 全炸。

POSIT_TRACE 定位：每個 transformer 的 **attention batched matmul（rank-4 `[1,12,1,64]×[1,12,64,1]`）+ 最後的 lm_head（rank-3 A `[1,1,768]` × rank-2 B `[768,50257]`）輸出全 0** → logits 均勻 → ppl=50257。兩個 bug：

1. **`PositToKrnl/Pattern/Math.cpp` `PositGemmOpLowering` 輸出維**：rank≥3 沒有對應 case，fallback 複製 A 的維 → QK^T 輸出算成 `[1,12,1,64]`（應 `[1,12,1,1]`，N 拿錯成 K）。修：加 rank≥3 batched 分支（M 從 A 第二末維、N 從 B 末維、leading batch 維從 A）。
2. **`posit_runtime.cpp` `gemm_kernel` 無 batched 計算路徑**：rank-4 落到 2D 分支 → 維度算錯 → `zeroY()`。修：加 batched matmul 分支，支援 (a) `A[..,M,K]×B[..,K,N]`（B 同 rank，attention）與 (b) `A[..,M,K]×B[K,N]`（B 共用 2D 權重，lm_head）；逐 batch slice 做 posit dot；含 alpha 與 beta*C（C 為 `[1]` scalar broadcast）。

**驗證**：p16e1 修後 **ppl=86.4、top1=22.2%**（f32 為 74.7 / 28.6%；p16 接近 f32，差距是正常 16-bit posit 精度）。attention 數值恢復（QK^T mean 0.97/max 10）、lm_head 不再全 0、probe 出現 `matmul_batched`×24。

**注意**：p8e1 也需重 build + 重測（之前的 50257 同樣是此 bug，非 p8 精度；真正的 p8 精度結果現在才測得出來）。

### batched matmul 平行化（method B 延伸，2026-06-26）

- `gemm_kernel` 的 batched 分支輸出迴圈加 `#pragma omp parallel for collapse(3) if(!samples.enabled) num_threads(positOmpThreadCount()) schedule(static)`（平行 batch×i×j 輸出元素；ao/bo/yo 移進內層使三層完美巢狀可 collapse）。
- **bit-identical**（每個輸出獨立 dot，未改累加順序）：p16e1 平行前後 ppl 都是 86.41060289089356。
- 速度：p16e1 max-tokens 64 從 **779s → 253s（≈3×）**（lm_head `768×50257` 是主瓶頸，現在 N 維平行）。仍受軟體 posit 限制（~0.25 tok/s）。預設 `POSIT_OMP_THREADS=1` 時序列、行為不變。

### batched matmul 加 POSIT_QOP_F32_MATH 支援（2026-06-26）

- 原本 batched 分支只走 posit quire，不理會 `POSIT_QOP_F32_MATH` → attention/lm_head 與線性層不一致。現加 `useF32Dot` 分支：開 `POSIT_QOP_F32_MATH=on` 時，attention/lm_head 也 decode posit→f32、f32 fma 累加、再 encode（與 2D/3D gemm 的 f32 路徑一致）。probe mode 正確顯示 `f32`。
- **驗證（p16e1, WikiText-2 64 tok）**：
  - 預設（posit quire）：ppl=86.41 / top1=0.222
  - `POSIT_QOP_F32_MATH=on`：probe 87 個 matmul_batched 全 `f32`；**ppl=74.79 / top1=0.286 ≈ native f32 baseline（74.7 / 0.286）** → 整模型 f32 計算一致，證明計算路徑正確。且更快（103s vs 253s）。
- 至此「**posit 計算 vs f32 計算**」對整個 GPT-2（含 attention/lm_head）一致可比。預設不開 f32 時行為不變（由 `useF32Dot` gate）。

---

## 2026-06-22

### 新增：IEEE 浮點格式 fake-quant 評估 `eval_fp_formats.py`（fp16/bf16/fp8 vs posit）

- posit pipeline **不支援** fp16/bf16/fp8（只有 posit）。底層 Universal 函式庫有 cfloat/bfloat，但接進 dialect/runtime 是大工程；改用 **PyTorch/ORT fake-quant** 做這四種浮點格式的精度對照（使用者選定路線）。
- `ImageNet100/eval_fp_formats.py`：載入**與 posit 同源的 ONNX**，把所有 float 權重 initializer 用 torch cast 成目標格式再轉回 fp32（fake-quant：fp16 / bf16 / fp8e4m3=float8_e4m3fn / fp8e5m2=float8_e5m2），用 ORT 跑（CPU），同一驗證集 + 同前處理（Resize256/CenterCrop224/ImageNet mean-std）算 Top1/Top5。
- **`--quant-mode weight|act|both`（預設 both，2026-06-23 擴充）**：
  - weight：只 cast 權重 initializer。
  - act：在 ONNX graph 每個 float activation（op 輸出 + graph 輸入）後插 `Cast→格式→Cast→fp32` roundtrip（已驗證 ORT CPU 支援 fp16/bf16/fp8e4m3/fp8e5m2 的 Cast）。最終 logits 無 consumer 故不量化（不影響 argmax）。
  - both = 全模型 fp8/fp16/bf16（權重+activation）。
  - 驗證：ResNet18 量化 W=42/A=50、MobileNetV2 W=108/A=101，both 模式兩模型五格式皆跑通。
- 環境：用 `gpt2` venv（已補裝 timm；torch 2.12 含 float8 dtype + torchvision + onnx + onnxruntime）。resnet18 無 .pth、ONNX 又是 BN-folded，onnx2torch 也不支援其 op → 所以走「ORT 跑 ONNX + torch 只做權重 cast」這條統一、免載模型的路。
- 小驗證（limit 20，僅 class 000，class-biased）：兩模型五格式皆跑通。ResNet18 f32/fp16/bf16/fp8e4m3≈80%、fp8e5m2≈75%；MobileNetV2 f32/fp16/bf16≈85%、**fp8e4m3=0%**、fp8e5m2≈35%（MobileNetV2 對 fp8 動態範圍/精度敏感，與其 depthwise/ReLU6 脆弱性一致）。完整數字需 `--limit 5000`。

---

## 2026-06-19

### ✅ 已修復：GPT-2 reshape `-1` 推斷漏算靜態維 → seq 維變 hidden（mapped_range 根因）

**根因**：`PositToKrnl/Pattern/Math.cpp` `PositReshapeOpLowering` 推斷 `-1` 維時，`inferredDim = totalInputElems / knownProd`，但 `knownProd` 只乘進了**動態**輸出維，**漏掉靜態維**（else 分支沒更新 knownProd）。於是 `reshape [1,1,768] -> [-1,768]`（shape `[-1,768]`）算成 `inferredDim = 768/1 = 768`（應為 768/768=1）→ seq 維 = 768 → 每個 transformer gemm 拿到 `[hidden,hidden]` 操作數 → nan/±4096 垃圾 + mapped_range guard 拒絕。

**修法（一行）**：else 分支（靜態輸出維）也 `knownProd *= s`。

**驗證（POSIT_TRACE，2026-06-19）**：修後 gemm 輸出變回 `[1,2304]/[1,768]/[1,3072]`（M=seq=1）、mean 不再 nan、`mapped_range` PFALLBACK **完全消失**。

**f32 端到端確認（重 build f32 baseline 後，WikiText-2 64 tokens）**：ppl=**74.7**、next-token top1=**0.286（28.6%）**、27 tok/s → 模型**計算正確**（top1 是合理的語言模型行為，非均勻亂猜；ppl 偏高是簡化 ASCII BPE tokenizer + 短 context，非 bug）。**對照 p8e1**：ppl=50257（≈均勻）、top1≈0 → 走相同 shape 路徑，差別純粹是 **8-bit posit 精度太低使 logits 崩潰**（與 MobileNetV2 p8 崩潰同類）。結論：**GPT-2 nqdq 經此修正後數值正確；低 bit posit 的精度衰減屬正常量化現象，由使用者自行做 bit-width 精度實驗。**

> 下方「調查中」段落為定位過程記錄（broadcast 修正方向錯誤、guard 是對的、POSIT_DESC_DEBUG/POSIT_DISABLE_MAPPED_GUARD 診斷 env），保留供參考。

### （定位過程）GPT-2 mapped_range = 動態序列維 `?` 在 runtime 算錯（gemm 操作數過大）

- 症狀：`[PFALLBACK] kind=mapped_range detail=pointer range not mapped`（GPT-2 p8e1，count 隨 token 增加到 17~20）。
- 後果（嚴重）：`has_mapped_dense_storage` 回 false → gemm `zeroY()` 把輸出歸零跳過 → GPT-2 logits/ppl 失真。
- **決定性診斷**（用新加的 `POSIT_DISABLE_MAPPED_GUARD=1` 繞過 guard 測試）：繞過後 **segfault / core dump** → 證明 descriptor 是**真的過大**、讀下去會 OOB。**guard 是對的**（正確防止崩潰）。
- **真正根因**：用 `POSIT_DESC_DEBUG=1` dump 出被拒絕的是 `gemm.A`，runtime sizes=`[3072,3072]`、`[768,768]`（dense、offset 0）。但 posit IR 裡這些 gemm 的 A 是 `tensor<?x768>` / `tensor<?x3072>`（`?`=序列長度），權重是 `768x768` 等——**IR 形狀正確**。對照可知：**動態的 `?`（序列維）在 runtime 被算成了 hidden 大小（768 或 3072）而非真實序列長度**，使 c_proj 的 A=`[seq,3072]`→`[3072,3072]`、attention 的 A=`[seq,768]`→`[768,768]`。
- 結論：這是 **GPT-2 動態 shape 的 `?` 維度在 krnl/llvm runtime 計算錯誤**的 bug（與 broadcast / guard / compact-const 無關）。需追 dynamic-shape memref descriptor 的維度計算來源。**尚未修復**。
- **POSIT_TRACE 決定性證據（2026-06-19）**：每個 transformer gemm 的輸出都是 `[768,2304]`、`[768,768]`、`[768,3072]`、`[3072,768]`，即 **M(輸出列數)=hidden(768/3072) 而非 seq(=1)**，且 `mean=nan`、值飽和到 ±4096 → **每層輸出都是垃圾**。早期 activation `[1,1,768]` 是對的，所以 M 維是在 reshape→gemm 之間（或 gemm 輸出 alloc）被弄成 hidden。
- **必須更正先前的過度宣稱**：「GPT-2 nqdq builds+runs / f32 端到端驗證」只代表**能跑不崩潰**，不代表算對。此 M 維 bug 讓輸出是垃圾；且 shape 計算 f32 與 posit **共用 krnl/llvm**，所以 **f32 版（ppl≈286）也很可能是此 bug 的垃圾結果**，非 context 短。GPT-2 nqdq **目前數值上不正確**，需修此 bug 才算真正可用。
- 附帶（保留，非此 bug 的解）：
  - `has_mapped_dense_storage` 改用 strides 算實際跨度（對 broadcast/strided 更正確，但不是此 bug 的成因）。
  - 新增 `POSIT_DISABLE_MAPPED_GUARD` 診斷 env（預設 off）：繞過 mincore 檢查，用來判別「descriptor 真壞」vs「guard 誤判」。

### Method B：posit conv/gemm 內部 OpenMP 平行（intra-op）

讓單次 forward 用多核（GPT-2 無法跨序列平行，這是唯一能加速單序列的方式；對 CNN 也能加速單張延遲）。

- `posit_runtime.cpp`：
  - 新增 `positOmpThreadCount()`：讀 `POSIT_OMP_THREADS`（**預設 1**）。用它當 `num_threads()`，**不靠 OMP_NUM_THREADS**（後者預設全核，會在 process 級 `--jobs` 下嚴重 oversubscribe）。
  - gemm 3D（`(M,K)x(N,K,S)`）與 gemm 2D 的輸出迴圈加 `#pragma omp parallel for collapse(3|2) if(!samples.enabled) num_threads(positOmpThreadCount()) schedule(static)`。
  - **`posit_to_f32` / `posit_from_f32` bridge kernel** 的逐元素迴圈也平行化（GPT-2 對這兩個呼叫最多：to_f32×164、from_f32×140，是最大熱點；gemm×73）。to_f32 無條件平行；from_f32 用 `if(!collect)`（qalign 收集時的 push_back 非執行緒安全）。
  - **平行「輸出/元素」而非切 K 累加 → 位元級相同（bit-identical），不改 Top1/ppl**。try/catch 都在迴圈內，例外不逸出平行區。
  - **conv2d 尚未平行化（待辦）**：CNN 的 conv 仍序列，所以 CNN 用 method B 加速有限；GPT-2（gemm/matmul + bridge）才是主受益者。
- `build_model11_sos.sh`：runtime 編譯與 .so 連結加 `-fopenmp`。
  - **踩雷紀錄**：此 LLVM clang 的 `-fopenmp=libgomp` **不會定義 `_OPENMP`、pragma 被靜默忽略**（一開始白做）。正解是**純 `-fopenmp`**（會定義 `_OPENMP`、生成 `__kmpc_*`），連結再指向 build 樹內的 libomp：`-L<...>/build/runtimes/runtimes-bins/openmp/runtime/src -Wl,-rpath,...`（從 `clangxx_bin` 推導）。
  - 驗證 .so 有 `nm | grep kmpc` 才算 pragma 真的編進去。
- 預設行為**完全不變**：`POSIT_OMP_THREADS` 未設=1=序列=bit-identical；現有 `--jobs 25` CNN 指令不受影響、不會 oversubscribe。
- 實測：`build_gpt2_nqdq_p8e1` 重 build 後，`POSIT_OMP_THREADS=24` 跑 GPT-2 量到 **%CPU≈2000（≈20 核生效）**，平行確實啟動（修正 `-fopenmp` flag 前是 ~100%＝沒生效）。

---

## 2026-06-18

### run_gpt2_text_eval.py：score 模式加時間 + 進度追蹤

- `run_score` 新增 `--progress N`（預設 100）：每預測 N 個 token 印一行 running `ppl / top1 / tok/s / elapsed / ETA`（類比 CNN runner 的 `--progress`）。
- 最終輸出新增 `elapsed_sec` 與 `tokens_per_sec`。
- 原本完全沒有計時/進度，posit .so 跑長語料時看不出是否在動。

**平行化現況（查證）**：posit 路徑**完全單執行緒**——`posit_runtime.cpp` 無 OpenMP/thread（grep=0），build script 無 `-fopenmp`，`.so` 未連 libomp/libgomp。且 score 是自迴歸逐 token（token N 依賴 N-1 的 KV cache）→ **token 層無法平行**。

### 方法 A 平行 wrapper：`run_gpt2_text_eval_parallel.py`（新增）

- 把語料 token 流切成**不重疊的獨立 window**（`--window`，預設 128），每個 window 從乾淨 KV cache 各自計分，多 process（`--jobs`）平行跑，最後彙整 NLL/top1。= GPT-2 版的 CNN multi-sample 平行，**免改 runtime / 免重編**。
- 重構：`run_gpt2_text_eval.py` 抽出 `score_token_ids(runner, token_ids, progress)` 回傳原始 nll 加總；`run_score` 改成包它。wrapper import 這個函式。
- 用 `multiprocessing spawn`（避免 fork-after-native-load 問題），每個 worker 在 initializer 各自 `build_runner` 一次。`--jobs 1` 走單 process（給你比「平行 vs 不平行」精度差異用）。
- 參數：`--model --text-file --window --jobs --max-tokens --limit-windows --progress`（progress 以「完成 window 數」為單位印 running ppl/top1/tok-s）。
- 驗證：f32 ONNX jobs=2 window=20 跑通，彙整正確。
- **注意**：independent-window ppl 會比單一連續序列略高（每 window 前幾 token context 短）；做 posit-vs-f32 相對比較時，只要各模型用**相同 `--window` 與 `--max-tokens`** 即公平。

---

## 2026-06-15

### GPT-2 nqdq 路徑修正（動態 shape 模型支援，進行中）

GPT-2（動態 batch/seq、整數 shape 運算、scalar/mask broadcast）暴露出 posit pipeline 原本只為靜態 shape CNN 設計的數個缺口。逐一修正：

**Fix 1 — onnx-mlir-opt 缺 tensor dialect 註冊**
- 症狀：posit→krnl 報 `Dialect 'tensor' not found for custom op 'tensor.from_elements'`（GPT-2 動態 shape 會 emit `tensor.from_elements`）。
- 檔案：`src/Tools/onnx-mlir-opt/onnx-mlir-opt.cpp` — 加 `#include <mlir/Dialect/Tensor/IR/Tensor.h>` 並在 `registry.insert<>` 加 `mlir::tensor::TensorDialect`。

**Fix 2 — i64 整數（shape）運算被誤轉 posit**
- 症狀：`memref.cast: memref<i64> → memref<*xi8> cast incompatible`。`POSIT_FORCE_NQDQ` 把 i64 的 Add/Sub/Mul/Reshape（shape/index 運算）也轉成 `posit.add`，但 posit 只給浮點，posit→krnl 把 i64 當 posit-i8 儲存→爆。
- 檔案：`src/Conversion/ONNXToPosit/ONNXToPosit.cpp` — 新增 `opIsFloatTyped` helper（result element type 是否為 FloatType）；所有 numeric op 的 dynamic legality 加上「非 float 即 legal（留給 onnx→krnl 整數運算）」（14 處）。驗證：i64 的 Add 現在保持 `onnx.Add ... tensor<i64>`，無 posit op 作用在 i64。

**根因確認 — posit 二元 op 禁止 broadcast（scalar/mask 路徑）**
- 症狀：`memref.dim op operand must be non-0-ranked, got memref<i8>`（指向 `posit.sub`）。
- 追根：`PositOps.td` 的 `Posit_AddOp/SubOp/MulOp/DivOp/ClipOp/ReluOp` 都帶 `SameOperandsAndResultType` trait → 強制操作數與結果**同型**。GPT-2 attention mask `(1-mask)*-10000` 是 scalar(rank-0) 對 tensor(rank-4) 的 broadcast，onnx→posit 為了滿足 trait，插入 `unrealized_conversion_cast %scalar : tensor<!posit>(rank0) → tensor<?x1x?x?x!posit>(rank4)`（CLAUDE.md 註解的 "posit.add requires identical operand/result types"）。posit→krnl 對這個假 rank-4 操作數推 broadcast 輸出維 → 對底層 rank-0 `memref<i8>` 做 `memref.dim` → 驗證失敗。
- runtime **已支援** broadcast：`posit_runtime.cpp::offset_with_broadcast`（含 rank-0 scalar，NumPy 規則）。所以缺口純在 dialect/lowering 端。
- 已做的部分修正（保留，最終解的一部分）：`PositToKrnl/Pattern/Math.cpp` 的 `PositBinaryOpLowering` 改用 NumPy 對齊逐維推 broadcast 輸出維（scalar/低 rank 操作數自動跳過，不再對 rank-0 做 memref.dim）。但因 trait 仍在，onnx→posit 仍插假 rank-4 cast，strip 後又回 rank-0，所以還沒通。
**Fix 3 — posit 二元 op 支援 broadcast（已解）**
- `PositOps.td`：`Posit_Add/Sub/Mul/Div` 移除 `SameOperandsAndResultType`（只留 `Pure`），assemblyFormat 改 `functional-type(operands, results)` 以印出不同操作數型別。
- `ONNXToPosit/Pattern/Math.cpp`：新增 `castOperandToPositKeepShape`（只轉 element type、保留操作數自然 rank；已是 posit tensor 則原樣保留）。Add/Sub/Mul/Div 的 lowering 不再把操作數硬轉成 result type（移除 4 處 rank-equalizing `UnrealizedConversionCastOp`）。
- `PositToKrnl/Pattern/Math.cpp`：`PositBinaryOpLowering` 改 NumPy 對齊逐維推 broadcast 輸出維（scalar/低 rank 操作數自動跳過，不對 rank-0 做 memref.dim）。
- 重 build：因改 `.td`，重編 `onnx-mlir-opt` + `onnx-mlir`（會重生 op class）。
- 驗證：`posit.sub %scalar(rank-0), %t(rank-4) -> rank-4` 正常生成，posit→krnl 通過，**GPT-2 nqdq build 全程成功**（產出 nqdq-p8e1 / nqdq-f32 / qdq-f32 三個 .so）。

**端到端驗證（2026-06-15）**
- `gpt2-hf-debug-nqdq-f32.so` 經 `run_gpt2_text_eval.py --mode score` 跑通（ppl/top1 算得出，OMTensor + KV-cache 介面正常）→ GPT2_NOTES 風險 #2 解除。
- `nqdq-p8e1.so` 可載入、posit 計算路徑可執行，但軟體模擬 posit 極慢（每元素 decode/encode），完整 WikiText-2 由使用者自行測。
- 注意：CNN 路徑（MobileNetV2/ResNet18）不受影響——broadcast 改動只在操作數 rank/shape 不同時生效，CNN 的 bias 仍走既有 `broadcastPositConstantIfNeeded`（已 full-broadcast 成 output shape，操作數同型，新 helper 直接原樣返回）。
- 後續若 GPT-2 要開 ALPS / 加速，再評估 LayerNorm/Softmax/Gelu 的精度與 quire。

---

## 2026-06-14

### New: 新增 posit 格式 p10~p15（e0/e1/e2）+ 為 p4~p15 開放 ALPS

**動機**: bit-width sweep（MobileNetV2 何時恢復精度）需要 p8 與 p16 之間的格式。原本只有 p4-p9, p16, p32。

**新增格式 p10-p15 × e0/e1/e2（18 變體）** — 跨 4 檔，用 `/tmp` 生成器照 p9 模式插入：
- `posit_runtime.cpp`：`FmtP{10..15}E{0..2}` 別名（UniversalFmt, int16 儲存）、ENABLE 區塊、`DEFINE_POSIT_RUNTIME_EXPORTS` 實例化
- `run_time.cpp`：`OutType` enum、字串解析、format→string、`p{N}e{e}_bits_to_double`（universal + fallback）、inferByType/benchmark dispatch
- `RegisterPasses.cpp`：`isSupportedPositFormat` 放行 nbits 10-15（es≤2）
- `build_model11_sos.sh`：`runtime_format_define_for` 映射 `POSIT_RUNTIME_FMT_P{N}E{e}`
- 驗證：p10e1/p12e2/p15e0 通過 onnx-mlir-opt；p10e1 .so build + 單張推論成功（class-0 圖預測正確，p8 是崩的→**p10 已露恢復曲線**）

**為 p4~p15 開放 ALPS（先準備，sweep 階段用 plain）**:
- `posit_runtime.cpp::fmtSupportsRuntimeOutputAlps`：改用 `Fmt::nbits`（新增 `UniversalFmt::nbits` static member）→ 放行 nbits 4-15（原本寫死 p8）。activation runtime/offline output ALPS 對 p4-p15 生效
- `Math.cpp::universalEncodeRawDispatch` / `universalRoundToDoubleDispatch`：加 nbits 4-15 × es 0,1,2 的 case（weight const ALPS 編碼/scoring）
- `posit_runtime.cpp` calib 工具 `dumpRuntimeOutputAlpsCalibToPath`：加 p4-p15 的 `emitRuntimeOutputAlpsCalibRowForFormat`（offline calibrate 支援）
- `time_model11_dataset_parallel.sh`：output-alps-auto skip gate 從 `^p8e[0-2]$` 放寬到 `^p([4-9]|1[0-5])e[0-2]$`
- `posit_runtime.cpp::qalignFormatName`：原本只認 p8（其他回 "unknown"，導致 p9-p15 的 collect/calibrate/offline format key 全撞）。補上 p4-p15 + p16/p32 的 case。**這是透過實測 calib 工具抓到的 gap**。
- 注意：GP（generalized posit）仍 p8-focused；p10-p15 的 ALPS 走標準 asinh companding（非 GP）。需重 build onnx-mlir-opt + .so。

**端到端驗證**: p10e1 plain .so（單張推論正確）+ p10e1 const-ALPS .so（build 成功）+ calib 工具 emit p10e1 ALPS row（compand_mode=1, theta=0.5）。p4-p15 的 plain 與 ALPS 路徑都通。

---

## 2026-06-08

### New: ALPS theta 選參指標 sqnr / cosine（取代 MAE，保留判別訊號）

**動機**: bias 修正後 MobileNetV2 nqdq-p8 脫離常數崩塌（特徵恢復輸入相關），但 Top1 仍 ~1.3%。診斷發現 GAP 特徵被壓到極小稀疏（max 1.3 vs f32 5+）。根因：ALPS 選 theta 的指標（`posit_runtime.cpp::runtimeOutputAlpsScoreForMeta`）是 **MAE over all values** = `mean(|q(x)-x|)`，ReLU6 後 ~98% 是零，這個指標被零主導 → 挑「保零、壓爛大值」的 theta → 判別訊號（大值）被壓掉。

**新增（`merge_layer_ranges_to_clamp.py`，純 python，clamp 路徑用，不需重 build）**:
- `--theta-mode sqnr`：grid-search theta 最大化 SQNR = `sum(x²)/sum((q(x)-x)²)`（訊號功率加權，大值主導）
- `--theta-mode cosine`：最大化 cosine similarity（保留 channel pattern）
- 用**真實 p8e1 round-trip**（256 值查表 + asinh companding）在收集的 samples 上算
- sqnr/cosine 模式下 clamp 放寬到 [p0.1, p99.9]（asinh 自己有界，只切極端 outlier）

**效果（GAP 特徵 max）**: MAE/median theta → 1.5；sqnr theta → 1.9（略升，但仍遠小於 f32 的 5+）。完整 Top1 eval 進行中。

**註**: 這對應 ALPS 論文用 signal-aware 指標（非單純 MAE）選 companding 參數的概念。

### Change: C++ runtime/offline ALPS 選參指標 MAE → NSR(=1/SQNR)（需重 build）

把 `posit_runtime.cpp::runtimeOutputAlpsScoreForMeta` 從 **MAE**（`mean(|q(x)-x|)`，被 ReLU6 的零主導）改成 **NSR = `sum((q(x)-x)²)/sum(x²)`**（= 1/SQNR）。lower=better，保留原本 minimize + minGain 邏輯（maximize SQNR == minimize NSR）。

- 影響 **offline calibrate 工具**與 **sampled 線上搜尋**（兩者都用這個 score）→ offline/sampled 的 theta 改由 SQNR 選，與 clamp 的 python SQNR 一致。
- **不影響 weight**（build 時 weight 用 `Math.cpp::meanAbsError`，weight 稠密、MAE 沒問題，未改）。
- **需重 build .so**（posit_runtime.cpp 編進 .so + output_alps_calibrate 工具），不需重 build onnx-mlir-opt。
- NSR scale 與 MAE 不同，`min_gain` 門檻可能要重調（NSR 典型 0.005~0.1）。

### Change: build-time WEIGHT ALPS 選參指標 MAE → NSR(=1/SQNR)（需重 build onnx-mlir-opt）

把 `Math.cpp` 的 weight ALPS scoring `meanAbsError` 改成 `noiseToSignalRatio`（NSR = `sum((a-b)²)/sum(a²)` = 1/SQNR），兩個呼叫處（directScore、candidate score）同步改。lower=better，保留 minimize + minGain。

- 讓 weight 的 const ALPS theta 也由 SQNR 選（與 activation 一致）。
- 注意：weight 稠密（非 ReLU6 稀疏），MAE 對 weight 本來就沒被零主導，**預期增益小**；此改動主要為一致性 + 試驗。
- **需重 build onnx-mlir-opt + .so**（這是 ONNXToPosit build-time pass）。
- min_gain 門檻（NSR scale）同樣可能要調小（已在 build 指令用 0.0001）。

**SQNR 實測（clamp activation，5000 張）**: median 1.38% → SQNR 1.54%（**僅微幅**）。證實瓶頸是跨層 per-element 精度累積，非單層 theta 指標。weight SQNR 預期同樣邊際。

## 2026-06-02

### Fix: qdq p8e1/p8e2 missing p8e0 helper symbol

**File**: `onnx-mlir/src/bash/build_model11_sos.sh`

**Symptom**:
```
dlopen failed: undefined symbol: _mlir_ciface_posit_dequantize_linear_axis_f32_p8e0
```
當使用 `--posit-source qdq --runtime-format-scope single` build p8e1 或 p8e2 時發生。

**Root cause**: case 分支只對 `p4e*|p5e*|p6e*|p7e*|p9e*` 加入 `-DPOSIT_RUNTIME_FMT_P8E0=1`，漏掉 p8e1/p8e2。QDQ per-axis weight dequantization 在 single format scope 下仍需要 p8e0 helper symbol。

**Fix**:
```bash
# Before
p4e*|p5e*|p6e*|p7e*|p9e*)
# After
p4e*|p5e*|p6e*|p7e*|p9e*|p8e1|p8e2)
```

**Action required**: qdq-p8e1.so / qdq-p8e2.so 需重新 build。

---

## 2026-06-05

### ROOT CAUSE FIX: bias 用 per-channel ALPS 但 Add kernel 只 per-tensor 解碼

**這是 MobileNetV2 nqdq-p8 崩塌（Top1=0）的真正根因。**

**症狀**: nqdq-p8e1 Top1=0%、預測幾乎常數。逐層 trace 對照 p16（正常）vs p8（崩）發現：expand gemm 後的 bias add，p16 把均值從 -6.76 拉到 +0.05（bias 貢獻 +6.8 置中），但 p8 只到 -5.79（bias 只貢獻 +0.3）→ **bias 損失 95%** → activation 均值停在很負 → ReLU6 把幾乎全部歸零 → feature 崩塌。

**根因**: 
- bias 常數在 build 時被 `shouldUsePerAxisConstMetadata` 判為 per-axis（per-channel）→ 編成 per-channel ALPS（每 channel 不同 theta）
- 但消費 bias 的 elemwise Add kernel（`posit_runtime.cpp:4968`）**只做 per-tensor metadata lookup**（`lookupTensorGPMetadata`），不做 per-channel
- → bias 用 per-channel theta 編碼、用 per-tensor metadata 解碼 → **encode/decode 不匹配** → bias 解成垃圾（損失 95%，不是精度 4% rounding）

**為什麼是 95% 不是 4%**: 純 p8 rounding 對 bias≈6.8 誤差才 ~4%；95% 損失證明是編解碼方式對不上（companding 不一致），不是精度不足。

**Fix 第一版（per-tensor ALPS）— 失敗**: 把 bias 從 per-axis 改 scalar（per-tensor）ALPS。trace 顯示 add 仍只 +0.32（bias 還是損失 95%）。原因：bias 的**per-tensor 動態範圍極端**（單一 bias vector 內達 137~5197 倍，值範圍 ±53，最小非零 0.0045），per-tensor ALPS grid search 挑了 theta≈theta_min(1e-4) 想涵蓋整個範圍，反而把所有 bias 值推進 posit 極小區壓爛。

**Fix 第二版（DIRECT，正確）**: bias-only 常數強制 **direct posit（完全不套 ALPS）**。
- `Math.cpp` 新增 `isBiasOnlyConstant()`：所有 use 都是 conv_bias/gemm_bias 才回 true
- `createCompactPositConstantIfEnabled`: bias → 跳過 scalarDecision（也跳過 per-axis）→ 落到 direct 編碼分支
- direct posit 是浮點式相對精度，每個 bias 值各給 ~幾% 精度（6.835→7 誤差 2%），足夠置中 activation
- 對應 INT8 把 bias 存 int32（高精度），不壓低位元

**為何 direct 對、ALPS 錯**: bias 動態範圍 5197 倍，一個 ALPS theta 蓋不住（顧大值就壓爛小值，反之亦然）。posit direct 本身就是 tapered 相對精度，天然適合高動態範圍的 bias。weight 維持 per-channel ALPS（動態範圍小、且 conv/gemm 走 per-channel decode）。

**需重 build**: onnx-mlir-opt + nqdq .so。

### Fix 第三版（真正根因）：bias 被「預先廣播」成完整常數，認不出

**第二版（direct）為何還是失敗**: trace 顯示即使原始 [16] bias 改 direct，add 仍只 +0.28。查 IR 發現——bias [16] 在 lowering 時被**展開廣播成 [1,16,12544] 完整常數**（200704 值）餵給 posit.add。這個展開常數**不是 Conv input[2]**（是 Add 運算元）→ `isBiasOnlyConstant` / `getConvOrGemmPerAxisConstantUseRole` 認不出（回 nullopt）→ 落到 scalar ALPS path → 被壓爛。我第一、二版只修到原始 [16] bias，但 add 實際用的是展開後的常數。

**根本原則 Fix**: **只有 Conv/Gemm WEIGHT 才套 ALPS，其他常數一律 direct**。
- `Math.cpp` 新增 `hasConvGemmWeightUse()`：常數有 conv_weight/gemm_weight use 才回 true
- `createCompactPositConstantIfEnabled`: `scalarDecision` 改成 `if (!perAxisDecision && isWeight)` ——非 weight 常數（含展開 bias）跳過 ALPS → direct
- 展開的 bias [1,16,12544] 餵 posit.add，無 weight use → isWeight=false → direct ✅

**驗證**: p16-plain 正常（bias p16）、p8+ALPS-bias 崩（bias 被 ALPS 壓爛）、INT8 85%（bias int32）。三方一致指向 bias 需要高精度/direct。bias 數值實測 ±53、per-tensor 動態範圍 5197 倍。原始 bias [16] 被廣播折疊成 [1,16,12544] 常數（200704 值）。

**v3 trace 驗證成功（2026-06-05）**: biasdirect_v2 build，第一個 bias add 從「gemm -5.18 → add -4.90（bias +0.28）」恢復成「gemm -5.19 → add +1.99（bias +7.18）」——bias 貢獻接近 p16 的 +6.8，add 均值轉正，ReLU6 不再歸零。崩塌根源解決。完整 Top1 eval 進行中。

## 2026-06-04

### Fix + New: clamp collect 改 per-part raw samples + merge + 整合 pipeline bash

**Problem**: 先前 `POSIT_COLLECT_LAYER_RANGES_FILE` 在 program exit 寫單一檔（atexit）。但 dataset runner 是「一張圖一個 process」（run_time.cpp 的 `--image` 只收單張），parallel 跑法下每個 process 覆蓋同一檔，500 張只留最後 1 張統計 → 壞的。

**Fix（posit_runtime.cpp，需重 build .so，不用 onnx-mlir-opt）**:
- `writeLayerRangeCSV` 改成寫 **raw reservoir 取樣值**（可 merge），不再寫 percentile。格式：`key,op,total_seen,n_samples,v0,v1,...`
- reservoir 上限 `kLayerRangesReservoirSize` 從 50000 降到 8000（per-image per-process，避免 part 檔過大）
- 設計改為 mirror ALPS offline：每張圖寫自己的 part 檔，再由 Python merge

**新增檔案**:
- `ImageNet100/merge_layer_ranges_to_clamp.py`：讀所有 `part_*.csv` raw samples → 每 key 合併（global-cap reservoir）→ 算 percentile → clamp [lo,hi] + theta → 輸出 clamp CSV
- `onnx-mlir/src/bash/run_clamp_pipeline.sh`：一鍵 pipeline wrapper
  - Step1 COLLECT：對指定 posit suffix 跑 N 張（parallel，f32 math），每張寫 `parts/part_i.csv`
  - Step2 MERGE：呼叫 merge python → `clamp_<suffix>/clamp.csv`
  - Step3 EVAL：用 `POSIT_OUTPUT_CLAMP_FILE` 跑完整 dataset eval（delegate 給 time_model11_dataset_parallel.sh）

**使用方式（env 控制）**:
```bash
CLAMP_DIR=/home/lai/onnx_mlir/ImageNet100/clamp_mobilenetv2_v1 \
CLAMP_COLLECT_LIMIT=500 \
CLAMP_PERCENTILE=99 \
CLAMP_THETA_MODE=auto \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/run_clamp_pipeline.sh \
  --model-name imagenet100_mobilenetv2 \
  --out-dir <nqdq build dir> \
  --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation \
  --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py \
  --shape 1x3x224x224 \
  --suffixes nqdq-p8e1 \
  --jobs 25 --limit 5000 \
  --warmup 0 --iters 1 --no-benchmark --quire off \
  --output-alps-auto off --record-preds on
```
其他 env: `CLAMP_COLLECT_JOBS` `CLAMP_SUFFIX` `CLAMP_MARGIN` `CLAMP_FIXED_THETA` `CLAMP_GAMMA` `CLAMP_SYMMETRIC` `CLAMP_SKIP_COLLECT`（重用既有 clamp.csv）`CLAMP_VERSION_A`（=1 只輸出 key,lo,hi，配合 ALPS 使用）。

### clamp 與 ALPS 並用：用 Version A

ALPS 與 clamp 目的重疊但作用不同：ALPS 設 companding theta（精度分佈），clamp 夾範圍（防漂移）。修 ReLU6 collapse 的關鍵是夾範圍。

- **ALPS + clamp Version A**（推薦）：ALPS 編碼，clamp 只夾範圍並用既有 ALPS meta re-encode（theta=0 → 不覆蓋 ALPS theta）。merge 加 `--version-a`，pipeline 設 `CLAMP_VERSION_A=1`
- **ALPS + clamp Version B**（衝突）：clamp 自己的 theta 會覆蓋 ALPS theta（applyPositOutputClamp 的重註冊邏輯），ALPS 白做
- merge script 新增 `--version-a`：只輸出 `key,lo,hi`（loader 視為 Version A）

### Fix: clamp 不可套用在最終分類器 logits 層

**症狀**: nqdq + weight ALPS + clamp Version B 跑 100 張，Top1 仍 ≈0%。

**診斷（非程式 bug，是套用範圍錯）**: clamp.csv 的最終分類器 gemm2d（key `6244868847920025062`, 輸出 100 類）被夾到 `[-20, -1.31]`。每張圖勝出類別的 logit 落在跨 100 類分佈的最高 1%，p99 上界把識別正確答案的峰值砍平 → argmax 全亂。INT8 不會這樣因為它最後一層 calibration 涵蓋完整 logit 範圍。

**Fix**: merge script 新增 `--exclude-keys`（pipeline env `CLAMP_EXCLUDE_KEYS`），排除分類器層不 clamp。MobileNetV2 分類器 key（此 build）= `6244868847920025062`，可用 `nm + DOT_PROBE` 找出 gemm2d 輸出=100 的 key（不同 build key 不同）。

**通則**: 最終 logits 層永遠不要 clamp（需要完整範圍取 argmax）。

### New: theta-mode median（修 MobileNet 高動態範圍 activation）

**數據發現**: 收集的 per-layer activation 統計顯示——中位數 |x|≈0.75~3.25（≈posit 甜蜜點），但 max 達 40~64，**max/median = 13~53 倍**（MobileNetV2 的 6× expand 層 + ReLU6 造成的極端動態範圍）。

**問題**: `theta=1/max`（auto/from_max）被離群值綁架，把典型值 x≈1 壓成 `asinh(1/64)≈0.016`，推離 posit 甜蜜點 → 帶判別訊號的典型值失去精度。ResNet 無此問題（動態範圍緊）。INT8 用 per-tensor uniform affine 不受影響。

**Fix**: merge script 新增 `--theta-mode median`（pipeline `CLAMP_THETA_MODE=median`）+ `--theta-target`（pipeline `CLAMP_THETA_TARGET`，預設 1.0）。
- `theta = sinh(target) / median_abs`，讓典型值落在 posit 甜蜜點（|y|≈1），離群值靠 asinh 飽和 + clamp [lo,hi] 上限。
- 實測 theta 提升 6~12 倍（0.04→0.52 等）。

**驗證**: per-channel activation（方向 B）**非必要**——INT8 QDQ 的 activation 是 per-tensor（`per_channel=True` 只作用在 weight）。INT8 per-tensor 能到 85%，所以問題在 per-tensor 編碼方式（posit tapered+theta vs uniform affine），不在粒度。A（median theta）是對的方向。

**Action required**: 用前需重 build nqdq .so（現有為 6/1 舊版，不含新 collect 程式碼）。

### Fix: clamp Version B metadata 一致性（已驗證修正）

**Bug（確認屬實，靠程式碼追蹤）**: `applyPositOutputClamp` Version B 用新 theta re-encode posit bits，但 producing kernel（conv2d:6005 / gemm / clip:6166）已先註冊自己的 outMeta。下游 op 透過 `lookupTensorGPMetadata(tensorMetaPtr(input))` 解碼，拿到舊 outMeta，但 bits 是新 theta 編的 → 解碼錯誤。

**Fix（posit_runtime.cpp，applyPositOutputClamp 結尾）**:
```cpp
if (versionB)
  registerTensorGPMetadata(tensorMetaPtr(t), encodeMeta);
```
re-encode 後重新註冊 encodeMeta，下游用相同 meta 解碼。`tensorMetaPtr` 回傳 data pointer（穩定），kernel 註冊與 clamp 重註冊命中同一 key，clamp 在 kernel 之後跑所以正確覆蓋。

**驗證方式**: 程式碼追蹤 + compile OK。尚未 runtime 實測（需重 build .so）。

### New: POSIT_STORE_DQ_AS_POSIT — qdq Variant B persistent（INT8 → posit bits 持久存儲）

**Problem**: 在 qdq source 路徑中，`posit_dequantize_linear_f32_ref` 輸出 f32，posit bits 只是函式內部瞬時的（transient）中間值。Layer boundary 實際存儲的是 f32，不像 INT8 QDQ 是實際存 int8 bits。

**Fix（build-time env var）**:
- **`POSIT_STORE_DQ_AS_POSIT=1`**（或 `ONNX_MLIR_POSIT_STORE_DQ_AS_POSIT=1`）
- 設定後：`boundaryF32Flow=true` 的 DQ 節點改用 `posit_dequantize_linear_ref`（輸出 posit bits，不是 f32），再插入 `posit.to_f32` 讓下游 onnx.Conv 收到 f32
- 效果：`INT8 → posit_dequantize_linear_ref → [posit bits buffer, 跨 layer 持久存在] → posit.to_f32（當 Conv 需要時才 decode）→ f32`
- 與 INT8 QDQ 的對比：INT8 存 int8 bits；本方法存 posit bits（p4=4bits/value, p8=8bits/value）

**Changed files**（需重 build `onnx-mlir-opt`，再重 build `.so`）:

| 檔案 | 改動位置 | 說明 |
|------|----------|------|
| `src/Conversion/ONNXToPosit/ONNXToPosit.cpp` | `runOnOperation` | 讀 `POSIT_STORE_DQ_AS_POSIT` env var；pass 給 `populateONNXToPositConversionPattern()` |
| `src/Conversion/ONNXToPosit/Pattern/Math.cpp` | `ONNXDequantizeLinearOpLowering` | 新增 `preferStoreAsPosit_` member；在 `boundaryF32Flow=true` 路徑，若 flag 設定，改用 `positOutType` 輸出（呼叫 `posit_dequantize_linear_ref`），再 `toFinalOutputType` 插入 posit.to_f32；`populateONNXToPositConversionPattern()` 簽名加新參數 |

**Parameter 設定位置**: **Build time**

### New: POSIT_PREFER_DIRECT_FROM_QDQ — qdq Variant A（直接 posit，不經 INT8）

**Problem**: qdq source 的 DequantizeLinear 一律走 INT8 → posit → f32 路徑（Variant B）。若想讓 activation 從 orig_f32 直接 posit round-trip 而不經過 INT8 量化（Variant A），需要改 lowering 邏輯。

**Root cause**: `ONNXDequantizeLinearOpLowering::matchAndRewrite` 中，
`rewriteDirectFromQSource()` 只在 `!boundaryF32Flow`（DQ 輸出是 posit type）時被呼叫。對 qdq source 而言，DQ 輸出永遠是 f32（下游 `onnx.Conv` 需要 f32），所以 `boundaryF32Flow=true`，direct posit 路徑永遠被跳過。

**Fix（build-time env var）**:
- **`POSIT_PREFER_DIRECT_FROM_QDQ=1`**（或 `ONNX_MLIR_POSIT_PREFER_DIRECT_FROM_QDQ=1`）
- 設定後：即使 `boundaryF32Flow=true` 也嘗試 `rewriteDirectFromQSource()`
- 效果：`orig_f32 → posit.from_f32(qalign_key) → posit → posit.to_f32 → f32`（posit round-trip，不觸碰 INT8）
- 下游 `onnx.Conv` 仍然接收 f32，只是值來自 posit-rounded orig f32 而非 int8 linear DQ

**Changed files**（需重 build `onnx-mlir-opt`，再重 build `.so`）:

| 檔案 | 改動位置 | 說明 |
|------|----------|------|
| `src/Conversion/ONNXToPosit/ONNXToPosit.cpp` | pass class + `runOnOperation` | 新增 `preferDirectPositFromQDQ` bool member；讀 `POSIT_PREFER_DIRECT_FROM_QDQ` env var；pass 給 `populateONNXToPositConversionPattern()` |
| `src/Conversion/ONNXToPosit/Pattern/Math.cpp` | `ONNXDequantizeLinearOpLowering` | 新增 `preferDirectPositFromQDQ` parameter + member；condition 改為 `(preferDirectPositFromQDQ || !boundaryF32Flow) && preferDirectF32FromQDQ`；`populateONNXToPositConversionPattern()` 簽名加新參數 |

**Parameter 設定位置**: **Build time**（在跑 onnx-mlir-opt 的 shell 環境中設定，由 build_model11_sos.sh 呼叫 onnx-mlir-opt 時生效）。

### New: self-calibrated per-layer clamp + ALPS re-encode (Version B)

**Problem**: nqdq-p8e1 MobileNetV2 Top1≈0%，posit 精度誤差跨層累積，ReLU6 將 ~98% activation 歸零。目標：不依賴 QDQ ONNX，從 nqdq-f32 自行收集每層 activation 分佈，推算 calibrated range + ALPS theta，在每個 conv/gemm 輸出後做 clamp + ALPS re-encode。

**實作（posit_runtime.cpp）**:

1. **`PositOutputClampEntry` 新增 theta/gamma 欄位** (`posit_runtime.cpp:242`)
   - 原本：`struct PositOutputClampEntry { float lo, hi; }`
   - 現在：`struct PositOutputClampEntry { float lo, hi, theta=0, gamma=1; }`
   - theta > 0 → Version B（新 ALPS re-encode）；theta == 0 → Version A（原有行為）

2. **`applyPositOutputClamp` 支援 Version B** (`posit_runtime.cpp` 附近 3740 行)
   - 若 clamp.theta > 0：decode（用原 metadata）→ clamp → re-encode（用新 theta/gamma 的 ALPS TensorGPMetadata）
   - 這模擬了 INT8 的 per-layer requantize：每層輸出被壓縮進 calibrated range 再重新編碼

3. **`POSIT_COLLECT_LAYER_RANGES_FILE` 收集機制**
   - 新增 `PositLayerRangeReservoir`：per-key reservoir sampler（最多 50000 samples，Vitter Algorithm R）
   - `collectLayerRangeValues<Fmt>()` 在每個 conv2d / gemm 計算後收集 decoded output values
   - `writeLayerRangeCSV()`：program exit 時寫出 `key,op,n_samples,min,p50,p95,p99,p99.9,max`
   - 新增 `#include <random>`

4. **`loadPositOutputClampTable` 解析 optional theta/gamma 欄位**
   - 原格式：`key,lo,hi`（Version A）
   - 新格式：`key,lo,hi,theta,gamma`（Version B）
   - 自動計算並打印 versionA 和 versionB entry 數量

**新增 Python scripts**:

- `ImageNet100/gen_nqdq_layer_clamp_csv.py`：從 INT8 QDQ ONNX 萃取 calibrated range（依賴 QDQ model）
- `ImageNet100/gen_nqdq_layer_clamp_from_ranges.py`：從 `POSIT_COLLECT_LAYER_RANGES_FILE` 輸出推算 clamp + ALPS theta（不依賴 QDQ model）
  - 參數：`--percentile`（預設 99）、`--margin`、`--theta-mode`（auto/from_max/fixed）、`--symmetric`

**使用流程**:

```bash
# Step 1: 收集 nqdq-f32 的 activation 分佈（乾淨的 f32 統計）
POSIT_QOP_F32_MATH=on \
POSIT_COLLECT_LAYER_RANGES_FILE=/tmp/mobilenetv2_layer_ranges.csv \
bash time_model11_dataset_parallel.sh ... --suffixes nqdq-p8e1 --limit 500

# Step 2: 從統計推算 clamp + theta
python3 gen_nqdq_layer_clamp_from_ranges.py \
    --ranges /tmp/mobilenetv2_layer_ranges.csv \
    --output mobilenetv2_nqdq_selfcalibrated_clamp.csv \
    --percentile 99 --theta-mode auto

# Step 3: 用 Version B clamp 跑 posit eval
POSIT_OUTPUT_CLAMP_FILE=mobilenetv2_nqdq_selfcalibrated_clamp.csv \
bash time_model11_dataset_parallel.sh ... --suffixes nqdq-p8e1 --output-alps-auto off
```

**Status**: compile OK（2026-06-03），待測試。

### New: per-layer activation clamp for nqdq MobileNetV2

**Problem**: nqdq-p8e1 MobileNetV2 Top1≈0%。Posit 精度誤差導致 project conv 輸出超出 INT8 calibrated range，殘差連接後進入 expand conv → depthwise conv，最終 ReLU6 input 負漂嚴重，~98% activation 歸零。

**Approach**: 使用現有 `POSIT_OUTPUT_CLAMP_FILE` 機制（`posit_runtime.cpp:3706`），對每一層 conv2d / gemm3d 輸出，在計算後立即 decode → clamp to INT8 calibrated [lo, hi] → re-encode to posit。這模擬了 INT8 QDQ 的 per-layer QuantizeLinear 語義。

**實作（無 C++ 改動，只新增 Python 腳本）**:

- **腳本**: `ImageNet100/gen_nqdq_layer_clamp_csv.py`
  - 讀取 INT8 QDQ ONNX model（`imagenet100_mobilenetv2-int8-qdq.onnx`）
  - 找出每個 Conv → QuantizeLinear pair 的 scale/zp，換算 [lo, hi]
  - 讀取 DOT_PROBE CSV（`/tmp/probe_with_clips.csv`）取得 qalignKey
  - 按執行順序 match（共 52 個 conv2d + gemm3d ops，全部 OK）
  - 輸出 clamp CSV

- **產出 CSV**: `ImageNet100/mobilenetv2_nqdq_perlayer_clamp_v1.csv`
  - 52 個 entries：ReLU6 後的 expand/depthwise conv 輸出夾到 [0, 6]，project conv 夾到 INT8 calibrated 較寬範圍（例如 [-82.8, 79.6]）

**使用方式（runtime，不需重 build）**:
```bash
POSIT_OUTPUT_CLAMP_FILE=/home/lai/onnx_mlir/ImageNet100/mobilenetv2_nqdq_perlayer_clamp_v1.csv \
bash time_model11_dataset_parallel.sh ... --output-alps-auto off
```

**Status**: 待測試。預期可改善 nqdq-p8e1 Top1 超過目前 1.52%。

---

## 2026-05-21

### Fix: per-channel ALPS metadata fallback corruption

**File**: `onnx-mlir/src/posit_runtime.cpp`

**Symptom**: nqdq-p8e* MobileNetV2 logits 呈現週期性 pattern（n%4==0/1 全為 0，n%4==3 固定為 -0.96875）。Top1 接近 0%。

**Root cause**: MobileNetV2 有 52 個 Conv，總輸出 channel 17056，但 .so 只有 ~14536 個 per-channel metadata 註冊。缺少 metadata 的 channel 繼承了前一個錯誤 channel 的 ALPS/GP 參數。

**Fix**: 加入 `gTensorGPSpecialChannelMetaPtrs` / `gTensorGPAlpsChannelMetaPtrs` tracking set。缺失 channel 改為 fallback 到 direct/default posit，不再借用其他 channel 的 metadata。

**Action required**: 重新 build .so（不需重新 cmake）。

---

## 已知未解問題

### ReLU6 activation sparsity collapse（nqdq path）

**Symptom**: nqdq-p8e1 MobileNetV2 Top1=0%（sampled_v1 驗證）。

**Root cause**: posit 精度誤差使 activation 往負值漂移 → ReLU6 clip 後 ~98% 為零 → GlobalAveragePool 全零 → 分類失敗。ResNet18 不受影響（用 ReLU，無上界 clip）。

**Current direction**: 改用 qdq source path（`--posit-source qdq`），INT8 calibrated range 避免 unbounded drift。
