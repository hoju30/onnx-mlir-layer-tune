# Qwen3.5-0.8B 跨架構驗證：原版 native BC 成本表、排序與自動搜尋

日期：2026-10-15　腳本：`experiments/mixed_precision/run_qwen35_bc.py`（共用：`qwen35_common.py`、`run_native_bc_pipeline.py` 的 `qwen35_spec`）　輸出：本目錄

本輪**沒有修改 criterion**：C_i(w,a) = W_i(w) + A_i(a)，權重項為每個輸出 channel 的梯度平方均值 × 量化誤差平方和，激活項為原版變異 + 平均偏差項；梯度來自 FP32 task loss 的 native backward，`input_grad="total"`。搜尋沿用 MILP → 2-D Pareto → normalized chord knee，沒有人工資源預算，也沒有用 held-out 結果選配置。

## 結論（回答六個問題）

1. **Native 支援**：128 tokens 時，文字骨幹的 forward、backward、四種統計（weights／grads／ranges／errors）與混合精度編譯全部在 onnx-mlir 內完成，並通過正確性門檻。
   - 512 tokens 的 native forward 也通過；
   - 但 512 tokens 的統計 build 在這台 39 GB 機器上 OOM（`onnx-mlir-opt` 33 GB），所以 512 的 BC 分數改用已和 native 表核對過的 PyTorch 實作（§7）。

   實作擴充有三項：
   - 匯出 adapter：把 Gated DeltaNet 的 chunk 遞迴改寫成靜態 op；
   - 四條 backward 規則：Exp、Softplus、ReduceSum、depthwise Conv1D；
   - pipeline 的 `qwen35_spec`。
   三項都沒有改變 BC 的公式、正規化或量化語意（§2）。
2. **排序**：同層、同資源的決定性配對，BC 對了 934/1057（88.4%）；SQNR 參考是 944/1057，差距的 95% 區間 [−0.025, +0.003] 包含 0。
   - **較好**：activation-only（95.4%）、L0（98.8%）、Down 與 Gate／Up 之間的 int↔posit 選擇（91–94%）、Down 對 Gate 同格式（88.1%）。
   - **較差**：
     - Gate 對 Up 同格式（76.2%）、Down 對 Up 同格式（79.0%）；
     - wa 模式的 linear-attention 層（79.2%，SQNR 86.8%）；
     - L12（78.7%）、L22（81.8%）。
   - **錯誤集中在小效應**：BC 判錯的決定性配對，真實差異中位數 0.003 nat／token（全部決定性配對是 0.012）。123 個錯誤中有 42 個牽涉到「量化反而讓 loss 顯著下降」的候選。
3. **聯合量化**：
   - **有可重現的加總偏差，但只出現在大誤差的 int4 組合**：L23 的 Gate+Up int4 次可加 29%，L0 超可加 16%，兩半窗口一致。
   - **多數候選接近可加**：|交互項|／聯合 ΔNLL 的中位數 1.3%、p90 8.9%。
   - **加總失準不是 BC 錯誤的主因**：把真實的單支損失直接相加，就能排對 90–100% 的聯合配對；BC 只有 46–93%。BC 的聯合錯誤來自單支估計（尤其 Gate／Up 的相對敏感度）。
4. **完整 knee 配置**（final test，64 個窗口，native 編譯）：

   | 版本 | 搜尋節點 | 權重大小／BOPs（相對 int8） | PPL | ΔNLL vs FP32 [95% CI] |
   |---|---:|---|---:|---|
   | FP32 | — | — | 43.800 | — |
   | FFN-only | 72 | 0.826／0.877 | 44.120 | +0.0073 [+0.0036, +0.0108] |
   | FFN + attention | 186 | 0.799／0.841 | 44.728 | +0.0210 [+0.0151, +0.0267] |

5. **成本**：成本表 1854 s（31 分鐘，CPU，從 ONNX 開始，含所有編譯），搜尋 53 s，選配置總共約 33 分鐘。同一台機器上，SQNR 表外推需要約 36 小時 CPU（或約 2 小時 GPU）。
   - 128 tokens 時，BC 仍然適合作為低成本方法；主要成本是編譯（grads 608 s、errors 1015 s），不是執行。
   - 序列加長時，瓶頸是編譯記憶體：512 tokens 的 grads 編譯需要 > 33 GB，改用 -O1 則要 98 分鐘。
6. **建議**：結果支持保留原版 BC。
   - 這裡沒有一個在多數類別都穩定勝過 BC 的修正。
   - 512 tokens 時 BC 的排序幾乎不變（BC512 vs BC128 的 Spearman ≥ 0.978），在 512 truth 上決定性配對對 89.9%。

   可以驗證的新假說有兩個（§6），都還沒驗證：
   - Gate／Up 的相對敏感度在 wa 模式的 linear-attention 層估錯；
   - 小擾動時的負 ΔNLL（一階效應）讓非負 criterion 在低誤差區系統性失準。

## 1. 設定

**模型**
- `Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`，transformers 5.19.0（隔離於 `qwen35_assets/python_deps`）。
- 只用文字骨幹；vision tower、MTP 層與 MoE 都不在範圍內。
- 共 24 層：18 層 Gated DeltaNet（linear attention），6 層 gated full attention（layer 3、7、11、15、19、23）；hidden 1024，FFN 3584，vocab 248320，tied embedding。
- checkpoint 的投影權重以 **BF16** 儲存：bf16 權重在表中 B = 0（186/186），f16 幾乎無損。
- Qwen3 基準表的 hash 記在 `environment.json` 的 `qwen3_baseline`。

**搜尋粒度**
- 每個 MatMul 投影是一個節點，fused projection 不拆：
  - linear-attention 層：`in_proj_qkv`（q／k／v 合一，1024→6144）、`in_proj_z`、`in_proj_b`、`in_proj_a`、`out_proj`；
  - full-attention 層：`q_proj`（query 與 output gate 合一）、`k_proj`、`v_proj`、`o_proj`；
  - 每層 FFN 的 `gate_proj`、`up_proj`、`down_proj`。
- 共 18×8 + 6×7 = 186 個節點。
- 節點與參數的對應用權重 hash 確認（`node_mapping.json`）。

**固定 FP32（不搜尋）**
- embedding／tied LM head：2.54 億參數；
- 所有 RMSNorm、causal Conv1D（44 萬參數）、A_log／dt_bias；
- delta rule 的遞迴（decay、β、l2norm、狀態）、softmax、rotary、attention output gate，以及 chunk 內的三角求解。

**資料**（WikiText-2，模型自己的 tokenizer，不加特殊 token；token 範圍互不重疊，記在 `plan.json`）
- calibration：validation token 0–1279（10 窗口 × 128）；
- diagnostic：validation 1280–5119（30 × 128），只用於評估；
- final test：test 0–8191（64 × 128）；
- seq 512：calibration 8192–10239（4 × 512），diagnostic 10240–14335（8 × 512）。

**順序控制**（時間戳都在對應檔案中）
1. `plan.json` 先凍結；
2. 成本表；
3. 候選與 BC 分數凍結（`freeze.json`，06:03:39）；
4. SQNR 參考凍結（`sqnr_freeze.json`，06:32:32）；
5. 之後才跑 diagnostic forward。truth stage 會先核對 hash 才執行。

搜尋只讀成本表；final test 只在搜尋結果寫出之後評估，沒有重新選配置。

## 2. Native 支援與正確性（問題 1）

### 實作擴充（不改 BC 語意）

| 擴充 | 內容 | 驗證 |
|---|---|---|
| 匯出 adapter `qwen35_common.TextExport` | 把 HF 的 chunked delta rule 改寫成靜態 op（見下方說明）；另外用 slice 取代 `torch.chunk`（否則會產生動態 index），並以 `int(x.shape[i])` 固定形狀 | 與 HF FP32 logits rel_l2 8.1e-7，CE 相同；FP64 adapter 與 FP32 rel_l2 7.8e-7；ORT 9.5e-7 |
| backward 規則（`MixedPrecisionBackward.cpp`） | Exp、Softplus、ReduceSum（含 V11）、rank-3 depthwise Conv1D（以 [1,C,1,T] 走 2-D 規則，只算輸入梯度，因為 conv 權重不搜尋） | lit test `collect_stats_grads_qwen35_ops.mlir`（目錄內 21/21 通過）；tiny 隨機模型（3 linear + 1 full layer）對 FP64 autograd：中位數 2.0e-6、最大 8.1e-5 |
| pipeline | `qwen35_spec`（節點、資源、onehot、metrics、統計 build 的優化等級），`Spec.stats_levels` 覆寫 | 本節下方的端到端核對 |

adapter 對 delta rule 的改寫：
- 三角反矩陣 (I + A)⁻¹ 用區塊倍增計算；
- cumsum 用常數三角 MatMul 計算；
- 每個 chunk 的輸出用 Concat 收集。

adapter 是在 PyTorch 端重寫同一個函數，量化節點與 criterion 的定義都沒變。tiny check 與 lit test 由同時進行的另一個 session 加入，本 session 重新執行並確認通過。

### 正確性結果（全模型，128 tokens；`correctness.json`）

| 項目 | 結果 | 門檻 |
|---|---|---|
| forward logits，native vs torch FP32／FP64（10 窗口） | rel_l2 最大 2.19e-6／2.08e-6 | ≤ 1e-4 |
| 每窗口 summed CE 差（vs FP32） | 最大 0.0068 | ≤ 0.05 |
| backward：186 個權重、282 個 tap、372 個量化點的 per-channel Σg²、Σg、ΣdW²（2 窗口，vs FP64） | max-relative 中位數 6.8e-4、p99 6.6e-3、最大 1.4e-2（layer 23 的 `q_proj` 權重） | 中位數 ≤ 1e-3、最大 ≤ 5e-2 |
| 表的端到端核對：6 個代表層的全部 45 個投影 × 10 種格式 | B 相對誤差中位數 3.4e-4（最大 3.6e-3）；C 中位數 1.7e-4（最大 1.6e-3）；range 誤差 ≤ 2.9e-6 × 寬度；權重 err2 ≤ 1.3e-10；每個節點內的格式排序完全一致 | 中位數 ≤ 1e-3 |
| seq 512 forward（4 窗口） | rel_l2 最大 1.96e-6；CE 差最大 0.025（511 個 token） | 同上 |
| seq 512 統計 build | **失敗**：grads／ranges 的 Krnl→LLVM 在這台 39 GB 機器上 OOM（§7） | — |
| 成本表 non-finite／overflow 選項 | 0 | — |

表的端到端核對：PyTorch 從頭重算 ranges、梯度統計、激活誤差統計與權重誤差，再用 native 的公式組成 B、C，與 native 表逐項比對。

**架構相關的檢查**
- recurrence、Conv1D、normalization 與 gating 都在 forward 與 backward 的路徑上。backward 在 186 個節點的梯度都通過，表示這些 op 的反向傳播正確。
- seq 512 有 8 個 chunk（128 只有 2 個），forward 也通過，表示跨 chunk 的遞迴是對的。
- Conv1D 的 layout 是 [B, C, T]，channel 在 axis 1；`in_proj_qkv` 輸出端的梯度要先經過 conv backward，與 torch 一致。
- 狀態處理：
  - **reset**：每個窗口從零狀態開始（adapter 內的常數），與 HF `use_cache=False` 的逐窗口 logits 一致；
  - **detach**：窗口之間不傳遞狀態，所以不需要 detach；窗口內的梯度會穿過狀態，與 HF 相同。
- 形狀全部是靜態的，每個序列長度各匯出一次。

**沒有支援／沒有做**：
- vision tower、MTP、KV cache 或增量解碼、動態序列長度；
- 512 tokens 的 native 統計：記憶體不足，見 §7。

## 3. 排序（問題 2、3）

### 設計

**層的選法**：依規則選每種 block 的第一層、中間層（index n//2）與最後一層。
- linear attention：L0、L12、L22；
- full attention：L3、L15、L23。

**候選**（每層、每模式 40 個，共 720 個）
- 模式：w（激活 FP32）、a（權重 FP32）、wa（同一格式）。
- 格式：int4、int6、int8、p6e1、p8e1。
- 每個 (layer, mode)：Gate／Up／Down 單獨量化 15 個，Gate+Up 每對格式（含混合）25 個。

**量化語意**：沿用既有的 per-consumer QDQ、每個輸出 channel 的 int 權重、calibration range 的 int 激活，以及與 Universal 一致的 posit LUT。

**分數**（都只用 calibration）
- BC 是 table 中 B + C 的加總。
- SQNR 參考用 `sqnr_tables.py` 的定義：只量化單一節點時，Σ‖logits − logits_FP32‖²。
  - 原版用 onnxruntime 模擬，但它無法模擬 posit 激活，所以這裡改用與 truth 相同的 PyTorch 重播。
  - 聯合候選的分數是兩個單節點分數的和。
  - SQNR 只作參考，不參與搜尋。

**真實損失**：從該層開始重播的 FP32 PyTorch 模擬，30 個 diagnostic 窗口，summed next-token CE 的變化，除以 127 個 token 得到每 token ΔNLL。
- 主要結果在 GPU 上算（RTX 3060，FP32，關閉 TF32）。
- 預先登記的 CPU 交叉核對（`truth_device_plan.json`）：90 個候選（每層 wa 模式的 15 個單支候選）。
  - 每 token 均值差中位數 1.4e-5、最大 1.3e-4；
  - 198 個同資源配對的順序全部相同；決定性配對 GPU 139 個、CPU 137 個；
  - BC 正確率兩邊都是 0.803。
- 在 calibration 窗口上，個別小效應候選的單一窗口差可到 0.5 summed CE：FP32 的運算順序會讓少數激活跨過量化的捨入邊界。這是實作層級的雜訊，下文的「決定性」只考慮窗口雜訊，沒有考慮它。

**同資源配對與統計方法**
- 配對條件：同層、同模式，整個 FFN 的權重 bytes（M）與 BOPs（B）都相同；沒被量化的節點以 FP32 計。共 1722 對。
- 「決定性」：2000 次窗口 bootstrap 中，至少 95% 保持同一順序。
- 所有區間都是**配對窗口 bootstrap**：每次重抽 30 個窗口，所有候選一起重算。
  - 配對與候選共用窗口，不是獨立樣本；配對數不能當成獨立樣本數。
  - 區間只反映窗口雜訊，不反映候選集合的變異。

### 同資源配對

| 類別 | 配對 | 決定性 | BC 對（決定性） | SQNR 對（決定性） | BC 全部 | SQNR 全部 | BC − SQNR 區間 |
|---|---:|---:|---:|---:|---:|---:|---|
| 全部 | 1722 | 1057 | 934（88.4%） | 944（89.3%） | 0.759 | 0.777 | [−0.025, +0.003] |
| w | 606 | 446 | 389（87.2%） | 389（87.2%） | 0.790 | 0.790 | [+0.000, +0.003] |
| a | 606 | 285 | 272（95.4%） | 269（94.4%） | 0.726 | 0.754 | [−0.045, +0.015] |
| wa | 510 | 326 | 273（83.7%） | 286（87.7%） | 0.761 | 0.788 | [−0.047, +0.004] |
| linear-attention 層 | 861 | 497 | 430（86.5%） | 442（88.9%） | 0.753 | 0.787 | [−0.039, +0.009] |
| full-attention 層 | 861 | 560 | 504（90.0%） | 502（89.6%） | 0.765 | 0.767 | [−0.020, +0.017] |
| wa × linear | 255 | 159 | 126（79.2%） | 138（86.8%） | 0.702 | 0.765 | [−0.071, +0.000] |
| wa × full | 255 | 167 | 147（88.0%） | 148（88.6%） | 0.820 | 0.812 | [−0.031, +0.031] |
| w × full | 303 | 237 | 206（86.9%） | 206（86.9%） | 0.743 | 0.743 | [+0.000, +0.000] |

**Gate／Up 與 INT／Posit 分組**

| 類別 | 配對 | 決定性 | BC 對 | SQNR 對 |
|---|---:|---:|---:|---:|
| 單支：Gate vs Up，同格式 | 90 | 42 | 32（76.2%） | 32（76.2%） |
| 單支：Gate vs Up，同位寬 int vs posit | 72 | 42 | 37（88.1%） | 37（88.1%） |
| 單支：Down vs Gate，同格式 | 90 | 59 | 52（88.1%） | 51 |
| 單支：Down vs Up，同格式 | 90 | 62 | 49（79.0%） | 49 |
| 單支：Down vs Gate／Up，int vs posit | 144 | 96 | 89（92.7%） | 87 |
| 單支：同分支 int vs posit（int6↔p6e1、int8↔p8e1） | 108 | 64 | 55（85.9%） | 55 |
| 聯合：Gate／Up 互換 | 180 | 101 | 85（84.2%） | 90（89.1%） |
| 聯合：一支 int↔posit | 360 | 192 | 164（85.4%） | 165 |
| 聯合：其他 | 588 | 399 | 371（93.0%） | 378（94.7%） |

**依層**（決定性配對，BC／SQNR）
- L0：164/166 vs 166/166
- L3：144/156 vs 145
- L12：122/155 vs 127
- L15：156/169 vs 153
- L22：144/176 vs 149
- L23：204/235 vs 204

**BC 錯在哪裡**（量測）
- 錯誤集中在小差異：BC 判錯的 123 個決定性配對，真實差異中位數 0.0030 nat／token；全部決定性配對是 0.0120。
- 只要其中一個候選的 |ΔNLL| ≥ 0.01，BC 對 667/706（94.5%）；兩個都 < 0.01 時是 267/351（76.1%）。
- 720 個候選中有 17 個的 ΔNLL 顯著小於 0（量化反而降低 loss），14 個是 weight-only。BC 的 123 個錯誤中，42 個牽涉這些候選。非負的二次 criterion 不可能把它們排在 FP32 之下。
- 也有大效應的錯誤：L22 wa 的 int4，BC 判 Gate 206 vs Up 197（幾乎相同），真實是 0.056 vs 0.077。

**組內 Spearman**（每個 (layer, mode) 40 個候選，BC 與 truth；完整表在 `analysis.json`）
- w：中位數 0.86，最低 0.65（L23）；
- a：中位數 0.94，最低 0.72（L12）；
- wa：中位數 0.96，最低 0.83（L12）。
- SQNR 幾乎相同：中位數 0.86／0.93／0.96。

**BC 選擇的損失差**
- 132 個至少有 2 個成員的同資源類別。
- BC 選到的候選，平均比該類最好的候選多 **0.0026** nat／token：w 0.0013、a 0.0020、wa 0.0042。
- 對照：隨機選平均多 0.0154，SQNR 多 0.0018。BC 選到最好的 48/132 類，SQNR 53。

### 聯合 vs 單支（問題 3）

交互項 I = ΔL(Gate=f₁, Up=f₂) − ΔL(Gate=f₁) − ΔL(Up=f₂)，逐窗口計算。

- **大小**：聯合 |ΔNLL| > 0.001 的 355 個聯合候選，|I|／|ΔL_joint| 中位數 0.013、p90 0.089、最大 0.49。
- **顯著性**：450 個聯合候選中，120 個的 I 區間不含 0（78 個為正、42 個為負），正負都有，沒有一致方向。
- **可重現的大偏差**（兩半窗口的 I 幾乎相同）：
  - L23 a／wa 的 int4+int4：I = −0.224／−0.215，次可加 29%。
  - L0 wa 的 int4+int4：+0.033，超可加 16%。
  - L0 wa 的 int4+p6e1：+0.026。
- **split-half 相關**：L0、L15、L22、L23 是 0.85–0.998（交互項可重現）；L3、L12 是 −0.69 到 0.78（雜訊主導）。
- **對排序的影響**：「加總 oracle」（真實的單支 ΔL 相加）在聯合同資源配對上對 90.4–100%；BC 只有 45.6–92.6%，SQNR 55.9–92.6%。Spearman(單支和, 聯合) ≥ 0.986。
- **結論（推論）**：加總失準確實存在，也可重現，但主要在大誤差的 int4 組合，對同資源排序的影響小。BC 聯合錯誤的主因是單支分數的相對大小。

## 4. 完整搜尋（問題 4）

**設定**
- `mp_auto.auto_select`（2-D 預設），R = max(M/M₈, B/B₈)；
- 17 個 coarse 樣本、2 次 refinement；
- GRID11 × GRID11；每次 MILP 限時 10 s。
- 成本表沒有 non-finite 或 overflow 選項，所以沒有排除任何選項。

| | FFN-only | FFN + attention |
|---|---|---|
| 搜尋節點 | 72 | 186 |
| MILP 求解 | 23 次，全部 optimal（status 0），gap ≤ 0.0056 | 23 次，全部 optimal，gap ≤ 0.016 |
| 失敗 | 0 | 0 |
| 搜尋時間 | 9.3 s | 42.9 s |
| frontier 點 | 21 | 23 |
| 端點：最便宜／最小 BC | R 0.5（BC 954）／R 8.0（BC 0，f16 權重 + FP32 激活） | R 0.5（BC 2890）／R 8.0（BC 0） |
| knee（score） | R 0.877，BC 35.4（0.913） | R 0.841，BC 81.6（0.926） |
| 權重格式 | int6 34、int7 32、int8 6 | int6 82、int7 64、int8 31、int5 9 |
| 激活格式 | p8e1 68、f16 4 | p8e1 136、f16 47、int8 3 |
| 搜尋權重大小 | 218 MB（int8 為 264 MB；bf16 checkpoint 為 528 MB，knee 是 41%） | 398 MB（int8 為 498 MB；bf16 為 995 MB，knee 是 40%） |

- **各模組選擇**（`selection.json` 的 `by_kind`）：
  - `in_proj_a`、`in_proj_b`（1024×16，BOPs 很小）全部選 f16 激活；
  - Gate 多為 int6/p8e1，Down 多為 int7/p8e1；
  - `o_proj` 多為 int8/p8e1。
- **資源的意義**：M 是「被搜尋權重」的 bytes，B 是 BOPs 代理指標（權重位元 × 激活位元 × MAC），兩者都相對全 int8。它們不包含固定 FP32 的模組（embedding／LM head 2.54 億參數，約 1.0 GB FP32），也不代表實際硬體加速。本報告不宣稱任何硬體加速。

**final test 結果**（64 個窗口，native onnx-mlir 編譯：Krnl -O3、clang -O0、每 channel 的 int 權重、calibration range；所有 logits 都是有限值）

| 版本 | PPL | NLL | ΔNLL vs FP32 [95% CI] | 編譯 | 執行（5 process） |
|---|---:|---:|---|---:|---:|
| FP32 | 43.800 | 3.7796 | — | 47 s | 692 s |
| FFN-only knee | 44.120 | 3.7869 | +0.0073 [+0.0036, +0.0108] | 56 s | 525 s |
| FFN + attention knee | 44.728 | 3.8006 | +0.0210 [+0.0151, +0.0267] | 64 s | 653 s |

區間是 64 個窗口的配對 bootstrap。

## 5. 成本（問題 5）

所有時間都是同一台機器量到的（12 核 CPU、39 GB RAM），除非標為外推。

| 階段 | wall time | 內容 |
|---|---:|---|
| prepare | 54 s | 匯出 16 s、constant folding、mapping、ORT 檢查 |
| correctness | forward 188 s、backward 341 s、stats 114 s | 只做一次，不是每次搜尋都要 |
| **成本表（cold）** | **1854 s** | import 16 s + bytecode 18 s；weights 206 s；grads build 608 s + runs 259 s；ranges build 343 s + runs 362 s；errors 7 份 1111 s（最慢的 build 1015 s，runs 132 s）。三條鏈平行。 |
| freeze | 0.1 s | |
| **搜尋** | **53 s**（9 s + 43 s） | MILP |
| 評估一個配置 | 編譯 56–64 s + 執行 525–653 s | final test，64 個窗口 |
| FP32 基準評估 | 739 s | |
| seq 512：匯出、forward 檢查 | 70 s、307 s | |
| seq 512：native 成本表 | 未完成（兩次嘗試約 2 小時 20 分，OOM） | grads -O1 編譯 98 分鐘、峰值約 31.9 GB |
| seq 512：PyTorch 表（18 節點）、truth（GPU） | 77 s、672 s | 驗證用 |

- **選配置的 wall time**：不含評估 ≈ 1854 + 53 + 54 = **約 33 分鐘**；含評估一個選中的配置約 45 分鐘，再加 FP32 基準約 57 分鐘。
- **cold／warm**：pipeline 每次都會重建統計用的 .so，所以只量了 cold。不含編譯的執行部分（weights 206 s、grads runs 259 s、ranges + errors runs 494 s）是從量到的分段推出來的，並不是實際跑出的 warm 時間。
- **記憶體**（量測）：
  - 統計執行：每個 process 的峰值 RSS 為 grads 5.6 GB、ranges 3.2 GB、errors 3.3 GB；
  - 編譯：同時進行的 build 受 26 GB 記憶體預算控制；觀察到單一 `onnx-mlir-opt` 約 10 GB；
  - 評估：每個執行 process 7.7 GB，父 process 15.6 GB（logits 8 GB）。
- **驗證用的花費**（不屬於 pipeline）：
  - truth：GPU 669 s（720 候選 × 30 窗口）；
  - CPU 交叉核對：90 個候選 1438 s；
  - SQNR 參考：270 個選項 × 10 窗口，CPU 1649 s。
- **外推比較**：SQNR 的完整表（186 節點 × 120 個 (w,a) 選項，10 個 calibration 窗口），用本次的逐層重播實作與量到的速度外推：
  - CPU：每個選項平均約 5.8 s（L0 9.2 s 到 L23 2.4 s），合計約 **36 小時**；
  - GPU：每個候選-窗口約 0.031 s，合計約 **1.9 小時**。
  - 對照：BC 的完整表是 31 分鐘 CPU。
- **與 Qwen3-0.6B 比**：Qwen3 的表是 196 節點 1074 s。Qwen3.5 的 vocab（248320）讓 grads 每窗口的 onehot 和 LM head 更重，errors 的編譯也更長。

## 6. 判斷與建議（問題 6）

**量測到的結果**
- native 的成本表、搜尋與編譯在 Qwen3.5 的混合架構上都能正確運作。
- knee 配置用約 40% 的 bf16 權重大小，FFN-only 的 PPL 增加 0.73%，FFN + attention 增加 2.1%。
- BC 的同資源排序與 SQNR 參考在統計上分不開（88.4% vs 89.3%，區間含 0），成本卻低 1–2 個數量級。

**推論的原因（未直接驗證）**
- BC 的錯誤主要在小差異與負 ΔNLL 的候選。負 ΔNLL 代表一階（方向）效應或窗口特有的抵消；Qwen3 的研究發現方向項不會跨窗口轉移。
- Gate／Up 相對敏感度的錯誤，與 Qwen3 上觀察到的 Gate／Up 問題一致，但這裡沒有分析 attention sink 或 position 0 的貢獻。不能把 SwiGLU、position 0、attention sink 或非局部性當成已證明的原因。

**值得驗證的新假說**（都需要新窗口與預先登記）
1. **wa 模式 linear-attention 層的 Gate／Up 估計**：wa × linear 的決定性正確率 79.2%，SQNR 86.8%，區間 [−0.071, +0.000] 接近排除 0。要先確認這不是事後挑出的子集（本表有 9 個以上子集）。
2. **低誤差區（|ΔNLL| < 0.01）的判斷**：BC 在這一區沒有優勢，但這一區對完整搜尋的 knee 影響可能很小，因為 knee 的格式多為 int6–int8 + p8e1。應先量化它對 knee 的實際影響，再決定是否需要修正。

**建議**：保留原版 BC 作為 Qwen3.5 的預設，本輪不修改 criterion。

## 7. seq 512 有限檢查

**匯出與 forward**
- 512 tokens 的匯出：op 類型與 128 相同，共 6372 個節點（delta rule 展開成 8 個 chunk，128 只有 2 個）。
- native FP32 forward（4 個 512-token calibration 窗口）：
  - logits rel_l2 vs FP32 最大 1.96e-6，vs FP64 1.86e-6；
  - summed CE 差最大 0.025（511 個 token，門檻 0.05）；
  - 編譯 61 s，執行 157 s（4 個 process，各 4.4 GB）。

**native 成本表沒有完成**（量測，`seq512/seq512.json` 的 `table_attempts`）
- **第 1 次**（統計 build 等級與 128 相同）：grads 的 Krnl→LLVM 被 OOM killer 終止。`onnx-mlir-opt` anon RSS 33.4 GB，39 GB RAM + 7 GB swap 用完；當時只有它在編譯。
- **第 2 次**（只把 grads 改成 Krnl -O1）：grads 編譯完成，但花了 98 分鐘，峰值約 31.9 GB（每 3 s 取樣）。之後 ranges（-O3）在 grads 執行期間被 OOM 終止（25.6 GB）。兩次共約 2 小時 20 分。
- 對照：128 tokens 的 grads build 沒有 OOM（608 s）。在這台機器上，512 tokens 的 native 統計受編譯記憶體限制，這是 pipeline 延伸到長序列時最直接的阻礙。

**備案：PyTorch 計算 512 的 BC 分數**
- 用 `torch_bc_table`（同樣的統計與公式，range 在同樣的 512 calibration 窗口上量），只算 18 個代表 FFN 節點，77 s。
- 這個實作在 128 tokens 時與 native 表逐項核對過（`seq512/torch_table_check_128.json`）：
  - B 相對差中位數 3.8e-4、C 1.4e-4，格式排序一致；
  - 720 個候選的 BC 相對差最大 3.1e-3，Spearman 0.9999995。
- 分數凍結（`seq512/scores_freeze.json`）之後，才在 8 個 512-token diagnostic 窗口上跑 truth（GPU，672 s）。

**結果**（`seq512/analysis.json`；同樣 720 個候選與 1722 個同資源配對；決定性用 8 個窗口的 bootstrap）

| 預測 | 決定性配對對（共 998） | 全部配對 |
|---|---:|---:|
| BC512（512 calibration） | 897（89.9%） | 0.782 |
| BC128（原表） | 878（88.0%） | 0.769 |
| 參考：128 的 truth 本身 | 909（91.1%） | 0.776 |

- **BC 的排序幾乎不隨序列長度改變**：18 個 (layer, mode) 的 BC512 vs BC128 組內 Spearman 都 ≥ 0.978。
- **每 token ΔNLL**：除 L0 外，512 約是 128 的 0.50–0.96 倍（每組候選比值的中位數）。
- **L0 在 512 幾乎沒有損傷**：
  - Down int4 的每 token ΔNLL 從 0.188 降到 0.030；
  - L0 有 47% 的候選均值為負，窗口間正負交錯；
  - 只有 8 個窗口，L0 由雜訊主導，BC 與 truth 的 Spearman 只有 0.09–0.69（128 時是 0.94–0.97）。
- **推論（未驗證）**：L0 的損傷集中在窗口開頭的位置，所以在 128-token 窗口中佔的比例較高。本輪沒有逐位置資料，不能當成結論。

## 8. 限制與未完成

- 排序只涵蓋 6 層的 FFN，沒有直接驗證 attention 投影（in_proj_qkv、q_proj 等）的排序。FFN + attention 的配置只有 final test 的整體結果。
- truth 是 PyTorch 模擬（GPU，CPU 交叉核對過），不是 native 編譯模型的 loss。native 與 torch 的 forward 差異是 2e-6 等級，但小效應候選對 FP32 實作雜訊敏感（§3）。
- 「決定性」只考慮窗口雜訊。
- SQNR 參考用 PyTorch 重播（為了支援 posit 激活），與原版 onnxruntime 的實作不同。
- 每個 knee 只評估一次（64 個窗口），沒有重複；沒有與其他預算的配置比較。
- cold／warm 只量了 cold。
- 本輪沒有分析 sink 位置、位置別誤差或非局部性，所以不對失準原因下結論。
- seq 512 的限制：
  - BC 分數是 PyTorch 計算的，不是 native；
  - 只有 4 個 calibration 窗口和 8 個 diagnostic 窗口，只涵蓋 18 個代表 FFN 節點；
  - 沒有做 512 的搜尋或評估。
- 本 session 的工作時間有一部分浪費在 512 native 表的兩次失敗嘗試（約 2 小時 20 分）。512 的 native 成本表沒有完成。

## 9. 重跑與檔案

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/qwen35_bc_20261015      # 新實驗請換新目錄
GPU="env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$PY run_qwen35_bc.py --stage prepare     --out-dir $O
$PY run_qwen35_bc.py --stage correctness --out-dir $O --parts forward,backward
$PY run_qwen35_bc.py --stage tables      --out-dir $O
$PY run_qwen35_bc.py --stage correctness --out-dir $O --parts stats
$PY run_qwen35_bc.py --stage search      --out-dir $O
$PY run_qwen35_bc.py --stage freeze      --out-dir $O
$PY run_qwen35_bc.py --stage sqnr        --out-dir $O --layers 23   # 依層分批，續跑
$GPU run_qwen35_bc.py --stage truth      --out-dir $O --device cuda
$PY run_qwen35_bc.py --stage truth       --out-dir $O --truth-file truth_cpu_crosscheck.json --layers 0,3 --names "<wa 單支候選>"
$PY run_qwen35_bc.py --stage analyze     --out-dir $O
$PY run_qwen35_bc.py --stage evaluate    --out-dir $O --versions fp32   # 再跑 ffn、ffn_attn
$PY run_qwen35_bc.py --stage seq512      --out-dir $O --parts export,forward
# --parts table（native）在 39 GB 機器上 OOM；改用 PyTorch 表：
$PY run_qwen35_bc.py --stage seq512      --out-dir $O --parts torch_table,scores
$GPU run_qwen35_bc.py --stage seq512     --out-dir $O --parts truth --device cuda --batch 2
$PY run_qwen35_bc.py --stage seq512      --out-dir $O --parts analyze
```

| 檔案 | 內容 |
|---|---|
| `plan.json`、`environment.json` | 凍結的計畫；版本與 hash（含 Qwen3 基準表的 hash） |
| `prepare.json`、`node_mapping.json` | op inventory、節點與參數的對應 |
| `correctness.json` | forward／backward／stats 核對 |
| `table.json`、`ranges_by_point.json`、`tables.json` | native BC 表（B = W，C = A）、range、成本表耗時 |
| `candidates.json`、`freeze.json`、`sqnr.json`、`sqnr_freeze.json` | 候選、BC 分數、SQNR 參考與凍結 hash |
| `truth.json`、`truth_cpu_crosscheck.json`、`truth_device_plan.json` | 逐窗口 ΔCE（GPU）、CPU 交叉核對與其計畫 |
| `analysis.json`、`candidates.csv`、`pairs.csv`、`joint.csv` | 排序指標與原始數據 |
| `selection.json`、`selection_{ffn,ffn_attn}.json`、`selected_formats_{ffn,ffn_attn}.json` | 搜尋摘要、frontier、knee、solver log 與選定格式 |
| `results.json`、`eval/*/compile_env.json` | final-test 結果與編譯設定 |
| `seq512/` | 512 tokens 的 export、正確性、表、分數、truth 與分析 |
| `timing.json`、`*.log` | 各 stage 每次執行的 wall time 與峰值 RSS |
