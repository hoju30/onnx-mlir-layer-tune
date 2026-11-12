# Qwen3-0.6B SwiGLU gate／up criterion 驗證報告

日期：2026-10-14　分支：`posit-work-continue`（HEAD `2a784020`，加上第 8 節列出的未提交修改）

所有數字都用 calibration 資料：`validation_tokens.npy`，10 個窗口 × 128 tokens，每窗口 127 個預測位置。loss 是 summed next-token CE，與 criterion 的 seed 相同。本研究完全沒有用到 held-out（`test_tokens.npy`）。

## 摘要

1. **SwiGLU backward 正確。**
   - 在小型獨立模型上，native 對 FP64 autograd 最差 2.4e-6。
   - 閉式解對 autograd ≤ 2.2e-16，中央有限差分 ≤ 4.3e-8。
   - gate 與 up 共用輸入 x 的 tap 正好等於兩支梯度之和（有 residual 時再加 residual），沒有重複計算。
2. **完整模型梯度差異的原因是 onnx-mlir 的 epsilon bug，不在 backward。**
   - `--recompose-onnx` 把 RMSNorm 重組回來時，讀不到 dense 常數形式的 epsilon，結果改用預設 1e-5（Qwen3 是 1e-6）。
   - 修正後，前向 native 對 FP64 的 rel_l2 從 3.3e-3 降到 3.0e-6，與 ORT 同級。
   - 梯度 median／max rel_max 從 8.7e-3／0.437 降到 7.3e-4／8.9e-3（對 FP64），落在原門檻內（median ≤ 1e-3、max ≤ 5e-2），**沒有放寬容差**。
   - 已修正並加了 lit 測試。BERT（eps 1e-12）同樣受影響；GPT-2 與 CNN 不受影響。
3. **criterion 聚合一致。**
   - native 統計重新聚合後，與 `table.json` 的差 ≤ 7.1e-16。
   - PyTorch 與 native 的 W／A／C median 相對差約 6e-5，最大 5.1e-3。
   - 56 節點 × 121 組 (w, a) 的排序 Spearman ≥ 0.999988；最小 C 的組合 56/56 相同。
4. **共享輸入的成本歸屬。**
   - 現行的輸入項用 x 的總梯度。「只量化 gate」或「只量化 up」時，這會把另一支的敏感度也算進來：A_x(總)/A_x(分支) 的 median 是 1.92。
   - 已加入 per-edge（分支）梯度選項：`MIXEDP_STATS_INPUT_GRAD=branch`，pipeline 用 `--input-grad branch`，輸出 `table_branch.json`。
   - 分支版讓量級接近模擬的一階值：A_x/Σf_X² 的 median 從 1.68 變成 0.91。
   - 但 108 組設定的排序幾乎不變：Spearman(C, |ΔL|) 0.856 vs 0.850。
   - Qwen 全模型 MILP 卻有 33–89/196 個節點的選擇改變，因為 q/k/v 也共用輸入。
5. **預測力：粗排序可以，細部決策不行。**
   - 跨層、跨格式、跨模式的 108 組中，C 對 |ΔL| 的 Spearman 是 0.856，高於模擬一階 Σf² 本身的 0.830。
   - 同層「gate 還是 up 傷害較大」，C 只對 14–15/24，Σf² 本身也只對 12/24。
   - 聯合量化的 ΔL 明顯不是加成的：L2 int4 激活的聯合是 7262，單獨和是 2534；L27 的聯合是 863，單獨和是 1736。
   - 一階交叉項的正負號只預測對 11/22；有 10 組聯合 ΔL 小於某一支單獨的 ΔL。加總型 criterion 表示不出這些。
6. **建議。**
   - criterion 形式不改，也不加 gate／up 交互項：沒有證據顯示交互項能預測非加成。
   - 分支歸屬保留為選項。要不要當預設，取決於一次全模型 calibration loss 比較；這次還沒做，列為下一步。
   - 搜尋粒度維持逐節點。共享輸入的激活可以考慮改成「一個 tensor 一個決策」，但本實驗沒有證據顯示它改善預測（第 6 節）。

## 0. 實驗條件與重現

**模型與資料**
- Qwen3-0.6B，來源 `runs/qwen3_prepare_20261011/model.onnx`（sha256 `1b0028fd…`）。
- 搜尋範圍是 196 個 projection。
- 第 1、3 階段用前 2 個窗口，第 4、5 階段用 10 個窗口；都沒有隨機抽樣。

**版本**
- 每個階段都寫出 `environment*.json`：git HEAD 和 dirty 狀態、各腳本與 C++ 原始檔的 sha256、`onnx-mlir-opt` 二進位的 sha256 與 mtime、模型檔 sha256。

**沒有 stale cache**
- 每個階段都寫進新目錄；第 1 階段用 `mkdir(exist_ok=False)`。native 的 `.so` 都在本次重新建置。
- 舊的 `qwen3_grad_check.py` 會重用已有的 grads `.so`。`phase1_grads` 在全新目錄重建後，重現了舊數字：median 8.6e-3、max 0.437。所以舊結果不是 cache 造成的。

**PyTorch 參考**
- HF eager 模型，FP32 與 FP64 兩種，窗口與 loss 定義都相同。
- CPU 上的 PyTorch FP32 在低層並不穩定。原因是 MKL 多執行緒加上低層統計量的條件數很差：每個陣列由單一位置主導，token 0 有巨大激活，例如 layer 2 的 down 輸入約 3653。layer 0–7 的 FP32 對 FP64 rel_max 約 8e-3。
- 因此以 **FP64 為參考**，FP32 只並列顯示。

**機器**：CPU 執行，記憶體 39 GB。

## 1. SwiGLU backward 正確性（第 2 階段）

**設定**
- 腳本 `swiglu_backward_check.py`，在小型獨立 ONNX 圖上測試，op 與 Qwen3 MLP 的 TorchScript 匯出相同：
  - zg = x·Wg，zu = x·Wu，h = SiLU(zg)·zu，y = h·Wd；
  - r = y（或 y + x），z = r·Wh，loss = Σ CE。
- zg 用精確求解的方式放在 16 個值上：−40、−12、−4、−1、−0.1、−1e-3、−1e-6、0、1e-6、1e-3、0.1、1、4、12、40、90。這樣涵蓋 SiLU 的飽和區、零點附近與線性區。
- L = 1 時，每 channel 的總和就是元素本身，正負號與值都逐元素比較。

**結果**

| case | native vs FP64 autograd（最差欄位） | 閉式解 vs autograd | 有限差分 vs autograd | 共用 tap vs 總梯度 | 共用 tap vs 兩倍 |
|---|---:|---:|---:|---:|---:|
| L=1, residual=False | 1.1e-06 | 2.0e-16 | 4.3e-08 | 1.6e-07 | 0.5 |
| L=1, residual=True | 5.6e-07 | 2.2e-16 | 1.0e-08 | 3.0e-07 | 0.5 |
| L=4, residual=False | 1.5e-06 | 4.6e-17 | 7.9e-09 | 4.2e-07 | 0.5 |
| L=4, residual=True | 2.4e-06 | 1.2e-16 | 1.2e-09 | 3.5e-07 | 0.5 |

- x 的 tap 等於 gate 分支加 up 分支（加 residual）的總和，只算一次；和「兩倍」比相差 0.5，代表沒有重複累加。
- 新增的分支梯度（第 4 節）對 autograd 的分支值，最差 3.5e-7。
- `phase2_swiglu_branch/swiglu_backward.json` 另有逐 zg 值的比較，以及 30 個有限差分點。

## 2. 完整模型梯度差異的原因（第 1、3 階段）

### 2.1 重現與定位

- **重現（修正前）**：native 對 FP64 的 median rel_max 是 8.66e-3，max 是 0.437，不符合門檻。
- **誤差分布**：layer 0–7 的 x／y／w 欄位都約 0.3，layer 8 以後約 1e-2。誤差均勻地分在所有欄位，看起來不像某個 op 的 VJP 錯了。
- **前向 tap**（第 3 階段，`qwen3_swiglu_forward.py`）：把 MLP 的中間 tensor 加成 ONNX 輸出後比較。
  - 修正前，native 對 FP64 的 rel_l2 在前向就已經是 3.3e-3；同一模型的 ORT 是 1.5e-6。所以問題在前向，不在 backward。
  - 接著把中間 tensor 加成輸出逐段二分。第一個偏離的是 RMSNorm 的輸出：用 epsilon 1e-5 計算正好吻合，用 1e-6 則不吻合。
  - 這個二分用的是一次性的暫存腳本，清理磁碟時已刪除。可重跑的證據是下面的 lit 測試，以及修正前後的第 1、3 階段結果。

### 2.2 根本原因與修正

- **出錯位置**：`src/Dialect/ONNX/Transforms/Recompose.cpp` 的 RMS／LayerNorm 重組 pattern 用 `epsilonOp.getValueFloatAttr()` 取 epsilon。
- **為什麼會讀不到**：
  - 匯出器寫的是 `value = dense<…>`；`--decompose-onnx` 也會把 `value_float` 轉成 dense。
  - 對 dense 常數，這個 getter 回傳 null，於是 pattern 用了預設的 1e-5。
- **修正**：沒有 float 屬性時，從 dense 的 `value` 取純量。
- **測試**：新增 `test/mlir/onnx/onnx_recompose_epsilon.mlir`，分別測 RMS（eps 1e-6）與 LayerNorm（eps 1e-12）。修正前失敗，修正後通過。

### 2.3 修正前後

| 版本 | 比較 | median rel_max | p99 rel_max | max rel_max | median rel_l2 | min cosine |
|---|---|---:|---:|---:|---:|---:|
| 修正前 | native vs FP64 | 8.66e-03 | 0.334 | 0.437 | 8.32e-03 | 0.968212 |
| 修正前 | PyTorch FP32 vs FP64 | 5.05e-05 | 4.25e-04 | 9.92e-04 | 4.45e-05 | 1.000000 |
| 修正後 | native vs FP64 | **7.25e-04** | 5.33e-03 | **8.85e-03** | 6.29e-04 | 0.999993 |
| 修正後 | PyTorch FP32 vs FP64 | 1.07e-04 | 8.19e-03 | 0.0118 | 9.92e-05 | 0.999987 |
| 修正後 | native vs PyTorch FP32 | 8.87e-04 | 8.43e-03 | 0.0131 | 8.03e-04 | 0.999984 |

| 前向 | native median／max rel_l2 | ORT | PyTorch FP32 | logits native rel_l2 |
|---|---|---|---|---:|
| 修正前 | 3.3e-03／7.9e-03 | 1.5e-06／7.2e-06 | 1.5e-06／2.9e-06 | 2.9e-03 |
| 修正後 | 3.0e-06／8.3e-06 | 1.5e-06／7.2e-06 | 1.5e-06／2.9e-06 | 3.7e-06 |

**修正後剩下的差異**：
- lm_head backward 要沿 K = 151936 累加 f32，這一步 native 的 rel_l2 是 8.6e-5，PyTorch FP32 是 4.3e-7。
- 再加上前面提到的低層條件數問題。
- 修正後 native 對 FP64 的最大誤差（8.9e-3）比 PyTorch FP32 對 FP64 的（1.2e-2）還小。
- 兩次修正後的 PyTorch FP32 欄位不同（5.0e-5 vs 1.1e-4），是 CPU 多執行緒的不確定性；FP64 參考不受影響。

### 2.4 對其他模型的影響

criterion 用修正後的 onnx-mlir 重建，存在 `runs/native_eps_fix_20261014/`：

| 模型 | LayerNorm eps | B（=W）median／max 相對變化 | C（=A）median／max | MILP 選擇相同的節點 |
|---|---|---|---|---|
| GPT-2 | 1e-5 | 0／0 | 0／0 | 48/48，8 個設定全同 |
| BERT | 1e-12 | 6.6e-4／5.6e-3 | 8.8e-4／0.28 | 74/74 的有 9 個設定，72/74 的有 1 個 |
| Qwen3-0.6B | 1e-6 | 2.0e-3／0.12 | 2.8e-3／0.12 | 191–196/196，8 個設定中 3 個全同 |

CNN 沒有 LayerNorm，不受影響。先前 BERT 與 Qwen 的下游結果（auto pick、PPL）是在 bug 下算的，**沒有重跑**。

## 3. criterion 聚合一致性（第 4 階段）

**定義**
- W_i(w) = Σ_c S_ic Σ_j ΔW²，其中 S_ic = Σ_s mean_j gW²。
- A_i(a) = Σ_{t∈{x,y}} Σ_c [G2·var + GΣ·mean²]，其中 G2 = Σ_s Σ_j g²，GΣ = Σ_s (Σ_j g)²，也就是每個窗口先在 channel 內加總再平方。
- C = W + A。table.json 裡的命名是 B = W、C = A。
- PyTorch 端（`qwen3_swiglu_criterion.py --stage torch`）用相同定義，從 hook 取得統計量。

| 項目 | 數值 |
|---|---:|
| native 重新聚合 vs table.json：W／A／A（分支表）最大相對差 | 0／5.9e-16／7.1e-16 |
| PyTorch vs native W：median／max 相對差（56 節點 × 10 格式） | 8.7e-05／5.1e-03 |
| PyTorch vs native A（總梯度） | 5.2e-05／3.6e-03 |
| PyTorch vs native A（分支梯度） | 6.0e-05／4.5e-03 |
| PyTorch vs native C | 5.9e-05／3.6e-03 |
| W、A 格式排序 Spearman（min） | 1.000000／1.000000 |
| 121 組 (w, a) 的 C 排序 Spearman（min） | 0.999988 |
| 最小 C 的格式組合相同的節點數 | 56/56 |
| int4 輸入項逐 channel 貢獻 cosine（min） | 1.000000 |
| 權重 channel 軸（weights pass = grads pass） | 輸出 channel，[1, 3072] |

- 「56 節點」是 28 層 × gate／up。
- **GΣ**：native 與 PyTorch 都在每個窗口內先加總再平方，一致。
- **權重 channel 軸**：MatMul 權重 [K, N] 的 axis 1 是輸出 channel（3072 個），weights pass 與 grads pass 用的是同一個軸。
- **posit 參考**：LUT 對 Universal 在每種格式約 150 萬個值上 0 個不同（p4e1、p6e1、p8e1）。

## 4. 共享輸入的成本歸屬

### 4.1 語意

**pass 的量化方式**
- mixed-precision pass 對每個 consumer 各插一個 QDQ，所以 gate 和 up 各自量化自己那份 x。
- 只量化 gate 時，擾動 dX 只進入 gate 那條邊。一階效應是 g_X^(gate)·dX，其中 g_X^(gate) = g_zg·Wgᵀ。
- Qwen3 的 MLP 輸入是 post-attention RMSNorm 的輸出，consumer 只有 gate 和 up（residual 繞過了 norm），所以總梯度 g_X = g^(gate) + g^(up)。

**現行 criterion 的問題**
- 現行做法是 x tap 一律用總梯度。
- 單獨量化 gate 或 up 時，這會把另一支的敏感度算進來。
- 兩支一起量化時，又會把同一個總梯度算兩次。

### 4.2 實作（未提交）

- **`MixedPrecisionBackward.cpp`／`.hpp`**：Conv／Gemm／MatMul 除了累加輸入梯度，也記錄該 op 自己對輸入梯度的貢獻，即 per-edge 梯度，存在 `operands[{op, 0}]`。
- **`ONNXToMixedPrecision.cpp`**：設定 `MIXEDP_STATS_INPUT_GRAD=branch` 時，每個權重節點多存兩個量：輸入 channel 的 Σg² 與 Σg。layout 新增 `x_branch_offset`、`x_channels`、`x_channel_axis`、`x_has_grad`。沒設定時 layout 完全不變。
- **測試**：`test/mlir/conversion/onnx_to_mixed_precision/collect_stats_grads_branch.mlir`，兩個 MatMul 共用輸入，分別檢查 branch 與 total 兩種 layout。
- **`run_native_bc_pipeline.py --input-grad branch`**：在同一個 grads build 裡同時產生 `table.json`（總梯度）與 `table_branch.json`。Qwen 全流程 1074 s，前一版是 1123 s，沒有明顯額外成本。

### 4.3 數值

**A_x(總)/A_x(分支)**

| layer | 節點 | int4 | int8 | p6e1 |
|---:|---|---:|---:|---:|
| 0 | gate | 2.3 | 2.3 | 1.12 |
| 0 | up | 1.72 | 1.72 | 1.94 |
| 2 | gate | 1.95 | 2.06 | 0.977 |
| 2 | up | 2.02 | 1.93 | 8.24 |
| 14 | gate | 2.53 | 2.48 | 2.01 |
| 14 | up | 1.45 | 1.48 | 1.3 |
| 27 | gate | 1.96 | 1.89 | 1.81 |
| 27 | up | 2.49 | 2.55 | 2.7 |
| 全部 28 層 | gate／up | median 1.92（範圍 0.977–8.24） | | |

**聯合量化時的輸入項**
- 兩條邊的量化範圍相同，所以 scale 與 zp 也相同，dX 完全一樣。輸入項的一階值因此是 (f_X,gate + f_X,up)，正好是總梯度只算一次所量的東西。

| 聯合量化的輸入項（a、wa 模式各 12 組） | median |
|---|---:|
| Σ(f_X,gate + f_X,up)² / [Σf_X,gate² + Σf_X,up²] | 1.49 |
| A_x(總梯度，算一次) / Σ(f_X 聯合)² | 0.53 |
| A_x(分支 gate + up) / Σ(f_X 聯合)² | 0.63 |
| A_x(總梯度，算兩次，即現行做法) / Σ(f_X 聯合)² | 1.07 |

- **兩支同號**：模擬中，兩支的一階輸入效應在各窗口之間正相關，比值 1.49。
- **criterion 估不到這個相關**：「總梯度算一次」約等於「分支和」（0.53 vs 0.63）。也就是說，在 criterion 的統計量裡，逐元素的交叉梯度 Σ g_gate·g_up 幾乎是 0。
- **相關從哪裡來**：f 的交叉項來自 dX 不同元素之間的相關。criterion 的逐元素獨立模型（G2·var + GΣ·mean²）表示不出這種相關。相關的具體來源本實驗沒有直接驗證；推測是 int4 下 outlier token 整列的誤差有結構。
- **現行做法看起來準是巧合**：算兩次的比值 1.07，只是兩次錯誤剛好互相抵消。

**MILP 選擇**
- 用 `table.json` 與 `table_branch.json` 各跑一次 MILP，比較 8 個設定（grid8／grid11 × budget 0.8／1.0／1.25／1.5）。結果在 `phase5_quant/picks_total_vs_branch.json`。

| 設定 | grid8 f0.80 | f1.00 | f1.25 | f1.50 | grid11 f0.80 | f1.00 | f1.25 | f1.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 選擇不同的節點（/196） | 89 | 74 | 82 | 72 | 66 | 64 | 71 | 33 |
| 其中 gate／up | 24 | 16 | 20 | 21 | 16 | 11 | 21 | 12 |

- 其餘不同的節點主要是 q／k／v，它們也共用輸入，而且有三個 consumer。剩下的是預算重新分配造成的連帶變化。
- 這些變化對全模型 loss 的影響，本實驗沒有量測（第 6 節）。

## 5. 單獨與聯合量化的預測力（第 5 階段）

### 5.1 設定

**組合**：layer 2、0、14、27 × {gate, up, both} × {w, a, wa} × {int4, int8, p6e1}，共 108 組；其餘節點都是 FP32。

**模擬**（`qwen3_swiglu_quant.py --stage sim`）是 PyTorch FP32，照 pass 的 QDQ 語意實作：
- **權重**：每個輸出 channel 對稱量化，用 `std::lround`。
- **int 激活**：用 native errors layout 的 scale／zp，順序是 Div → Add(zp) → Round（half-even）→ Clip → Sub(zp) → Mul。
- **posit**：用和 Universal 逐值相同的 LUT。
- **共用輸入**：每個 consumer 量化自己那份。

**每組記錄的量**
- 帶號 ΔL：每窗口一個值，以及總和。
- f_s = Σ g_y·dy：每個節點輸出在上游為 FP32 時的局部擾動，再拆成權重、輸入、輸出三部分。
- Σ_s f_s 與 Σ_s f_s²。
- 「both」另外記錄一階交叉項 2Σ f_g f_u，以及 SwiGLU 二階項 Σ g_h·SiLU′(zg)·dzg·dzu。

**native 驗證**（`--stage native`）：用 onnx-mlir 實際編譯 7 組代表性設定和 FP32（`--convert-onnx-to-mixed-precision`，評估用的 build 等級），在相同 10 個窗口上執行。**以下除了 5.2 節，其餘都是 PyTorch 模擬的結果。**

### 5.2 native 驗證：onnx-mlir 編譯執行 vs PyTorch 模擬

| 設定 | native ΔL（onnx-mlir 編譯執行） | PyTorch 模擬 ΔL | 差 | 相對差 | 每窗口最大 abs 差 |
|---|---:|---:|---:|---:|---:|
| L2_gate_wa_int4 | 589 | 589 | -1.45e-03 | 2.5e-06 | 4.28e-04 |
| L2_up_wa_int4 | 1.13e+03 | 1.13e+03 | -1.99e-03 | 1.8e-06 | 5.65e-04 |
| L2_both_wa_int4 | 6.98e+03 | 6.98e+03 | -8.77e-03 | 1.3e-06 | 2.08e-03 |
| L2_both_a_p6e1 | -0.531 | -0.694 | 0.163 | 0.23 | 0.093 |
| L2_both_w_int4 | 27 | 27 | 1.01e-04 | 3.7e-06 | 3.50e-04 |
| L27_both_wa_int4 | 909 | 909 | -1.26e-03 | 1.4e-06 | 8.62e-04 |
| L0_both_a_int8 | -1 | -1.03 | 0.024 | 0.023 | 0.0276 |

FP32 基準（native）：10 窗口 summed CE 合計 4443.8437。

- **有明顯 ΔL 的 5 組**：模擬與 native 的相對差 ≤ 3.7e-6。這 5 組包括單獨與聯合、權重與激活，還有 L2 聯合的 2.9 倍非加成（6977）和 L27 的次加成（909）。所以模擬重現了 pass 的 QDQ 語意，包括「每個 consumer 各自量化共用輸入」。
- **ΔL 接近 0 的 2 組**（p6e1 激活、int8 激活）：絕對差是 0.16 與 0.024，約每 token 1.3e-4 與 1.9e-5 nats，和第 5 節討論的 ΔL 量級比可以忽略。原因沒有追查；可能是前向的 FP32 差異讓少數元素跨過捨入邊界。

### 5.3 layer 2 明細（模擬）

| 對象 | 模式 | 格式 | C（總梯度） | C（分支） | Σ_s f_s² | Σ_s ΔL_s（帶號） | Σ_s abs(ΔL_s) |
|---|---|---|---:|---:|---:|---:|---:|
| gate | w | int4 | 57.8 | 57.8 | 24.3 | 0.0679 | 12.8 |
| gate | w | int8 | 0.177 | 0.177 | 0.124 | -0.877 | 0.895 |
| gate | w | p6e1 | 66.2 | 66.2 | 67.2 | 6.12 | 21.5 |
| gate | a | int4 | 4.63e+03 | 3.68e+03 | 6.22e+03 | 587 | 587 |
| gate | a | int8 | 65.8 | 38.5 | 13.4 | 12 | 15.9 |
| gate | a | p6e1 | 85.5 | 86.8 | 11.2 | 4.24 | 6.43 |
| gate | wa | int4 | 4.68e+03 | 3.74e+03 | 5.87e+03 | 589 | 589 |
| gate | wa | int8 | 66 | 38.6 | 12.4 | 10.7 | 16.5 |
| gate | wa | p6e1 | 152 | 153 | 97.6 | 17.9 | 27.4 |
| up | w | int4 | 76.1 | 76.1 | 86 | 25.9 | 28.3 |
| up | w | int8 | 0.234 | 0.234 | 0.189 | -0.79 | 1.23 |
| up | w | p6e1 | 142 | 142 | 64.5 | 24.6 | 24.7 |
| up | a | int4 | 3.08e+03 | 2.1e+03 | 2.44e+03 | 1.95e+03 | 1.95e+03 |
| up | a | int8 | 135 | 109 | 129 | 23.2 | 30.9 |
| up | a | p6e1 | 60.9 | 12.6 | 29.1 | -4.46 | 12.1 |
| up | wa | int4 | 3.16e+03 | 2.18e+03 | 2.39e+03 | 1.13e+03 | 1.13e+03 |
| up | wa | int8 | 135 | 109 | 104 | 21 | 28.8 |
| up | wa | p6e1 | 203 | 155 | 116 | 25.4 | 25.5 |
| both | w | int4 | 134 | 134 | 83.7 | 27 | 27 |
| both | w | int8 | 0.411 | 0.411 | 0.406 | -1.68 | 1.76 |
| both | w | p6e1 | 208 | 208 | 145 | 35 | 38 |
| both | a | int4 | 7.71e+03 | 5.78e+03 | 1.10e+04 | 7.26e+03 | 7.26e+03 |
| both | a | int8 | 201 | 148 | 192 | 30.3 | 35.8 |
| both | a | p6e1 | 146 | 99.4 | 42 | -0.694 | 9.99 |
| both | wa | int4 | 7.84e+03 | 5.92e+03 | 1.05e+04 | 6.98e+03 | 6.98e+03 |
| both | wa | int8 | 201 | 148 | 129 | 28.6 | 34.4 |
| both | wa | p6e1 | 355 | 308 | 223 | 39.9 | 43.5 |

layer 0、14、27 的完整數字在 `phase5_quant/analysis.csv`。

**int4 激活下，ΔL 遠大於一階和。** 以 L2 gate a int4 為例：ΔL 587，Σ_s f_s 只有 −50。這是二階（曲率）主導的區域，Fisher 型 criterion 在原理上適用，但量級比值會分散。

### 5.4 各項 proxy 的量級（模擬一階 Σf² 為分母）

| 項目 | 單獨（gate 或 up）median（範圍） | 聯合 median（範圍） |
|---|---|---|
| W / Σ(f_W)² | 1.11（0.587–78） | 1.38（0.502–9.59） |
| A_x(總梯度) / Σ(f_X)² | 1.68（0.336–14.2） | 1.07（0.436–12.5） |
| A_x(分支梯度) / Σ(f_X)² | **0.914**（0.173–7.53） | 0.631（0.22–7.13） |
| A_y / Σ(f_Y)² | 1.32（0.215–28.4） | 1.64（0.514–28.2） |
| C(總梯度) / Σf² | 1.37（0.567–78） | 1.21（0.45–9.59） |
| C(分支) / Σf² | 0.987（0.293–78） | 0.907（0.281–9.59） |

10 個窗口的 Σ_s f_s² 本身就有雜訊；若當成 10 個自由度的卡方，相對標準差約 45%。所以 0.3–3 倍的分散有一部分是抽樣造成的。

### 5.5 聯合 vs 單獨

| layer | 模式 | 格式 | ΔL 聯合 | ΔL gate + ΔL up | 差 | Σf² 聯合 | Σf² gate + up | 2Σ f_g f_u | SwiGLU 二階項 | C 聯合（分支） |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | w | int4 | 27 | 26 | 0.989 | 83.7 | 110 | -26.6 | 1.05 | 134 |
| 2 | a | int4 | 7.26e+03 | 2.53e+03 | 4.73e+03 | 1.10e+04 | 8.66e+03 | 2.3e+03 | 50.1 | 5.78e+03 |
| 2 | a | int8 | 30.3 | 35.3 | -4.97 | 192 | 142 | 49.5 | -2.96 | 148 |
| 2 | wa | int4 | 6.98e+03 | 1.72e+03 | 5.26e+03 | 1.05e+04 | 8.25e+03 | 2.29e+03 | 49.1 | 5.92e+03 |
| 0 | a | int4 | 1.33e+03 | 1.39e+03 | -52 | 3.78e+03 | 2.62e+03 | 1.15e+03 | -23.9 | 1.51e+03 |
| 0 | wa | int4 | 1.36e+03 | 1.29e+03 | 63.8 | 3.19e+03 | 2.33e+03 | 861 | -44.9 | 1.58e+03 |
| 14 | a | int4 | 85.4 | 104 | -18.5 | 303 | 222 | 81 | -11.6 | 222 |
| 14 | wa | int4 | 84.3 | 109 | -24.5 | 308 | 200 | 107 | -7.2 | 241 |
| 27 | w | int4 | -11.3 | -16.2 | 4.85 | 529 | 375 | 153 | 0.318 | 5.07e+03 |
| 27 | a | int4 | 863 | 1.74e+03 | -873 | 3.18e+04 | 1.60e+04 | 1.58e+04 | -320 | 8.96e+03 |
| 27 | a | p6e1 | 44.8 | 24.4 | 20.4 | 756 | 503 | 253 | 23.4 | 718 |
| 27 | wa | int4 | 909 | 1.76e+03 | -849 | 3.27e+04 | 1.65e+04 | 1.62e+04 | -329 | 1.40e+04 |

- 這裡只列有代表性的列，36 組完整的表在 `qwen3_swiglu_tables.py` 的 `joint` 段。
- C 聯合（分支）一律等於 C gate + C up，因為 criterion 是加總型的。
- **兩個主要的例子**：
  - L2 int4 激活：聯合 ΔL 是單獨和的 2.9 倍。
  - L27 int4 激活：聯合 ΔL 863，比單獨量化 gate（912）還小；但一階交叉項 +1.58e4 是正的。
- **SwiGLU 乘法二階項**：相對一階交叉項很小，L2 是 50 vs 2300，L27 是 −320 vs 1.58e4。
- **一階交叉項也解釋不了**：它的正負號解釋不了非加成的方向（見 5.6 節，11/22）。

### 5.6 排序相關與決策

| 模式 | criterion | Spearman(C, Σf²) | Spearman(C, abs ΔL) | Spearman(C, ΔL 帶號) | log-log Pearson(C, abs ΔL) | n |
|---|---|---:|---:|---:|---:|---:|
| w | 總梯度＝分支 | 0.947 | 0.753 | 0.341 | 0.754 | 36 |
| a | 總梯度 | 0.960 | 0.873 | 0.889 | 0.872 | 36 |
| a | 分支 | 0.960 | 0.873 | 0.894 | 0.869 | 36 |
| wa | 總梯度 | 0.923 | 0.897 | 0.929 | 0.838 | 36 |
| wa | 分支 | 0.920 | 0.880 | 0.912 | 0.819 | 36 |
| all | 總梯度 | 0.943 | **0.856** | 0.775 | 0.813 | 108 |
| all | 分支 | 0.940 | 0.850 | 0.769 | 0.805 | 108 |
| all | 上限：Σf² 本身 | — | 0.830 | — | — | 108 |

| 決策 | 依據 | 正確／總數 |
|---|---|---:|
| gate 與 up 哪支傷害較大（w 模式） | C（兩種歸屬相同） | 9 / 12 |
| 同上（a、wa 模式） | C 總梯度 | 14 / 24 |
| 同上（a、wa 模式） | C 分支 | 15 / 24 |
| 同上（w 模式） | 模擬一階 Σf² | 7 / 12 |
| 同上（a、wa 模式） | 模擬一階 Σf² | 12 / 24 |
| 聯合 ΔL 比單獨和大或小（差的絕對值 > 1） | 一階交叉項 2Σ f_g f_u 的正負號 | 11 / 22 |
| 聯合 ΔL 低於某一單獨 ΔL（單獨 ΔL > 1） | 加總型 criterion 恆預測聯合 ≥ 單獨 | 10 組違反 |

**解讀**
- **粗排序**：跨層、格式、模式的大小，C 排得很好。0.856 甚至高於模擬一階 Σf² 本身（0.830）；C 用每 channel 的統計量聚合，比 10 個窗口的 Σf² 雜訊少。
- **w 模式的帶號相關低（0.341）**：int8 權重的 ΔL 常是小的負值，例如 L2 gate 是 −0.88，屬於雜訊。
- **細部決策**：同層選 gate 或 up、選聯合或單獨，C 都接近隨機。
  - 理想的一階量 Σf² 一樣接近隨機，所以這不是歸屬問題，而是二階、可加、逐元素獨立模型在 int4 激活擾動下的解析度極限。
  - 誤判的例子差距並不小：L2 a int4 的 gate 是 587、up 是 1947；L0 a int4 的 gate 是 1208、up 是 178。
- **分支歸屬**：改善量級的校準，但沒有改善排序，也沒有改善這些細部決策（15/24 vs 14/24）。

## 6. 是否修改 criterion 或搜尋粒度

1. **criterion 的形式不改。** W／A 的定義、GΣ、channel 軸、聚合都已驗證正確，而且 native 與 PyTorch 一致。
2. **不加 gate／up 交互項。** 一階交叉項連非加成的正負號都只預測對 11/22，SwiGLU 二階項更小。加了不會讓聯合預測更準，反而會讓 MILP 多出需要線性化的二次項。
3. **分支歸屬：保留為選項，預設暫時不變。**
   - 支持採用的理由：它是 per-edge QDQ 的正確敏感度，消除了單獨量化時約 1.9 倍的高估（量級比值 1.68 → 0.91）。
   - 暫緩的理由：
     - 108 組的排序沒有改善；
     - 全模型 MILP 選擇改變了 33–89/196 個節點；
     - 聯合量化時，現行「算兩次」的量級（1.07）反而接近真值，分支和是 0.63。
   - 要決定預設，需要一次全模型比較：兩種 table 的 knee 或固定 budget 選擇各 native 編譯一次，比 calibration loss；決定之後才報 held-out PPL。估計 10–15 分鐘，本次沒有做。
4. **搜尋粒度：權重維持逐節點；共享輸入的激活有一個選項。**
   - 粗排序已經達到 10 窗口一階量的上限，逐節點的 MILP 沒有改的理由。
   - 選項是把共享輸入的激活改成「一個 tensor 一個決策」：只插一個 QDQ，成本用總梯度算一次。
   - 好處：
     - 語意更乾淨：同一個 tensor 不會有兩份不同格式的量化複本，部署時本來也只量化一次；
     - 歸屬問題直接消失；
     - 在 q／k／v 上少兩個 QDQ。
   - 本實驗沒有證據顯示它能改善預測：聯合時，算一次的 A_x 只有真值的 0.53。另外它需要同時修改 pass 與 MILP。所以列為設計選項，不是修正。
5. **激活項的量級誤差不是梯度歸屬造成的。** 聯合時 L2 int4 的輸入項低估約 4.5 倍，L2 p6e1 高估 3.7–7 倍，來源是逐元素獨立假設不成立。推測（未驗證）與 massive activation 和 outlier token 有關，這和先前在 mbv2 上的發現方向一致：不均勻的激活範圍造成 int4 激活的大部分損傷。若要處理，應從範圍或量化方式著手。

## 7. 時間與限制

### 7.1 各階段耗時

| 階段 | 內容 | 時間 |
|---|---|---:|
| 1 | 梯度重現（2 窗口）：native grads build + 每窗口 50 s；PyTorch FP32 每窗口 3.8 s、FP64 每窗口 4.6 s | 約 220 s（修正前後各一次） |
| 2 | 獨立 SwiGLU，4 種 case（建置 + 執行 + FD） | 約 7 s |
| 3 | 前向 tap：onnx-mlir build 約 22 s 加執行、ORT、PyTorch | 約 97 s（修正前後各一次） |
| — | 二分定位（暫存腳本） | 未記錄 |
| — | criterion 重建（Qwen，eps 修正 + 分支表）：import 10 s、bytecode 12 s、grads build 86 s、weights 178 s、ranges build 188 s、grads 執行 238 s、errors 723 s（7 shards） | 1074 s |
| — | criterion 重建：GPT-2、BERT | 見 `native_eps_fix_20261014/summary.json` |
| 4 | native 統計 80 次執行，平均 65 s，並行 | 715 s |
| 4 | PyTorch 統計（每窗口 3.7 s）、比較 | 約 46 s |
| 5 | PyTorch 模擬：基準每窗口 1.3 s；108 組每組 5.2 s | 584 s |
| 5 | native 驗證：8 次編譯（每次約 40–66 s）+ 10 窗口執行，並行 2 | 774 s |
| 5 | 分析、表格、MILP 選擇比較 | 每項數十秒以內 |

### 7.2 限制

1. **樣本量小**：只有 10 個 calibration 窗口 × 127 個預測位置。每組設定的 Σf² 與 ΔL 都只有 10 個樣本，小的 ΔL（abs 值 < 1，約占總 loss 的 0.02%）基本上是雜訊。
2. **第 5 階段主要是 PyTorch 模擬**，只有 7 組用 native 編譯驗證（5.2 節）。有明顯 ΔL 的組，模擬與 native 的相對差 ≤ 4e-6；但其餘 101 組沒有逐一用 native 執行。ΔL 接近 0 的組有約 0.02–0.16 的絕對差，原因沒有追查。
3. **只測了 gate／up 的 4 層。** q／k／v 也共用輸入，而且有 3 個 consumer，第 5 階段沒有測；分支歸屬對它們的影響只反映在 MILP 選擇的變化上。
4. **單獨與聯合只測了同一格式。** 聯合時 gate 與 up 用同一格式，沒有測混合格式（例如 gate int4、up int8）。
5. **先前 BERT 與 Qwen 的下游結果**（auto pick、PPL、frontier）是在 epsilon bug 下算的，沒有重跑。criterion 本身的變化見 2.4 節。
6. **分支歸屬對全模型的影響沒有量測。**
7. **修正後 native 梯度還有少量誤差**：lm_head 的長 K f32 累加，rel_l2 8.6e-5；低層的條件數差。都在門檻內，但並不比 PyTorch FP64 精確。
8. **PyTorch FP32 在 CPU 上不完全確定**（多執行緒），所以以 FP64 為參考。
9. **定位 RMSNorm 的二分腳本已刪除**，不能直接重跑。可重跑的證據是 lit 測試與第 1、3 階段的修正前後結果。

## 8. 程式修改與回歸結果（全部未提交）

| 檔案 | 修改 |
|---|---|
| `src/Dialect/ONNX/Transforms/Recompose.cpp` | RMS／LayerNorm 重組讀取 dense 形式的 epsilon |
| `test/mlir/onnx/onnx_recompose_epsilon.mlir`（新增） | epsilon 1e-6、1e-12 保留；修正前失敗，修正後通過 |
| `src/Conversion/ONNXToMixedPrecision/MixedPrecisionBackward.cpp`、`.hpp` | 記錄每個 Conv／Gemm／MatMul 對輸入梯度的 per-edge 貢獻 |
| `src/Conversion/ONNXToMixedPrecision/ONNXToMixedPrecision.cpp` | `MIXEDP_STATS_INPUT_GRAD=branch`；weights pass 的 fake-quant 改成只取值、不建 attribute（Qwen weights pass 峰值記憶體 22.4 → 7.6 GB，mbv2 與 Qwen 輸出完全相同） |
| `test/mlir/conversion/onnx_to_mixed_precision/collect_stats_grads_branch.mlir`（新增） | branch 與 total 兩種 layout |
| `experiments/mixed_precision/run_native_bc_pipeline.py` | `--input-grad {total,branch}`、`table_branch.json`、A 聚合抽成 helper；`--criterion-only` 在沒有 results.json 時改用 criterion.json |
| `experiments/mixed_precision/qwen3_swiglu_*.py`、`swiglu_backward_check.py`（新增） | 本研究的腳本 |

**回歸結果**
- **lit**：`onnx_to_mixed_precision` 目錄 20/20 通過；加上 epsilon 與 bytecode 測試，共 22/22。
- **既有失敗**：`krnl_to_affine_matmul.mlir`、`onnx_recompose*.mlir`（properties 列印）、`driver/static_quantization.mlir`、`driver/compile_phases.mlir`。修改前就失敗，與本次修改無關。
- **criterion 重建**：GPT-2 完全不變；BERT 與 Qwen 的變化來自 epsilon 修正（2.4 節）。
- **預設行為**：沒有設定 `MIXEDP_STATS_INPUT_GRAD` 時，layout 與 table 不變。

## 9. 重跑

在 `experiments/mixed_precision/` 下，`PY=../../venv/bin/python`，`R=runs/qwen3_swiglu_20261013`：

```bash
$PY qwen3_swiglu_grads.py --out-dir $R/phase1_grads_fixed                      # 第 1 階段
$PY swiglu_backward_check.py --out-dir $R/phase2_swiglu_branch                # 第 2 階段
$PY qwen3_swiglu_forward.py --out-dir $R/phase3_forward_fixed                 # 第 3 階段
$PY run_native_bc_pipeline.py --models gpt2,bert,qwen3 --criterion-only \
    --compare-dir runs/native_bytecode_20261012 --out-dir runs/native_eps_fix_20261014        # eps 修正後重建
$PY run_native_bc_pipeline.py --models qwen3 --criterion-only --input-grad branch \
    --compare-dir runs/native_bytecode_20261012 --out-dir runs/native_eps_fix_20261014        # Qwen：table + table_branch
N=runs/native_eps_fix_20261014/qwen3
$PY qwen3_swiglu_criterion.py --stage native --native-run $N --out-dir $R/phase4_criterion   # 第 4 階段
$PY qwen3_swiglu_criterion.py --stage torch --native-run $N --out-dir $R/phase4_criterion
$PY qwen3_swiglu_criterion.py --stage compare --native-run $N --out-dir $R/phase4_criterion
$PY qwen3_swiglu_quant.py --stage sim --native-run $N --out-dir $R/phase5_quant             # 第 5 階段
$PY qwen3_swiglu_quant.py --stage native --native-run $N --out-dir $R/phase5_native
$PY qwen3_swiglu_quant.py --stage analyze --native-run $N --criterion-dir $R/phase4_criterion --out-dir $R/phase5_quant
$PY qwen3_swiglu_tables.py                                                    # 本報告的表格
```

原始資料（JSON、CSV）都在 `$R/phase*/` 與 `runs/native_eps_fix_20261014/`。
