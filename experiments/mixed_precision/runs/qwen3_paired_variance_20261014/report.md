# Qwen3-0.6B：activation 變異項的位置配對（V_paired）對照

日期：2026-10-14　腳本：`experiments/mixed_precision/run_qwen_paired_variance.py`　輸出：本目錄

**這是事後分析（post-hoc）**：calibration 與 tuning 窗口都已被前面的研究分析過，本結果不是獨立驗證，也不代表最終 accuracy 有任何改善。

## 結論

1. **中心化的位置配對沒有改善，整體反而變差。**

   | 指標（tuning） | A：原完整 W + V_old + U | B：新完整 W + V_paired + U |
   |---|---:|---:|
   | 全部配對（3151） | 2721 | 2692（修正 29，弄壞 58，淨 −29） |
   | 相同 M 與 B 的配對（166） | 108 | 98 |
   | activation-only 組內 Spearman 平均 | 0.805 | 0.782 |

   - **bootstrap**（B − A，逐窗口 paired bootstrap 的 95% 區間）：
     - 全部配對正確率：tuning [−0.019, −0.0006]，calibration [−0.015, −0.002]；
     - 相同資源配對：tuning [−0.088, 0.000]，calibration [−0.093, −0.012]。
   - **只看變異項**（C → D）也一樣：2739 → 2711，修正 38、弄壞 66。
   - **改善只出現在 L2**：L2 activation-only 修正 17、弄壞 9；L2 weight+activation 修正 5、弄壞 1。另外兩邊都含 int4 的配對修正 7、弄壞 2。
   - **主要退步在 L27**：activation-only 修正 4、弄壞 31；weight+activation 修正 3、弄壞 13。不含 int4 的配對修正 22、弄壞 56。
2. **修正與弄壞的案例**
   - 修正的 29 對：22 對在 L2，主要是 p6e1、int8 相關的比較；7 對在 L27，是 int4 與 p6e1／p8e1 的組合。
   - 弄壞的 58 對：44 對在 L27，56 對不含 int4，多半是 p6e1、p8e1、int8、bf16 之間的比較，loss 差多在 20–100。
   - 其中包括 knee 會用到的格式，例如 L27 的 int6/p8e1 對 int8/int8（ΔS_tune −207）。
3. **配對打亂控制支持「新舊差異來自位置關聯」。**
   - 在每個 channel 內把 r 隨機重排 20 次，平均剛好回到 V_old：60 個 tap × 格式的 z 平均 +0.01。全部配對的正確率平均 2717.8，範圍 [2696, 2733]，和 A 的 2721 一致。
   - B 的 2692 低於打亂的最小值；L2 activation-only 的 B（459）高於打亂最大值（456）；L27 activation-only 的 B（447）低於打亂最小值（458）。
   - 所以 L2 的改善與 L27 的退步，都是真的來自 g² 與 r² 在同位置的關聯，不是隨機波動。只是這個關聯在 L27 把權重推向 position 0：x tap 的 p6e1 有 96% 的 V_paired 落在 position 0，int4 有 57%。而前面的介入實驗已顯示，L27 的 position 0 實際 loss 響應約為 0 或為負。
4. **仍然較符合大擾動／下游非線性解釋的錯誤：**
   - L0 與 L2 的「int4 放 Gate 還是 Up」，A、B、C、D 全錯，只有舊 raw diagonal（E）在 L2 判斷對。
   - L0 兩組完全沒有變化（修正 0、弄壞 0）。
   - |ΔS| > 100 的大差異配對幾乎不變（1470 → 1469）。
   - 這些和先前 α 掃描的發現一致：int4 誤差把 position 0 的 attention sink 推出二次範圍，任何二階、局部的變異項都改善不了。
   - L27 的 int4 Gate／Up 判斷，只有 D（新 variance-only）與 E 判斷對；B 因為保留了平均偏差項 U（L27 int4 的 U 佔 34–38%），仍然判斷錯。這個改善也沒有延伸到 L27 的其他格式，那些格式反而退步。
5. **不建議進入新資料驗證，也不建議做 native 實作。** 主要對照 A → B 在兩組資料上都退步，CI 不含 0；改善只局限在 L2。依照事前約定接受這個負結果，不再依 tuning loss 調整公式。

## 1. 符號與對應

| 本報告 | 意義 | 現有程式／table.json |
|---|---|---|
| W | weight criterion | table.json 的 `B` |
| A | activation criterion | table.json 的 `C` |
| C | W + A | `BC_total`（scores.json） |
| M | model（weight）size | `resources[0]`（權重位元和） |
| B | BOPs | `resources[1]`（Σ bits(w)·bits(a)） |

方法標籤 A–E（見第 2 節）與上表的符號是兩套東西：方法 A 指「原完整 criterion」，不是 activation criterion。

## 2. 公式（float64；calibration 窗口 s = 0–9，全部 128 個位置 t，channel c）

**每個 layer、branch、tap 與格式**
- tap 有兩種：x 是該分支的 X 複本，用 total input gradient；y 是 zg 或 zu。
- d = QDQ(x) − x，用 pass 的 QDQ。int 的 scale／zp 來自 calibration range，posit 用和 Universal 一致的 LUT。
- g = FP32 loss 對 tap 的梯度，由快取的 gY 經 FP32 block 精確推得。
- n = S·T = 1280，μ_c = Σ_{s,t} d / n，r = d − μ_c。
- G2_c = Σ_{s,t} g²；GS_c = Σ_s(Σ_t g)²，每個窗口先加總再平方。

**各項**
- V_old = Σ_c G2_c · Σ_{s,t} r² / n（native 的變異項）
- U = Σ_c GS_c · μ_c²（native 的平均偏差項）
- V_paired = Σ_{s,t,c} g² r²（不除以 n）

**候選成本**（量化分支的 x、y tap 相加；W 取自 table.json 的 `B`）

| 標籤 | 公式 | 用途 |
|---|---|---|
| A | W + V_old + U | 原完整，主要 baseline |
| B | W + V_paired + U | 新完整，主要對照 |
| C | W + V_old | 原 variance-only |
| D | W + V_paired | 新 variance-only |
| E | W + Σ g²d² | 舊 raw diagonal，未中心化，只作歷史對照 |
| 打亂 | W + V_shuffled + U 與 W + V_shuffled | r 在每個 channel 內對全部 n 個位置均勻重排，20 個固定 seed |

**刻意不改的部分**
- 不刪 position 0、不刪 outlier channel、不加交互項、不重算量化後的梯度、不改係數。
- 量化 range、scale、zp、posit LUT、候選、資源成本都不變。
- 不使用 branch gradient。

## 3. 範圍

**候選**：程式核對後是 activation 相關的 236 個。
- activation-only：每層 35 個（Gate 5 + Up 5 + 聯合 25），共 140 個。
- weight+activation：每層 24 個（4 + 4 + 16），共 96 個。

**層**：0、2、14、27。

**activation 格式**：只有 int4、int8、p6e1、p8e1、bf16 五種，**不是全部十種**。weight+activation 的選項是 int4/int4、int6/p8e1、int8/int8、p6e1/p6e1。

**loss**：既有的 PyTorch 模擬每窗口 summed CE 變化（block criterion 研究已用 native 核對）。
- S = Σ_s ΔL_s，帶號。
- 每 token 的平均 NLL 變化 = S ÷（窗口數 × 127）：tuning 是 S/2540，calibration 是 S/1270。

**配對**：
- 只在同一組（層 × 模式）內比較，|ΔS| > 1（TAU，沿用既有設定）。
- 所有方法用同一組配對：tuning 3151 對，calibration 3061 對，全部候選配對 3484 對。

## 4. 數值檢查與 baseline 一致性

**數值檢查**

| 檢查 | 結果 |
|---|---|
| QDQ 不改變輸入（fp32） | 變異項、平均偏差項都是 0 |
| 誤差在 channel 內為常數 | 兩種變異項 ≈ 6.6e-30（即 0），U > 0 |
| 梯度在 channel 內為常數 | V_paired = V_old，相對差 5.7e-16 |
| 逐 channel 恆等式 V_paired,c − V_old,c = n·Cov_pop(g², r²) | 最大相對誤差 2.3e-16 |
| 打亂期望 E[V_shuffled] = V_old | 60 個 tap × 格式，z = (平均 − V_old)/SE：平均 +0.01，最大 3.7（1 個 > 3，以 20 次重排的 t 分佈尾部來看合理） |
| W、U 在 A 與 B 中完全相同 | 是 |
| NaN／Inf | 0 |

**baseline 一致性**
- A 的 activation 部分對 native table C：median 相對差 1.3e-4，max 7.9e-4。A 對 BC_total：max 7.9e-4。
- 舊評估完全重現（tuning，全部配對）：A 2721、C 2739、E 2750；相同資源 A 108、C 112，與先前的 attribution 和 variance stage 相同。
- 以 native BC_total 排序，結果也與 A 完全一樣（2721）。

## 5. 結果

### 5.1 配對排序（tuning；calibration 見 `tables.md`，結論相同）

表中每格是正確的配對數；「修正／弄壞」是相對 A（B 欄）或相對 C（D 欄）。

| 配對類別 | 配對數 | A | B | B vs A | C | D | D vs C | E | 打亂平均 [min, max] |
|---|---:|---:|---:|---|---:|---:|---|---:|---|
| 全部 | 3151 | 2721 | 2692 | +29／−58 | 2739 | 2711 | +38／−66 | 2750 | 2717.8 [2696, 2733] |
| 相同 M 與 B | 166 | 108 | 98 | +1／−11 | 112 | 107 | +8／−13 | 116 | 107.4 [106, 108] |
| activation-only | 2084 | 1780 | 1757 | +21／−44 | 1790 | 1768 | +29／−51 | 1793 | 1776.5 |
| weight+activation | 1067 | 941 | 935 | +8／−14 | 949 | 943 | +9／−15 | 957 | 941.3 |
| Gate／Up 擇一（同選項） | 26 | 14 | 11 | +0／−3 | 15 | 12 | +1／−4 | 14 | 14.1 |
| Gate／Up 互換（聯合） | 58 | 29 | 23 | +1／−7 | 31 | 29 | +6／−8 | 34 | 28.8 |
| 兩邊都含 int4 | 331 | 159 | 164 | +7／−2 | 175 | 190 | +17／−2 | 231 | 158.6 |
| 只有一邊含 int4 | 1596 | 1596 | 1596 | 0 | 1596 | 1596 | 0 | 1596 | 1596 |
| 都不含 int4 | 1224 | 966 | 932 | +22／−56 | 968 | 925 | +21／−64 | 923 | 963.2 |
| 兩支單獨 | 255 | 221 | 214 | +2／−9 | 222 | 215 | +3／−10 | 217 | 220.1 |
| 單獨 vs 聯合 | 1329 | 1148 | 1141 | +11／−18 | 1153 | 1147 | +15／−21 | 1161 | 1147.7 |
| 兩支聯合 | 1567 | 1352 | 1337 | +16／−31 | 1364 | 1349 | +20／−35 | 1372 | 1350.0 |
| \|ΔS\| 1–5 | 501 | 364 | 360 | +1／−5 | 364 | 356 | +1／−9 | 356 | 363.6 |
| \|ΔS\| 5–20 | 336 | 220 | 217 | +12／−15 | 220 | 217 | +12／−15 | 217 | 218.6 |
| \|ΔS\| 20–100 | 726 | 667 | 646 | +9／−30 | 669 | 644 | +8／−33 | 642 | 665.5 |
| \|ΔS\| > 100 | 1588 | 1470 | 1469 | +7／−8 | 1486 | 1494 | +17／−9 | 1535 | 1470.2 |

**各組**（tuning，全部配對；B 對 A）

| 組 | 配對數 | A | B | 修正／弄壞 | 打亂平均 [min, max] |
|---|---:|---:|---:|---|---|
| L0 a | 526 | 410 | 410 | 0／0 | 409.6 |
| L0 wa | 267 | 235 | 235 | 0／0 | 234.9 |
| L2 a | 544 | 451 | **459** | +17／−9 | 451.1 [442, 456] |
| L2 wa | 270 | 237 | **241** | +5／−1 | 237.2 [236, 239] |
| L14 a | 453 | 445 | 441 | +0／−4 | 444.9 |
| L14 wa | 256 | 245 | 245 | 0／0 | 245.0 |
| L27 a | 561 | 474 | **447** | +4／−31 | 470.9 [458, 482] |
| L27 wa | 274 | 224 | **214** | +3／−13 | 224.2 [222, 227] |

### 5.2 組內 Spearman（tuning S）

| 組 | A | B | C | D | E | 打亂（完整）平均 |
|---|---:|---:|---:|---:|---:|---:|
| L0 a | 0.677 | 0.677 | 0.677 | 0.677 | 0.677 | 0.676 |
| L2 a | 0.799 | **0.829** | 0.804 | 0.829 | 0.864 | 0.800 |
| L14 a | 0.869 | 0.861 | 0.869 | 0.861 | 0.861 | 0.869 |
| L27 a | 0.876 | **0.761** | 0.891 | 0.772 | 0.772 | 0.863 |
| activation-only 平均 | 0.805 | 0.782 | 0.810 | 0.785 | 0.793 | 0.802 |
| L0 wa | 0.897 | 0.897 | 0.897 | 0.897 | 0.897 | 0.896 |
| L2 wa | 0.896 | 0.916 | 0.896 | 0.916 | 0.971 | 0.898 |
| L14 wa | 0.959 | 0.959 | 0.959 | 0.959 | 0.959 | 0.959 |
| L27 wa | 0.841 | 0.779 | 0.873 | 0.814 | 0.808 | 0.842 |

calibration 的 activation-only 平均：A 0.808、B 0.785、C 0.813、D 0.788、E 0.797。

### 5.3 int4 Gate 對 Up（activation-only 單支；判斷對／錯為 calibration／tuning）

| 層 | 實際 Gate − Up | A | B | C | D | E |
|---|---|---|---|---|---|---|
| L0 | +1030／+371 | 錯／錯 | 錯／錯 | 錯／錯 | 錯／錯 | 錯／錯 |
| L2 | −1360／−2536 | 錯／錯 | 錯／錯 | 錯／錯 | 錯／錯 | 對／對 |
| L14 | −26／−67 | 對／對 | 對／對 | 對／對 | 對／對 | 對／對 |
| L27 | +87／+125 | 錯／錯 | 錯／錯 | 錯／錯 | **對／對** | 對／對 |

### 5.4 逐窗口 paired bootstrap

**方法**
- 窗口有放回地重抽 2000 次，每次所有方法使用同一個重抽。
- 每次重新依 |ΔS| > 1 篩選配對；預測值固定，因為它只用 calibration 計算。
- 配對之間高度相關，所以不用二項信賴區間。

**95% 區間（差值）**

| 差值 | 資料 | 全部配對正確率 | 相同資源 | activation-only Spearman 平均 |
|---|---|---|---|---|
| B − A | tuning | [−0.019, −0.0006] | [−0.088, 0.000] | [−0.042, +0.003] |
| B − A | calibration | [−0.015, −0.002] | [−0.093, −0.012] | [−0.037, −0.0004] |
| D − C | tuning | [−0.018, +0.0006] | [−0.057, +0.039] | [−0.043, +0.004] |
| D − C | calibration | [−0.014, −0.002] | [−0.063, +0.028] | [−0.039, +0.0005] |

### 5.5 為什麼 L2 變好、L27 變差：position 0（同一個全域 μ）

V_paired／V_old 與 position 0 佔比（全表見 `tables.md`；逐 channel、逐 position 的數值見 `terms.npz`）：

| tap | 格式 | V_paired／V_old | position 0 佔 V_paired | position 0 佔 V_old（梯度側） | position 0 佔 V_old（誤差側） |
|---|---|---:|---:|---:|---:|
| L2 x | int4 | 1.26 | 0.002 | 0.001 | 0.016 |
| L2 x | p6e1 | 0.50 | 0.034 | 0.005 | 0.015 |
| L2 x | p8e1 | 0.50 | 0.026 | 0.004 | 0.017 |
| L27 x | int4 | 0.51 | 0.568 | 0.798 | 0.003 |
| L27 x | p6e1 | 5.15 | 0.957 | 0.726 | 0.094 |
| L27 y_gate | p6e1 | 0.11 | 0.671 | 0.973 | 0.001 |
| L27 y_up | p8e1 | 0.65 | 0.729 | 0.820 | 0.009 |

- **L2**：position 0 的份量很小，配對主要降低 posit 的成本（約 0.5 倍）。這符合 posit 誤差集中在梯度小的 outlier channel，排序因此改善。
- **L27**：position 0 的梯度極大。配對後，V_paired 在多個 tap 與格式上大半落在 position 0，比例又因格式差異很大（0.11–5.2 倍），格式之間的相對成本被打亂。而 L27 的 position 0 實際 loss 響應約為 0 或為負（見 `runs/qwen3_error_decomposition_20261014` 的介入與 α 掃描）。

## 6. 耗時

| stage | 內容 | 時間 |
|---|---|---:|
| freeze | 設定、候選清單、hash | 4 秒 |
| compute | 60 個 tap × 格式的逐 channel／position 項，以及 20 次重排 | 160 秒 |
| evaluate | 236 個候選、3484 對、2000 次 bootstrap | 20 秒 |
| tables | — | < 1 秒 |

沒有執行任何模型推論；只用既有的 cache 與逐窗口 loss。

## 7. 限制

- **事後分析**：calibration 與 tuning 窗口都已被前面的研究分析過。
- **格式與範圍**：只有 5 種 activation 格式、4 層、Down 固定 FP32、單一 block 的局部量化。沒有 MILP、knee 或全模型搜尋。
- **配對不獨立**：大量配對高度相關，正確對數不能當成獨立樣本解讀；區間只來自逐窗口 bootstrap。
- **全模型 loss 是 PyTorch 模擬**：native 的一致性已在 block criterion 研究中核對。

## 8. 是否進入下一步

**不建議。** 主要對照 A → B 在 calibration 與 tuning 上都退步，bootstrap 區間不含 0；局部的 L2 改善被 L27 的退步抵消有餘；大擾動錯誤（L0、L2 的 int4 Gate／Up）完全不受影響。依約定接受負結果，不再依 tuning loss 調整公式，也不替換正式 pipeline。

若未來仍要研究位置配對，native 端需要的統計如下（本次不實作）：

- **需要的累加量**：V_paired,c = Σ g²d² − 2μ_c Σ g²d + μ_c² Σ g²。除了既有的 G2_c 與 μ_c，每個 channel 還需要 Σ g²d² 與 Σ g²d。
- **與現有架構不相容**：這兩項要求同一個 run 裡同時有逐元素的 g 與 d。目前 native 的 grads（backward）與 errors（forward）是分開的 run，只輸出 channel 層級的聚合量，無法還原。

## 9. 重跑與檔案

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/qwen3_paired_variance_20261014   # 新實驗請換新目錄
$PY run_qwen_paired_variance.py --stage freeze   --out-dir $O
$PY run_qwen_paired_variance.py --stage compute  --out-dir $O
$PY run_qwen_paired_variance.py --stage evaluate --out-dir $O
$PY run_qwen_paired_variance.py --stage tables   --out-dir $O
```

| 檔案 | 內容 |
|---|---|
| `config.json` | 凍結的設定、候選清單、符號對應 |
| `environment.json` | 程式、模型、cache、table、loss 檔的 sha256 |
| `terms.json` | 各 tap × 格式的總值、打亂結果、數值檢查 |
| `terms.npz` | 逐 channel 的 V_old、V_paired、U、Cov、raw diag、每次重排；逐 position 的 V_paired 與 V_old（梯度側、誤差側） |
| `costs.csv` | 每個候選的 W、V_old、V_paired、U、raw diag、native A、A–E 與 S |
| `pairs.csv` | 每個配對的類別、ΔS 與各方法的對錯 |
| `summary.json` | 所有指標、bootstrap、B 對 A 改變的配對清單 |
| `tables.md` | 完整表格 |
| `timing.json` | 各 stage 耗時 |
