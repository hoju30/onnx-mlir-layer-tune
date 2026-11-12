# Qwen3-0.6B：FD 曲率能否穩定改善同層候選選擇（擴大、預先登記驗證）

日期：2026-10-15　腳本：`experiments/mixed_precision/run_qwen_fd_validation.py`　輸出：本目錄

## 結論

**未通過預先登記的標準。依計畫不進入第二關（完整搜尋的 accuracy 與總時間），保留原 BC。**

| 通過標準 | 結果 |
|---|---|
| (1) 決定性同資源配對上，FD 正確率高於 BC，且 bootstrap 95% 區間大於 0 | **未通過**：FD 200/264，BC 216/264；區間 [−0.072, 0.000] |
| (2) 5 層中至少 4 層、兩種模式都不比 BC 差 | **未通過**：只有 2 層（L14、L21）；weight-only 模式明顯較差（88 vs 109） |

**細節**
- **activation-only 有微弱優勢**：決定性配對 FD 112、BC 107（FD 修正 15、弄壞 10）。其中 Gate／Up 擇一 FD 7/9、BC 5/9；int4 Gate／Up 是 FD 2/3、BC 1/3。
- **先前 3/3 的 int4 結果沒有完全延伸**：新加的 L7 上 FD 錯、BC 對。
- **weight-only 是主要退步**：決定性配對 FD 88、BC 109（修正 7、弄壞 28）。組內 Spearman 在 L0 是 0.02 vs 0.64，L27 是 −0.14 vs 0.17。
- **Gate+Up 聯合**：互換配對 19 vs 19，其他聯合配對 98 vs 111，FD 沒有改善。
- **原因推測（未驗證）**：weight-only 的誤差同樣作用在 position 0 的巨大輸入上。先前已經發現 sink 位置的曲率隨尺度改變，h = 1/16 的有限差分在那裡不可靠。

## 1. 設計（執行前凍結於 `plan.json`）

**範圍**
- **層**：0、7、14、21、27。L7、L21 先前從未分析過。
- **模式**：activation-only（權重 FP32）與 weight-only（激活 FP32）。

**候選**（每層每模式 28 個，共 280 個）
- Gate-only、Up-only × {int4, p4e1, int6, p6e1, int8, p8e1}；
- Gate+Up × {int4, int6, int8, p6e1}²。

**量化**：per-consumer QDQ；校準 range 同 `runs/native_eps_fix_20261014/qwen3`；權重每個輸出 channel 量化；posit 用和 Universal 一致的 LUT（含 p4e1）。

**分數**（只用 calibration 窗口 0–9）
- **FD**：在 block 輸出 Y，把 Y_fp32 加減 h·e（h = 1/16）重播下游，取對稱差分的二階項 Σ_s (ΔL_s(h) + ΔL_s(−h)) / 2h²。不使用一階項，因為它不會跨窗口轉移。
- **BC**：native table 中量化分支的 W + A，table.json 的 B + C。

**實際損傷**：在全新的 validation 窗口 90–119（30 個，任何先前研究都沒用過）上，完整量化的 summed next-token CE 變化。

**配對**
- 範圍：同層、同模式，model size 與 BOPs 都相同的配對，共 450 個；其中 |ΔS| > 1 的有 420 個。
- 「決定性」：新窗口 bootstrap 2000 次中，至少 95% 保持同一順序。共 264 個。

**順序控制**：計畫凍結 → FD 分數算完後凍結（sha256 記在 `freeze.json`，02:24）→ 才量新窗口。`truth` stage 會先核對 hash 才執行。

## 2. 結果

**決定性同資源配對**（264）

| 類別 | 配對 | FD 對 | BC 對 | FD 修正／弄壞 |
|---|---:|---:|---:|---|
| 全部 | 264 | 200 | **216** | +22／−38 |
| L0 | 48 | 32 | 40 | +7／−15 |
| L7 | 51 | 40 | 45 | +1／−6 |
| L14 | 53 | 49 | 48 | +2／−1 |
| L21 | 30 | 22 | 22 | 0／0 |
| L27 | 82 | 57 | 61 | +12／−16 |
| activation-only | 130 | **112** | 107 | +15／−10 |
| weight-only | 134 | 88 | **109** | +7／−28 |
| Gate／Up 擇一 | 27 | 18 | 17 | +5／−4 |
| Gate／Up 互換（聯合） | 32 | 19 | 19 | +8／−8 |
| 同分支 int 對 posit | 38 | 34 | 36 | +1／−3 |
| 不同分支、不同格式 | 35 | 31 | 33 | +1／−3 |
| 其他聯合配對 | 132 | 98 | 111 | +7／−20 |

**全部同資源配對**（|ΔS| > 1，420 個）：FD 303、BC 313（修正 42、弄壞 52）。activation-only 165 vs 159，weight-only 138 vs 154。

**組內 Spearman**（新窗口 S，每組 28 個候選）

| 組 | FD | BC |
|---|---:|---:|
| L0 a | 0.723 | 0.670 |
| L0 w | 0.016 | 0.644 |
| L7 a | 0.869 | 0.956 |
| L7 w | 0.837 | 0.730 |
| L14 a | 0.921 | 0.924 |
| L14 w | 0.929 | 0.924 |
| L21 a | 0.887 | 0.880 |
| L21 w | 0.432 | 0.428 |
| L27 a | 0.956 | 0.900 |
| L27 w | −0.141 | 0.174 |

**Gate／Up 擇一的決定性配對，依格式**（事後拆解）

| 模式 | 格式 | 配對 | FD 對 | BC 對 |
|---|---|---:|---:|---:|
| a | int4 | 3 | 2 | 1 |
| a | int6 | 2 | 2 | 1 |
| a | int8 | 1 | 1 | 1 |
| a | p6e1 | 1 | 0 | 1 |
| a | p8e1 | 2 | 2 | 1 |
| w | int4 | 2 | 0 | 0 |
| w | int6 | 2 | 1 | 1 |
| w | int8 | 3 | 2 | 1 |
| w | p4e1 | 5 | 4 | 5 |
| w | p6e1 | 4 | 4 | 4 |
| w | p8e1 | 2 | 0 | 1 |

activation-only 的 int4 Gate／Up：L0（ΔS +821）FD 對、BC 錯；L27（+420）FD 對、BC 錯；L7（−132）FD 錯、BC 對。

**檢查**
- calibration 窗口重播 Y 的 rel_max ≤ 5.4e-7。
- 新窗口下游與完整模型 loss 的最大差 ≤ 1.6e-4。
- 重新 tokenize 的 calibration 窗口和 `validation_tokens.npy` 完全一致。

## 3. 成本

| 階段 | 時間 | 內容 |
|---|---:|---|
| 擷取 FP32 張量（calibration 10 + 新窗口 30，5 層） | 27 秒 | 共用前處理 |
| FD 分數（280 個候選） | 1567 秒（L0 517、L7 412、L14 324、L21 200、L27 115） | 5650 個下游 forward 窗口次，約 5.6 秒／候選 |
| 新窗口實際損傷（只用於驗證） | 2319 秒 | — |

外推：若對完整表格（196 節點 × 10 格式）做 FD，約 3 小時（PyTorch、CPU）；原 native BC 的完整表格是 1074 秒。

## 4. 判斷與建議

- **保留原 BC**，FD 不進入搜尋。依預先登記的規則，不做第二關。
- **activation-only 有微弱訊號**（決定性配對多對 5 對，Gate／Up 7/9 vs 5/9），但：
  - 只有 9 個 Gate／Up 判斷；
  - weight-only 的退步遠大於這個改善；
  - 這是事後拆解，不足以支持「只在 activation-only 用 FD」這種選擇性採用。若要測，必須重新預先登記，並用新的窗口。
- **weight-only 的退步原因推測**：權重誤差 × position 0 的巨大輸入，讓 Y 誤差集中在 sink；FD 在那裡的有限尺度曲率不穩定，與 `runs/qwen3_curvature_transfer_20261015` 的發現一致。本研究沒有直接驗證。
- **對整條研究線的意義**：Gate／Up 的「同層選擇」目前沒有任何低成本、可轉移的改善方法。剩下的誤差主要來自 sink 層的非局部性，以及窗口雜訊。

## 5. 限制

- 5 層、2 種模式、每支 6 種格式。沒有測 weight+activation 同時量化，也沒有量化 Down。
- 30 個新窗口；決定性配對 264 個，但配對之間不獨立。
- 全部是 PyTorch 模擬；native 一致性已在 block criterion 研究中核對。
- FD 只用 h = 1/16。

## 6. 重跑

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/qwen3_fd_validation_20261015   # 新實驗請換新目錄
$PY run_qwen_fd_validation.py --stage plan    --out-dir $O
$PY run_qwen_fd_validation.py --stage capture --out-dir $O
$PY run_qwen_fd_validation.py --stage scores  --out-dir $O --layers 27,21,14,7,0
$PY run_qwen_fd_validation.py --stage freeze  --out-dir $O
$PY run_qwen_fd_validation.py --stage truth   --out-dir $O --layers 27,21,14,7,0
$PY run_qwen_fd_validation.py --stage analyze --out-dir $O
$PY run_qwen_fd_validation.py --stage tables  --out-dir $O
```

| 檔案 | 內容 |
|---|---|
| `plan.json` | 凍結的計畫與通過標準 |
| `environment.json` | 版本與 hash |
| `capture.json`、`cache_*.npz`、`tokens_*.npy` | FP32 張量與 tokens |
| `scores_L*.json`、`freeze.json` | FD 分數、逐窗口值與凍結 hash |
| `truth_L*.json` | 新窗口逐窗口 ΔL |
| `analysis.json`、`pairs.csv`、`candidates.csv` | 指標與原始數據 |
| `tables.md` | 完整表格 |
| `timing.json` | 各 stage 耗時 |
