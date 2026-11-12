# Qwen3.5-0.8B：換一批 calibration 重建 BC，Gate／Up 的判斷是否不準

日期：2026-10-15　腳本：`experiments/mixed_precision/run_qwen35_gateup_recalib.py`　輸出：本目錄　主實驗：`runs/qwen35_bc_20261015`

## 結論

**依預先登記的規則，判定為「mixed」：本研究不支持修改 criterion，也不支持只靠換 calibration 來改善。**

| 規則條件 | 結果 |
|---|---|
| 系統性：兩次樣本內正確率都 ≤ 0.85 | 不成立：BC_A 0.856（107/125），BC_B 0.896（138/154） |
| 系統性：樣本外錯誤重疊 ≥ 0.6 | 成立：BC_B 的 23 個錯誤中，22 個也是 BC_A 的錯誤（0.96） |
| 系統性：樣本內偏差方向在 ≥ 2/3 的組一致 | 不成立：7/11（0.64） |
| 抽樣：兩次樣本內都 ≥ 0.90，且重疊 < 0.5 | 不成立 |

**白話**
1. **換 calibration 幾乎不改變 BC 對 Gate／Up 的相對判斷**。
   - BC 的 log(Up／Gate) 比值，在兩批 calibration 之間的 Spearman 是 0.99，差的中位數 0.05；只有比值接近 0 的 10/90 個翻號。
   - 絕對大小會變（例如 L0 約差 40%），但 Gate 和 Up 誰比較敏感幾乎不變。
   - 所以換一批或多加 calibration，修不了 BC 的 Gate／Up 錯誤：樣本外錯誤從 32 個降到 23 個，其中 22 個沒變。
2. **錯誤不是 calibration 抽樣造成的**。在建表用的同一批窗口上（樣本內），BC 的決定性正確率是 85.6%／89.6%；在 diagnostic 窗口上（樣本外）是 82.9%／87.5%。兩者差不多，表示錯誤來自 BC 公式的近似，不是從 calibration 推到新窗口時才出現。
3. **但沒有一致的偏差方向**，所以沒有簡單的修正方法：
   - BC 有時高估 Up、有時高估 Gate，看層與模式而定；
   - 樣本內的偏差中位數是 +0.08／+0.11，樣本外是 −0.18／−0.02（log 比值）；
   - 「把 Up 或 Gate 乘一個係數」這類修正得不到支持。
4. **Gate+Up 的實際損傷幾乎等於兩支單獨損傷相加**。
   - 樣本內的交互項中位數 1.6–1.9%，p90 11–12%；
   - 用實際的單支損傷相加來排序聯合配對，對 97–98%；BC 只有 78–83%；
   - 所以聯合的錯誤來自單支估計，BC 的「相加」本身沒有問題。
5. **錯誤集中在小差異與 weight-only**。
   - 判錯配對的真實差異中位數只有 0.002–0.005 nat／token，全部決定性配對是 0.009–0.016；
   - 樣本內的錯誤中，weight-only 佔 13/18（A）、10/16（B）。

**建議**：維持原版 BC，不改 calibration 數量。若要繼續追，具體目標只有一個：weight-only 時 BC 對 Gate／Up 的相對判斷（§4）。改之前應先量化這些小差異對 knee 配置的實際影響，並另外預先登記。

## 1. 設計（執行前凍結於 `plan.json`）

- **calibration B**：validation token 16384–17663（10 個窗口 × 128），和主實驗用過的所有範圍（0–5119、8192–14335）都不重疊。
- **BC_B**：用 native onnx-mlir 重建原版 BC（`native_tables`，`input_grad="total"`），範圍是 6 個代表層（L0、3、12、15、22、23）的 `gate_proj` 與 `up_proj`，共 12 個節點，903 s，沒有 non-finite。
- **BC_A**：主實驗原本的表（calibration A，validation 0–1279）。
- **候選**：主實驗中 Gate 單獨、Up 單獨、Gate+Up 的 630 個候選（w／a／wa × int4、int6、int8、p6e1、p8e1）。BC_B 的分數在 13:38:14 凍結（`freeze.json`），之後才跑使用 B 的 truth。
- **真實損傷**：PyTorch 重播（GPU、FP32、關閉 TF32；與主實驗相同，主實驗做過 CPU 交叉核對）。每個 truth 都用和被比較的 BC 相同的 int 激活 range。

  | truth | 窗口 | range | 用途 |
  |---|---|---|---|
  | T_cal_a | calibration A（10） | A | BC_A 樣本內 |
  | T_cal_b | calibration B（10） | B | BC_B 樣本內 |
  | T_diag_a | diagnostic（30，主實驗 truth.json） | A | BC_A 樣本外 |
  | T_diag_b | diagnostic（30）：有 int 激活的 324 個候選重跑，其他沿用 T_diag_a（不受 range 影響） | B | BC_B 樣本外 |

- **配對**：同層、同模式，整個 FFN 的 M 與 B 都相同。
  - Gate／Up 判斷配對共 342 個：單支 Gate vs Up 同格式 90 個、Gate vs Up 的 int vs posit 72 個、聯合 Gate／Up 互換 180 個。
  - 「決定性」：2000 次窗口 bootstrap 中至少 95% 同向。樣本內只有 10 個窗口，決定性配對較少。

## 2. Gate／Up 判斷的正確率（決定性配對）

| | 合併 342 對 | Gate vs Up 同格式 | Gate vs Up int vs posit | 聯合互換 |
|---|---:|---:|---:|---:|
| BC_A 樣本內（T_cal_a） | 107/125（85.6%） | 18/25（72.0%） | 31/32 | 58/68（85.3%） |
| BC_B 樣本內（T_cal_b） | 138/154（89.6%） | 28/34（82.4%） | 31/34 | 79/86（91.9%） |
| BC_A 樣本外（T_diag_a） | 155/187（82.9%） | 32/43（74.4%） | 37/42 | 86/102（84.3%） |
| BC_B 樣本外（T_diag_b） | 161/184（87.5%） | 35/43（81.4%） | 35/39 | 91/102（89.2%） |

**其他同資源配對**（參考）
- 聯合「一支 int↔posit」：84–92%；
- 聯合「其他」：93–95%；
- 全部 1362 對：88.2–92.0%。

**錯誤的分布**
- 判錯配對的 |真實差異| 中位數：樣本內 0.0052／0.0028，樣本外 0.0026／0.0018 nat／token。
- 依模式（A 樣本內／B 樣本內／A 樣本外／B 樣本外）：
  - w：13／10／13／13
  - a：4／2／4／1
  - wa：1／4／15／9
- 樣本外共同的 22 個錯誤中，13 個在 L12（w 與 wa 模式）。

**BC_B 修正與新增的錯誤**
- BC_B 修正了 BC_A 的 10 個樣本外錯誤，大多是 BC 比值接近 1 的情況：L22 wa int4、L3 wa int6、L15 a p6e1。
- 新增 1 個（L3 wa int8 vs p8e1）。

## 3. BC 本身的穩定性

- **Up／Gate 比值**（90 個「同層、同模式、同格式」的單支 Gate vs Up）：log(BC_up／BC_gate) 在 A 與 B 之間：
  - Spearman 0.990；
  - |差| 中位數 0.052；
  - 翻號 10/90，全部是比值接近 0 的：L3 a／wa 的 int、L22 a／wa 的 int、L15 a 的 posit。
- **630 個候選的 BC 整體**：A vs B 的 Spearman 0.995。
- **絕對大小**：節點的 B、C 在兩批之間可以差 10–40%，例如 L0 gate 的 C(int6) 是 62.3 vs 38.9。但同一層的 Gate 與 Up 會一起變。

## 4. 偏差方向

**定義**：e = log(BC_up／BC_gate) − log(T_up／T_gate)，只取兩支的真實損傷都顯著 > 0 的同格式配對；e > 0 表示 BC 高估了 Up 相對於 Gate。

**樣本內，各 (layer, mode) 的中位數**（A／B）
- **weight-only**：
  - L3：+1.15／+1.04
  - L23：+1.06／+1.16
  - L15：+0.85／（B 無資料）
- **activation-only**：L15 −0.27／−0.28，L3 −0.32／−0.04，L23 −0.24／+0.11，L0 −0.10／+0.72
- **wa**：L15 +0.21／−0.35，L22 0.00／−0.54，L23 +0.10／+0.71

**樣本外**（A／B）
- L23 w：+2.28／+2.37（BC 把 Up／Gate 高估約 10 倍）
- L12 w：+0.61／+0.72
- L22 w：−0.92／−0.87（方向相反）
- L3 w：0.00／−0.01

**讀法**
- 有兩層的 weight-only，BC 在樣本內與樣本外都一致地高估 Up：L23（兩批都是）、L3（只有樣本內）。
- 但 L22 的 weight-only 方向相反，activation-only 與 wa 也沒有一致方向。
- 整體同向的組只有 7/11，未達規則的 2/3。

## 5. Gate+Up 相加 vs 實際

| truth | 交互項／聯合（中位數、p90） | 單支實際相加的排序正確率 | BC_A | BC_B |
|---|---|---:|---:|---:|
| T_cal_a | 1.6%、10.6% | 98.0% | 82.7% | 82.4% |
| T_cal_b | 1.9%、11.9% | 97.4% | 77.7% | 79.0% |
| T_diag_a | 1.3%、8.9% | 98.4% | 75.5% | 77.7% |
| T_diag_b | 1.4%、8.2% | 98.4% | 76.9% | 78.2% |

（聯合同資源配對 1128 對，全部配對的正確率。）

實際的 Gate+Up 損傷很接近兩支單獨損傷之和，所以 BC 用相加的方式組合兩支沒有問題；不準的是單支的估計。

## 6. 判斷

**量測**
- 換 calibration 後，BC 的 Gate／Up 相對判斷幾乎不變；樣本外的錯誤大多相同。
- 樣本內的錯誤率與樣本外相近。
- 錯誤集中在小差異的配對，以及 weight-only。
- 實際損傷可加。

**推論**（未驗證）
- 錯誤來自 BC 公式本身的近似，不是 calibration 抽樣，也不是 Gate／Up 的交互作用。
- 但偏差的方向依層與模式而變，所以沒有簡單的係數修正。
- weight-only 的權重項使用「每個輸出 channel 的梯度平方均值 × 量化誤差平方和」，忽略了輸入 channel 之間的差異；在有 outlier 輸入 channel 的層，可能誤判 Gate／Up。這是假說，本研究沒有測。

**接續建議**
1. 不修改 criterion：本研究依規則判定為 mixed。
2. 不要增加或更換 calibration 來修 Gate／Up：BC 的比值在兩批之間幾乎不變。
3. 若要追，先做兩件事，並另外預先登記：
   - 量化這些小差異（約 0.002–0.005 nat／token）對 knee 配置的實際影響；knee 的 FFN-only 總損傷只有 0.007；
   - 用新窗口測 weight-only 的權重項換成逐元素形式（Σ_i G_ci ΔW_ci²）是否能修正 L3／L23 的 Up 高估與 L22 的反向，並且不讓其他類別變差。

## 7. 限制

- 只有 6 層的 Gate／Up，沒有 Down 與 attention 投影。
- 樣本內 truth 只有 10 個窗口，決定性配對較少，bootstrap 較粗。
- truth 是 PyTorch 模擬（GPU）；主實驗的 CPU 交叉核對顯示配對順序不受影響。
- 只換了一批 calibration（B）；「穩定」是兩批之間的結論。
- 偏差分析只用兩支都顯著 > 0 的配對，每組只有 1–4 個。

## 8. 成本

| 階段 | 時間 |
|---|---:|
| native BC 表 B（12 節點，與 GPU truth 同時執行） | 903 s |
| truth：T_cal_a（630 × 10 窗口，GPU） | 223 s |
| truth：T_cal_b（630 × 10 窗口，GPU） | 221 s |
| truth：T_diag_b（324 × 30 窗口，GPU） | 279 s |
| 分析 | < 1 s |

## 9. 重跑與檔案

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/qwen35_gateup_recalib_20261015   # 新實驗請換新目錄
GPU="env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$PY  run_qwen35_gateup_recalib.py --stage plan    --out-dir $O
$PY  run_qwen35_gateup_recalib.py --stage table   --out-dir $O
$GPU run_qwen35_gateup_recalib.py --stage truth   --out-dir $O --parts calib_a --device cuda   # 不需要 B 的 freeze
$PY  run_qwen35_gateup_recalib.py --stage freeze  --out-dir $O
$GPU run_qwen35_gateup_recalib.py --stage truth   --out-dir $O --parts calib_b,diag_b --device cuda
$PY  run_qwen35_gateup_recalib.py --stage analyze --out-dir $O
```

| 檔案 | 內容 |
|---|---|
| `plan.json` | 設計與判斷規則（先凍結） |
| `tokens_calibration_b.npy` | calibration B |
| `table_b.json`、`ranges_b.json`、`table_b_info.json` | native BC_B 表、range、耗時 |
| `candidates.json`、`freeze.json` | 630 個候選（`bc` = BC_A，`bc_b` = BC_B）與凍結 hash |
| `truth_calib_a.json`、`truth_calib_b.json`、`truth_diag_b.json` | 逐窗口 summed ΔCE |
| `analysis.json`、`pairs.csv` | 指標、判斷與逐配對資料 |
| `timing.json`、`*.log` | 執行時間 |
