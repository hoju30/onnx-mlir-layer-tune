# Qwen3.5-0.8B：BC 的 Gate／Up 誤判會不會讓完整 knee 配置變差

日期：2026-10-15　腳本：`experiments/mixed_precision/run_qwen35_knee_gateup.py`　輸出：本目錄
來源：`runs/qwen35_bc_20261015`（兩個 knee 與 native final-test 結果）；前兩步：`runs/qwen35_gateup_recalib_20261015`、`runs/qwen35_weight_elem_20261015`

## 結論

**沒有可量測的影響。兩個 knee 都判定「不實質」，維持原 pipeline。**

| knee | 選擇窗口上採用的調整 | final test：修正後 − knee（nat／token） | 判定 |
|---|---|---|---|
| FFN-only | 3 層（L4、L20、L21），選擇窗口上 −0.0036 | GPU 模擬 +0.0018 [−0.0016, +0.0048]；CPU 模擬 +0.0001 [−0.0038, +0.0043]；**native −0.0004 [−0.0038, +0.0031]**（PPL 44.101 vs 44.120） | 不實質：區間含 0、方向不一致、低於雜訊下限 0.0012 |
| FFN + attention | 0 層（76 個調整中沒有一個決定性較好，14 個決定性較差） | 修正後即 knee | 不實質 |

**重點**
1. **在 knee 的位元分配上，BC 的 Gate／Up 判斷大多站得住**：
   - FFN-only 的 74 個替代分配中，15 個決定性較差、3 個決定性較好；
   - FFN + attention 的 76 個中，14 個較差、0 個較好。
2. **選擇窗口上的改善沒有轉移到 final test**：
   - FFN-only 修正後的配置在選擇窗口上改善 0.0036，在 final test 上消失；
   - 採用的 3 個調整，都是把更多位元給 Up（例如 L20：Gate int6／Up int6 → int5／int7）。這和之前看到的「BC 在 weight-only 高估 Up」方向相反；
   - 推論：這 3 個是 30 個選擇窗口的抽樣雜訊加上實作雜訊造成的，不是 BC 的系統性錯誤。
3. **新發現：完整量化模型的 loss 有實作雜訊**（§2）。
   - 同一個 knee 配置，GPU 模擬、CPU 模擬、native 三者的單一窗口差可達 0.04–0.05 nat／token；
   - 64 個窗口的平均 ΔNLL 差 0.0012（FFN-only）到 0.0023（FFN + attention）；
   - 同一組窗口上，FP32 三者只差 ≤ 6e-5。
   - 所以 knee 之間 < 約 0.002 nat／token 的差異，在不同實作之間無法穩定重現。之前觀察到的 Gate／Up 誤判（0.002–0.005／配對）大多落在這個尺度。

**對研究線的意義**：不再為 Gate／Up 修改 criterion。三件事已經依序排除：
- 換 calibration：BC 的比值不變；
- 逐元素權重項：結果與原版相同；
- 對 knee 的實際影響：不實質。

## 1. 設計（`plan.json`，執行前凍結；修正記在 `amendments`）

**起點**：主實驗兩個 knee 的 `selected_formats_{ffn,ffn_attn}.json`。
- 兩者的 Gate／Up 激活都是 p8e1，權重是 int5–int8。
- 每一層 Up 拿到的位元都 ≥ Gate。

**調整**
- 每一層把 Gate 與 Up 的權重位元，在總和不變下重新分配（int4–int8，激活不變）。因為 Gate、Up 形狀相同，M 與 BOPs 都不變。
- 數量：FFN-only 74 個、FFN + attention 76 個。
- 這正是 BC 在 Gate／Up 之間做的判斷；在 BC 眼中所有調整都比 knee 差（BC 差值全部 > 0，MILP 最佳解）。

**選擇**
- 全新 validation 窗口：token 24576–28415，30 × 128，之前沒用過。
- 整個配置量化，用 PyTorch 在 GPU 上重播（FP32，關閉 TF32）。
- 每層取 loss 最低的替代分配；若 2000 次窗口 bootstrap 中 ≥ 95% 比 knee 低，就採用。修正後的配置 = knee + 所有採用的調整。

**評估**：主實驗的 final test（64 個窗口），修正後 vs knee 逐窗口配對比較。

**判定**
- 原規則：native 差值的 95% 上界 < 0，且 |平均| ≥ knee 本身 ΔNLL 的 10%。
- 修正（選擇開始前加入，理由見 §2）：
  - 另外要求 |native 差值| 大於雜訊下限（三種實作的 knee 平均 ΔNLL 之差：FFN-only 0.0012、FFN + attention 0.0023）；
  - 而且 GPU、CPU、native 的方向一致。

## 2. 模擬核對與實作雜訊（`simcheck_cuda.json`、`simcheck_cpu.json`）

final test 64 個窗口，平均 NLL（nat／token）：

| 配置 | GPU 模擬 | CPU 模擬 | native | 單窗口最大差 |
|---|---:|---:|---:|---|
| FP32 | 3.77964 | 3.77959 | 3.77964 | ≤ 6.0e-5 |
| FFN-only knee | 3.78599 | 3.78564 | 3.78691 | GPU–CPU 0.039，CPU–native 0.044 |
| FFN + attention knee | 3.80089 | 3.80285 | 3.80060 | GPU–CPU 0.051，CPU–native 0.045 |

| ΔNLL vs FP32 | GPU | CPU | native | 逐窗口相關（GPU–CPU／CPU–native） |
|---|---:|---:|---:|---|
| FFN-only | +0.00635 | +0.00605 | +0.00727 | 0.46／0.44 |
| FFN + attention | +0.02125 | +0.02327 | +0.02096 | 0.67／0.70 |

- 兩個 PyTorch 模擬之間的差，和它們各自與 native 的差一樣大，所以這不是 native 的語意問題，而是完整量化模型對 FP32 運算順序敏感（推論：微小差異讓部分激活跨過量化捨入邊界，經 24 層放大）。
- 單節點量化時這種雜訊小很多（主實驗的 CPU 交叉核對：單窗口差 ≤ 0.004），整個配置同時量化時就大得多。

## 3. 選擇結果（`selection.json`）

| | FFN-only | FFN + attention |
|---|---:|---:|
| 調整數 | 74 | 76 |
| 選擇窗口上 loss 較低 | 22 | 13 |
| 決定性較好（≥ 95%） | 3 | 0 |
| 決定性較差（≤ 5%） | 15 | 14 |
| ΔNLL 範圍 | −0.0038 到 +0.0081 | −0.0019 到 +0.0089 |

採用的調整（FFN-only）：

| 層 | knee（Gate／Up） | 調整後 | 選擇窗口 ΔNLL | bootstrap 較好比例 | BC 差值 |
|---|---|---|---:|---:|---:|
| L4 | int7／int7 | int6／int8 | −0.0038 | 0.950 | +0.141 |
| L20 | int6／int6 | int5／int7 | −0.0026 | 0.976 | +0.437 |
| L21 | int6／int6 | int5／int7 | −0.0023 | 0.985 | +0.446 |

合起來在選擇窗口上 −0.0036。

## 4. final test 評估（`evaluation.json`）

**FFN-only 修正後 − knee**（逐窗口 NLL 差，配對 bootstrap）
- GPU 模擬：+0.0018 [−0.0016, +0.0048]
- CPU 模擬：+0.0001 [−0.0038, +0.0043]
- native：−0.0004 [−0.0038, +0.0031]；修正後 NLL 3.78648、PPL 44.101，knee 是 3.78691／44.120；編譯 84 s、執行 585 s、logits 全部有限

**判定**
- 原規則不成立；
- 低於雜訊下限（0.0004 < 0.0012）；
- 三者方向不一致。
- → 不實質。

FFN + attention 沒有採用任何調整。

## 5. 成本

| 階段 | 時間 |
|---|---:|
| simcheck：GPU／CPU（3 個配置 × 64 窗口） | 27 s／247 s |
| select（150 個調整 × 30 窗口，GPU） | 387 s |
| evaluate：GPU／CPU 模擬 | 19 s／137 s |
| evaluate：native 編譯與執行 | 約 700 s |

## 6. 限制

- 只考慮 Gate／Up 之間的位元重新分配（int4–int8、激活不變），沒有考慮 Gate／Up 與其他節點之間的分配，也沒有 posit 權重。
- 選擇窗口只有 30 個，每層取最佳的替代分配，本身就會有選擇偏差；這是為什麼評估必須用獨立的 final test。
- 雜訊下限是三種實作的差距，只有一次量測；更精確的估計需要更多種實作或擾動。
- 只看了 knee 這一個資源點。

## 7. 重跑與檔案

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/qwen35_knee_gateup_20261015   # 新實驗請換新目錄
GPU="env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$PY  run_qwen35_knee_gateup.py --stage plan     --out-dir $O
$GPU run_qwen35_knee_gateup.py --stage simcheck --out-dir $O --device cuda
$PY  run_qwen35_knee_gateup.py --stage simcheck --out-dir $O --device cpu
$GPU run_qwen35_knee_gateup.py --stage select   --out-dir $O --device cuda
$GPU run_qwen35_knee_gateup.py --stage evaluate --out-dir $O --device cuda
$PY  run_qwen35_knee_gateup.py --stage evaluate --out-dir $O --device cpu --native
```

| 檔案 | 內容 |
|---|---|
| `plan.json` | 設計、判定規則與修正（`amendments`） |
| `tokens_select.npy`、`moves.json` | 選擇窗口、所有替代分配 |
| `simcheck_cuda.json`、`simcheck_cpu.json` | 模擬 vs native，以及實作雜訊 |
| `selection.json`、`corrected_formats_{ffn,ffn_attn}.json` | 每個調整的逐窗口結果、BC 差值、採用與修正後配置 |
| `evaluation.json`、`eval/` | final-test 結果與 native 編譯設定 |
| `timing.json`、`*.log` | 執行時間 |
