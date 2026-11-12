# Qwen3.5-0.8B：BC 權重項改成逐元素計算（預先登記，全新窗口）

日期：2026-10-15　腳本：`experiments/mixed_precision/run_qwen35_weight_elementwise.py`　輸出：本目錄
前一步：`runs/qwen35_gateup_recalib_20261015`（換 calibration 後判定 mixed，建議測這個假說）

## 結論

**未通過（P1 不成立）。保留原版權重項。**

| 標準 | 結果 |
|---|---|
| P1：決定性配對 BC_elem 對的數量 > BC，且正確率差的區間下界 > 0 | **不成立**：650 vs 650（共 725 對），區間 [−0.002, +0.002] |
| P2：Gate／Up 判斷不比 BC 差 | 成立：98 vs 98（共 120 對） |
| P3：每層最多少對 2 個 | 成立：L15 −1、L3 +1，其餘相同 |

**為什麼幾乎沒有差別**（量測：W_elem／W，18 個節點 × 5 種格式）
- **int 格式：0.97–1.03**，只有 L15 up int4 是 1.06。int 量化誤差在一個輸出 channel 內大致均勻，而且和各元素的梯度大小無關，所以 channel 平均 × 誤差和已經等於逐元素加總。
- **posit 格式：0.67–1.03**，多數是 0.8–0.97。posit 在大權重上誤差較大，而這些權重的梯度平方相對較小。這會改變 int vs posit 的比較，但在決定性配對上沒有翻轉任何一個判斷。
- **Gate／Up 的偏差完全沒變**：L23 weight-only 的 log(Up／Gate) 偏差，原版 +1.55、逐元素 +1.57（全新窗口）；在舊的 diagnostic 窗口上是 +2.28 vs +2.30。

**推論**（未驗證）
- BC 在 Gate／Up 上的誤判，不是「channel 內平均」造成的。剩下可能的來源是：
  - 經驗 Fisher（梯度平方）和真正曲率之間的落差；
  - BC 沒有的一階項；
  - 非局部效應。
- 這些在 Qwen3 上都測過：FD 曲率在 weight-only 反而更差，方向項不跨窗口轉移。目前沒有低成本、可轉移的修正方向。

**建議**
- 維持原版 BC；不移植逐元素權重項到 native。
- 這些 Gate／Up 誤判的差異都很小（0.002–0.005 nat／token）。如果還要決定是否值得繼續追，下一步應該量它們對 knee 配置的實際影響，而不是再換 criterion。

## 1. 設計（`plan.json`，執行前凍結）

**criterion**
- 原版 W(w) = Σ_c Ḡ_c Σ_i ΔW_ci²，Ḡ_c = (1/n_in) Σ_i G_ci；
- 逐元素 W_elem(w) = Σ_c Σ_i G_ci ΔW_ci²；
- G_ci = Σ_calibration windows (∂L/∂W_ci)²，與 native grads pass 的定義相同；
- 激活項 A（native C）不變：BC_elem = W_elem + A。

**計算方式**
- W_elem 用 PyTorch 計算（native grads pass 目前只輸出每個 channel 的總和），用的是主實驗的 10 個 calibration 窗口。
- 同一段程式算出的原版 W 和 native 表的 B 核對：相對差中位數 3.4e-4、最大 3.6e-3，因此兩邊的定義一致。

**範圍**
- 6 個代表層（L0、3、12、15、22、23）的 FFN（Gate／Up／Down 單獨與 Gate+Up），模式 w 與 wa。
- 共 480 個候選（mode a 的權重是 FP32，W = 0，不受影響）。

**真實損傷**
- 全新窗口：validation token 20480–24319（30 × 128），之前任何研究都沒用過。
- 主實驗的 diagnostic 窗口參與了假說的形成，所以只列為次要參考。
- PyTorch 從該層開始重播（GPU、FP32、關閉 TF32），int 激活使用主實驗的 range。

**順序**：分數凍結（`freeze.json`，15:39:16）→ 才跑全新窗口的 truth。

## 2. 結果（全新窗口，主要）

| 類別 | 配對 | 決定性 | BC | BC_elem |
|---|---:|---:|---:|---:|
| 全部（w + wa） | 1116 | 725 | 650 | 650 |
| Gate／Up 判斷（合併） | 228 | 120 | 98 | 98 |
| 單支 Gate vs Up 同格式 | 60 | 23 | 18 | 18 |
| 單支 Gate vs Up int vs posit | 48 | 31 | 28 | 28 |
| 聯合 Gate／Up 互換 | 120 | 66 | 52 | 52 |
| 單支 Down vs Up 同格式 | 60 | 42 | 32 | 32 |
| 單支 Down vs Gate 同格式 | 60 | 38 | 33 | 33 |
| 單支 同分支 int vs posit | 72 | 48 | 43 | 43 |
| 聯合 一支 int↔posit | 240 | 155 | 137 | 137 |
| 聯合 其他 | 360 | 254 | 240 | 240 |
| w | 606 | 411 | 370 | 370 |
| wa | 510 | 314 | 280 | 280 |

**各層**（決定性配對，BC／BC_elem）
- L0：113／113（共 116）
- L3：105／106（共 110）
- L12：89／89（共 118）
- L15：129／128（共 139）
- L22：101／101（共 116）
- L23：113／113（共 126）

**次要（主實驗的 diagnostic 窗口）**：全部 662 vs 664（共 772），Gate／Up 126 vs 126。不用於判定。

**順便確認原版 BC 在全新窗口上的表現**
- w + wa 的決定性配對對 89.7%；Gate／Up 81.7%。
- L12 最差（89/118 = 75.4%），與主實驗一致。

## 3. 成本

| 階段 | 時間 |
|---|---:|
| W_elem 與 W（PyTorch，CPU，18 節點，10 窗口） | 36 s |
| 全新窗口 truth（480 × 30，GPU） | 431 s |
| 分析 | < 1 s |

## 4. 限制

- W_elem 是 PyTorch 計算的；因為假說沒有通過，所以沒有移植到 native。
- 只有 6 層的 FFN；沒有測 attention 投影（以 int 格式為主的話，結論應該相同，因為 int 的比值接近 1，但沒有驗證）。
- 單一批全新窗口（30 個）。

## 5. 重跑與檔案

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/qwen35_weight_elem_20261015   # 新實驗請換新目錄
$PY run_qwen35_weight_elementwise.py --stage plan    --out-dir $O
$PY run_qwen35_weight_elementwise.py --stage scores  --out-dir $O
env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python run_qwen35_weight_elementwise.py --stage truth --out-dir $O --device cuda
$PY run_qwen35_weight_elementwise.py --stage analyze --out-dir $O
```

| 檔案 | 內容 |
|---|---|
| `plan.json` | 假說、資料、通過標準（先凍結） |
| `tokens_fresh.npy` | 全新窗口 |
| `weight_terms.json` | 每個節點與格式的 W、W_elem，以及和 native B 的核對 |
| `candidates.json`、`freeze.json` | 480 個候選（`bc`、`bc_elem`）與凍結 hash |
| `truth_fresh.json` | 全新窗口逐窗口 summed ΔCE |
| `analysis.json`、`pairs_fresh.csv`、`pairs_diagnostic_secondary.csv` | 判定、各類別、偏差、W_elem／W |
| `timing.json`、`*.log` | 執行時間 |
