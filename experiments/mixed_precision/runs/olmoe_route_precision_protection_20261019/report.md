# OLMoE aggressive 配置：逐層權重精度保護 vs 同資源 BC

日期：2026-10-19
腳本：`run_olmoe_route_precision_protection.py`（plan → data_selection → control_prev → selection → bc_solve → selection_bc → data_final → final → analyze）
輸出：本目錄

**本輪的範圍**
- 所有品質數字都在正常 dynamic route 下量測。FP32 只用來當基準，以及離線量測 router 比例偏移；沒有任何 FP32 資訊被餵回量化推論。
- BC 公式、router、量化器、range 都沒有改動，也沒有加入一階項。
- 本輪測的是 GPU 上的 QDQ 行為，不代表 native kernel 的結果；沒有整合進正式 pipeline。

**流程**
- 預先登記 `plan.json` 和 `plan.md` 在任何候選評估之前寫定。
- 每個階段用的都是同一份腳本（sha `6f17b5fb…`）。
- 事後只改了圖表版面，數字沒有變，記錄在 `amendments.json`。

**標記**
- [預先登記]：寫在 `plan.json` 的分析。
- [探索]：selection 階段的數字，只用來挑選，不能當確認結果。
- [描述]：不做推論檢定的描述統計。

## 結論

**以路由比例為導向的逐層權重保護，在 final test 上沒有比原 aggressive 好，而且明確輸給相同資源的 BC 配置。這個選擇方式在本輪沒有得到支持（預先判讀 C）。**

| 配置 | R | size／int8 | BOPs／int8 | dynamic 損傷 vs FP32 | PPL | G（比例偏移） |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | — | — | — | — | 19.216 | 0 |
| 原 aggressive | 0.5946 | 0.5946 | 0.5946 | +0.0328 | 19.856（+3.33%） | 0.0414 |
| route-focused（保護第 0 層） | 0.6234 | 0.6234 | 0.6186 | +0.0332 | 19.864（+3.37%） | 0.0395 |
| **同資源 BC** | 0.6234 | 0.6234 | 0.6186 | **+0.0293** | **19.787（+2.97%）** | **0.0383** |

final test 共 169 篇新的 WikiText-103 文章（676 個窗口），和 selection 的文章整篇隔離。

| 對比 | 平均 | 95% CI | Bonferroni ×2（97.5%） | 判定 |
|---|---:|---|---|---|
| **Δ_primary = route-focused − 同資源 BC** | **+0.00390** | **[+0.00169, +0.00613]** | — | **BC 明確較好** |
| route-focused − aggressive | +0.00043 | [−0.00145, +0.00235] | [−0.00173, +0.00262] | 無法和 0 區分 |
| 同資源 BC − aggressive | −0.00347 | [−0.00559, −0.00140] | [−0.00589, −0.00114] | 明確改善 |

單位 nat/token；負值代表前者較好。

**判讀**（依預先固定的規則）
- **C 成立**：route-focused 配置的 G 確實比原 aggressive 小（0.0395 對 0.0414），但 task loss 沒有改善（+0.0004，CI 含 0）。所以本輪的 G 不足以作為選擇依據。
- A 和 B 都不成立：route-focused 既沒有勝過同資源 BC，也沒有確認勝過原 aggressive。
- 同資源 BC 把相同的額度分散到 730 個節點，loss 明確改善（−0.0035），而且 **G 也比 route-focused 更小**（0.0383）。G 下降似乎反映整體量化品質變好，不是「保護上游層」特有的效果。
- 本輪不改 BC。結論只限於「單層 int<8 → int8 的權重保護，加上這個 G／ΔR 選擇規則」；不能推論到 activation 保護或其他路由修正。

## 候選（`plan.json` → `candidates`、`selection_table.csv`）

**規則**
- 第 l 份配置：把第 l 層所有 expert 的 gate／up／down 投影中，int4–int7 的權重都改成 int8。
- int8、bf16、activation、range、其他層，以及 21 個未進成本表的投影都不變。
- 資源沿用固定成本定義（calibration 的 routed rows、int8 參考）；R = max(size／M8, BOPs／B8)。

**檢查**
- 16 份配置都只改了該層的權重格式，全部在 GRID11² 候選空間內。
- 原 aggressive 用格式重新計算的資源，和 frontier 紀錄一致。

| 保護層 | 修改節點（主要變化） | ΔM_rel | ΔB_rel | ΔR | BC 目標值（原 aggressive 為 42.18） |
|---:|---|---:|---:|---:|---:|
| 0 | 192（int4→8 132、int5→8 55） | +0.0288 | +0.0240 | +0.0288 | 40.50 |
| 4 | 192 | +0.0270 | +0.0212 | +0.0270 | — |
| 8 | 189 | +0.0255 | +0.0218 | +0.0255 | — |
| 15 | 188（int5→8 76、int6→8 68） | +0.0216 | +0.0187 | +0.0216 | 40.42 |

完整 16 列見 `plan.md` 和 `selection_table.csv`。保護之後，R 改由 size 決定。

## Selection（32 篇文章，`selection.json`）[探索]

**基準**：原 aggressive 的 selection CE 為 2.85343，G 為 0.04128。

| 保護層 | CE 改善（aggressive − 候選） | G 下降 | (G 下降)／ΔR | 合格 |
|---:|---:|---:|---:|---|
| 0 | +0.0003 | 0.00178 | **0.0617** | ✓（選中） |
| 1 | −0.0030 | 0.00088 | 0.032 | ✗（CE 變差） |
| 2 | −0.0029 | 0.00096 | 0.036 | ✗ |
| 3 | +0.0021 | 0.00136 | 0.051 | ✓ |
| 4 | +0.0032 | 0.00100 | 0.037 | ✓ |
| 5 | −0.0000 | 0.00085 | 0.033 | ✗ |
| 6–14 | +0.0000 到 +0.0029 | 0.00026–0.00063 | 0.011–0.024 | ✓ |
| 15 | +0.0024 | 0 | 0 | ✗（G 不變；後面沒有 router） |

**選擇與檢查**
- 16 個候選中 12 個合格；依規則（(G 下降)／ΔR 最大）選中**第 0 層**，雜湊為 `5392b1b0…`，並已凍結。
- 第 0 層的 selection CE 改善只有 +0.0003，幾乎是 0。它是因為對後面所有 router 的比例偏移降幅最大而被選中，見 `fig_protection_route_drift.png`。
- 保護第 l 層時，第 ≤ l 層的 router 在 16 個候選中**全部**和原 aggressive 逐位元相同。所以 G 的變化只來自保護層之後的 router，符合預期。
- 各層的 selection CE 差落在 ±0.003 之間，32 篇文章不足以分辨，只作探索。

## 同資源 BC（`bc_solve.json`）

**求解設定**
- 目標：min 原 BC 成本。
- 限制：size ≤ 3,988,783,104 bytes（M_rel 0.62340），而且 BOPs ≤ 4.0810e13（B_rel 0.61861）。兩個分量都受限，不是只限制 R。
- 求解器：已整合修正的 `mp_milp.milp_select`，GRID11²，精確的支配剪枝，HiGHS。

**結果**
- 狀態：**證明最優**（HiGHS status 7，gap 0），230 秒；過程中用到 1 ppm 的預算重試。
- BC 目標值：**34.13**；route-focused 為 40.50，原 aggressive 為 42.18。
- 實際資源：size 用了預算的 99.993%，BOPs 用了 99.99996%；R 0.62336。
- 和 route-focused 有 865 個節點不同，和 aggressive 有 730 個節點不同。

**可比性**
- route-focused 的所有格式都在 GRID11²，屬於同一個候選空間。
- 其中 107 個選項被支配剪枝刪除。剪枝是精確的，不影響最優性，所以 MILP 的最優解在完整空間中也成立；它的 BC 目標值也確實低於 route-focused（34.13 < 40.50）。
- BC 配置沒有使用任何 selection loss。

## 樣本數（`sample_size.json`）

- 在 selection 上，「route-focused − BC」的文章級 SD 為 0.0132。
- n = ⌈(1.96·s／0.002)²⌉ = **169 篇**，預期 95% CI 半寬 0.0020。
- 實際主要 CI 半寬為 0.0022。
- selection 參與了挑選，所以這個變異估計可能偏樂觀；這點已在 plan 中揭露。
- 在 selection 上，同資源 BC 已經比 route-focused 好 0.0036。這只用來估樣本數，不算確認結果。

## 資料隔離（`selection_segments.json`、`final_segments.json`）

**來源**
- 去重後的 WikiText-103 train，規則和前幾輪相同：嚴格文章邊界、排除 WikiText-2 的全部文章、排除 WikiText-103 內部重複、5-gram 近重複檢查。
- 另外排除先前所有 OLMoE 用過的 WikiText-103 文章：103 + 242 篇。
- 排除後可用 27,050 篇。

**抽樣**
- 依同一個隨機順序（seed 18191）：前 32 篇接受的文章是 selection，之後的 169 篇是 final。
- 兩者整篇隔離（內容 hash 重疊 0）。
- 每篇 4 個連續窗口。抽樣不依 loss、路由或難度。

## 數值與完整性控制（全部通過）

| 控制 | 結果 |
|---|---|
| 程式控制（舊資料：`runs/olmoe_route_dynamic_prop_20261018` 的窗口 0–127） | FP32 基準的 S0、w0、p0 和 CE，以及 aggressive 的 dynamic CE、ID、權重，都逐位元相同 |
| 量測掛勾不影響推論 | aggressive 在新函式下的 CE、ID、權重，和上一輪的完整 dynamic 逐位元相同 |
| 關閉量化 | `forward_measure` 和 FP32 基準逐位元相同；量到的 p[S0] 恰好等於 w0（舊資料及 final 第一批） |
| 保護第 l 層時，第 ≤ l 層的 router | 16 個候選全部逐位元不變 |
| final 重跑（route-focused、BC；第一批 128 個窗口） | 逐位元相同，路由也相同 |
| final 重播：記錄的 ID 和權重交給 `run_olmoe_dynamic_route.forward` | 兩個配置都 676／676 逐位元相同 |
| 沒有用 FP32 路由替換 | 三個配置都有約 30% 的 (token, 層) 選擇和 S0 不同 |
| 配置完整、router 未動、hook 只在 expert 投影、有限值、每窗口 127 token | 全部是 |

## 路由描述（final，`final_results.json` → `route`）[描述]

| 配置 | G | 和 S0 不同的集合比例 | 權重總和 − m0 |
|---|---:|---:|---:|
| aggressive | 0.0414 | 32.5% | −0.00146 |
| route-focused | 0.0395 | 30.8% | −0.00143 |
| 同資源 BC | 0.0383 | 30.4% | −0.00121 |

route-focused 降低了路由偏移，但 loss 沒有跟著改善；同資源 BC 的偏移更小，loss 也更好。這只是關聯，不是因果。

## 圖

- `fig_selection_loss_and_dR.png`：各保護層的 selection CE 改善（探索）和 ΔR。
- `fig_protection_route_drift.png`：保護第 l 層（列）對第 l′ 層 router（欄）比例偏移的下降，藍色代表偏移變小。對角線以下（l′ ≤ l）都是 0，顯示保護只影響之後的 router。
- `fig_final_damage_vs_R.png`：
  - 左：三個配置的資源和 dynamic 損傷。絕對損傷的 CI 互相重疊，但這不代表沒有差異。
  - 右：配對差和預先登記的區間。

## 下一步的含意（不在本輪執行）

- 以「單層 int8 保護＋G／ΔR」選位置的做法，**不值得繼續**：
  - 在相同資源下明確輸給 BC 本身；
  - 也沒有確認比原 aggressive 好。
- 同樣多出約 0.029 的 R，交給 BC 自己分配，效果明顯較好（−0.0035）。也就是說，dynamic route 下的損傷，沿著 BC frontier 增加資源就能有效降低；不需要額外的路由導向啟發式規則。
- 若之後仍要考慮路由，應該在 criterion 層級預先登記（例如 BC 中的 router-logit 敏感度項），並和同資源 BC 比較。本輪結果顯示，只看路由比例偏移的指標，不足以預測 loss。
- 本輪沒有測 activation 精度保護和其他保護形式，不能據此排除它們。

## 時間成本

以下是研究流程的成本，不代表部署時的 latency。

| 項目 | 時間 |
|---|---|
| 候選 selection（FP32 基準＋17 個配置 × 128 個窗口） | 372 秒 |
| 抽 selection 資料 | 29 秒 |
| BC 求解（MILP 230 秒） | 238 秒 |
| BC 在 selection 上的評估（估樣本數） | 26 秒 |
| 抽 final 資料 | 34 秒 |
| final 評估：FP32 37 秒；aggressive／route-focused／BC 各約 80 秒 | 約 278 秒 |
| 控制：程式控制 47 秒；final 關閉量化 10 秒、重播 2 × 80 秒、重跑 2 次（子集） | 約 260 秒 |
| final stage 合計 | 500 秒 |
| 統計分析與作圖 | 0.7 秒 |
| plan（含候選資源與資料清點） | 23 秒 |

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_route_precision_protection_20261019      # 新實驗請改用新目錄
CPU=../../venv/bin/python
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_route_precision_protection.py
$CPU $S --stage plan --out-dir $O
$CPU $S --stage data_selection --out-dir $O
$GPU $S --stage control_prev --out-dir $O
$GPU $S --stage selection --out-dir $O          # 約 6 分鐘
env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE $CPU $S --stage bc_solve --out-dir $O   # 約 4 分鐘
$GPU $S --stage selection_bc --out-dir $O
$CPU $S --stage data_final --out-dir $O
$GPU $S --stage final --out-dir $O              # 約 8 分鐘
$CPU $S --stage analyze --out-dir $O
```

## 檔案

- **預先登記**：`plan.json`、`plan.md`、`amendments.json`。
- **配置**：`configs/`（aggressive、protect_L00…L15、route_focused、bc_same_budget）。
- **資料**：`selection_segments.json`、`selection_tokens.npy`、`selection_data_freeze.json`、`final_segments.json`、`final_tokens.npy`、`final_data_freeze.json`。
- **selection**：`control_prev.json`、`selection.json`、`selection_pS0.npz`、`selection_baseline.npz`、`choice_freeze.json`、`selection_table.csv`。
- **BC 與樣本數**：`bc_solve.json`、`sample_size.json`。
- **final**：`final_results.json`、`final_baseline.npz`、`routes/`、`final_per_article.csv`。
- **分析**：`analysis.json`、`fig_*.png`。
- **紀錄**：`timing.jsonl`、`logs/`。
