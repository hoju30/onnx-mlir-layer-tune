# OLMoE：多資源點的 fixed-route／dynamic-route 對照

日期：2026-10-18
腳本：`run_olmoe_dynamic_multipoint.py`（沿用上一輪已驗證的 `run_olmoe_dynamic_route.forward`）
輸出：本目錄

**本輪沒有改動的東西**
- BC、搜尋、配置、量化器、range。
- 沒有加入一階項，也沒有依結果選點或加樣。

**流程**
- 從 plan 到分析，每個階段用的都是同一份腳本（sha `b2ccbaed…`）。
- 事後唯一的修改是圖表版面（標籤碰撞），數字沒有變，記錄在 `amendments.json`。

**標記**
- [預先登記]：寫在 `plan.json` 的分析。
- [描述]：不做推論檢定的描述統計。
- [推論]：由資料推得、尚未直接驗證的判斷。

## 結論

**額外的路由損傷在最激進的資源點明顯變大，而且明確超過容忍界線；knee 的結果在新文章上重現；knee 和 relaxed 是否超過 ε 仍不確定。**

| 配置 | R | D_fixed | D_dynamic | **Δ_route** [96.7% 決策 CI] | 95% CI | 判定（Bonferroni） |
|---|---:|---:|---:|---|---|---|
| aggressive | 0.5946 | +0.02686 | +0.03439 | **+0.00753** [+0.00593, +0.00912] | [+0.00606, +0.00897] | **超過 +ε** |
| knee | 0.7071 | +0.01246 | +0.01448 | +0.00203 [+0.00095, +0.00311] | [+0.00103, +0.00301] | 不確定 |
| relaxed | 0.8409 | +0.00664 | +0.00799 | +0.00135 [+0.00044, +0.00227] | [+0.00051, +0.00221] | 不確定 |

- 單位都是 nat/token；300 篇新 train 文章，1200 個窗口；ε = 0.002。
- 判定用 96.7% 決策區間：三個配置做 Bonferroni 校正，每個單側檢定 α = 0.05/3。

**五個問題**

1. **額外路由損傷是否隨壓縮加重？** 兩端確實不同，但不能說是單調增加。
   - 主要對比 T = Δ(aggressive) − Δ(relaxed) = **+0.00618，95% CI [+0.00469, +0.00763]**，支持兩端點不同。
   - 次要對比（Bonferroni ×2，97.5% CI）：
     - aggressive − knee：+0.00550 [+0.00366, +0.00733]，支持；
     - knee − relaxed：+0.00067 [−0.00060, +0.00193]，無法判定。
   - 所以額外損傷在 knee 和 aggressive 之間明顯增加；knee 和 relaxed 之間的差距本輪無法分辨。
   - 三個點不能當成已驗證的連續曲線。
2. **knee 的結果是否重現？** 重現。
   - 新的 knee Δ_route 是 +0.00203，95% CI [+0.00103, +0.00301]，包含上一輪的 +0.00130。判定一樣是「不確定」。
   - 新資料較多（300 篇），95% CI 已不含 0：dynamic 確實額外增加損傷。只是它是否超過 ε，仍無法判定。
   - 在決策區間下，可以排除 Δ < 0.00095 和 Δ > 0.0031。
3. **哪個配置需要機制診斷？** aggressive（R 0.5946）。依預先固定的判讀規則 2，下一輪應優先對它拆解 expert ID 和 routing weight 各自的影響。
4. **是否有證據需要修改現有 pipeline？** 沒有證據需要修改搜尋或 criterion，但最終品質報告應該改用 dynamic route。
   - 在 dynamic 下，三個配置的品質順序和 fixed 相同：aggressive > knee > relaxed，相鄰差的 95% CI 都不含 0。
   - 但 fixed-route 評估一致地低估實際損傷：knee 約 +0.002 nat/token，aggressive 約 +0.0075。
   - 這是建議，本輪沒有修改任何東西。
5. **哪些仍未確定？**
   - knee 和 relaxed 是否超過 ε；knee 和 relaxed 之間的差異。
   - 額外損傷來自換 expert 還是權重改變。
   - 更低 R 的點、其他模型、native kernel。
   - BC 在 dynamic route 下的候選排序能力。

## 品質與 PPL

FP32 dynamic：CE 2.95281，PPL 19.160。

| 配置 | PPL fixed | PPL dynamic | 相對 FP32（fixed → dynamic） |
|---|---:|---:|---|
| aggressive | 19.681 | 19.830 | +2.72% → +3.50% |
| knee | 19.400 | 19.439 | +1.25% → +1.46% |
| relaxed | 19.287 | 19.313 | +0.67% → +0.80% |

- PPL 是由平均 CE 取 exp 得到的，不是平均各窗口的 PPL。
- D_dynamic / D_fixed（只作描述，不用來判斷）：1.28、1.16、1.20。

**品質順序**（相鄰差，95% CI）

| | aggressive − knee | knee − relaxed |
|---|---|---|
| D_fixed | +0.0144 [+0.0130, +0.0159] | +0.0058 [+0.0048, +0.0068] |
| D_dynamic | +0.0199 [+0.0179, +0.0219] | +0.0065 [+0.0051, +0.0078] |

兩種路由下的順序都沒有改變，所以不觸發判讀規則 5。

## 配置（`plan.json`，在任何新評估前固定）

**選點規則**
- 來源：`runs/olmoe_search_final_20261016/B/search_milp.json` 的修正後 MILP frontier（預設 pipeline 產生，上一輪用其 knee）。
- 依配置本身的實際資源選點，不看求解預算：R = max(權重 bytes／int8, BOPs／int8)。
- aggressive：最接近 0.55 的是 R 0.5946，距離 0.0446；R 0.5 的點距離 0.05。
- relaxed：最接近 0.82 的是 R 0.8409，距離 0.0209；R 0.7711 的點距離 0.049。

| 配置 | frontier 項目 | size／int8 | BOPs／int8 | 決定 R 的分量 | BC | 權重格式 | sha256 |
|---|---|---:|---:|---|---:|---|---|
| aggressive | cand03 | 0.594600 | 0.594603 | BOPs | 42.18 | int4 1161、int5 1549、int6 310、其他 31 | `e51dd107…` |
| knee | cand20 | 0.690593 | 0.707106 | BOPs | 18.29 | int5 1341、int6 1390、int4 170、int7 125、其他 25 | `8ebcc86e…`（＝上一輪） |
| relaxed | cand05 | 0.791544 | 0.840896 | BOPs | 9.11 | int6 1433、int7 1148、int5 346、其他 124 | `8e98082e…` |

- 用格式重新計算的 size、BOPs、R，都和 frontier 紀錄一致（差 ≤ 1e-9）。
- 配置之間不同的節點數：
  - aggressive 對 knee：2576 個（權重 2280、activation 759）；
  - knee 對 relaxed：2687 個；
  - aggressive 對 relaxed：3036 個。
- 21 個沒有 calibration token 的投影，在三個配置中都是 bf16 權重和 FP32 activation（歷史處理方式，未改動）。

## 資料與樣本數

**來源**
- WikiText-2 raw train，使用上一輪的嚴格文章邊界規則（train 共 600 篇）。
- 整篇排除先前任何 OLMoE 用過的文本，包括上一輪的 100 篇 train 和 28 篇 validation／test。
- 排除後可用（≥512 token）：456 篇。

**樣本數**（在任何新評估之前固定）
- 公式：n = ⌈(z·s／h)²⌉。
  - z = Φ⁻¹(1 − 0.05／3) = 2.128，對應 Bonferroni 校正的決策區間。
  - s = 上一輪 knee 的文章級 Δ_route SD，0.00858。
  - h = 0.001。
- 結果需要 334 篇，取上限 **300 篇**（1200 個窗口）。
- 預期決策區間半寬：0.00105。

**實際精度**

| 配置 | 文章級 SD | 決策區間半寬 |
|---|---:|---:|
| aggressive | 0.0128 | 0.0016 |
| knee | 0.0088 | 0.0011 |
| relaxed | 0.0075 | 0.0009 |

aggressive 的變異比規劃值大，和事先揭露的風險一致；但它的判定不受影響。

**抽樣**
- 均勻抽樣，seed 18071；每篇 4 個連續窗口。
- 三個配置和所有條件共用同一批窗口。
- 文章標題、offset、token hash 記在 `segments.json`、`data_freeze.json`。

**剩餘資料**：train 還剩約 156 篇可用（≥512 token）。下一輪若需要類似的樣本數，資料會不夠，需要另行規劃，例如 WikiText-103 的 train。

## 數值與完整性控制（`analysis.json` → `controls`）

全部通過，所有比較都逐位元一致。

| 控制 | 結果 |
|---|---|
| 程式控制：重現上一輪 knee 前 128 個窗口（舊資料，不進確認集） | fp32_dyn 的 CE、hidden state、路由，以及 q_fix、q_dyn 的 CE，都差 0 |
| fp32_fix vs fp32_dyn | 1200／1200 個窗口 CE 和最終 hidden state 逐位元相同 |
| 不量化的 dynamic 路徑重跑（控制子集：窗口 0–127，也就是第一批） | 128／128 相同，路由相同 |
| 各配置 q_dyn 重跑（控制子集） | 三個配置都 128／128 相同，路由相同 |
| 各配置的 q_fix 改用該配置 q_dyn 自己的路由 vs q_dyn | 三個配置都 1200／1200 逐位元相同 |
| 配置完整性 | 格式都在明確清單內；所有量化 activation 都有 range；posit LUT 齊全；router 物件未被替換；hook 只在 expert 投影上 |
| 有限值、路由 ID 在 [0, 64)、每個 token 8 個不同 expert、每窗口 127 個預測 token | 是 |

**推論設定**
- 每批 128 個窗口，最後一批是剩下的 48 個窗口；所有條件相同。
- GPU FP32，TF32 關閉；torch 2.14.0+cu130，RTX 3060。
- 本輪測的是 GPU 上的 QDQ 行為，不代表 native kernel 的結果。

## 逐篇文章（`per_article.csv`）[描述]

| 配置 | Δ_route > 0 的篇數 | 範圍 | 中位數 | 前 5 篇占正值總和 | 去掉最大 1 篇後的平均 |
|---|---:|---|---:|---:|---:|
| aggressive | 218／300 | −0.025 到 +0.041 | +0.0076 | 6% | +0.0074 |
| knee | 173／300 | −0.018 到 +0.038 | +0.0015 | 9% | +0.0019 |
| relaxed | 173／300 | −0.022 到 +0.027 | +0.0011 | 12% | +0.0013 |

- 結果不是由少數文章主導。
- 206／300 篇文章的 Δ(aggressive) > Δ(relaxed)。
- 同一篇文章在不同配置下的 Δ 只弱相關（Spearman 0.15–0.25）：單篇結果大多是雜訊，整體平均才有訊號。

## 路由變動（`per_layer_routing.csv`）[描述]

範圍：各配置的 q_dyn 對 FP32 dynamic，位置 0–126。

| 配置 | top-8 集合改變 | 重疊率 | 權重 L1 | dispatch TVD（第 1–15 層平均） | 依 margin 五分位的改變率（低 → 高） | 換入 expert 原本是 FP32 第 9 名 |
|---|---:|---:|---:|---:|---|---:|
| aggressive | 30.4% | 0.959 | 0.035 | 0.36% | 56%、45%、31%、17%、4.3% | 63% |
| knee | 23.4% | 0.970 | 0.026 | 0.26% | 50%、36%、21%、9%、1.8% | 70% |
| relaxed | 18.1% | 0.977 | 0.020 | 0.19% | 45%、27%、13%、5%、0.9% | 75% |

**逐層**（集合改變率／權重 L1）：

| 層 | aggressive | knee | relaxed |
|---|---|---|---|
| 1 | 28.6%／0.021 | 20.9%／0.015 | 15.0%／0.011 |
| 5 | 37.3%／0.036 | 29.2%／0.027 | 22.9%／0.021 |
| 8 | 35.4%／0.039 | 27.5%／0.029 | 21.7%／0.022 |
| 12 | 28.2%／0.043 | 21.9%／0.032 | 17.1%／0.024 |
| 15 | 30.7%／0.049 | 23.3%／0.036 | 18.8%／0.028 |

- 第 0 層在三個配置中都完全不變，因為第 0 層 router 的輸入還沒經過任何量化。
- 集合改變率在第 5–8 層最高。
- 權重 L1 隨深度增加。
- 三個配置的路由改變幅度依壓縮程度排列，和 Δ_route 的順序一致。

**文章層級**：每篇的路由改變比例和 Δ_route 只弱相關。
- 集合改變比例的 Spearman：0.13、0.13、0.11。
- 權重 L1 的 Spearman：0.18、0.08、0.10。

[推論]
- 壓縮越激進，路由偏移越大，額外損傷也越大，兩者在配置層級同向。
- 但只有三個點，而且文章層級幾乎不相關，所以不能說路由改變就是損傷來源，也不能歸因到某一層或某種變動。

## 判讀（依預先固定的規則）

- **規則 2（某配置明確超過 +ε）適用於 aggressive**：下一輪優先拆解 expert ID 與 routing weight。本輪不修改 criterion。
- **規則 3（不確定）適用於 knee 和 relaxed**：
  - 不宣稱等價或通過，也不繼續加樣。
  - 可排除的範圍：knee 的 Δ 在 0.00095–0.0031 之外；relaxed 在 0.00044–0.00227 之外。
  - 兩者的 95% CI 都不含 0。
- 規則 1、4、5 都不適用：
  - 沒有配置在 ε 內被判定等價；
  - dynamic 的損傷都高於 fixed；
  - 品質順序沒有改變。

## 下一輪建議（需另行預先登記）

**對 aggressive 配置（R 0.5946）拆解路由損傷**

| 條件 | expert ID | routing weight |
|---|---|---|
| (a) | 固定 FP32 | 量化模型對同一組 ID 重新計算（只放開權重） |
| (b) | 量化模型自選 | 用 FP32 router 機率（只放開 ID；需記錄 FP32 全 64 個 expert 的機率） |
| (c) | 依 FP32 margin 分層，只放開邊界附近 token 的路由 | |

- 主要比較預先固定，例如 (a) 和 (b) 各自占 Δ_route 的多少。
- 資料：train 只剩約 156 篇，需要新來源（例如 WikiText-103 train），並依來源分層。

**在 pipeline 層面**（尚未實作）
- 可以預先登記「最終報告改用 dynamic-route 評估」。
- 搜尋本身（BC 建立在 fixed route 上）在 dynamic 下是否會選出不同的 knee，需要另設 selection 資料和最終測試資料。本輪的確認集不能拿來挑配置。

**目前不建議**：依路由統計直接提高某層精度，或修改 BC。

## 時間成本

所有條件都在同一套 GPU 程式中執行，以下只是本次實驗的成本，不代表部署時的 latency。

| 項目 | 時間 |
|---|---|
| baseline 與路由擷取（FP32 dynamic，1200 個窗口） | 63.7 秒（stage 69.6 秒） |
| aggressive：fixed／dynamic | 137.4／137.6 秒 |
| knee：fixed／dynamic | 137.8／138.1 秒 |
| relaxed：fixed／dynamic | 138.6／139.0 秒 |
| 控制組合計 | FP32 fixed 63.4 秒；FP32 重跑子集 9.7 秒；三個「q_fix 改用 q_dyn 路由」共 413.7 秒；三個 q_dyn 重跑子集共 62.5 秒；上一輪程式控制 57.5 秒 |
| 統計（含路由統計與圖） | 16.7 秒 |
| plan／抽資料（CPU） | 9.1／8.4 秒 |
| eval stage 合計 | 1387 秒（約 23 分鐘） |

## 圖

![損傷與資源](fig_damage_vs_R.png)

![額外路由損傷與資源](fig_delta_route_vs_R.png)

- 圖中只有三個配置，點之間沒有連線，不代表已驗證的連續曲線。
- 誤差棒：第一張圖是 95% CI；第二張圖是 96.7% 決策區間，另畫出 0 和 ±ε 參考線。
- 顏色用 dataviz 參考色票的分類色第 1、2 槽：fixed 藍、dynamic 橙。文件記載這兩色在淺色模式通過所有檢查；本機沒有 node，無法執行驗證腳本。

## 已確立與未確定

**已確立**
- 在 aggressive 配置（R 0.5946），dynamic route 額外增加 +0.0075 nat/token 的損傷，明確超過 ε = 0.002（Bonferroni 決策區間下界 0.0059）。
- 主要對比 aggressive − relaxed 支持兩端點不同。
- knee 的額外損傷在新文章上重現（+0.0020，包含上一輪的 +0.0013），而且 95% CI 已不含 0。
- relaxed 的額外損傷 95% CI 也不含 0。
- fixed-route 評估在三個點都低估實際的 dynamic 損傷；三個配置的品質順序不變。
- 所有數值與執行路徑控制都逐位元通過，也逐位元重現上一輪。

**未確定**
- knee 和 relaxed 是否超過 ε。
- knee 和 relaxed 之間的額外損傷是否不同，以及是否單調。
- 額外損傷來自換 expert 還是權重改變；也不知道是哪些層造成的。
- R < 0.59 的點、其他 MoE 模型、native kernel。
- BC 在 dynamic route 下的候選排序，以及在 dynamic 下搜尋是否會選出不同的 knee。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_dynamic_multipoint_20261018      # 新實驗請改用新目錄
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
../../venv/bin/python run_olmoe_dynamic_multipoint.py --stage plan --out-dir $O
../../venv/bin/python run_olmoe_dynamic_multipoint.py --stage data --out-dir $O
$GPU run_olmoe_dynamic_multipoint.py --stage control_prev --out-dir $O     # 約 1 分鐘
$GPU run_olmoe_dynamic_multipoint.py --stage capture --out-dir $O          # 約 1 分鐘
$GPU run_olmoe_dynamic_multipoint.py --stage eval --out-dir $O             # 約 23 分鐘
../../venv/bin/python run_olmoe_dynamic_multipoint.py --stage analyze --out-dir $O
```

**環境**
- GPU：torch 2.14.0+cu130，RTX 3060，TF32 關閉，matmul precision highest。
- CPU：torch 2.14.0+cpu。
- 需要 `gpt2_assets/data/train-00000-of-00001.parquet`（sha256 `e83889ba…`）。

## 檔案

- **預先登記與凍結**：`plan.json`、`plan.md`、`configs/{aggressive,knee,relaxed}.json`、`segments.json`、`data_freeze.json`、`tokens.npy`、`amendments.json`。
- **控制**：`control_prev.json`、`eval_fp32.json`、`eval_{aggressive,knee,relaxed}.json`。
- **路由**：`routes/fp32_dyn.npz`、`routes/q_dyn_{config}.npz`、`capture.json`。
- **分析**：`analysis.json`、`per_article.csv`、`per_layer_routing.csv`、`fig_damage_vs_R.png`、`fig_delta_route_vs_R.png`。
- **紀錄**：`timing.jsonl`、`logs/`。
