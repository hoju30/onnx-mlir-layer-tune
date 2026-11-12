# OLMoE：預設 knee 點的 selective sensitivity refresh

日期：2026-10-22
腳本：`run_olmoe_knee_selective_sensitivity.py`（plan → auto ×3 → controls → refresh ×6 → 端到端 ×3 → check → freeze → data → eval → analyze）
輸出：本目錄

**本輪的範圍**
- 只驗證上一輪的策略能否移植到正常的自動 knee 配置。
- 沒有修改 BC 公式；沒有訓練 router；沒有測新模型；沒有調整 low-bit 門檻；沒有修改正式 pipeline 預設。
- 結果只適用於本輪的 q0_knee、calibration 和 GPU QDQ，不代表 native kernel。

**流程**
- `plan.json`／`plan.md` 在任何搜尋、統計與評估之前寫定。
- 所有 stage 都用同一份腳本（sha `8047cbeb…`），沒有修訂（`amendments.json` 為空）。

## 結論

**上一輪在 aggressive 點的 D 改善，在正常 knee 點沒有重現。** 只更新 ≤4-bit 節點的 D 和 static knee 幾乎完全相同：Δ_low = +0.000001，95% CI [−0.00063, +0.00065]。而且 D 確認比完整 refresh 差，超過 +0.0005 的界線。

**完整 refresh（B）在 knee 本身確認有收益**：−0.00120 nat/token，約為 knee 損傷的 7.4%。

依預先登記的解讀，屬於「**完整 refresh 有收益，但 D 沒有保留**」。

| 條件 | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| FP32 | 2.98935 | 19.873 | — |
| A：static knee（q0_knee） | 3.00553 | 20.197 | +0.01618 [+0.01510, +0.01728] |
| B：knee 完整 refresh | 3.00433 | 20.173 | +0.01498 [+0.01391, +0.01607] |
| D：knee low-bit sensitivity-only | 3.00553 | 20.197 | +0.01618 [+0.01507, +0.01730] |

600 篇新的 WikiText-103 文章 × 4 個窗口；三個量化條件都用原 router 和自己的 dynamic route。

| 對比（nat/token） | 平均 | 95% CI | 較好／較差篇數 | 判定 |
|---|---:|---|---|---|
| **Δ_low = D − A（主要）** | **+0.000001** | **[−0.00063, +0.00065]** | 303／297 | **方向未確認** |
| **Δ_keep = D − B（非劣）** | **+0.00120** | **[+0.00057, +0.00185]** | 256／344 | **下界 > +0.0005：D 確認比完整 refresh 差，超過界線** |
| B − A（次要） | −0.00120 | [−0.00184, −0.00057] | 344／256 | 完整 refresh 確認較好 |

- 10,000 次文章級 paired bootstrap，seed 22511；文章級 SD 約 0.0081。
- Δ_low 的 CI 包含 0，但**不包含** aggressive 點的效果 −0.00173。依預先登記的規則，這不算「CI 太寬無法判定」：資料足以排除 knee 上的 D 有 aggressive 那麼大的收益。

## 回答五個問題

**1. 前次 aggressive 的改善，能否在正常 knee 配置重現？**
- **D 不能。** 點估計幾乎為 0，CI 排除了 −0.00173 這個量級。
- **完整 refresh 可以**：B − A = −0.00120 [−0.00184, −0.00057]，損傷從 0.01618 降到 0.01498（−7.4%），PPL 從 20.197 降到 20.173（−0.12%）。這是完整 refresh 第五次在新文章上得到負的點估計，並和 aggressive 的結果並列（見後文）。
- **[推論] D 在 knee 失效的原因**：低位元集合太小，而且不在 refresh 真正改動的位置。
  - knee 的 q0 只有 169 個 ≤4-bit 投影（5.5%），它們在 calibration 只拿到 2.7% 的 routed rows（4,451／163,838）。aggressive 時是 1161 個（38.1%）、25% 的 rows。
  - B 改動的 201 個投影中，**168 個（84%）在低位元集合之外**，主要是：
    - weight 的 int6↔int5 互換（64 個）、int6→int7（11 個）；
    - 中位元節點的 activation 格式互換。
  - D 無法更新這些節點的成本，只改了 37 個投影，其中 32 個在集合內（主要是 int4→int5 共 17 個）。
  - 本輪沒有直接檢驗 B 的收益具體來自哪些改動。

**2. knee 中 4-bit 節點減少後，省時比例如何變化？**
- 比例變大了：D 的額外時間是 27.9 秒，B 是 79.4 秒，**少 65%**（aggressive 時是 39.9 對 80.6，少 50%）。
- **但這個「省時」主要是因為 D 幾乎沒做事**：
  - D 只算 169 個投影：weight sensitivity 0.8 秒、梯度統計 0.14 秒、ΔW 讀取 0.8 秒。
  - 27.9 秒中有 **92% 是每次 refresh 都要付的固定成本**：載入與量化準備 16.8 秒、forward 1.6 秒、backward 2.3 秒、剪枝和搜尋 4.9 秒。
  - 節點再少，D 的額外時間也不會低於約 26 秒。
- **D 相對 B 省下的 51.5 秒**：
  - weight sensitivity 25.4 秒；
  - activation 候選誤差 12.2 秒（改讀快取）；
  - ΔW 讀取 10.5 秒；
  - activation 梯度統計 3.0 秒。

**3. D 是否同時滿足「優於 static」與「相對 full 非劣」？**
- **兩者都不滿足。**
  - 對 static 的改善未確認（點估計 0）；
  - 對完整 refresh，D 確認差超過 0.0005（CI 下界 +0.00057）。

**4. 品質收益與完整 offline 成本相比，是否值得作為可選步驟？**
- **D 在 knee 不值得**：沒有可偵測的收益，卻多 25.6 秒（從零的時間 +15%）。
- **完整 refresh B 在 knee 有收益，但幅度小**：
  - 從零到最終配置（端到端實測）：A 170.5 秒，B 246.6 秒（+76 秒，+45%）。
  - 換到的是損傷 −7.4%、PPL −0.12%。
- 0.0012 nat/token 和先前測得的跨後端實作差異（GPU、CPU、native 之間約 0.001–0.002）同一量級，native kernel 上是否成立沒有驗證。
- 若要保留可選步驟，證據支持的是**完整 refresh**（aggressive 與 knee 都確認），而不是 D。即使是完整 refresh，也只適合作為研究選項，不足以設為預設。

**5. 下一輪測另一個 MoE，還是先維持 static BC？**
- **建議先維持 static BC**，不要把 D 帶到另一個 MoE：同一個模型換一個資源點，「≤4-bit」這個選擇規則就失效了，跨模型移植沒有根據。
- 若要繼續這條研究線，比較有資訊量的下一步（未執行）：
  1. 用 native kernel 驗證 OLMoE knee 上的 B − A；
  2. 若想保留「選擇性更新」，需要預先登記一個新的選擇規則，並在新資料上檢驗。例如依「refresh 後成本變化的大小」或依 routed rows 選節點，而不是依位元數。不要用本輪結果事後調門檻。

## knee 的取得（`knee/knee.json`、`work/runs/auto_r1/selection.json`）

**流程**
- 表：修正後的 static BC 表 `runs/olmoe_proxy_refresh_20261022/table_static.json`。它和上一輪重建的結果逐位元相同；本輪端到端的 3 次 static 重建也都逐位元相同。
- 搜尋：先做精確支配剪枝（70,153 個選項），再在 `mp_auto.auto_select` 中以 `make_solver("lpfirst")` 取代每個預算的求解。
- 設定：17 個粗掃描樣本、2 輪 refinement、原本的二維 normalized-chord knee、時限 300 秒。

**結果**
- **選中的 knee**：
  - 預算 0.70710678（2^−1/2，第 0 輪粗掃描的一個樣本）；
  - 實際 R 0.7070606（B 0.7070606、M 0.6905932），BC 18.2973191；
  - 求解路徑為 LP（status 1）；兩輪 refinement 都沒有移動 knee。
- **精確共同上限**（auto_select 給求解器的值，B、D 都用它）：
  - weight bytes ≤ 4,524,359,631.556；
  - BOPs ≤ 46,648,327,679,632.125。
  - q0_knee 的實際用量為 4,418,699,264 bytes（97.66%）和 46,645,278,408,704 BOPs（99.993%）。
  - 用 static 表在這組上限重解，得到的正是 q0_knee。
- **求解路徑**：23 次求解中，LP 被接受 21 次，退回 MILP 2 次（R 3.36 重試、R 6.73）。這和歷史的 LP 優先搜尋相同。
- **重複**：3 次 auto 的 knee、嘗試過的預算和配置完全相同；3 次端到端執行的 knee 也相同。
- **和歷史 knee 的比較**：
  - 和 `runs/olmoe_search_final_20261016` 的 LP 優先 knee **完全相同**（0 個投影不同）。那次用的是修正前量化器的歷史表；兩張表的 int activation 成本只差 1e-6 到 2e-5（相對），目標值只差 7.1e-9（18.297319103 對 18.297319096）。
  - 和歷史 MILP knee（R 0.7071056）差 7 個投影：4 個 weight（int4→int5 3 個、int6→int5 1 個）、4 個 activation。
- **q0_knee 的格式**：
  - weight：int6 1389、int5 1343、int4 169、int7 125、int8 18、bf16 7；
  - activation：p8e1 2188、p6e1 506、int7 210、int8 62、int6 40、f16 35、int5 5、p4e1 3、int4 2。

**目標節點**（`lowbit_nodes.json`；依 q0_knee 重算並凍結）
- 169／3051（5.5%），全部是 weight int4。
- 依投影：gate 65、up 51、down 53；分布在 74 個 expert、16 層。第 0 層最多：29 個投影、15 個 expert。
- activation 格式：p6e1 104、int6 29、p8e1 13、int7 12、int5 5、p4e1 3、int4 2、int8 1。
- **calibration 覆蓋**：目標 expert 在 FP32 和 q0_knee calibration 都拿到 4,451 個 routed rows（佔 2.7%）。沒有無 token 的目標節點，也沒有任何 fallback。

## 方法

**數學與梯度語義**：和上一輪相同，`build` 逐字複製並維持相同的運算順序。
- summed CE 梯度尺度；
- per-output-channel weight sensitivity 搭配 static ΔW；
- activation 的 variance 加 coherent bias；
- gate／up 共用輸入的 total gradient；
- activation QDQ 經 identity STE；
- q0_knee 正常 dynamic forward，backward 把記錄的路由視為常數；
- 不更新任何參數。

**D 的成本**
- activation：A_new = Σ (G2_new·var_static + GS_new·μ_static²)，以 static routed rows n_s 正規化。
- 非目標節點逐位元沿用 static 表。
- 搜尋時所有節點都可以改格式。

**重用上一輪的 FP32 static 統計**
- 包括 static 表、ΔW（`weight_error_d2.npz`）、activation 誤差快取。
- 重用前核對了 20 項依賴的 hash（plan 中逐項記錄），全部一致：model 檔、calibration、ranges、stage-1 路由、static 表、兩個快取，以及 12 份量化器與搜尋的程式碼。
- q0_knee 工作狀態的梯度與 activation 誤差全部重算。

**B、D**：各在 knee 上限重新搜尋一次，搜尋方式同 knee 的單一預算求解（LP 優先、精確支配剪枝）。沒有重跑 frontier，也沒有依 loss 回退。

## 控制（`check.json`、`controls.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 1. static | static 表的 hash 和上一輪記錄一致；端到端 A／B／D 的 3 次 static 重建逐位元相同；ΔW 和 activation 快取的重建也相同；在 knee 上限重解得到 q0_knee |
| 2. q0_knee forward | 6 次 refresh 和 2 次端到端 refresh 的逐窗口 CE 與最終 hidden state（sha256）、6 次 refresh 的路由，都和獨立的 `run_olmoe_dynamic_route.forward` 逐位元相同 |
| 3. 輸入梯度 | 6 次 refresh 和 2 次端到端 refresh 的 16 層輸入梯度 sha256 全部相同；重算和 forward 的差為 0 |
| 4. D 非目標 | 2882 列和 static 表逐位元相同 |
| 5. D 目標 | weight 列和 B 逐位元相同；用記錄的梯度統計、快取和 ΔW 以 numpy 獨立重算，weight 最大相對差 0、activation 3.8e-16（門檻 1e-12）。改寫成 G2·(S2/n_s − μ²) 的形式也只差 4.3e-16；若誤用 q0_knee 的 rows 正規化，差異達 100%，表示這個檢查分辨得出錯誤的正規化 |
| 6. 有效性 | 有限；格式合法；都在精確上限內；所有執行的上限都和 knee 的記錄一致 |
| 7. 輸入 | checkpoint（含 router）、ranges、calibration、量化器程式碼、所有輸入的 hash 前後一致 |
| 8. 重複 | auto 3 次、B／D 各 3 次、端到端 3 次：knee、表、配置完全相同（端到端 B／D 的表與配置也等於計時重複的結果） |
| 回歸 | 複製的 `build` 重現上一輪 aggressive D 表（B、C 兩項都逐位元相同） |
| 其他 | STE 單元測試通過 |
| 評估 | 程式控制（上一輪的窗口 0–127）：FP32、A、D 的 CE 差 0；D 第一批重跑逐位元相同；三個配置都不同，各自評估；router 未被替換 |

## 計時（`analysis.json` → `timing`）

**方式**
- 每次一個 process，所有計時區塊前後都有 CUDA synchronization，沒有清除系統快取。
- 每次執行前，checkpoint、ΔW 快取、activation 快取都 100% 在 page cache 中。
- 順序：auto r1（正式）→ controls → auto r2、r3 → B1 D1 D2 B2 B3 D3 → 端到端 A、B、D。
- 「方法時間」= 執行時間減去控制區塊；process 牆鐘另含約 2–3 秒的啟動與 CUDA 初始化。

**a. 已有 static 快取與 q0_knee 時的額外更新成本**（中位數，3 次）

| 類別（秒） | B 完整 refresh | D low-bit |
|---|---:|---:|
| 載入、搬移、量化準備（checkpoint 6.3、q0 量化 7.1、量化器初始化 2.3、其他 1.1） | 16.7 | 16.8 |
| forward／路由 | 1.6 | 1.6 |
| 真正的 backward／重算 | 2.4 | 2.3 |
| weight sensitivity 矩陣運算 | 26.2 | 0.8 |
| activation 梯度統計 | 3.2 | 0.14 |
| activation 候選誤差 | 12.2 | — |
| 快取讀取（ΔW＋activation） | 11.3 | 0.8 |
| 組表 | 0.2 | 0.05 |
| 剪枝＋搜尋 | 4.9 | 4.9 |
| 輸出、未分類 | 0.9 | 0.5 |
| **方法時間：中位數（範圍）** | **79.4（79.3–79.5）** | **27.9（27.8–28.0）** |
| process 牆鐘：中位數 | 82.3 | 30.1 |
| 峰值 GPU／CPU（GB） | 4.21／14.5 | 4.14／14.6 |

**完整 auto knee 搜尋**（3 次）
- 方法時間 80.0 秒（80.0–80.6）：剪枝 4.1 秒，auto_select 75.4 秒（LP 21 次＋MILP 退回 2 次）。
- 只用 CPU，峰值 RSS 1.1 GB。

**b. 從 static 統計、完整 auto knee 搜尋到最終配置的總成本**

| | **實測：端到端單次執行（方法／process 牆鐘）** | 各階段中位數相加的估算 | 相對 A |
|---|---:|---:|---:|
| A：static → auto knee | **170.5／173.5** | 170.7 | — |
| B：static（含 ΔW 快取）→ auto knee → 完整 refresh → 搜尋 | **246.6／250.5** | 251.3 | +76.1（+45%） |
| D：static（含 ΔW 與 activation 快取）→ auto knee → D → 搜尋 | **196.1／199.4** | 200.9 | +25.6（+15%） |

- **實測值**：每個條件只量了 1 次，不是中位數。
- **估算值**：「static 統計」用上一輪的 3 次重複中位數（扣除它的固定預算搜尋：A 90.7、B 91.8、D 92.9 秒），加上本輪 auto 和 refresh 的中位數；這只是各階段相加，不是整條 pipeline 的實測中位數。
- 估算比實測 B、D 多約 5 秒。[推論] 一部分是因為估算把量化器初始化（約 2.3 秒）等準備工作算了兩次（static 和 refresh 各一次），而端到端在同一個 process 中只做一次。
- 各階段的分界時間（端到端）：
  - static 統計完成於 91 秒左右；
  - auto knee 完成於 171–173 秒左右。
- **快取成本**（都計入了 B／D，沒有加給 A）：
  - A 的端到端不寫 ΔW 快取；
  - B 多寫 ΔW 快取 1.0 秒；
  - D 再多建 activation 快取 1.0 秒、寫入 0.3 秒，快取檔 1.17 GB；峰值 CPU RSS 16.1 GB（A、B 為 14.8 GB）。

**和 aggressive 點並列**（額外成本中位數）

| 資源點 | 目標節點 | B | D | D 省下 |
|---|---|---:|---:|---:|
| aggressive（R 0.595） | 1161（38%，25% rows） | 80.6 秒 | 39.9 秒 | 50% |
| knee（R 0.707） | 169（5.5%，2.7% rows） | 79.4 秒 | 27.9 秒 | 65% |

## 配置與描述性指標（不用於重新選擇配置）

**格式變化**（`freeze.json`）

| | 不同的投影 | weight | activation | 在低位元集合內／外 |
|---|---:|---:|---:|---|
| B 對 A | 201 | 114 | 93 | 33／168 |
| D 對 A | 37 | 18 | 22 | 32／5 |
| D 對 B | 177 | 96 | 85 | |

- B 的 weight 改變：int6→int5 33、int5→int6 31、int4→int5 18、int5→int4 13、int6→int7 11、int7→int6 7、int7→int8 1。
- D 的 weight 改變：int4→int5 17、int6→int5 1。

**搜尋目標值**

| 條件 | 目標值（各自的表）／LP 下界 | size 用量 | BOPs 用量 |
|---|---|---:|---:|
| A | 18.29732（auto） | 97.66% | 99.993% |
| B | 17.96180／17.96081 | 97.71% | 99.999% |
| D | 18.30320／18.30216 | 97.76% | 99.999% |

**交叉目標值**（列：用哪張表；欄：哪個配置）

| 表 | A | B | D |
|---|---:|---:|---:|
| static | 18.29732 | 18.32288 | 18.29394 |
| B | 18.01559 | 17.96180 | 18.00874 |
| D | 18.30982 | 18.33187 | 18.30320 |

- 用 static 表評分時，D 的配置甚至比 A 低 0.0034（0.02%）。
- 原因是 A 來自 auto_select 的 LP 取整，留下約 0.007% 未用的 BOPs；D 在同一上限重新搜尋時用掉了這部分資源。
- 這是一個描述性的小差異；proxy 數值不是 loss。

**實際路由的計算成本**（評估資料，2400 個窗口）

| | 搜尋節點 BOPs，相對全 int8 | 含未搜尋 expert，相對全 int8 | 送到未搜尋 expert 的 token-slot | 搜尋 proxy BOPs（≤ 4.66483e13） |
|---|---:|---:|---:|---:|
| A | 0.67837 | 0.71149 | 177,701 | 4.66453e13 |
| B | 0.67844 | 0.71157 | 177,756 | 4.66478e13 |
| D | 0.67867 | 0.71180 | 177,717 | 4.66478e13 |

- FP32 有 175,582 個 token-slot 送到未搜尋的 expert。
- 路由：(token, 層) 的 expert 集合和 FP32 不同的比例，A／B／D 都約 23.2–23.3%；B、D 和 A 不同的比例分別是 16.7%、15.0%。

**完整 refresh（B − A）的歷次結果**（各輪文章不同，不合併）

| 資源點、輪次 | 文章數 | 平均 | 95% CI |
|---|---:|---:|---|
| aggressive：proxy refresh | 128 | −0.00196 | [−0.00415, +0.00017] |
| aggressive：確認實驗 | 600 | −0.00121 | [−0.00219, −0.00024] |
| aggressive：router FT 的評估 | 600 | −0.00211 | [−0.00305, −0.00116] |
| aggressive：selective sensitivity | 600 | −0.00108 | [−0.00201, −0.00016] |
| **knee：本輪** | 600 | **−0.00120** | **[−0.00184, −0.00057]** |

## 資料（`segments.json`、`dedup_checks.json`）

**來源**：WikiText-103 raw train，使用嚴格文章邊界。

**排除**

| 理由 | 篇數 |
|---|---:|
| 標題或內容和 WikiText-2 重疊 | 613 |
| WikiText-103 內部重複 | 464 |
| 所有歷史 OLMoE 用過的 WikiText-103 文章 | 2474 |

2474 篇的組成：proxy refresh 128、確認實驗 600、router FT 600、selective sensitivity 600、dynamic_prop 242、mass_restore 103、precision_protection 的 final 169 與 selection 32。這是實際掃描 `runs/**/*segments*.json` 得到的，只依路徑排除本輪自己的目錄。

**抽樣**
- 在凍結之後才抽。
- 依 seed 22501 的順序看了 626 篇：接受 600 篇；跳過 24 篇近重複（≥10% 的 5-gram）、2 篇不足 512 token。窗口起點用 seed 22502。
- 和歷史文章的重疊為 0。

## 研究成本（不屬於方法成本）

| 項目 | 時間 |
|---|---|
| plan（含 20 項重用 hash 核對與資料可用性掃描） | 43.4 秒 |
| auto r1 | 87.7 秒 |
| controls（獨立 forward、STE、回歸測試） | 45.5 秒 |
| auto r2–r3 | 165.5 秒 |
| refresh 計時 6 次 | 340.6 秒 |
| 端到端 3 次 | 625.3 秒 |
| check | 71.8 秒 |
| 抽資料 | 101.8 秒 |
| 評估 | 1009.9 秒（FP32 123.1 秒、A／B／D 各約 268 秒、程式控制約 60 秒、重跑 20.6 秒；峰值 GPU 7.7 GB） |
| 分析 | 15.6 秒 |

**磁碟**：`work/` 1.8 GB，主要是端到端 D 的快取副本，內容已確認和既有快取相同。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_knee_selective_sensitivity_20261022      # 新實驗請改用新目錄
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_knee_selective_sensitivity.py
$CPU $S --stage plan --out-dir $O
$GPU $S --stage auto --rep 1 --out-dir $O          # 正式 knee，約 1.5 分鐘
$GPU $S --stage controls --out-dir $O
$GPU $S --stage timing --phase auto --out-dir $O   # r2、r3
$GPU $S --stage timing --phase refresh --out-dir $O
$GPU $S --stage timing --phase e2e --out-dir $O    # 約 10 分鐘
$CPU $S --stage check --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage eval --out-dir $O                  # 約 17 分鐘
$CPU $S --stage analyze --out-dir $O
```

**環境**
- GPU：torch 2.14.0+cu130，RTX 3060，TF32 關閉。
- 搜尋：scipy 1.18.1（HiGHS），和量化器在同一個 HTQ venv 中執行。
- nvidia-smi 的 NVML 版本不一致不影響 CUDA，系統驅動未改動。

## 檔案

- **預先登記**：`plan.json`（含重用 hash 核對）、`plan.md`、`amendments.json`（空）、`input_hashes_before.json`。
- **knee**：
  - `knee/knee.json`：上限、精確資源、frontier、各輪 knee、歷史比較；
  - `knee/q0_knee.json`；
  - `work/runs/auto_r{1,2,3}/selection.json`：完整 auto 輸出、求解紀錄、solver log；
  - `logs/`：HiGHS log。
- **目標**：`lowbit_nodes.json`。
- **各次執行**：`work/runs/{auto_r*,refresh_{B,D}_r*,e2e_{A,B,D}}/`（`run.json`、表、配置；`refresh_D_r1/target_stats.npz` 用於控制 5）。
- **凍結**：`configs/{A,B,D}.json`、`tables/{static,B,D}.json`、`freeze.json`。
- **控制**：`controls.json`、`check.json`、`controls_summary.json`。
- **資料**：`segments.json`、`tokens.npy`、`dedup_checks.json`、`data_freeze.json`。
- **結果**：`eval.json`、`routes_{fp32,A,B,D}.npz`、`per_article.csv`、`per_window.csv`、`analysis.json`。
- **時間**：`timing.json`、`timing.jsonl`。
