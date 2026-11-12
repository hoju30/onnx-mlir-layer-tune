# OLMoE：sensitivity-only 與 low-bit selective sensitivity refresh

日期：2026-10-22
腳本：`run_olmoe_selective_sensitivity.py`（plan → controls → static ×7 → refresh ×9 → check → freeze → data → eval → analyze）
輸出：本目錄

**本輪的範圍**
- 只做編譯器分析與自動量化：沒有模型訓練、router fine-tuning 或參數更新；沒有修改 BC 公式；也沒有整合為預設 pipeline。
- 所有工作狀態都由原始 q0、原 checkpoint 和原 router 建立；沒有載入 router FT 的 checkpoint。
- 結果只適用於本輪的 q0、calibration、資源點（aggressive，R ≈ 0.595）和 GPU QDQ，不代表 native kernel。

## 結論

**只更新 q0 中 4-bit 節點的 sensitivity（D），在 600 篇新文章上確認優於 static BC；依預先登記的 +0.0005 界線，它也不比完整 refresh 差。和完整 refresh 相比，額外時間少一半（80.6 → 39.9 秒）。不過收益本身很小，只有 q0 損傷的 5%，PPL 約 −0.17%。**

**全部節點只更新 sensitivity（C）的收益則沒有得到確認。**

| 條件 | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| FP32 | 2.96044 | 19.306 | — |
| A：static BC（q0） | 2.99499 | 19.985 | +0.03455 [+0.03304, +0.03608] |
| B：完整 refresh（= q1） | 2.99391 | 19.964 | +0.03348 [+0.03199, +0.03499] |
| C：全部節點 sensitivity-only | 2.99440 | 19.973 | +0.03396 [+0.03245, +0.03552] |
| **D：low-bit sensitivity-only** | **2.99326** | **19.951** | **+0.03282 [+0.03133, +0.03434]** |

600 篇新的 WikiText-103 文章 × 4 個窗口；每個條件都用原 router 和自己的 dynamic route。

| 對比（NLL 差，nat/token） | 平均 | 95% CI | 較好／較差篇數 | 判定 |
|---|---:|---|---|---|
| **Δ_low = D − A（主要）** | **−0.00173** | **[−0.00264, −0.00080]** | 325／275 | **確認 D 優於 static BC** |
| **Δ_keep = D − B（非劣）** | **−0.00066** | **[−0.00146, +0.00016]** | 320／280 | **上界 < +0.0005：D 不比完整 refresh 差超過 0.0005** |
| C − A（次要、探索） | −0.00059 | [−0.00156, +0.00037] | 311／289 | 未確認 |
| C − B（次要、探索） | +0.00048 | [−0.00029, +0.00125] | 284／316 | 未確認 |
| D − C（次要、探索） | −0.00114 | [−0.00191, −0.00036] | 335／265 | D 較好（探索性） |
| B − A（背景） | −0.00108 | [−0.00201, −0.00016] | 313／287 | refresh 較好 |

- 10,000 次文章級 paired bootstrap，seed 22411；PPL = exp(平均 NLL)。
- 次要比較沒有預先指定多重比較校正。事後用 Bonferroni（4 個比較，98.75% 區間）檢查（`posthoc_multiplicity.json`）：
  - D − C 仍排除 0：[−0.00213, −0.00014]；
  - B − A 變成跨 0：[−0.00225, +0.00011]；
  - 這不改變預先登記的判定。

## 回答五個問題

**1. 只更新 sensitivity，是否保留收益？**
- **部分保留，要分兩種情況看。**
- **全部節點（C）**：
  - 點估計保留了 B 約一半的收益（−0.00059 對 −0.00108），但對 A 的 CI 跨 0，所以**沒有確認**。
  - 對 B 的 CI 也跨 0（−0.00029 到 +0.00125）。不能說 C 和完整 refresh 一樣好，也不能說它失去了收益：**證據不足**。
  - C 和 B 的 weight 項逐位元相同（原本的 refresh 就已經沿用 static ΔW），所以兩者只差在 activation 誤差統計的來源。兩者配置只差 54 個投影，其中 52 個是 activation 格式。
- **只更新低位元節點（D）**：同樣只用 static 誤差統計，卻確認了改善。所以 sensitivity-only 本身不必然失去收益。

**2. 只更新 4-bit 節點，額外犧牲多少品質？**
- **本輪沒有偵測到犧牲。**
- D − B = −0.00066，CI [−0.00146, +0.00016]，點估計反而是 D 較好；CI 上界低於預先登記的 +0.0005，所以非劣判定成立。
- 探索性比較 D − C = −0.00114，D 比全部節點更新的 C 還好，事後 Bonferroni 之後仍排除 0。
- 這個方向出乎預期，本輪沒有檢驗原因。一個可能的解釋（推論，未驗證）：
  - D 只改了 177 個投影，C 改了 284 個；D 在低位元集合外只改了 57 個，C 改了 164 個。
  - 早先 OLMoE 的研究發現，per-node 排序錯誤主要來自 calibration 到新文本的轉移。限制更新範圍，可能減少了高位元節點上不可靠的重新分配。

**3. 真正省下哪些工作？總 offline 時間有沒有下降？**
- **有下降。**
  - 已有快取時，D 的額外時間為 39.9 秒，B 為 80.6 秒（−50%）。
  - 從零到最終配置：D 137.9 秒，B 177.4 秒（−22%）；static 本身是 95.7 秒。
- **D 相對 B 省下的 40.7 秒**（中位數差）：

  | 工作 | 省下（秒） | 說明 |
  |---|---:|---|
  | weight sensitivity 矩陣運算（GYᵀX，FP64） | 19.3 | 26.5 → 7.2 |
  | 候選 activation 誤差 | 12.5 | 改讀快取；C 也省這一項 |
  | 讀取快取 | 6.7 | ΔW 11.4 → 4.5（取決於 npz 格式），activation 快取 +0.15 |
  | activation 梯度統計 | 2.1 | 3.2 → 1.1 |
  | 組表、搜尋與未分類 | 約 0.2 | 各類中位數的差，加總不必然等於 40.7 |

- **backward 沒有變短。** 真正的逐層重算 backward 在 B、C、D 都是約 2.4 秒（static 為 1.5 秒）。上一輪「backward＋sensitivity 28 秒」中，絕大部分其實是 FP64 的 weight sensitivity 矩陣運算（26.5 秒），不是反傳本身。
- **為什麼省下的比例不是 38%**：
  - D 的目標雖然占投影數的 38.1%，但這些低位元 expert 在 q0 calibration 只拿到 25% 的 routed rows（41,151／163,838）。
  - weight sensitivity 的時間和 rows 與窗口成正比，所以降到 B 的 27%。
- **每次 refresh 都要付的固定成本**：約 26 秒，占 D 額外時間的 65%。
  - checkpoint 載入 6.5 秒；
  - q0 量化 7.1 秒；
  - 量化器初始化 2.3 秒；
  - 其他準備 1.1 秒；
  - forward 1.6 秒；
  - backward 2.4 秒；
  - 剪枝和搜尋 4.9 秒。
- **只更新 sensitivity 的 C** 只省了 11.3 秒（−14%）：省下 activation 誤差的 12.5 秒，多了 0.8 秒的組表和 0.3 秒的快取讀取。
- **不計快取讀取的比較**（避免把 npz 格式造成的差異算成方法的收益）：
  - B 69.2 秒，D 35.2 秒，仍省 34.0 秒（−49%）。

**4. 快取成本是否抵消省時？**
- **沒有。**
- **建立成本**：static 建表加上 activation 快取的中位數為 98.0 秒，不加為 96.8 秒，只多約 1.2 秒（收集 0.8 秒、寫入 0.3 秒）。
- **大小與記憶體**：檔案 1.17 GB；峰值 CPU RSS 從 14.8 GB 增加到 15.9 GB。
- **讀取成本**：
  - C 讀 activation 快取 0.34 秒，D 讀 0.15 秒；
  - 一次在 page cache 外的讀取（C1，執行前只有 4% 駐留）約多 0.9 秒。
- **相對省下的時間**：每次 refresh 省下 activation 誤差 12.5 秒，遠大於快取成本。
- **真正較貴的是既有的 ΔW 快取**（`weight_error_d2.npz`，341 MB，30,510 個 zip 成員）：
  - 每次 refresh 讀它要 11.2–11.4 秒（D 只讀目標節點，4.5 秒）。
  - 它在原本的完整 refresh 就已經存在，計時中沒有把它的計算算成省時。

**5. 值得保留為可選分析步驟，還是維持 static BC？**
- **建議把 D 保留為可選的分析步驟，不設為預設，預設仍維持 static BC。**
- 支持保留的理由：
  - 在本資源點，它是第一個以預先登記的主要終點確認優於 static BC 的省時變體。
  - 它在品質上不輸完整 refresh，額外成本只有一半。
- 不足以設為預設的理由：
  1. **收益很小**：損傷減少 5.0%（B 為 3.1%），PPL 從 19.985 降到 19.951。換算從零的時間，是用 +44% 的 offline 時間（95.7 → 137.9 秒）換來的。
  2. **可能被實作差異蓋過**：先前測得同一配置在 GPU、CPU、native 之間的 loss 差異約 0.001–0.002 nat/token，和 0.0017 同一量級。native kernel 上是否仍有效，本輪沒有驗證。
  3. **只有一個資源點**：其他 q0、calibration、資源點和模型都沒有測。
  4. **D 優於 C 的原因不明**：這是探索性結果。

## 方法（`plan.json`）

**固定的輸入**
- q0：`runs/olmoe_proxy_refresh_20261022/configs/q0.json`（sha256 `2fe46627…`）。
- q1：同目錄 `configs/q1.json`（sha256 `346fd805…`）。
- ranges：`runs/olmoe_multipoint_20261015/ranges.json`。
- calibration：`runs/olmoe_fixed_route_20261015/tokens_calibration.npy`（10 × 128）。
- 量化器：`dev_quant(corrected=True)`，含 Posit LUT 和修正後的 int activation QDQ。
- 3051 個搜尋投影，GRID11 × GRID11；21 個未搜尋投影維持 bf16 權重和 FP32 activation。
- **資源上限直接讀原 plan**，不是 q0 的實際用量：
  - weight bytes ≤ 3,804,495,872；
  - BOPs ≤ 39,226,380,910,592，用 FP32 calibration 的 routed rows 計算。

**低位元集合**（`lowbit_nodes.json`，在 plan 中固定）
- 定義：q0 的 weight 或 activation 位元數 ≤ 4，依 `FORMAT_BITS` 解析。
- 結果：1161／3051（38.1%），和預期一致。
  - 1161 個 weight 為 int4；其中 15 個的 activation 也 ≤ 4 bit。
  - 依投影：gate 437、up 370、down 354；分布在 16 層、450 個 expert（每層 40–132 個投影）。
- 目標集合依初始 q0 凍結，不做第二輪。

**四個條件**
- **A**：原 q0，不做 refresh。
- **B**：完整 refresh。所有投影在 q0 工作狀態下重算 weight／activation sensitivity 和 activation 誤差；ΔW 沿用 static。
- **C**：全部投影只更新 sensitivity；誤差統計全部來自 FP32 static reference。
- **D**：只有 1161 個目標投影更新 sensitivity（重算它們所有候選格式的成本）；其他 1890 個投影逐位元沿用 static 表。

**數學**（BC 的形式不變）
- **Weight 項**：
  - W_new[f] = Σ_c S_new[c]·D_static[c,f]。
  - S_new[c] = G_c／n_in，G_c 是各窗口 GYᵀX 逐列平方和的總和。GY 是對 y-QDQ 前輸出的梯度，X 是 x-QDQ 後的輸入。
  - D_static 是原 checkpoint 權重的 ΔW²，沿用原快取。
- **Activation 項**：
  - A_new[f] = Σ_slots Σ_c (G2_new[c]·var_static[c,f] + GS_new[c]·μ_static[c,f]²)。
  - **快取內容**：每個搜尋 expert × 5 個 slot 張量（gate／up 共用的 x_in、gate_y、up_y、down_x、down_y）× 每個 channel × 10 種量化格式，存 static 的 Σerr 和 Σerr²（float64），再加上 static routed rows n_s。
  - μ = Σerr／n_s，var = Σerr²／n_s − μ²。
  - 實際計算採用原 BC 的運算順序 G2／n_s·(S2 − n_s·μ²) + GS·μ²，代數上等於上式。
  - 正規化用 n_s，不用 q0 的 rows；不額外除以 token 數；維持原 BC 的 summed-CE 梯度尺度。
  - gate／up 的 x ranges 在 1017 個 expert 上完全相同，所以共用一份 x_in 統計（plan 中已檢查）。
  - 原本只存合併後 B／C 的 `table_static.json` 拆不出這些統計，所以本輪擴充了快取；沒有用「合併成本 × 梯度比例」來代替。
- **Forward／backward**：
  - B、C、D 都跑完整的 q0 calibration forward，使用 q0 自己的 dynamic expert ID 和 routing weight。
  - backward 時路由視為常數，不對 router 或 top-k 求導；activation QDQ 經 identity STE；沒有 optimizer。
  - 逐層重算時經過**所有** expert，不 detach。
- **D 的選擇性**：只省去非目標投影的 retain_grad／tap、weight sensitivity 矩陣運算、梯度統計和成本重建。輸入梯度逐位元不變（控制 6）。
- **搜尋**：
  - B、C、D 都用 `make_solver("lpfirst")`：LP → 取整 → 修復，必要時退回修正後的 MILP。
  - 各自做精確支配剪枝，所有節點都可以重新分配格式。
  - 本輪四次搜尋都由 LP 接受。

**實作上的一處簡化**（四個條件共用）
- gate 和 up 共用的 x_in，其 activation 誤差與梯度統計只算一次。原實作會算兩次。
- 數值不變，B 的表仍和原本逐位元相同，但 B 的 activation 誤差時間從上一輪的 18.3 秒降到 12.5 秒。

## 控制（`check.json`、`controls.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 1. 快取重建 static | 重建的 static 表和原表逐位元相同（相對差 0）；q0 完全重現；ΔW 和原快取逐位元相同。用快取統計加上 static 梯度、套 sensitivity-only 公式重建的 C，最大相對差 4.8e-16（門檻 1e-12），用它搜尋也得到 q0。static FP32 路由和 CE 與 stage-1 快取相同；routed rows、ranges 與歷史相同 |
| 2. 完整 refresh B | 表和 `table_refresh.json` 逐位元相同（B、C 的相對差都是 0）；配置 = q1 |
| 3. q0 forward | 9 次 refresh 的逐窗口 CE、最終 hidden state（sha256）、路由，都和獨立的 `run_olmoe_dynamic_route.forward` 以及上一輪記錄的 q0 路由逐位元相同 |
| 4. D 的目標列 | 1161 列和 C 逐位元相同 |
| 5. D 的非目標列 | 1890 列和 static 表逐位元相同 |
| 6. 輸入梯度 | 9 次 refresh 的 16 層輸入梯度 sha256 全部相同；重算和 forward 的差為 0 |
| 7. 輸入 | model 檔、ranges、q0／q1、量化器程式碼、所有輸入的 hash 前後一致 |
| 8. 有效性 | 所有統計與成本有限；格式合法；都在預算內 |
| 另加 | C 的 weight 項和 B 逐位元相同；STE 單元測試通過；所有 refresh 都沒有 fallback（q0 calibration 中 1017 個搜尋 expert 都有 token） |
| 重複 | static plain／cache 各 3 次、B／C／D 各 3 次：表與配置完全相同；activation 快取檔的 sha 相同；所有 ΔW 快取內容相同 |
| 評估 | 程式控制（上一輪的窗口 0–127）：FP32、q0、q1 的 CE 差 0；D 第一批重跑逐位元相同；四個配置都不同，各自評估；router 未被替換 |

## 計時（`analysis.json` → `timing`、`work/runs/*/run.json`）

**方式**
- 每次執行一個獨立 process；每個計時區塊前後都有 CUDA synchronization。
- 執行順序預先固定：
  - static：r0（正式快取與額外控制，不列入計時）→ plain1、cache1、cache2、plain2、plain3、cache3；
  - refresh：B1 C1 D1｜C2 D2 B2｜D3 B3 C3。
- 沒有清除系統快取。checkpoint 在每次執行前都 100% 在 page cache 中；ΔW 快取為 96–100%。
- 在會讀 activation 快取的 C、D 執行中，只有 C1 前快取駐留 4%（被之前 static 重複寫出的副本擠出），其餘 5 次都是 100%。
- 「方法時間」= 執行時間減去控制區塊。

**各類時間（中位數，秒）**

| 類別 | static | static＋快取 | B 完整 | C | D |
|---|---:|---:|---:|---:|---:|
| 載入、搬移、量化準備 | 9.8 | 10.2 | 17.0 | 17.0 | 17.0 |
| forward／路由 | 0.8 | 0.8 | 1.6 | 1.6 | 1.6 |
| 真正的 backward／重算 | 1.5 | 1.5 | 2.4 | 2.4 | 2.4 |
| weight sensitivity | 26.4 | 26.3 | 26.5 | 26.3 | 7.2 |
| activation 梯度統計 | 3.2 | 3.1 | 3.2 | 3.2 | 1.1 |
| weight 誤差 ΔW | 35.5 | 35.4 | — | — | — |
| 候選 activation 誤差 | 12.5 | 12.4 | 12.5 | — | — |
| 組表 | 0.2 | 0.2 | 0.2 | 1.0 | 0.4 |
| 剪枝＋搜尋 | 5.0 | 5.1 | 5.0 | 4.9 | 4.9 |
| 快取建立 | — | 0.8 | — | — | — |
| 快取寫入（ΔW／activation） | 1.1 | 1.4 | — | — | — |
| 快取讀取（ΔW＋activation） | — | — | 11.4 | 11.6 | 4.7 |
| 未分類 | 0.7 | 0.7 | 0.8 | 0.8 | 0.5 |
| **方法時間：中位數** | **96.8** | **98.0** | **80.6** | **69.3** | **39.9** |
| 方法時間：範圍 | 96.3–97.3 | 97.4–99.2 | 80.4–81.4 | 68.7–69.7 | 39.6–40.0 |
| process 牆鐘：中位數 | 99.9 | 101.2 | 83.4 | 72.1 | 42.3 |
| 峰值 GPU／CPU（GB） | 3.82／14.8 | 3.82／15.9 | 4.21／14.5 | 4.21／15.6 | 4.15／15.0 |

- 各類時間的中位數加總不必然等於方法時間的中位數。
- 峰值 CPU RSS 包含以 mmap 讀入的 checkpoint 頁面。
- process 牆鐘另含 Python 啟動、import 和 CUDA 初始化，約 2–3 秒。

**兩種成本**（中位數，秒）

| | A static | B 完整 refresh | C | D |
|---|---:|---:|---:|---:|
| 已有快取時的額外成本 | 0 | 80.6 | 69.3 | 39.9 |
| 從零到最終配置 | 95.7 | 177.4 | 167.3 | 137.9 |

**計算方式**
- A = static 方法時間，扣除只有 refresh 才需要的 ΔW 快取寫入 1.1 秒。
- B = static（無 activation 快取）+ B。
- C、D = static（含 activation 快取）+ C／D。

**和上一輪的單次量測比較**：上一輪 refresh 額外約 108 秒。差異有兩個來源：
- 本輪共用 x_in 的誤差統計（activation 誤差 18.3 → 12.5 秒）；
- 上一輪約 45 秒的 I/O 沒有分項計時，本輪已拆開：checkpoint 載入 6.5 秒、ΔW 讀取 11.4 秒、量化 7.1 秒、量化器初始化 2.3 秒等。

兩者條件不完全相同，不作精確比較。

## 配置與描述性指標（不用於重新選擇配置）

**格式變化**（`freeze.json`）

| | 不同的投影 | weight | activation | 在低位元集合內／外 |
|---|---:|---:|---:|---|
| B 對 A | 292 | 125 | 182 | 129／163 |
| C 對 A | 284 | 127 | 171 | 120／164 |
| D 對 A | 177 | 93 | 95 | 120／57 |
| C 對 B | 54 | 4 | 52 | |
| D 對 B | 203 | 84 | 131 | |
| D 對 C | 183 | 86 | 109 | |

- 雖然 D 只更新低位元節點的成本，搜尋仍會牽動其他節點，所以集合外也有 57 個投影改變。
- weight 的改變大多在 int4／int5／int6 之間。

**搜尋目標值**（各自的表）
- 四次搜尋都由 LP 接受；size 用量 100%，BOPs 用量 99.99%。

| 條件 | 目標值／LP 下界 | 剪枝後選項數 |
|---|---|---:|
| A | 42.2003／42.1827 | 70,153 |
| B | 43.9457／43.9402 | 70,115 |
| C | 43.9756／43.9719 | 70,127 |
| D | 43.3898／43.3806 | 70,135 |

**交叉目標值**（列：用哪張表；欄：哪個配置）

| 表 | A | B | C | D |
|---|---:|---:|---:|---:|
| static | 42.20 | 42.95 | 42.93 | 42.52 |
| B | 50.70 | 43.95 | 44.06 | 46.10 |
| C | 50.50 | 44.00 | 43.98 | 45.86 |
| D | 48.05 | 43.83 | 43.81 | 43.39 |

B 的表認為 D 比 B 差 4.9%，但實測 D 不比 B 差。proxy 數值不是 loss 的估計。

**實際路由的計算成本**（評估資料，2400 個窗口）

| | 搜尋節點 BOPs，相對全 int8 | 含未搜尋 expert，相對全 int8 | 送到未搜尋 expert 的 token-slot | 搜尋 proxy BOPs（≤ 3.92264e13） |
|---|---:|---:|---:|---:|
| A | 0.5603 | 0.5930 | 172,625 | 3.92215e13 |
| B | 0.5611 | 0.5938 | 172,519 | 3.92249e13 |
| C | 0.5609 | 0.5937 | 172,651 | 3.92254e13 |
| D | 0.5606 | 0.5933 | 172,905 | 3.92239e13 |

- FP32 有 169,906 個 token-slot 送到未搜尋的 expert。
- 四個配置的實際計算量相差不到 0.15%。
- 路由：(token, 層) 的 expert 集合和 FP32 不同的比例，A 30.4%、B 29.9%、C 30.0%、D 30.1%；B、C、D 和 A 之間約 22%。

**和歷史結果並列**（B − A = NLL(q1) − NLL(q0)；各輪文章不同，不合併）

| 輪次 | 文章數 | 平均 | 95% CI |
|---|---:|---:|---|
| proxy refresh | 128 | −0.00196 | [−0.00415, +0.00017] |
| 確認實驗 | 600 | −0.00121 | [−0.00219, −0.00024] |
| router FT 的評估 | 600 | −0.00211 | [−0.00305, −0.00116] |
| 本輪 | 600 | −0.00108 | [−0.00201, −0.00016] |

## 資料（`segments.json`、`dedup_checks.json`）

**來源**：WikiText-103 raw train，使用嚴格文章邊界。

**排除**

| 理由 | 篇數 |
|---|---:|
| 標題或內容和 WikiText-2 重疊 | 613 |
| WikiText-103 內部重複 | 464 |
| 所有歷史 OLMoE 用過的 WikiText-103 文章 | 1874 |

1874 篇的組成：proxy refresh 128、確認實驗 600、router FT 600、dynamic_prop 242、mass_restore 103、precision_protection 的 final 169 與 selection 32。

- 這是實際掃描 `runs/**/*segments*.json` 得到的，只依路徑排除本輪自己的目錄，沒有硬編碼篇數。
- 確認實驗的 600 篇已正確排除。

**抽樣**
- 在凍結（`freeze.json`）之後才抽。
- 依 seed 22401 的順序看了 636 篇：接受 600 篇，跳過 36 篇近重複（≥10% 的 5-gram）；窗口起點用 seed 22402。
- 和歷史文章的重疊為 0。

## 修訂（`amendments.json`）

1. **controls 第一次執行**：量化器需要的 `work/` 目錄還沒建立，執行在寫出任何檔案前就失敗了。建立目錄後，用原程式重跑。
2. **freeze 第一次執行**：在 `relative_to` 處失敗，因為 `--out-dir` 是相對路徑。
   - 失敗時 configs 和 tables 已經寫出，但 freeze.json 還沒有。
   - 加上 `.resolve()` 後重跑；重寫的 configs 和 tables 逐位元相同。
   - 腳本 sha：plan 到 check 是 `73753b69…`，freeze 之後是 `2b6508f6…`，只改了這一行。
3. **事後新增 `posthoc_multiplicity.json`**（Bonferroni 檢查）：不參與任何判定。

三項都發生在抽評估資料之前，或屬於事後的描述；沒有修改方法、門檻或主要終點。

## 研究成本（不屬於方法成本）

| 項目 | 時間 |
|---|---|
| plan | 32.6 秒 |
| controls | 14.1 秒 |
| 正式快取 r0 | 102.4 秒 |
| static 計時 6 次 | 607.6 秒 |
| refresh 計時 9 次 | 599.2 秒 |
| check | 178.3 秒（含 7 份 ΔW 快取的內容比對） |
| 抽資料 | 87.8 秒 |
| 評估 | 1277.9 秒（FP32 123.4 秒、A／B／C／D 各約 267.5 秒、程式控制約 60 秒、重跑 20.7 秒；峰值 GPU 7.7 GB） |
| 分析 | 16.2 秒 |

**磁碟**：`work/` 共 6.7 GB，大多是計時重複所寫出的快取副本，內容已確認和正式版本相同；其餘檔案約 0.5 GB。

## 下一步建議（未執行）

1. **在第二個資源點（例如 knee）用同一流程重複 A／B／D**，預先登記，600 篇新文章。
2. **用 native kernel 驗證 A 和 D 的差**：0.0017 和已知的跨後端實作差異同一量級。
3. **若要了解 D 為何優於 C**：預先登記一個「只更新非低位元節點」的互補條件。不要事後調整 4-bit 門檻。
4. **降低工程成本**（不改變結果）：
   - ΔW 快取改成單一連續陣列：讀取目前要 11 秒，主要來自 npz 的 3 萬多個成員；
   - weight sensitivity 的 FP64 矩陣運算；
   - checkpoint 載入與 q0 量化：每次 refresh 固定約 14 秒。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_selective_sensitivity_20261022      # 新實驗請改用新目錄
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_selective_sensitivity.py
$CPU $S --stage plan --out-dir $O && mkdir -p $O/work
$GPU $S --stage controls --out-dir $O
$GPU $S --stage static --variant cache --rep 0 --out-dir $O    # 正式快取與控制 1，約 2 分鐘
$GPU $S --stage timing --phase static --out-dir $O             # 6 次，約 10 分鐘
$GPU $S --stage timing --phase refresh --out-dir $O            # 9 次，約 10 分鐘
$CPU $S --stage check --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage eval --out-dir $O                              # 約 21 分鐘
$CPU $S --stage analyze --out-dir $O
```

**環境**
- GPU：torch 2.14.0+cu130，RTX 3060，TF32 關閉。
- CPU：scipy 1.18.1（HiGHS）。
- nvidia-smi 的 NVML 版本不一致不影響 CUDA，系統驅動未改動。

## 檔案

- **預先登記與修訂**：`plan.json`、`plan.md`、`lowbit_nodes.json`、`amendments.json`、`input_hashes_before.json`。
- **快取**：`work/runs/static_cache_r0/act_err_cache.npy`（1.17 GB）、`act_err_cache_index.json`（快取規格：列順序、n_s、欄位、格式、統計量）。
- **各次執行**：`work/runs/{static_plain,static_cache,refresh_B,refresh_C,refresh_D}_r*/`（`run.json`、`table.json`、`config.json`）。
- **凍結**：`configs/{A,B,C,D}.json`、`tables/{static,B,C,D}.json`、`freeze.json`。
- **控制**：`controls.json`、`check.json`、`controls_summary.json`。
- **資料**：`segments.json`、`tokens.npy`、`dedup_checks.json`、`data_freeze.json`。
- **結果**：`eval.json`、`routes_{fp32,A,B,C,D}.npz`、`per_article.csv`、`per_window.csv`、`analysis.json`、`posthoc_multiplicity.json`。
- **時間**：`timing.json`、`timing.jsonl`；`logs/`。
