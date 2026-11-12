# OLMoE 正常 knee：activation error-only refresh（E）

日期：2026-10-22
腳本：`run_olmoe_knee_error_only.py`（plan → static（建梯度快取）→ controls → refresh ×9 → 端到端 E → check → freeze → data → eval → analyze）
輸出：本目錄

**本輪的範圍**
- A、B、C 沿用已凍結的配置；D4、D5 只用於交叉評分的診斷。
- 新增 E：保留原始 FP32 BC 的 sensitivity，只在原始 q0_knee 工作狀態下更新 activation 量化誤差；在同一組精確上限下重新搜尋一次。
- 沒有重選 knee；沒有訓練或更新參數；沒有修改 BC 公式、ranges 或 scale；沒有新的搜尋演算法；沒有位元門檻實驗；沒有修改正式 pipeline 預設。
- 結論只適用於本模型、本 calibration、本 knee 和 GPU QDQ。

## 結論

**E 沒有改善 static BC**：E − A = +0.00048，點估計反而是 E 較差；校正 CI [−0.00039, +0.00135]。

**E 也沒有保留完整 refresh 的品質**：E − B = +0.00105，校正 CI [+0.00017, +0.00195]。非劣沒有成立，而且校正區間的下界大於 0：在這批文章上，E 比 B 差。

**E 也沒有確認優於 sensitivity-only 的 C**：E − C = +0.00063，CI 跨 0。

**省時很明顯**：E 的額外時間只有 25.1 秒（B 80.2 秒、C 69.2 秒），因為它完全不需要 backward。但本輪沒有可以讓它保留的品質收益。

| 條件 | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| FP32 | 2.95625 | 19.226 | — |
| A：static knee | 2.97220 | 19.535 | +0.01595 [+0.01468, +0.01721] |
| B：全節點 sensitivity＋activation error | 2.97163 | 19.524 | +0.01538 [+0.01405, +0.01662] |
| C：全節點 sensitivity | 2.97204 | 19.532 | +0.01579 [+0.01462, +0.01697] |
| **E：全節點 activation error** | **2.97268** | **19.544** | **+0.01643 [+0.01521, +0.01761]** |

600 篇新的 WikiText-103 文章 × 4 個窗口；全部用原 router、各自的 dynamic route。

**預先登記的三個主要比較**（以 Bonferroni 98.33% CI 判定）

| 對比（nat/token） | 平均 | 95% CI | **98.33% CI（校正）** | 較好／較差篇數 | 判定 |
|---|---:|---|---|---|---|
| E − A | +0.00048 | [−0.00022, +0.00119] | **[−0.00039, +0.00135]** | 289／311 | 未確認較好 |
| E − B | +0.00105 | [+0.00033, +0.00178] | **[+0.00017, +0.00195]** | 281／319 | 非劣證據不足（上界 > +0.0005） |
| E − C | +0.00063 | [−0.00008, +0.00136] | **[−0.00023, +0.00150]** | 277／323 | 未確認較好 |

**次要比較**（95% CI）

| 對比 | 平均 | 95% CI | 較好／較差篇數 | 方向 |
|---|---:|---|---|---|
| B − A | −0.00058 | [−0.00134, +0.00018] | 318／282 | 未確認 |
| C − A | −0.00016 | [−0.00091, +0.00060] | 306／294 | 未確認 |

10,000 次文章級 paired bootstrap，seed 22811；各差值的文章級 SD 為 0.0087–0.0093。

## 回答五個問題

**1. E 是否改善原始 BC？**
- **沒有。** 點估計 +0.00048，方向是變差，但未確認。
- 校正區間能排除的：大於 0.00039 nat/token 的改善。
- 不能排除的：沒有效果、極小的改善（≤0.0004），或最多約 0.0013 的變差。

**2. E 是否保留 B 的品質？**
- **沒有。** 非劣沒有成立（校正上界 +0.00195 > +0.0005）。
- 校正區間的下界 +0.00017 大於 0，所以在這批文章上 E 比 B 差。差距是否超過 0.0005，沒有確認（下界沒有超過 +0.0005）。
- 本批的 B − A 本身也沒有確認（−0.00058，95% CI 跨 0），所以 E 和 A 之間「要保留的收益」本身就不確定。

**3. 相較 B／C，實際省下多少時間？**
- **已有快取時的額外時間**（3 次中位數）：

  | 條件 | 秒 | E 少多少 |
  |---|---:|---|
  | E | 25.1 | — |
  | B | 80.2 | 55.1 秒（−69%） |
  | C | 69.2 | 44.1 秒（−64%） |

- **E 省下的工作**：
  - 整段 backward 與重算（2.4 秒）；
  - weight sensitivity（26.3 秒）；
  - activation 梯度統計（3.2 秒）；
  - ΔW 快取讀取（11.3 秒）：weight 表整張沿用 A；
  - 第二次 checkpoint 載入與量化（約 6.7 秒）：只需要一次 forward。
- **E 仍要付的**：
  - 載入與量化準備 10.2 秒；
  - forward 1.6 秒；
  - 候選 activation 誤差 7.3 秒（B 為 12.4 秒，E 的組表較有效率）；
  - 剪枝與搜尋 4.8 秒。
- **從零到最終配置**：
  - E 單次實測 192.1 秒；
  - 歷史單次實測：A 170.5、B 246.6、C 240.3 秒。
  - 換言之，E 比 A 多約 22 秒（+13%），比 B 少約 55 秒。
- **梯度快取的成本**：建立與寫入只多 0.26 秒，檔案 117 MB；讀取 0.07 秒。

**4. 結果是否支持「只更新 activation error」這個方向？**
- **本輪的資料不支持。** E 在單獨使用時沒有收益，點估計還是變差的方向。
- 依指示，不能據此宣稱 error 更新無用：
  - 本批的 B − A 也沒有確認；
  - 只憑「B 顯著、E 不顯著」，也不能推論必須兩者同時更新。
- **[描述，非因果證據] 配置面的觀察**：
  - E 對 A 只改了 37 個投影：8 個 weight、34 個 activation 格式。
  - C 對 A 改了 198 個；B 對 A 改了 201 個；B 和 C 之間差 34 個。
  - 換句話說，不論單獨使用（E 對 A）或疊加在新 sensitivity 之上（B 對 C），activation error 的刷新都只牽動約 35 個投影；大部分的格式重新分配來自 sensitivity 的更新。
  - 這些改動對 loss 的影響，本批 600 篇解析不出來。
- **[推論] 四批 knee 資料的整體圖像**（各批單獨配對，不合併、不跨批相減）：
  - B − A 四批都是負的點估計（−0.0012、−0.0005、−0.0008、−0.0006），其中兩批在 95% 水準確認；
  - 單獨更新 sensitivity（C，三批）或單獨更新 error（E，一批），都沒有確認。
  - 這組資料無法分辨 B 的小幅收益來自哪個成分，或來自兩者的交互作用。

**5. 是否值得保留為可選步驟？**
- **不值得。** E 很便宜（額外 25 秒、從零多 13%），但本輪沒有品質收益，而且確認比 B 差。
- 依指示，E 沒有收益，所以不提出「選擇性 error 更新」的下一輪假說。
- 若仍要追究 B 的收益來源，需要以事先計算的樣本量設計的 A／B／C／E 因子實驗。依本輪觀察到的文章級 SD（約 0.009），要讓校正 CI 半寬達到約 0.0003，約需 5,000 篇文章；或先用 native kernel 驗證 B − A。在那之前，**建議維持 static BC 預設**。

## E 的定義與實作（`plan.json`）

**成本**
- **weight**：W 和候選量化器固定，ΔW(f) = Q_f(W) − W 不變；E 的 weight 表整張逐位元沿用 A（控制 3）。
- **activation**：A_E(f) = Σ_slots,c [G2_static,c × var_new,c(f) + GS_static,c × μ_new,c(f)²]。
  - err_new(f) = Q_f(a_q0) − a_q0，在和 B 相同的位置取得：
    - x 槽：gate／up 共用的 QDQ 前 expert 輸入，down 用 QDQ 前的 SiLU(g)·u；
    - y 槽：QDQ 前的 linear 輸出；
    - 使用固定的 ranges。
  - μ_new = Σerr／n_q0，var_new = Σerr²／n_q0 − μ_new²；n_q0 是 q0 實際的 routed rows。
  - G2／GS 維持 static 的尺度，以 BC 的運算順序 G2／n_q0·(S2 − n_q0·μ²) + GS·μ² 計算。
- **範圍**：全部 3051 個投影、全部候選格式；q0 中 1017 個 expert 都有 token，fallback 為 0。

**static 梯度統計快取**（`work/runs/static_gc/`）
- 本輪之前沒有可驗證的 G2_static／GS_static 快取，所以在一次 static 重建中補存。重建出的 static 表和 A 逐位元相同。
- **內容**：每個 expert（1017）× [G2, GS] × 7168 個 channel。channel 分成 5 段：x_in 2048、gate_y 1024、up_y 1024、down_x 1024、down_y 2048。
  - 型別 float64，形狀 (1017, 2, 7168)，116.6 MB。
  - G2 = 對 static routed rows 的 Σg²；GS = Σ_窗口 (窗口內 Σg)²。
  - 梯度來自 summed CE、FP32 reference、固定路由；x_in 使用 gate／up 共用輸入的 total gradient。
  - index 記錄 static rows、欄位位置與來源 hash。
- **成本**：這次 static 建表單次 90.2 秒，其中梯度快取的收集 0.24 秒、寫入 0.03 秒。只算給 E。

**forward-only 更新**
- 整段在 `torch.inference_mode()` 中執行，使用新寫的觀測路徑 `experts_observe`：和既有 forward 相同的運算順序，只保存 QDQ 前張量的參考，沒有 `retain_grad`。
- 以 `AutogradGuard` 計數 `torch.autograd.backward`、`torch.autograd.grad`、`Tensor.backward` 的呼叫。E 的 3 次執行和端到端執行，呼叫次數都是 0，執行中梯度都是關閉的。
- 候選格式只用來計算誤差；forward 本身維持 q0 配置，路由與 hidden state 和獨立 forward 逐位元相同。
- 原始 static BC 的建立仍需要 backward；只有 E 的「更新」這一段不需要。

## 控制（`check.json`、`controls.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 1. 輸入 | A／B／C／D4／D5 的配置與表、knee、三輪的 check，以及 20 項重用依賴的 hash，全部一致 |
| 2. static 重建 | 建梯度快取的 static 重建，和 A 表逐位元相同；static rows 和既有誤差快取一致；用 static 梯度快取加 static 誤差快取，以 numpy 重建 A 的 activation 成本，最大相對差 6.9e-16 |
| 3. weight 列 | E 的 weight 列和 A 逐位元相同 |
| 4. forward | E 的 3 次執行和端到端 E，CE 與 hidden state 都和獨立的 `dr.forward` 逐位元相同；3 次執行的路由也相同。本輪的 B、C 執行，輸入梯度和歷史記錄相同 |
| 5. 獨立重算 | 用 static 梯度快取和 E 第 1 次存下的新誤差統計，以 numpy 重算全部 3051 列：最大相對差 5.0e-16（門檻 1e-12）。若改用 static rows 正規化，差異達 100%，表示檢查分辨得出錯誤的正規化 |
| 6. 重現 B | 用 C 存下的 q0 梯度統計，加上本輪 E 的新誤差，重現了 B 的 activation 成本：最大相對差 7.4e-16；兩者的 q0 rows 也逐 expert 相同。這證實 E 的觀測位置和誤差計算與 B 一致。此項不計入 E 的方法時間 |
| 7. 無梯度計算 | autograd 呼叫 0 次、梯度關閉；E 的 backward、weight sensitivity、梯度統計、ΔW 讀取的計時都是 0 |
| 8. 有效性 | fallback 0；有限；恰好涵蓋 3051 個投影；格式合法；在精確上限內 |
| 9. 重複 | E 3 次的表與配置相同（主要配置 = 第 1 次）；本輪的 B、C 各 3 次和凍結版本逐位元相同；端到端 E 重現了 static 表、梯度快取、knee，以及 E 的表和配置 |
| 評估 | 程式控制（D5 輪的窗口 0–127）：FP32、A、B、C 的 CE 差 0；E 第一批重跑逐位元相同；router 未被替換 |

## 搜尋與近似解檢查

**E 的搜尋**
- 目標值 18.311804，LP 下界 18.308548；絕對差距 0.00326，相對 1.8e-4；由 LP 接受。
- 剪枝後 70,133 個選項；size 用量 97.68%，BOPs 用量 99.996%。

**各配置在 E 表上的分數**（都在精確上限內）

| 配置 | A | B | C | D4 | D5 | **E** |
|---|---:|---:|---:|---:|---:|---:|
| E 表上的目標值 | 18.33147 | 18.33673 | 18.36471 | 18.32810 | 18.35225 | **18.31180** |

- E 自己的搜尋解在 E 表上分數最低，**沒有出現**像 D5 那一輪「其他配置的成本反而比搜尋解低」的情況。所以本輪的品質結果，沒有受到近似搜尋失誤的干擾。
- 交叉評分只是診斷：不同成本表的目標值不能直接比較，也不代表品質。

## 配置（`freeze.json`、`analysis.json`）

**精確上限**：weight bytes ≤ 4,524,359,631.556，BOPs ≤ 46,648,327,679,632.125。

| | weight bytes | BOPs | R |
|---|---:|---:|---:|
| A | 4,418,699,264 | 46,645,278,408,704 | 0.7070606 |
| B | 4,420,796,416 | 46,647,774,019,584 | 0.7070984 |
| C | 4,421,058,560 | 46,648,180,867,072 | 0.7071046 |
| E | 4,419,223,552 | 46,646,519,922,688 | 0.7070794 |

**格式差異**

| | 不同的投影 | weight | activation |
|---|---:|---:|---:|
| E 對 A | 37 | 8 | 34 |
| E 對 B | 188 | 110 | 83 |
| E 對 C | 211 | 113 | 105 |
| B 對 A | 201 | 114 | 93 |
| C 對 A | 198 | 113 | 89 |

- 依原本的低位元分區，E 對 A 的改動：≤4-bit 集合 11 個、新增的 5-bit 集合 18 個、其餘 8 個。
- E 的 weight 改變：int5→int6 3、int6→int5 3、int4→int5 2。
- E 的 activation 改變：主要是 p6e1→p8e1 6、int6→p6e1 4、int8→p8e1 4，以及 p8e1→int8／p8e1→p6e1／int7→p6e1／p6e1→int6 各 3。

**實際路由的計算成本**（評估資料）

| | 搜尋節點 BOPs，相對全 int8 | 含未搜尋 expert，相對全 int8 | 送到未搜尋 expert 的 token-slot |
|---|---:|---:|---:|
| A | 0.67795 | 0.70777 | 159,996 |
| B | 0.67801 | 0.70783 | 159,974 |
| C | 0.67798 | 0.70780 | 159,988 |
| E | 0.67803 | 0.70784 | 159,912 |

FP32 有 157,915 個 token-slot 送到未搜尋的 expert。B／C／E 和 A 的 expert 集合不同的比例分別是 16.8%、16.8%、14.8%。

**四批 knee 資料並列**（各批不同的文章，各自配對；不合併、不跨批相減；95% CI）

| 對比 | 第 1 批（selective） | 第 2 批（all-node） | 第 3 批（≤5-bit） | 第 4 批（本輪） |
|---|---|---|---|---|
| B − A | −0.00120 [−0.00184, −0.00057] | −0.00049 [−0.00117, +0.00022] | −0.00084 [−0.00158, −0.00011] | −0.00058 [−0.00134, +0.00018] |
| C − A | — | −0.00029 [−0.00096, +0.00039] | −0.00039 [−0.00112, +0.00031] | −0.00016 [−0.00091, +0.00060] |
| E − A | — | — | — | +0.00048 [−0.00022, +0.00119] |

## 計時（`analysis.json` → `timing`）

**方式**
- B／C／E 各 3 次，交錯執行：B1 C1 E1｜C2 E2 B2｜E3 B3 C3。
- 每次一個 process，前後都有 CUDA synchronization，沒有清除系統快取。
- 每次執行前，checkpoint、ΔW 快取、activation 快取、梯度快取都 100% 在 page cache 中。
- 控制區塊不計入方法時間。

**已有快取與 q0_knee 時的額外更新＋搜尋**（中位數，3 次；秒）

| 類別 | B | C | **E** |
|---|---:|---:|---:|
| 準備與載入（checkpoint、q0 量化、量化器初始化、設定） | 17.0（6.5＋7.1＋2.3＋1.1） | 17.3（6.8＋7.1＋2.3＋1.1） | **10.2（3.3＋3.5＋2.3＋1.0）** |
| forward | 1.6 | 1.6 | 1.6 |
| backward | 2.4 | 2.4 | **0** |
| weight sensitivity | 26.3 | 26.2 | **0** |
| activation 梯度統計 | 3.2 | 3.2 | **0** |
| activation 誤差計算 | 12.4 | — | 7.3 |
| static sensitivity 快取讀取（梯度快取） | — | — | 0.07 |
| static 誤差快取讀取（ΔW＋activation） | 11.3 | 11.6 | — |
| 組表 | 0.2 | 1.0 | 0.6 |
| 剪枝與搜尋 | 4.9 | 4.9 | 4.8 |
| 輸出、未分類 | 0.9 | 0.9 | 0.6 |
| **方法時間：中位數（範圍）** | **80.2（79.7–80.5）** | **69.2（68.8–69.3）** | **25.1（25.1–25.2）** |
| process 牆鐘：中位數 | 83.0 | 72.1 | 27.1 |
| 峰值 GPU／CPU（GB） | 4.21／14.5 | 4.21／15.6 | 3.95／15.4 |

**快取**
- 梯度快取 117 MB（只有 E 需要）；activation 誤差快取 1.17 GB（只有 C 需要）；ΔW 快取 0.34 GB（B、C 需要）。

**從零到最終配置**

| | 秒 | 性質 |
|---|---:|---|
| **E：static BC＋梯度快取 → auto knee → E → 搜尋** | **192.1（process 194.8）** | 本輪單次實測 |
| E：各階段相加 | 196.1 | 估算：static 90.7（selective sensitivity 輪的中位數，不含快取）＋梯度快取 0.26（本輪單次）＋auto 80.0（歷史中位數）＋E 25.1 |
| A | 170.5 | 歷史值（單次實測） |
| B | 246.6 | 歷史值（單次實測） |
| C | 240.3 | 歷史值（單次實測） |
| D4 | 196.1 | 歷史值（單次實測） |
| D5 | 215.4 | 歷史值（單次實測） |

- 端到端 E 的分界時間：static 統計完成於 91.1 秒，auto knee 完成於 171.7 秒，其後 E 更新與搜尋約 20 秒。
- 峰值 GPU／CPU 為 3.96／14.9 GB。

## 資料（`segments.json`、`dedup_checks.json`）

**來源**：WikiText-103 raw train，使用嚴格文章邊界。

**排除**

| 理由 | 篇數 |
|---|---:|
| 標題或內容和 WikiText-2 重疊 | 613 |
| WikiText-103 內部重複 | 464 |
| 所有歷史 OLMoE 用過的 WikiText-103 文章 | 4274 |

- 4274 篇包含 D5 輪的 600 篇。這是實際掃描所有 manifest 得到的，只依路徑排除本輪自己的目錄。
- calibration 屬於 WikiText-2，已隨 WikiText-2 一起排除。

**抽樣**
- 在凍結之後才抽。
- 依 seed 22801 的順序看了 666 篇：接受 600 篇；跳過 63 篇近重複、3 篇太短。窗口起點用 seed 22802。
- 和歷史文章的重疊為 0。

## 修訂（`amendments.json`）

1. **check 第一次執行在最後一行 log 中止**：
   - `check.json` 已完整寫入（pass 為 true），但印 log 時，用 `json.dumps` 處理了 numpy bool。`dr.write` 會先轉型，log 卻沒有。
   - 修正：check 和 analyze 的 log 也先轉型。原輸出另存為 `check_logcrash_20261022T1852.json`，再用修正後的程式重跑 check；兩次結果完全相同。
   - 腳本 sha：plan 到端到端是 `cd90dada…`，check（重跑）之後是 `44c08d4f…`。
   - 這發生在凍結與抽資料之前，不影響任何表、配置、控制或計時。

## 研究成本（不屬於方法成本）

| 項目 | 時間 |
|---|---|
| plan | 43.8 秒 |
| static（建梯度快取，即控制 2 的來源） | 92.9 秒 |
| controls | 24.7 秒 |
| refresh 計時 9 次 | 552.4 秒 |
| 端到端 E | 195.5 秒 |
| check（重跑） | 29.9 秒 |
| 抽資料 | 147.0 秒 |
| 評估 | 1300.8 秒（FP32 123.2 秒、A／B／C／E 各約 268 秒、程式控制約 85 秒、重跑約 21 秒；峰值 GPU 7.7 GB） |
| 分析 | 17.0 秒 |

**磁碟**：`work/` 1.4 GB，主要是 E 第 1 次存下的新誤差統計（1.17 GB，控制 5、6 使用）。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_knee_error_only_20261022      # 新實驗請改用新目錄
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_knee_error_only.py
$CPU $S --stage plan --out-dir $O
$GPU $S --stage static --out-dir $O                    # static 重建＋梯度快取，約 1.5 分鐘
$GPU $S --stage controls --out-dir $O
$GPU $S --stage timing --phase refresh --out-dir $O    # 9 次，約 9 分鐘
$GPU $S --stage timing --phase e2e --out-dir $O        # 約 3 分鐘
$CPU $S --stage check --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage eval --out-dir $O                      # 約 22 分鐘
$CPU $S --stage analyze --out-dir $O
```

**環境**
- GPU：torch 2.14.0+cu130，RTX 3060，TF32 關閉。
- 搜尋：scipy 1.18.1（HiGHS）。
- nvidia-smi 的 NVML 版本不一致不影響 CUDA，系統驅動未改動。

## 檔案

- **預先登記與修訂**：`plan.json`、`plan.md`、`amendments.json`、`input_hashes_before.json`。
- **梯度快取**：`work/runs/static_gc/grad_stats_cache.npy`、`grad_stats_cache_index.json`（欄位、shape、dtype、static rows、來源 hash）。
- **各次執行**：`work/runs/{refresh_{B,C,E}_r*,e2e_E}/`。`refresh_E_r1/new_error_stats.npy` 和 `new_error_stats_index.json` 用於控制 5、6。
- **凍結**：`configs/{A,B,C,D4,D5,E}.json`、`tables/E.json`、`freeze.json`。
- **控制**：`controls.json`、`check.json`、`check_logcrash_20261022T1852.json`、`controls_summary.json`。
- **資料**：`segments.json`、`tokens.npy`、`dedup_checks.json`、`data_freeze.json`。
- **結果**：`eval.json`、`routes_{fp32,A,B,C,E}.npz`、`per_article.csv`、`per_window.csv`、`analysis.json`。
- **時間**：`timing.json`、`timing.jsonl`。
