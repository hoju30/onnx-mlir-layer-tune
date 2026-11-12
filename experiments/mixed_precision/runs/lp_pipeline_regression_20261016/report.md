# 跨架構 LP 搜尋 pipeline 回歸驗證

日期：2026-10-16

- **腳本**：`experiments/mixed_precision/run_lp_pipeline_regression.py`（實驗 A、B）、`experiments/mixed_precision/run_lp_pipeline_eval.py`（實驗 C）
- **輸出**：本目錄的 `A/`、`B/`、`C/`、`summary.json`
- **預先登記**：`plan.json` 在 19:51 寫入，早於任何求解；`plan_budgets.json` 是 A 的全部預算。amendment 只有 1 筆，在 21:41 寫入，內容是實驗 C 的具體設定，發生在讀取任何 C 資料之前。

**沒有修改**：BC criterion、BC 表、calibration、range、候選格式、資源模型、`mp_milp`／`mp_auto`、正式 pipeline 的預設與既有輸出。
**沒有用評估結果**決定任何規則、門檻、預算、knee 或配置。只做 fixed route，沒有做 dynamic route。

標記說明：**[量測]** 實際量到；**[推論]** 由證據推得；**[未驗證]** 本輪沒有資料。


## 更正與後續（2026-10-16）

以下是報告完成後，由後續實驗發現的更正。原文保留不改，閱讀時請以本節為準。

- 原文說「沒有修改 `mp_milp`／`mp_auto`」，已過時：本報告完成後，經使用者同意，`mp_milp` 加入了最小修正（重試失敗時才修復），共用的 `int_act_qdq` 也加入了裝置修正（`runs/fix_integration_20261016`）。
- 已驗證：8 個 dense 問題的 auto_select 結果和本報告完全相同；本報告的評估本來就已套用量化器修正。

## 結論：逐題回答

**1. 能否採用跨架構共用的 LP 優先 pipeline？——不能，也沒有必要。**
- [量測] 8 個 dense 問題上，MILP 的整個固定預算掃描只要 2–34 秒。LP 優先在 23–29 個預算中，只直接接受 2–11 個 LP 解，其餘都退回 MILP：
  - CNN、GPT-2、BERT、Qwen3.5 FFN：時間和 MILP 相同或略多；
  - Qwen3、Qwen3.5 FFN+attn：約少 20%。
- [量測] 只有 OLMoE 真正受益：MILP 2776 秒，LP 優先 75 秒（約 37 倍），21 個預算接受 19 個 LP 解。

**2. 哪些模型適用、哪些保留 MILP？**
- **保留 MILP**：MobileNetV2、ResNet18、EfficientNet-B0、GPT-2、BERT、Qwen3、Qwen3.5 FFN-only、Qwen3.5 FFN+attn。
  - 理由：LP 優先沒有可觀的時間效益。
  - 在 Qwen3、Qwen3.5 FFN+attn 上，搭配原 knee 規則時，LP 優先選出的配置實測**明顯較差**（實驗 C）。
- **使用 LP 優先**：OLMoE fixed route。在兩種 knee 規則下，它選出的配置都和先前在 512 個新窗口上驗證過「品質等價」的配置相同（雜湊相同）。

**3. 是否值得自動選擇求解器？——目前只能提出假說，證據不足以固定規則。**
- [推論] 可以在求解前取得的特徵中，問題大小（選項數或節點數）能把 9 個問題分成兩群：
  - dense：≤ 23,716 個選項、≤ 196 個節點，每次 MILP 求解 ≤ 數秒；
  - OLMoE：70,153 個選項、3051 個節點，每次 MILP 100–250 秒，其中 95% 是 presolve。
- 像「選項數 > 5 萬就用 LP 優先」的規則能分對這 9 個問題，但大型問題只有 OLMoE 一個，門檻落在 2.4 萬到 7 萬之間的哪裡都無法辨識。
- [未驗證] 規則本身沒有被獨立驗證。

**4. knee 規則修改（近重複點分組）是否有足夠證據採用？——可以作為 LP 優先的研究選項，不足以改成預設。**
- **好處** [量測]：
  - 在原本 MILP 和 LP 優先分歧的兩個模型上，兩者選出的配置差異大幅縮小：Qwen3 從 37 個投影降到 3 個，Qwen3.5 FFN+attn 從 31 個降到 8 個。
  - 新窗口上，新規則下 LP 優先的配置相對 MILP **非劣且等價**；原規則下則**明顯較差**。
  - MILP 的 knee 在 9 個問題中有 7 個不變；求解次數不變。
- **代價** [量測]：新規則改變了 ResNet18 和 EfficientNet-B0 的 knee（兩種求解器都改變）。
  - ResNet18：新 knee 的 knee 分數和原 knee 相同（差 < 1e-6）。
  - EfficientNet-B0：新 knee 的分數低 1.3e-4。
  - 兩者在 1000 張新圖片上的品質差異都**無法判定**：CI 包含 0，但寬度遠超過容許差距，檢定力約 5%（事先已登記）。
- **離線穩健性** [量測]：
  - 打亂候選順序、加入完全重複點：兩種規則都完全不變。
  - 只擾動 BC，ε ≤ 1e-6：knee 和細化區間都不變；唯一例外是 EfficientNet 新規則那次，ε = 1e-6 時有 9% 的抽樣 knee 改變。
- [推論] 這個規則確實能減少「求解器的小差異被 adaptive search 放大」。但它也會在近重複點附近，把單側的細取樣換成兩側的粗取樣，進而改變 CNN 的選點，而這部分的品質證據不足。

**5. 是否足以固定搜尋流程，進入 MoE dynamic-route？**
- [推論] fixed-route OLMoE 的搜尋端可以固定為研究流程：LP 優先，搭配哪一種 knee 規則都會得到相同的配置。可以拿這個流程進入 dynamic-route 研究。
- 但它不是預設，而且以下兩件事尚未處理：
  - knee 分數非常平；
  - 「最低預算端點精確平手」的脆弱性（見實驗 B）。

**6. 尚未完成的驗證**：見最後一節。

## 逐模型總表

**時間**：實驗 A 的固定預算掃描（17 個粗掃描預算加上診斷預算），3 次重複的中位數，範圍見下方說明。
**knee**：實驗 B 的結果，格式為 R／BC／M／B（M、B 是相對全 int8 的比例）。「差」是相對「原規則＋MILP」不同的投影數。

| 模型 | 節點／選項 | MILP／LP 優先時間（s） | LP 接受 | 原規則＋MILP knee | 原規則＋LP 優先 | 新規則＋MILP | 新規則＋LP 優先 | 實際品質（C） | 建議 |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 53／6,413 | 4.63／4.65 | 3/23 | 0.7303／0.758／0.711／0.730 | 相同 | 相同 | 相同 | 配置相同，不需評估 | MILP |
| ResNet18 | 21／2,541 | 2.22／2.27 | 3/23 | 0.7912／0.304／0.777／0.791 | 相同 | 0.7698／0.334（差 5） | 同新規則 MILP | 新規則 − 原規則：loss −0.0035 [−0.0122, +0.0051]，無法判定 | MILP、原規則 |
| EfficientNet-B0 | 82／9,922 | 13.08／12.32 | 7/23 | 0.7505／0.356／0.751／0.751 | 相同 | 0.7711／0.327（差 17） | 同新規則 MILP | loss +0.0068 [−0.0028, +0.0165]，無法判定 | MILP、原規則 |
| GPT-2 | 48／5,808 | 3.33／3.46 | 3/23 | 1.1884／21.64／0.922／1.188 | 相同 | 相同 | 相同 | 配置相同 | MILP |
| BERT | 74／8,954 | 5.97／6.22 | 2/23 | 1.3987／2.03e-6／0.885／1.399 | 相同 | 相同 | 相同 | 配置相同 | MILP |
| Qwen3 | 196／23,716 | 32.38／26.47 | 11/29 | 0.8780／330.5／0.760／0.878 | 0.8406／392.3（差 37） | 相同 | 0.8778／330.7（差 3） | 原規則 LP 優先：**+0.0111 [+0.0060, +0.0161]，較差**；新規則：−0.0001 [−0.0033, +0.0032]，**等價** | MILP（LP 優先需搭配新規則） |
| Qwen3.5 FFN | 72／8,712 | 5.33／5.63 | 6/23 | 0.8767／35.44／0.826／0.877 | 相同 | 相同 | 相同 | 配置相同 | MILP |
| Qwen3.5 FFN+attn | 186／22,506 | 33.49／26.16 | 10/29 | 0.8409／81.58／0.799／0.841 | 0.8228／88.30（差 31） | 相同 | 0.8408／81.60（差 8） | 原規則：**+0.0037 [+0.0019, +0.0054]，較差**；新規則：−0.0014 [−0.0030, +0.0002]，**等價** | MILP（LP 優先需搭配新規則） |
| OLMoE fixed route | 3051／70,153 | 2775.6／74.9 | 19/21 | 0.7071／18.292／0.691／0.707 | 0.70706／18.297（差 4） | 相同 | 同原規則 LP 優先 | 配置和上一輪相同：+0.00062 [−0.0003, +0.0015]，**等價**（引用，非新評估） | **LP 優先** |

表格說明：
- 時間範圍：MILP 的 3 次重複相差 ≤ 0.4%；OLMoE 的 MILP 只完整重複 1 次，另外在 3 個預算上重複了 3 次，相差 ≤ 0.6%。
- 「無法判定」：CI 包含 0，但寬度遠超過容許差距（CNN 的 loss 容許差距約 0.002）。
- 品質欄的數值都是「替代配置 − 參考配置」的 nat/token（LM）或每張圖片的 loss（CNN）。LM 用 97.5% CI（Bonferroni，k = 2），CNN 和 OLMoE 用 95% CI。
- 可行性與數值警告：
  - 9 個問題的所有求解都在精確預算內，沒有失敗。
  - 數值警告見實驗 A。

## 實驗 A：固定預算比較求解器（`A/`）

**設定**：
- 每個模型都用成本表推導出的 17 個粗掃描預算，和前幾輪逐一相同；另加上 plan 中列出的診斷預算，也就是前幾輪 refinement 用過的預算：dense 6–12 個，OLMoE 4 個。
- 方法依序執行，每次是全新的程序，不使用任何快取。

| 模型 | 純 LP 時間（s） | 純 LP 相對 MILP 的 BC 損失（中位數／最大） | LP 優先退回原因 | 可疑下界（MILP／LP／LP 優先） |
|---|---:|---|---|---|
| MobileNetV2 | 0.56 | 0.99%／8.7% | 取整 20 | 0／0／0 |
| ResNet18 | 0.22 | 2.2%／19.7% | 取整 20 | 0／0／0 |
| EfficientNet-B0 | 1.00 | 1.9%／18.8% | 取整 15、LP 下界可疑 1 | 1／2／1 |
| GPT-2 | 0.48 | 1.7%／6.5% | 取整 19、取整＋修復 1 | 0／0／0 |
| BERT | 0.74 | 0.54%／7.5% | 取整 20、取整＋修復 1 | 0／0／0 |
| Qwen3 | 5.17 | 0.056%／90%（目標值極小） | 取整 15、取整＋修復 3 | 1／3／3 |
| Qwen3.5 FFN | 1.41 | 0%／2.7% | 取整 15、取整＋修復 2 | 0／0／0 |
| Qwen3.5 FFN+attn | 5.01 | 0.19%／180%（目標值極小） | 取整 18、取整＋修復 1 | 2／1／1 |
| OLMoE | 12.87 | 0.007%／0.50% | 取整 2 | 1／0／0 |

**[量測] 對四個問題的回答**：
1. **LP 能否在相同預算下找到夠好的可行配置**：
   - 每次都能找到精確可行的配置。
   - dense 上，BC 中位數損失 0–2%，最差單點約 20%。換算成 frontier 跨度（span_S）都 ≤ 0.4%；換算成 knee 用的跨度最大 7%（BERT）。
   - OLMoE 上損失只有 0.007%。
2. **哪些模型真正省時間**：只有 OLMoE。MILP 每個預算 100–250 s，其中 presolve 82–109 s；LP 每個預算 0.5–0.9 s。
3. **哪些只是多跑一次 LP 又回到 MILP**：MobileNetV2、ResNet18、EfficientNet-B0、GPT-2、BERT、Qwen3.5 FFN。LP 優先的時間和 MILP 相同或略多。Qwen3、Qwen3.5 FFN+attn 約少 20%。
4. **問題的來源**：
   - **取整**：dense 問題只有 21–196 個節點，取整 1–2 個分數節點，就造成大於 1e-3 的相對損失。OLMoE 有 3051 個節點，同樣的取整只是 1e-4 的量級。所以 LP 被接受的比例主要取決於問題大小。
   - **修復**：很少需要（0–3 次）。
   - **數值**：
     - 所有問題都有 HiGHS 警告「excessively small costs」；ResNet18 和 MobileNetV2 另有「untransformed violations: row = 0.0008–0.001」（presolve 空間中的違反量），但最終解經過精確檢查都可行。
     - 有 1–3 個可疑下界，都出現在 R ≥ 4.76、目標值極小的預算上。
     - 有 9 個「純 LP 比 MILP 好」的預算，同樣都在 R ≥ 4.76。MILP 雖然回報 status 0，但差距（例如 Qwen3 的 8.9e-5）小於 HiGHS 的絕對 gap 容許（1e-6 × 目標尺度，例如 6.2e-4）。
     - [推論] 這些點的「最佳」或「下界」都不能當成嚴格保證，但它們都遠離 knee。
- **計時**：問題建立（spec／表讀取、支配刪除）另外計時：dense 0.1–2.0 s，OLMoE 4.8 s（支配刪除 4.1 s）。分析時間可忽略。

## 實驗 B：knee 搜尋穩健性（`B/`）

**新規則**（`plan.json` 中預先登記的唯一方案）：
- frontier 依實際 R 排序後，由左往右分組：當 R 超過該組第一點的 (1 + δ) 倍時，開始新的一組。δ = ((upper/lower)^(1/64) − 1)/2，也就是計畫中最細 refinement 間距的一半：CNN、GPT-2、BERT 為 2.78%，Qwen 和 OLMoE 為 2.21%。
- 細化區間取「knee 所在組的下一組最近點」到「上一組最近點」。
- 選點規則、粗掃描、每輪 3 次求解、2 輪 refinement，全部不變。
- 所有 (求解器, 預算) 在本輪最多只求解一次，各方法、各規則一律適用：實驗 A 第 1 次重複的結果會被重用。
- 原規則版本已和 `mp_auto.auto_select` 逐一核對：9 個問題 × 3 種求解器，嘗試過的預算和選出的配置都完全相同。

**[量測] 結果**（見總表）：
- MILP 的 knee：原規則和新規則在 7/9 個問題上相同；ResNet18、EfficientNet-B0 不同。
- LP 優先和 MILP 的分歧：在 Qwen3 和 Qwen3.5 FFN+attn 上大幅縮小；在其他問題上，兩種規則下都一致。
- R 最低預算的端點：36 次搜尋全部保留。
- 搜尋成本：新規則的求解次數相同。OLMoE MILP 的新規則花 3226 s，原規則 3296 s（重用相同預算後的組合成本）；LP 優先兩者都是 76 s。

**knee 改變的機制**（`B/*/search_*.json` 的 trace）：
- **Qwen3、Qwen3.5 FFN+attn（原規則＋LP 優先）**：LP 取整後剩下的資源，和 refinement 點一起形成近重複點（例如 0.8406 和 0.8408）。knee 的鄰居被擠到同一側，第 2 輪往錯的方向細取樣。新規則把這兩點視為同一組，兩側都會取樣，因此找回和 MILP 相同區域的 knee。
- **ResNet18、EfficientNet-B0（新規則）**：第 1 輪後也出現近重複點對（例如 0.77038 和 0.77107）。原規則把它當鄰居，只往一側細取樣，找到了 0.7912 或 0.7505。新規則改成往兩側粗取樣，沒有取樣到那一點。
  - ResNet18 兩個 knee 的分數相同（0.966808）。
  - EfficientNet 新規則的 knee 分數低 1.3e-4。
- **knee 分數非常平**：最佳點和 R 差超過 0.5% 的次佳點，分數只差 1e-5 到 5e-4（各問題的數字見 `B/analysis.json`）。

**離線檢查**（`B/offline.json`，人為擾動後的點不是真實配置）：

| 擾動 | 結果 |
|---|---|
| 100 種輸入順序、全部加入重複點 | 所有執行結果都不變 |
| 只擾動 BC，ε = 1e-8 | 全部不變 |
| 只擾動 BC，ε = 1e-6 | 只有 EfficientNet 新規則那次：knee 改變 9%、原規則區間改變 1% |
| 只擾動 BC，ε = 1e-4 | Qwen3 原規則 LP 優先 24%、Qwen3.5 FFN+attn 新規則 45%、EfficientNet 新規則 60% |
| 只擾動 R，任何 ε | 約 40% 的 knee 改變 |

- **只擾動 R 的探索性發現**：最便宜的端點（例如 Qwen3 的 BC 212,289）和 R 最低的 MILP 點（BC 11,606）資源都恰好等於下界 0.5。chord 的低端點取決於這個精確平手怎麼打破。真實求解時兩者一定精確平手，所以不會發生，但這是 knee 規則的一個脆弱點。
- [未驗證] 明確定義低端點（例如固定取 R = 下界時 BC 最小的點）可以消除這個脆弱性，但本輪沒有測試。

## 實驗 C：補測實際品質（`C/`）

**評估範圍**：只評估 B 的四種組合（原／新規則 × MILP／LP 優先）選出不同配置的問題。
- **OLMoE**：配置和上一輪相同，引用既有證據，沒有重新評估。
- **其他 4 個問題**：配置完全相同，省略評估。

**判定方式**：
- 主要主張是「非劣性」：替代配置減參考配置（原規則＋MILP）的 (1 − 0.05/k) 配對 CI，上界要小於容許差距。
- 次要主張是「等價」：同一個 CI 落在 ±容許差距內。

**容許差距**：參考 knee 先前 native 量到的損傷 × 10%。
- Qwen3：0.0075 nat/token。
- Qwen3.5 FFN+attn：0.0032。
- CNN 的 loss：約 0.0020（ResNet18）和 0.0019（EfficientNet）。
- CNN 的 top-1：先前的降幅是 0.5 pp 和 0，10% 不足以構成有意義的容許差距，所以只報告效應量。

**樣本量**：依計畫的公式，在評估前固定。
- Qwen3：256 個窗口，檢定力 0.86。
- Qwen3.5：896 個窗口，檢定力 0.84。
- CNN：1000 張圖片（全部可用的），檢定力約 0.05，已事先標明。

**後端**：
- LM：GPU PyTorch 重播，**不是 native 全模型結果**。使用裝置修正後的量化器，關閉 TF32。
  - 事先在舊窗口上和 native 比對過：FP32 每窗口最大差 3e-6，knee 的 ΔNLL 差 +0.0032 ± 0.0052（Qwen3）和 −0.0025 ± 0.0026（Qwen3.5）。
  - 另做 native 交叉檢查，每個配置 32 個窗口，分開報告。
- CNN：native，和 `run_native_auto` 的編譯方式相同。

**新資料**：
- Qwen3：validation／test 的 token 65536 起，各 128 個窗口。
- Qwen3.5：validation 從 32768 起、test 從 16384 起，各 448 個窗口。
- 都和該模型先前所有窗口範圍核對過，沒有重疊。
- CNN：HTQ/imagenet1k_2050 中，本 repo 沒用過的 1000 張 held-out 圖片（排除 2050 張中的 calibration 和已用過的 1000 張）。

| 模型 | 配置（來源） | ΔNLL／loss 相對 FP32 | 相對參考 [CI] | 判定 |
|---|---|---|---|---|
| Qwen3（256 窗口，FP32 NLL 3.7280） | 參考（MILP，兩種規則） | +0.0672 [0.0623, 0.0722] | — | — |
| | alt1（LP 優先＋原規則） | +0.0783 | **+0.0111 [+0.0060, +0.0161]** | 較差（CI > 0），非劣性未成立 |
| | alt2（LP 優先＋新規則） | +0.0671 | −0.0001 [−0.0033, +0.0032] | **非劣且等價** |
| Qwen3.5 FFN+attn（896 窗口，FP32 3.6579） | 參考 | +0.0298 [0.0279, 0.0316] | — | — |
| | alt1（LP 優先＋原規則） | +0.0334 | **+0.0037 [+0.0019, +0.0054]** | 較差，非劣性未成立 |
| | alt2（LP 優先＋新規則） | +0.0284 | −0.0014 [−0.0030, +0.0002] | **非劣且等價** |
| ResNet18（1000 張；FP32 top-1 0.695、loss 1.2042） | 參考（原規則，兩種求解器） | top-1 0.689、loss +0.0169 | — | — |
| | 新規則（兩種求解器） | top-1 0.690、loss +0.0134 | loss −0.0035 [−0.0122, +0.0051]；top-1 +0.1 pp [−0.6, +0.9] | 無法判定 |
| EfficientNet-B0（1000 張；FP32 0.785、0.9423） | 參考 | top-1 0.765、loss +0.0150 | — | — |
| | 新規則 | top-1 0.766、loss +0.0218 | loss +0.0068 [−0.0028, +0.0165]；top-1 +0.1 pp [−0.8, +1.0] | 無法判定 |
| OLMoE（引用 `olmoe_search_final`，512 窗口） | LP 優先（兩種規則）對 MILP（兩種規則） | +0.0201 對 +0.0195 | +0.00062 [−0.0003, +0.0015] | 等價（引用，非新評估） |

- **native 子集**（32 個窗口，分開報告，樣本少）：

| 比較 | native | 相同窗口的重播 |
|---|---|---|
| Qwen3 alt1 − 參考 | +0.0078 ± 0.0062 | +0.0185 |
| Qwen3 alt2 − 參考 | −0.0032 ± 0.0041 | +0.0047 |
| Qwen3.5 alt1 − 參考 | +0.0091 ± 0.0050 | +0.0021 |
| Qwen3.5 alt2 − 參考 | +0.0031 ± 0.0034 | −0.0041 |

  方向和主評估一致，差異都在單窗口混沌雜訊的範圍內（見 `runs/olmoe_search_final_20261016`）。
- **OLMoE 的容許差距**：若依本輪「10% × 參考 knee 損傷」的規則，容許差距是 0.00195。但這個損傷值取自同一次 512 窗口的評估，不是獨立的先驗。上一輪事先登記的是 ±0.002，兩者結論相同。
- **[推論] 資源差異**：
  - Qwen3 和 Qwen3.5 的 alt1，R 比參考低 4.3% 和 2.1%，屬於端到端的選點比較，不是同資源下誰的求解較好。
  - alt2 的 R 和參考相差 ≤ 1e-4。
  - CNN 新規則的 knee R 和參考不同：ResNet18 低 2.7%，EfficientNet 高 2.7%。

## 背景核對與修正紀錄

- 題目列的 5 點背景，和程式及原始數據一致（`plan.json` 的 `background_verified`）。另外記錄了兩點：
  - 上一輪 dense 回歸的 LP 優先，退回時用的是「單次 MILP＋修復」，不是重試版。不過 dense 問題上重試從未失敗，所以修正後的 MILP 在 dense 上就等於 `mp_milp`。
  - `qwen3_swiglu_common.Quantizers.act` 在 GPU 上也有同樣的「除以 CPU 純量」問題，本輪所有 GPU 重播都套用了裝置修正。
- **amendment**（21:41，在 C 讀取任何資料之前）：C 的評估範圍、後端、樣本量、新資料來源、容許差距數值。
- **執行中的程式修正**（都在產生結果之前）：
  - Qwen3 的載入改成在腳本內重寫 `qwen3_prepare.CausalLM`：GPU 環境沒有 onnxruntime。
  - Qwen3 的 tokenizer 改用 `gpt2_assets/python_deps` 的 transformers：CPU 環境沒有。
  - 只影響套件匯入，不影響任何計算。

## 尚未完成／未驗證

- OLMoE MILP 的固定預算掃描只完整量了 1 次（每次 46 分鐘），另在 3 個預算上重複 3 次，是事先登記的成本上限。所有 LP 和 LP 優先的時間都量了 3 次。
- LM 的主評估是 GPU 重播，不是 native 全模型；native 只有 32 個窗口的交叉檢查。
- CNN 的品質檢定力只有約 5%；要用 0.002 的容許差距判定，需要約 10 萬張圖片。
- GPT-2、BERT、MobileNetV2、Qwen3.5 FFN 的配置在各組合間相同，沒有評估。
- 自動選擇求解器的門檻（只有一個大型問題）、新規則 δ 的敏感度、R 下界端點平手的修正，都沒有測試。
- dynamic route、其他 MoE 模型。
- BOPs 的降低不代表硬體加速；儲存估算不代表 runtime 記憶體。

## 重現

```
PY="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_lp_pipeline_regression.py; E=run_lp_pipeline_eval.py; O=runs/lp_pipeline_regression_20261016   # 新實驗請換新目錄
for k in mobilenetv2 resnet18 efficientnet_b0 gpt2 bert qwen3 qwen35_ffn qwen35_ffn_attn olmoe; do $PY $S --stage plan --problem $k --out-dir $O; done
$PY $S --stage plan --out-dir $O
for rep in 1 2 3; do for k in <8 個 dense 問題>; do for m in milp lp lpfirst; do $PY $S --stage a_solve --problem $k --method $m --rep $rep --out-dir $O; done; done; done
for rep in 1 2 3; do for m in milp lp lpfirst; do $PY $S --stage a_solve --problem olmoe --method $m --rep $rep --out-dir $O; done; done
$PY $S --stage a_analyze --out-dir $O
for k in <9 個問題>; do $PY $S --stage b_search --problem $k --solvers milp,lpfirst,lp --out-dir $O; done
$PY $S --stage b_offline --out-dir $O; $PY $S --stage b_analyze --out-dir $O
$GPU $E --stage c_freeze --problem qwen3 --out-dir $O; $PY $E --stage c_windows --problem qwen3 --out-dir $O
$GPU $E --stage c_check --problem qwen3 --out-dir $O; $GPU $E --stage c_replay --problem qwen3 --out-dir $O
$PY $E --stage c_native --problem qwen3 --out-dir $O; $PY $E --stage c_analyze --problem qwen3 --out-dir $O
（qwen35_ffn_attn 同上）
$PY $E --stage c_cnn --problem resnet18 --out-dir $O; $PY $E --stage c_cnn --problem efficientnet_b0 --out-dir $O
```

實際耗時共約 3 小時（`timing.json`，123 筆）：

- A：dense 約 35 分鐘，OLMoE 約 75 分鐘；
- B：dense 數分鐘，OLMoE 約 35 分鐘；
- C：約 70 分鐘。

## 檔案

- `plan.json`（含 `background_verified`、amendment）、`plan_budgets.json`（每個問題的端點、粗掃描與診斷預算、δ、span）、`timing.json`、`summary.json`（逐模型總表的資料）。
- `A/<問題>/<方法>_rep<n>.json`：每個預算的路徑、狀態、BC、M、B、精確可行性、各步驟時間、HiGHS 解析結果；第 1 次重複另存配置。`A/<問題>/logs/`：HiGHS log。`A/analysis.json`。
- `B/<問題>/search_<求解器>_<規則>.json`：每輪 trace（knee、鄰居、分組、下一輪預算）、嘗試過的預算、frontier、候選與配置雜湊、成本。`B/<問題>/store_<求解器>.json`：本輪每個 (求解器, 預算) 唯一一次的解。`B/offline.json`、`B/analysis.json`。
- `C/<問題>/`：`freeze.json`、`configs/`、`windows.json` 或 `images.json`、`tokens_eval.npy`、`replay_check.json`、`replay.json`、`native/`、`analysis.json`。
