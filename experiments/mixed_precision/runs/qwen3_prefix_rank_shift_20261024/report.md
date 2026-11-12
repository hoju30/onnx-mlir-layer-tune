# Qwen3-0.6B：前層量化是否改變後層 Down activation 候選排名

日期：2026-10-24。腳本：`run_qwen_prefix_rank_shift.py`（sha256 `da072a892eeb24b1…`）。計畫：`plan.json`／`plan.md`（04:45:57 寫入）。設計來源：`reports/qwen3_prefix_rank_shift_prompt_20261024.md`。

## 性質與邊界

- **做法**：GPU FP32 + QDQ 的單點介入診斷，語意與 LMReplay 相同。
- **沒有修改**：BC、搜尋器、knee、range、共享量化函式、權重；沒有 fine-tune；沒有新增 criterion；沒有重算 BC 統計。
- **不宣稱**：任何 native 或低位元 kernel 的收益。
- **結論範圍**：只限本模型的 Down activation、本輪的候選與兩種背景。

## 結論摘要

**新資料支持的結果**（128 篇新 WT103 文章 × 2 窗口；文章 bootstrap 20000 次）

- **前層量化沒有改變任何同位元候選的實測排名。**
  - 81 對（27 層 × int4／p4e1、int6／p6e1、int8／p8e1）中，背景 U（前層 FP32）與背景 Q（前層依 A 量化）的 D 點估計符號全部相同。
  - 觀察到的反轉 0 對；未校正 CI 支持的反轉 0 對；Holm 校正後支持的反轉 0 對。
- **相對效應的變化只出現在 L27。** J = D_Q − D_U 經 Holm（81 個）顯著的只有 2 對，都是同號、都是 Posit 的優勢在 Q 背景更大：
  - L27 int4–p4e1：J = +0.0084
  - L27 int6–p6e1：J = +0.0299
- **原 BC 在兩種背景下的正確率幾乎相同。**
  - U：可判定 76 對中排對 75；Q：可判定 77 對中排對 76。
  - 兩背景都可判定的共同 74 對：U、Q 各排對 73。
  - 唯一的錯誤是 L27 int4–p4e1，兩種背景都錯（原本就錯）。
  - 「原本對、背景改變後才錯」0 對；「原本錯、後來對」0 對。
- **依新背景重選沒有實際價值。**
  - G_shift = +0.000008 nat/token，Bonferroni 97.5% CI [−0.000019, +0.000036]。上界遠低於 ε = 0.001。
- **G_BC 超過門檻，但收益幾乎全部來自一個舊案例。**
  - G_BC = +0.00455，97.5% CI [+0.00424, +0.00488]，下界 > ε。
  - 收益幾乎全部來自 L27 int4–p4e1：BC 選 int4，但 p4e1 每 token 好約 0.39。
  - 排除 L2／L14／L27 後，G_BC = −0.000051，97.5% CI [−0.000072, −0.000029]，在其餘層反而是 BC 的選擇略好。

**依預先規則的判讀**
- G_BC 通過而 G_shift 沒有證據：較可能是原 proxy 的局部排名誤差，不能歸因於前層量化造成的排名改變。
- G_shift 上界 < ε：平均收益低於門檻，不做全模型重新評分。
- G_BC 的收益集中在 L27 這個歷史案例：只支持局部追查，不支持普遍增加 offline 成本。

**未確認或屬次要的結果**
- calibration 選擇不同（q_U ≠ q_Q）的 5 對全是 int8–p8e1。在新資料上 q_Q 對 3 對、錯 2 對，子集收益 +0.00013，CI 跨 0。
- L2 int8–p8e1：calibration 偏好 p8e1，新文章上卻是 int8 較好；BC 選 int8，是對的。
- f16 vs p8e1（次要，資源不同）：L27 與 L26 的 p8e1 比 f16 好；L2 的 f16 好得多；L23 的 f16 略好。

## 1. 設計與實作

### 背景與設定

對每個目標層 i（L0–L27 的 `mlp.down_proj`，層號由 node_mapping 解析為整數）建立兩個背景：

| 背景 | blocks < i | block i 的其他投影 | blocks > i |
|---|---|---|---|
| U | FP32 | FP32 | FP32 |
| Q | 全部搜尋投影依 A（w 與 a，attention 與 FFN 都含） | FP32 | FP32 |

- **目標 Down 權重**：兩背景都用 A 的 w，逐位元相同。
- **activation 設定**：fp32（只關目標 a）、int4、p4e1、int6、p6e1、int8、p8e1、f16，同時作用於 Down 的 input x 與 output y。INT 用 native ranges，Posit 用 LUT，都固定不變。
- **條件數**：28 × 2 × 8 = 448。L0 的前綴為空（U = Q），作為負控制，不計入主要統計。
- **A**：`runs/qwen3_down_fp16_protection_20261024/configs/A.json`，sha256 `ac10afa9…`，與 `ref.json` 相同。

### 實作方式

- 每個 chunk 先做兩次完整前向：全 FP32 一次、全 A 一次，擷取進入每個 block 的 hidden state。由於深度上的因果關係，這就分別是 U 與 Q 的前綴。
- 每個條件只從 block i 重播到 L27，再接 norm、LM head 與 CE。
- 在 calibration 上，全部 448 個條件也各做一次完整前向，與重播結果比對。

### 三個 selector（看新文章前凍結：`bc_choices.json`、`choices.json`）

- **q_BC**：native 表 C[node][a]（input + output）較小者。沒有任何 proxy tie。
- **q_U、q_Q**：原 10 個 calibration 窗口上，各背景實測 NLL 較小者。
  - 沒有數值 tie。
  - low margin（差 < 0.001）：U 7 對、Q 6 對，標記但保留。

### 資料（`segments.json`、`exclusions.json`、`data_freeze.json`、`audit.json`）

- 128 篇新 WT103 文章，每篇 2 個連續、不重疊的 128-token 窗口（254 個 target）。seed：順序 30101、起點 30102。
- verified 掃描排除 8489 篇歷史文章，其中包含 spike 128、fp16 512、l27 128（已逐一核對）。
- 歷史 Qwen token audit：55 個檔，無法解析的來源 0 個。
- 抽樣：合格池 18906 篇，訪問 159 篇。跳過：歷史 Qwen 窗口重疊 19、近重複歷史 WT103 10、近重複 WT2 2。
- 新規則「任一窗口與 WT2 的 5-gram 重疊 ≥ 50% 即拒收」沒有觸發；接受窗口的最高重疊為 14%。

## 2. 控制（`controls.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 448 個條件「重播 vs 完整前向」（calibration，每窗口 summed CE） | 最大差 0（全部逐位元相同；容差 1e-6） |
| L0 前綴 U = Q | 逐位元相同；L0 全部 8 個設定的 U、Q 結果也相同 |
| 全 FP32 重播 vs 無 hook 模型 | 逐位元相同 |
| 切回同一條件 | 逐位元重現 |
| 每個條件的權重 | 每個模組的權重都指向預期的快取 tensor（原始 FP32，或由原始權重量化的 A 格式）；activation map 等於配置；不會重複量化 |
| 前綴 | Q 恰好量化 7·i 個投影，都在 block < i；U 沒有前綴量化 |
| 目標 x／y 的實際格式（全部 448 個條件） | 都是預期量化器的不動點；fp32 設定下不可被 f16 表示（確認沒被量化） |
| hook | 每個投影 1 個 pre-hook 與 1 個 forward hook，前後不變 |
| GPU 設定 | TF32 關閉、deterministic、float32、matmul precision highest |
| Posit LUT（p4e1／p6e1／p8e1）vs Universal | 0 不一致 |
| 其他 | ranges = native_eps_fix；calibration tokens = prepare；RMSNorm epsilon 1e-6 |
| 新資料的非有限值 | 0；主要矩陣完整 |

**裁切與飽和（描述，`calibration_scores.json`）**
- L2 與 L27 的 Down output |y| 最大值為 6850–6960 與 5033–5169，超過 p8e1 的上限 4096。超出的元素比例約 1e-5。
- Q 背景下，L27 Down input 的最大絕對值 1808，超出原 range 下限 −1682，比例低於 1e-6。
- 量化器沒有因此更改。

## 3. 問題一：前層量化是否改變真實排名？

沒有。D_s = E_s(INT) − E_s(Posit)，正值代表 Posit 較好；見 `fig_scatter_DU_DQ.png`、`pairwise_results.csv`。

| 項目 | 結果 |
|---|---|
| D 點估計符號相反的對（觀察到的反轉） | 0／81 |
| 兩背景 CI 都排除 0 且符號相反 | 0 |
| Holm（162 個 D 檢驗）支持的反轉 | 0 |
| Holm 顯著的 D | U 71、Q 69 |
| J 未校正 CI 排除 0 | 9 對：L1、L8、L10、L26、L27 的 int4–p4e1；L18 的 int6–p6e1 與 int8–p8e1；L27 的 int6–p6e1 與 int8–p8e1 |
| J 經 Holm（81 個）顯著 | 2 對：L27 int4–p4e1（+0.0084）、L27 int6–p6e1（+0.0299） |

- J 不為 0 只代表相對影響改變。L27 的兩對在 U、Q 都是 Posit 大幅較好，前層量化只讓差距更大，不是反轉。
- 散點圖中所有點都在第一或第三象限。L2 三對都在第三象限（INT 較好），L27 與大多數層在第一象限。

## 4. 問題二：原 BC 在兩種背景下的正確率

見 `fig_bc_classes.png`。

| | U | Q |
|---|---|---|
| 可判定配對（95% CI 排除 0） | 76 | 77 |
| BC 排對 | 75 | 76 |
| 未定 | 5 | 4 |
| proxy tie | 0 | 0 |
| 覆蓋率 | 93.8% | 95.1% |

- 兩背景都可判定的共同 74 對：U 排對 73、Q 排對 73。

**分類**

| 類別 | 對數 |
|---|---|
| 兩邊都對 | 73 |
| 兩邊都錯 | 1（L27 int4–p4e1） |
| 原本對、後來錯 | 0 |
| 原本錯、後來對 | 0 |
| 未定 | 7 |

未定的 7 對全是 int8–p8e1：只在 U 未定的是 L3、L4、L15；只在 Q 未定的是 L12、L18；兩邊都未定的是 L13、L16。

**判讀**
- 唯一的 BC 錯誤是 L27 int4–p4e1，在前層 FP32 時就已經錯，不是背景改變造成的。
- BC 預測 int4 較好（C 6482 vs 7177），實測 p4e1 每 token 好約 0.39（U +0.387、Q +0.396）。
- 這與前幾輪「L27 position-0 massive activation 使二階 proxy 高估 Posit 誤差」的觀察一致；這一點是事後解釋。

## 5. 問題三：凍結 selector 在未見資料上的收益

數值都是 81 個單點介入的平均（nat/token），不是任何配置的收益；見 `selection_gain.json`、`fig_selection_gain.png`。

| 指標 | 平均 | 95% CI | Bonferroni 97.5% CI | 對數 |
|---|---|---|---|---|
| **G_shift** = E_Q(q_U) − E_Q(q_Q) | +0.000008 | [−0.000016, +0.000032] | [−0.000019, +0.000036] | 81 |
| **G_BC** = E_Q(q_BC) − E_Q(q_Q) | +0.004552 | [+0.004281, +0.004829] | [+0.004243, +0.004875] | 81 |
| U 背景：E_U(q_U) − E_U(q_Q) | +0.000010 | [−0.000014, +0.000034] | | 81 |
| U 背景：E_U(q_BC) − E_U(q_U) | +0.004467 | [+0.004210, +0.004730] | | 81 |
| G_shift，只看 q_U ≠ q_Q（次要） | +0.000128 | [−0.000255, +0.000519] | | 5 |

**分層（探索性定位）**

| 子集 | G_shift | G_BC |
|---|---|---|
| 4-bit | 0（q_U = q_Q） | +0.01465（全部來自 L27） |
| 6-bit | 0 | 0（三個 selector 全同） |
| 8-bit | +0.000024 [−0.00005, +0.00010] | −0.00099 [−0.00166, −0.00031] |
| L1–8 | +0.00011 [+0.00007, +0.00016] | −0.00096 |
| L9–17 | −0.00002 | −0.00005 |
| L18–27 | −0.00005 | +0.01311 |
| 排除 L2／L14／L27 | +0.000009 [−0.000018, +0.000036] | −0.000051 [−0.000069, −0.000032] |

- G_BC 的 8-bit 為負，來自 L2 int8–p8e1：calibration 在兩背景都偏好 p8e1（差 +0.018），但新文章上 int8 較好（D_Q = −0.023，CI 排除 0）。BC 選 int8，在這一對是對的。
- 「在新資料上直接取最佳格式」的差距是在同一批資料上取最小值，偏樂觀，只作描述：
  - q_Q 與該最佳格式差 +0.00033；
  - q_BC 與該最佳格式差 +0.00488。

**q_U ≠ q_Q 的 5 對（全是 int8–p8e1）在新資料（Q 背景）的結果**

| 層 | calibration 選擇 | 新資料 D_Q | q_Q 是否較好 |
|---|---|---|---|
| L6 | q_U = int8，q_Q = p8e1 | +0.0027，p8e1 較好 | 是 |
| L11 | q_U = p8e1，q_Q = int8 | +0.0014，p8e1 較好 | 否 |
| L15 | q_U = int8，q_Q = p8e1 | +0.0008，p8e1 較好 | 是 |
| L18 | q_U = int8，q_Q = p8e1 | +0.0007，CI 跨 0 | 點估計是 |
| L24 | q_U = p8e1，q_Q = int8 | +0.0022，p8e1 較好 | 否 |

這 5 對的 BC 選擇都是 p8e1，在新資料上都是點估計較好的一方。calibration 的差距多半 < 0.004，屬 low margin。

## 6. 次要：f16 vs p8e1（資源不同，不是等成本排名）

D16 = E(f16) − E(p8e1)；正值代表 p8e1 較好。Holm family 為 54 個 D 檢驗。

| 層 | A 的 a | D16_U | D16_Q | 換成 p8e1 的 BOPs 變化（佔 A 總 BOPs） |
|---|---|---|---|---|
| L2 | f16 | −0.2932（Holm p 0.003） | −0.2957（0.003） | −0.814%（但品質大幅變差） |
| L26 | f16 | +0.0024（0.003） | +0.0017（0.014） | −0.610% |
| L27 | f16 | +0.0019（CI 跨 0） | +0.0047（0.003） | −0.814% |
| L23 | p8e1 | −0.0004（n.s.） | −0.0007（0.005） | （A 已是 p8e1；改成 f16 會 +0.610%） |
| L24 | p8e1 | −0.0007（0.005） | −0.0003（n.s.） | 同上 |

- L27 的方向與前兩輪一致（完整配置的 PyTorch 與 native 都是 p8e1 較好）。
- L26 是新觀察到、同樣由 A 以 f16 保護的位置，單點顯示 p8e1 較好。
- 依計畫，只看單點品質不能決定取消保護；這些只作描述。BOPs 是 proxy。

## 7. 耗時（`timing.json`；研究掃描耗時，不是 pipeline latency）

| 項目 | 時間 |
|---|---|
| plan（含語料掃描與 hash） | 23 s |
| calibration 階段：重播 448 條件 × 10 窗口 | 81 s |
| calibration 階段：完整前向對照 | 166 s |
| calibration 階段：初始化與權重準備 | 16 s |
| calibration 階段總計 | 270 s |
| audit | 33 s |
| 抽樣 | 101 s |
| 正式評估（256 窗口 × 448 條件） | 1996 s（前綴前向 31 s，重播 1956 s，8 個 chunk 各約 245 s），GPU 峰值 7.0 GB |
| 分析 | 3 s |

計畫中依冒煙測試估計正式評估約 35 分鐘，實際 33 分鐘。

## 8. 決策與下一步

**依預先規則**

| 規則 | 本輪情況 | 結論 |
|---|---|---|
| 只有 G_BC 明顯改善、G_shift 沒有證據 | 符合 | 收益是原 proxy 的局部排名誤差，不能歸因於前層量化改變排名 |
| G_shift 上界 < ε | 上界 0.000036 | 平均收益低於門檻，不做全模型背景重新評分；但不能說完全沒有交互作用，L27 的 J 已顯示相對效應會變 |
| 效應集中在 L27 等少數歷史案例 | G_BC 幾乎全來自 L27 int4–p4e1 | 只支持局部追查，不支持普遍增加 offline 成本 |
| calibration selector 是否改善未見資料 | q_Q 相對 q_U 沒有驗證收益；L2 int8–p8e1 甚至選錯，而 BC 選對 | 不推廣到其他重新評分方法 |

**不做的事**
- 不進入「在新背景更新統計」的下一輪。
- 不啟動刷新或搜尋，也不把各單點選擇組成新配置。

**用數據回答**

| 問題 | 回答 |
|---|---|
| 多少排名反轉？ | 0／81（觀察、未校正 CI、Holm 都是 0） |
| 哪些不是 BC 原本就排錯？ | 沒有。唯一的 BC 錯誤（L27 int4–p4e1）在前層 FP32 時就已存在 |
| 錯選代價多大？ | L27 int4–p4e1 每 token 約 0.39，攤到 81 對平均約 0.0048；L2 int8–p8e1 上 q_Q 選錯，抵銷約 0.0003，所以 G_BC 為 0.0046。其餘層 BC 的平均錯選代價不為正（排除 L2／L14／L27 後，q_BC 反而比 q_Q 好 0.00005） |
| 是否值得下一輪重新評分？ | 以背景相關重新評分而言，不值得 |
| 證據還缺什麼？ | 見下 |

**證據還缺什麼**
1. 只有 Down activation、一個模型、一種前綴配置（A）。
2. 單點介入（後層 FP32），不是完整配置。
3. 128 篇 × 2 窗口的 Wikipedia 文本。
4. GPU QDQ，不是 native。
5. L2 int8–p8e1 的 calibration 與新文章不一致，原因沒有檢驗。
6. L27 int4–p4e1 的 BC 誤排機制（position 0）沒有在本輪重新檢驗。

## 9. 冒煙測試、修改紀錄與重現

**冒煙測試（計畫前，在 scratchpad，不在 runs/；細節在 `plan.json["time_estimate"]`）**
- 完整跑過 calibration 階段，看過全部 448 個 calibration 分數，以及 5 對 q_U ≠ q_Q。
- 一個評估 chunk 與分析，用重複填充的 calibration 窗口，沒有新文章。
- 以其他 seed（999301／999302）抽 4 篇，沒有評估。
- 修正了 3 處：權重核對改用 data_ptr、Q 前綴數量的檢查、圖標題。設計沒有改。

**修改紀錄**：計畫寫入後腳本未變更（每個階段都檢查程式與輸入 hash），沒有 amendment。

```
O=runs/qwen3_prefix_rank_shift_20261024
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$CPU run_qwen_prefix_rank_shift.py --stage plan    --out-dir $O --smoke-estimate <smoke json>   # 只能寫一次
$GPU run_qwen_prefix_rank_shift.py --stage calib   --out-dir $O    # 控制 + calibration 矩陣 + selector
$CPU run_qwen_prefix_rank_shift.py --stage freeze  --out-dir $O
$CPU run_qwen_prefix_rank_shift.py --stage audit   --out-dir $O
$CPU run_qwen_prefix_rank_shift.py --stage data    --out-dir $O
$GPU run_qwen_prefix_rank_shift.py --stage eval    --out-dir $O    # 可續跑：每 chunk 依 hash 標頭核對
$CPU run_qwen_prefix_rank_shift.py --stage analyze --out-dir $O
```

**hash（sha256 前 16 碼）**

| 檔案 | sha256 |
|---|---|
| 腳本 | `da072a892eeb24b1` |
| plan.json | `ba920ecc4a7532a8` |
| plan.md | `9846fcef30139aa8` |
| bc_choices.json | `9a382ae8fa82e811` |
| choices.json | `52339af3b4016545` |
| calibration_scores.json | `0b7f076b35b0aad6` |
| controls.json | `edd3b3b1ece82fd3` |
| freeze.json | `e6c04a2e7cf1d8aa` |
| configs/conditions.jsonl | `5459eb49fd80db6b` |
| configs/backgrounds.json | `50abcce41453a385` |
| audit.json | `bf0151842d33c5f3` |
| segments.json | `c29ff836082c84f9` |
| exclusions.json | `4d92a9f94bc7a930` |
| data_freeze.json | `74ee075f53a3cdca` |
| tokens.npy | `4054c5665d2783bd` |
| eval_ce.npz | `551eca236b31c8ad` |
| pairwise_results.csv | `3741b9223ae1a202` |
| per_article_losses.csv | `1c633cb6637c3a79` |
| per_window_losses.csv | `4ea2fd8b5f61b846` |
| selection_gain.json | `6dff97d595f3b58d` |
| stats.json | `cd78b339848bccc3` |
| timing.json | `02c670ea93f35b94` |

模型、ranges、BC 表、mapping 與歷史資料的 hash 記錄在 `plan.json["inputs"]`。
