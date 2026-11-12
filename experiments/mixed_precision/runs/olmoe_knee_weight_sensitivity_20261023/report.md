# OLMoE 正常 knee：只刷新 weight sensitivity（W）

日期：2026-10-23
腳本：`run_olmoe_knee_weight_sensitivity.py`（plan → controls → refresh ×9 → 端到端 W → check → freeze → data → eval → analyze）
輸出：本目錄

**本輪的範圍**
- 維持原本的 BC criterion，只借用 GEMQ「在量化後的工作狀態重新取得統計」這個想法。
- 不改成 GEMQ 的 output-error criterion，也不改成 weight-only 量化；這不是 GEMQ 的完整復現。
- A、B、C 沿用凍結的配置；D4、D5、E、E5 只用於交叉評分的診斷。
- 新增 W：只刷新 BC weight 項的 sensitivity，activation 項整項沿用 static；在原 knee 的精確上限下，用 W+A 重新搜尋一次。
- 沒有 ≤4／5-bit 篩選；沒有修改 BC 公式；沒有微調或更新參數；沒有重選 knee 或第二次 refresh；沒有修改正式 pipeline 預設。
- 所有 stage 都用同一份腳本（sha `192d28a3…`），沒有修訂（`amendments.json` 為空）。

## 結論

**在這批新的 600 篇文章上，W 沒有確認改善 static BC**：W − A = −0.00021，校正 CI [−0.00098, +0.00056]。

**W 和 C 之間沒有可確認的差異**：W − C = +0.00021，校正 CI [−0.00059, +0.00102]。

**W 相對 B 的非劣性證據不足**：W − B = −0.00003，點估計幾乎相同，但校正上界 +0.00081 超過 +0.0005。

本批的 B − A（−0.00019）和 C − A（−0.00042）也都沒有確認。W 的額外時間比 B 少 16 秒（−20%），比 C 少 5 秒（−7%）。

| 條件 | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| FP32 | 2.96699 | 19.433 | — |
| A：static knee | 2.98213 | 19.730 | +0.01514 [+0.01410, +0.01621] |
| B：全節點 sensitivity＋activation error | 2.98194 | 19.726 | +0.01496 [+0.01393, +0.01601] |
| C：全節點 sensitivity（weight＋activation），error static | 2.98171 | 19.721 | +0.01472 [+0.01363, +0.01582] |
| **W：只刷新 weight sensitivity** | **2.98192** | **19.726** | **+0.01493 [+0.01387, +0.01598]** |

600 篇新的 WikiText-103 文章 × 4 個窗口；全部用原 router、各自的 dynamic route。

**預先登記的三個主要比較**（以 Bonferroni 98.33% CI 判定）

| 對比（nat/token） | 意義 | 平均 | 95% CI | **98.33% CI（校正）** | 較好／較差篇數 | 判定 |
|---|---|---:|---|---|---|---|
| W − A | 只刷新 weight sensitivity 的整體效果 | −0.00021 | [−0.00084, +0.00041] | **[−0.00098, +0.00056]** | 312／288 | 方向未確認 |
| W − C | weight 已刷新後，再刷新 activation sensitivity 的差別（取負號） | +0.00021 | [−0.00045, +0.00087] | **[−0.00059, +0.00102]** | 304／296 | 方向未確認 |
| W − B | weight 已刷新後，再刷新 activation sensitivity 與 error 的差別（取負號） | −0.00003 | [−0.00071, +0.00066] | **[−0.00086, +0.00081]** | 299／301 | 非劣證據不足 |

**次要比較**（95% CI）

| 對比 | 平均 | 95% CI | 較好／較差篇數 | 方向 |
|---|---:|---|---|---|
| B − A | −0.00019 | [−0.00085, +0.00048] | 316／284 | 未確認 |
| C − A | −0.00042 | [−0.00108, +0.00023] | 304／296 | 未確認 |

10,000 次文章級 paired bootstrap，seed 23011；各差值的文章級 SD 為 0.0078–0.0085。

## 回答五個問題

**1. W 是否改善原始 BC？**
- **沒有確認。** 點估計 −0.00021，約為 A 損傷的 1.4%。
- 校正區間能排除的：大於 0.00098 的改善，或大於 0.00056 的變差。
- 不能排除的：沒有效果，或 0.001 以內的小改善。

**2. W 是否保留完整 B 的品質？**
- **證據不足。** W − B 的點估計幾乎為 0（−0.00003），但校正 CI 寬達 ±0.0008，上界超過 +0.0005。
- 本批的 B − A 本身也沒有確認，所以 W 要保留的收益本身就不確定。

**3. 相較 C，額外刷新 activation sensitivity 有什麼證據？**
- **沒有。** W − C = +0.00021（點估計是 C 稍好），校正 CI [−0.00059, +0.00102] 包含 0。
- 依預先登記：W − B 的差異包含 activation sensitivity 與 error 兩者，不能單獨歸因於 error；本批這項差異也不確定。
- **[描述] 配置面**：
  - W 對 A 改了 124 個投影：只改 weight 104 個、只改 activation 9 個、兩者都改 11 個。activation 格式並非固定，W+A 搜尋仍改了 20 個 activation 格式。
  - C 對 A：只改 weight 109、只改 activation 85、兩者 4。
  - W 和 C 差 103 個投影，其中 95 個是 activation 格式。
  - 換言之，weight sensitivity 的刷新主要帶來 weight 格式的重新分配（數量和 C 相近），activation sensitivity 的刷新則帶來大量 activation 格式的改動。這些差異對 loss 的影響，本批解析不出來。

**4. W 實際省下多少時間？**
- **額外時間**（3 次中位數）：

  | 條件 | 秒 | W 少多少 |
  |---|---:|---|
  | W | 66.1 | — |
  | B | 82.4 | 16.3 秒（−20%） |
  | C | 70.9 | 4.8 秒（−7%） |

- **W 省掉的**：activation 梯度統計（3.4 秒）、候選 activation 誤差（12.5 秒，相對 B），或 activation 組表與快取讀取（約 1.2 秒，相對 C）。
- **W 仍要付的**（這是大部分）：
  - weight sensitivity 27.5 秒（FP64 的 GYᵀX）；
  - ΔW 快取讀取 11.7 秒；
  - 載入與量化準備 17.0 秒；
  - 剪枝與搜尋 5.0 秒；
  - backward 2.4 秒、forward 1.6 秒。
- **從零到最終配置**：
  - W 單次實測 240.1 秒；
  - 歷史值：A 170.5、B 246.6、C 240.3 秒。
  - W 比 A 多約 70 秒（+41%），比 B 只少約 7 秒。

**5. 是否值得保留為可選步驟？**
- **不值得。** W 沒有確認改善 A，也沒有證明保留 B 的品質；而它仍要付 refresh 的主要成本（約 66 秒額外時間）。
- **[推論] 六批 knee 資料的整體圖像**（各批單獨配對，不合併、不跨批相減；見後文表格）：
  - 完整 refresh B 相對 A，六批的點估計都是負的，但幅度在 −0.0002 到 −0.0012 之間浮動，只有兩批確認；
  - C、D4、D5、E、E5、W 這些只更新一部分的變體，相對 A 都沒有確認過。
  - knee 上任何 refresh 的效果都很小，不超過約 0.001 nat/token，已在跨後端實作差異（0.001–0.002）的範圍內。以每批 600 篇的設計，無法分辨它們之間的差別。
- 即使 W 有效，結論也只適用於本模型、本 calibration、本 knee 和 GPU QDQ 下「刷新原 BC 的 weight 項」這個做法，不是 GEMQ 的復現。

## W 的定義與實作（`plan.json`）

**成本**
- **weight 項**（全部 3051 個投影）：B_i(f) = Σ_c S_i,c^q0 × D_i,c^static(f)。
  - D_static(f) = Σ_j (Q_f(W) − W)²，來自既有的 ΔW 快取。
  - S_c 的定義和既有實作完全相同：
    - GY 是對 output-QDQ 前 linear 輸出的梯度；X 是 input-QDQ 後實際送進 linear 的輸入。
    - 每個 calibration 窗口先算 GYᵀX，再平方、依 output channel 加總、跨窗口累加，最後除以 input dimension。
- **activation 項**：整項沿用 static（A），包含原本的 sensitivity 與誤差。
- **預期表**：weight 列 = 歷史 C 的 weight 列（也等於歷史 B 的，已核對）；activation 列 = static。
- **fallback**：q0_knee 中 1017 個 expert 都有 token，fallback 為 0。

**梯度語義**
- q0_knee 的 calibration forward 使用它自己的 dynamic route。
- 逐層 backward 把記錄的路由當成常數，activation QDQ 經 identity STE；不對 router 或 top-k 求導，沒有 optimizer。

**只收集 weight 統計的路徑**（`experts_tap_w`、`build_w`）
- 完整的 backward 照常經過每個 expert，STE 和路由語義不變。
- 只對 weight 項需要的 linear 輸出（zg、zu、zd）保留梯度，並保留對應的輸入參考（xg、xu、hd）。
- 不計算 activation 梯度統計、候選 activation 誤差、activation 組表，也不讀 activation 快取。
- 既有的 `build()` 即使換了 variant 名稱，仍會計算這些項目，所以本輪另寫了專用路徑。

**搜尋**
- W+A：weight 和 activation 格式都可以重新分配。activation 成本固定，不代表 activation 格式固定，也沒有把 activation 改成 FP32。
- 和前幾輪相同：LP 優先、精確支配剪枝、knee 的精確上限。

## 控制（`check.json`、`controls.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 1. 輸入 | A／B／C／D4／D5／E／E5 的配置、q0、各輪的 check、20 項重用依賴的 hash 都和記錄一致；歷史 B 和 C 的 weight 列逐位元相同 |
| 2. weight 列 | W 的全部 3051 個 weight 列，和歷史 C（也和 B）逐位元相同 |
| 3. activation 列 | W 的全部 activation 列和 A 逐位元相同；整張 W 表等於預期表 |
| 4. forward | 9 次 refresh 和端到端 W 的 CE、hidden state，以及 9 次的路由，都和 q0_knee 的獨立 forward 記錄逐位元相同 |
| 5. 梯度 | 全部執行的 16 層輸入梯度 sha256 等於歷史記錄；重算差為 0；W 的 3051 個 G_c 和 C 第 1 次存下的值逐位元相同（n_in 也相同） |
| 6. 獨立重算 | 用 W 第 1 次存下的 G_c 和 ΔW，以 numpy 重算全部 weight 列：最大相對差 0 |
| 7. 工作內容 | W 的 3 次執行：backward（2.4 秒）和 weight sensitivity（27.5 秒）都有；activation 梯度統計、誤差、組表、activation 快取讀取全為 0。端到端裡的 activation 統計屬於 static 建表；W 階段本身沒有 activation 組表或快取讀取 |
| 8. 有效性 | fallback 0；有限；恰好涵蓋 3051 個投影；格式合法；在精確上限內 |
| 9. 重複 | W 3 次的表與配置相同（主要配置 = 第 1 次）；本輪的 B、C 各 3 次和凍結版本相同；端到端 W 重現了 static 表、ΔW 快取、knee，以及 W 的表和配置 |
| 評估 | 程式控制（E5 輪的窗口 0–127）：FP32、A、B、E5 的 CE 差 0；W 第一批重跑逐位元相同；router 未被替換 |

## 搜尋近似診斷

**W 的搜尋**
- 目標值 18.081776，LP 下界 18.073330；絕對差距 0.00845，相對 4.7e-4；由 LP 接受。
- 剪枝後 70,045 個選項；size 用量 97.67%，BOPs 用量 99.990%。

**各已知配置在 W 表上的分數**（都在精確上限內）

| 配置 | A | B | C | D4 | D5 | E | E5 | **W** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W 表上的目標值 | 18.10673 | 18.08713 | 18.08550 | 18.10028 | 18.11322 | 18.10661 | 18.10651 | **18.08178** |

- 沒有已知配置比 W 的搜尋解更低；C 最接近，只差 0.0037。
- 但 W 的 LP 差距（0.0084）比這個分數差還大，所以不能宣稱已經排除了搜尋近似的影響。
- 交叉評分只是診斷，不同成本表的數字彼此不可比，也不代表品質。

## 配置（`freeze.json`、`analysis.json`）

**精確上限**：weight bytes ≤ 4,524,359,631.556，BOPs ≤ 46,648,327,679,632.125。

| | weight bytes | BOPs | R |
|---|---:|---:|---:|
| A | 4,418,699,264 | 46,645,278,408,704 | 0.7070606 |
| B | 4,420,796,416 | 46,647,774,019,584 | 0.7070984 |
| C | 4,421,058,560 | 46,648,180,867,072 | 0.7071046 |
| W | 4,418,961,408 | 46,643,598,589,952 | 0.7070351 |

**格式差異**

| | 不同的投影 | 只改 weight | 只改 activation | 兩者都改 |
|---|---:|---:|---:|---:|
| W 對 A | 124 | 104 | 9 | 11 |
| W 對 B | 106 | 7 | 95 | 4 |
| W 對 C | 103 | 8 | 93 | 2 |
| B 對 A | 201 | 108 | 87 | 6 |
| C 對 A | 198 | 109 | 85 | 4 |

- W 的 weight 改變：int6→int5 37、int5→int6 28、int4→int5 19、int5→int4 13、int6→int7 10、int7→int6 7、int7→int8 1。
- W 的 activation 改變：p6e1→p8e1 7、int7→p8e1 6、p6e1→int7 4，以及另外 3 個。

**實際路由的計算成本**（評估資料）

| | 搜尋節點 BOPs，相對全 int8 | 含未搜尋 expert，相對全 int8 | 送到未搜尋 expert 的 token-slot |
|---|---:|---:|---:|
| A | 0.67859 | 0.70931 | 164,768 |
| B | 0.67863 | 0.70935 | 164,758 |
| C | 0.67861 | 0.70932 | 164,723 |
| W | 0.67900 | 0.70974 | 164,902 |

FP32 有 162,538 個 token-slot 送到未搜尋的 expert。B／C／W 和 A 的 expert 集合不同的比例分別是 16.7%、16.7%、16.1%。

**六批 knee 資料並列**（各批不同的文章，各自配對；不合併、不跨批相減；95% CI）

| 對比 | 第 1 批 | 第 2 批 | 第 3 批 | 第 4 批 | 第 5 批 | 第 6 批（本輪） |
|---|---|---|---|---|---|---|
| B − A | −0.00120* | −0.00049 | −0.00084* | −0.00058 | −0.00057 | −0.00019 [−0.00085, +0.00048] |
| C − A | — | −0.00029 | −0.00039 | −0.00016 | — | −0.00042 [−0.00108, +0.00023] |
| W − A | — | — | — | — | — | −0.00021 [−0.00084, +0.00041] |

\* 該批的 95% CI 排除 0。其餘都包含 0，細節見各輪報告。

## 計時（`analysis.json` → `timing`）

**方式**
- B／C／W 各 3 次，交錯執行：B1 C1 W1｜C2 W2 B2｜W3 B3 C3。
- 每次一個 process，前後都有 CUDA synchronization，沒有清除系統快取；每次執行前，所有輸入與快取都 100% 在 page cache 中。
- 背景負載：每次執行前的 1 分鐘 load average 為 1.1–1.7。交錯執行能讓各方法承受相近的負載，但不能完全排除干擾。
- 控制區塊不計入方法時間。

**已有快取與 q0_knee 時的額外更新＋搜尋**（中位數，3 次；秒）

| 類別 | B | C | **W** |
|---|---:|---:|---:|
| 準備與載入（checkpoint、q0 量化、量化器初始化、設定） | 17.1（6.5＋7.1＋2.4＋1.1） | 17.0（6.4＋7.1＋2.4＋1.1） | 17.0（6.4＋7.1＋2.3＋1.1） |
| forward | 1.6 | 1.6 | 1.6 |
| backward | 2.4 | 2.4 | 2.4 |
| weight sensitivity | 27.3 | 27.3 | 27.5 |
| activation 梯度統計 | 3.4 | 3.4 | **0** |
| activation 誤差計算 | 12.5 | — | **0** |
| activation 快取讀取 | — | 0.38 | **0** |
| ΔW 快取讀取 | 12.0 | 11.9 | 11.7 |
| 組表 | 0.2 | 1.0 | 0.2 |
| 剪枝與搜尋 | 5.0 | 5.0 | 5.0 |
| 輸出、未分類 | 0.9 | 1.0 | 0.8 |
| **方法時間：中位數（範圍）** | **82.4（82.2–82.8）** | **70.9（70.8–71.0）** | **66.1（66.1–66.2）** |
| process 牆鐘：中位數 | 85.3 | 73.9 | 68.3 |
| 峰值 GPU／CPU（GB） | 4.21／14.5 | 4.21／15.6 | 4.18／14.5 |

**快取**：W 只需要 ΔW 快取（0.34 GB），和 B 相同；不需要 activation 誤差快取或梯度快取。

**從零到最終配置**

| | 秒 | 性質 |
|---|---:|---|
| **W：static BC（含 ΔW 快取）→ auto knee → W → 搜尋** | **240.1（process 243.3）** | 本輪單次實測 |
| W：各階段相加 | 238.0 | 估算：static 91.8＋auto 80.0（皆為歷史中位數）＋W 66.1 |
| A | 170.5 | 歷史值（單次實測） |
| B | 246.6 | 歷史值（單次實測） |
| C | 240.3 | 歷史值（單次實測） |
| E | 192.1 | 歷史值（單次實測） |

端到端 W 的分界時間：static 統計完成於 93.4 秒，auto knee 完成於 177.1 秒；峰值 GPU／CPU 4.18／14.8 GB。

## 資料（`segments.json`、`dedup_checks.json`）

**來源**：WikiText-103 raw train，使用嚴格文章邊界。

**排除**

| 理由 | 篇數 |
|---|---:|
| 標題或內容和 WikiText-2 重疊 | 613 |
| WikiText-103 內部重複 | 464 |
| 所有歷史 OLMoE 用過的 WikiText-103 文章 | 5474 |

- 5474 篇包含 E5 輪的 600 篇。這是實際掃描所有 manifest 得到的，只依路徑排除本輪自己的目錄。
- calibration 屬於 WikiText-2，已隨 WikiText-2 一起排除。

**抽樣**
- 在凍結之後才抽。
- 依 seed 23001 的順序看了 657 篇：接受 600 篇；跳過 54 篇近重複、3 篇太短。窗口起點用 seed 23002。
- 和歷史文章的重疊為 0。

## 研究成本（不屬於方法成本）

| 項目 | 時間 |
|---|---|
| plan | 43.5 秒 |
| controls | 25.0 秒 |
| refresh 計時 9 次 | 688.5 秒 |
| 端到端 W | 244.0 秒 |
| check | 59.9 秒 |
| 抽資料 | 187.0 秒 |
| 評估 | 1305.9 秒（FP32 123.5 秒、A／B／C／W 各約 269 秒、程式控制與重跑約 106 秒；峰值 GPU 7.7 GB） |
| 分析 | 17.0 秒 |

**磁碟**：`work/` 0.4 GB，主要是端到端 W 的 ΔW 快取副本與各次執行的表。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_knee_weight_sensitivity_20261023      # 新實驗請改用新目錄
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_knee_weight_sensitivity.py
$CPU $S --stage plan --out-dir $O
$GPU $S --stage controls --out-dir $O
$GPU $S --stage timing --phase refresh --out-dir $O    # 9 次，約 11 分鐘
$GPU $S --stage timing --phase e2e --out-dir $O        # 約 4 分鐘
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

- **預先登記**：`plan.json`、`plan.md`、`amendments.json`（空）、`input_hashes_before.json`。
- **各次執行**：`work/runs/{refresh_{B,C,W}_r*,e2e_W}/`（`run.json`、`table.json`、`config.json`）。`refresh_W_r1/weight_sens_stats.npz` 和 `weight_sens_stats_meta.json` 用於控制 5、6。
- **凍結**：`configs/{A,B,C,D4,D5,E,E5,W}.json`、`tables/W.json`、`freeze.json`。
- **控制**：`controls.json`、`check.json`、`controls_summary.json`。
- **資料**：`segments.json`、`tokens.npy`、`dedup_checks.json`、`data_freeze.json`。
- **結果**：`eval.json`、`routes_{fp32,A,B,C,W}.npz`、`per_article.csv`、`per_window.csv`、`analysis.json`。
- **時間**：`timing.json`、`timing.jsonl`。
