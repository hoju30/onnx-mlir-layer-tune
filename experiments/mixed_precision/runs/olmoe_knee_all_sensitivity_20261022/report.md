# OLMoE 正常 knee：全節點 sensitivity-only refresh（C）

日期：2026-10-22
腳本：`run_olmoe_knee_all_sensitivity.py`（plan → controls → refresh ×9 → 端到端 C → check → freeze → data → eval → analyze）
輸出：本目錄

**本輪的範圍**
- 在上一輪（`runs/olmoe_knee_selective_sensitivity_20261022`）凍結的 A／B／D 之外，補上 C：在同一個 q0_knee、同一組精確上限下，更新全部搜尋節點的 sensitivity，量化誤差沿用 static。
- 沒有重選 knee；沒有微調 router 或更新參數；沒有修改 BC 公式或 low-bit 門檻；沒有修改正式 pipeline 預設。
- 結論只適用於本模型、本 calibration、本 knee 和 GPU QDQ。

## 結論

**在這批新的 600 篇文章上，四個條件之間的差異都沒有確認。**

- 全節點 sensitivity-only（C）相對 static（A）的點估計是 −0.00029，但校正後的 CI 跨 0。
- 預先登記的三個主要比較（C − A、C − D、B − C）都包含 0，所以**本輪無法區分「更新範圍」和「activation error 更新」各自的作用**。
- 在這批文章上，**連完整 refresh 相對 static（B − A = −0.00049）也沒有確認**；上一輪 knee 那批是 −0.00120 並且確認。refresh 在 knee 的真實效果，可能比上一輪的點估計小。

| 條件 | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| FP32 | 2.97719 | 19.632 | — |
| A：static knee | 2.99240 | 19.933 | +0.01521 [+0.01411, +0.01635] |
| B：全節點 sensitivity＋activation error | 2.99191 | 19.924 | +0.01473 [+0.01360, +0.01586] |
| C：全節點 sensitivity，static 誤差 | 2.99211 | 19.928 | +0.01492 [+0.01380, +0.01604] |
| D：≤4-bit 節點 sensitivity | 2.99203 | 19.926 | +0.01484 [+0.01370, +0.01599] |

600 篇新的 WikiText-103 文章 × 4 個窗口；全部用原 router、各自的 dynamic route。

**預先登記的三個主要比較**（負值代表前者較好；以 Bonferroni 98.33% CI 判定方向）

| 對比（nat/token） | 平均 | 95% CI | **98.33% CI（校正）** | 較好／較差篇數 | 判定 |
|---|---:|---|---|---|---|
| C − A | −0.00029 | [−0.00096, +0.00039] | **[−0.00110, +0.00053]** | 310／290 | 未確認 |
| C − D | +0.00008 | [−0.00057, +0.00075] | **[−0.00072, +0.00090]** | 299／301 | 未確認 |
| B − C | −0.00020 | [−0.00079, +0.00040] | **[−0.00092, +0.00054]** | 303／297 | 未確認 |

**對照**（同一批文章，95% CI）

| 對比 | 平均 | 95% CI | 較好／較差篇數 | 判定 |
|---|---:|---|---|---|
| B − A | −0.00049 | [−0.00117, +0.00022] | 311／289 | 未確認 |
| D − A | −0.00037 | [−0.00102, +0.00028] | 302／298 | 未確認 |

**C 是否保留 B 的品質**
- C − B = +0.00020，校正 CI [−0.00054, +0.00092]（由 B − C 的校正區間反號）。
- 上界 +0.00092 大於 +0.0005，所以**非劣沒有成立**。
- 這不表示 C 比 B 差：校正 CI 也包含 0。

**套用預先登記的結論規則**：只有「C − A 未確認」這一條適用。C − D 和 B − C 的校正區間都包含 0，所以**無法區分原因**。其他三條規則（C 優於 A 且優於 D、C 優於 A 且對 B 非劣、B 優於 C）都不成立。

其他設定：10,000 次文章級 paired bootstrap，seed 22611；各差值的文章級 SD 為 0.0074–0.0086。

## 回答四個問題

**1. 全節點 sensitivity-only 在正常 knee 有沒有收益？**
- **本輪沒有確認。**
- 點估計 −0.00029，大約是 A 損傷的 1.9%；校正 CI [−0.00110, +0.00053]。
- 這個區間同時容許「沒有效果」和「約 0.001 的改善」，所以不能說 sensitivity 更新無用，也不能說它有用。

**2. 證據比較支持更新範圍的問題、error 更新的問題，還是仍無法區分？**
- **仍無法區分。**
  - 擴大更新範圍（C − D）：+0.00008，校正 CI ±0.0008 左右。
  - 再更新 activation error（B − C）：−0.00020，校正 CI 約 ±0.0007。
  - 兩者都包含 0。
- **這批文章上要分解的總效果本身也很小**：B − A 只有 −0.00049，而且未確認。
- **[推論] 樣本量**：依本輪觀察到的文章級 SD，要讓校正 CI 半寬縮到 0.0002（足以確認 −0.0002 的 B − C），需要約 7,800 篇文章；要確認 −0.0003 的 C − A，需要約 4,600 篇。這不是預先登記的分析，只說明 600 篇的解析度不夠。
- **配置面的觀察**（描述性，不是因果證據）：
  - C 和 B 只差 34 個投影：3 個 weight、33 個 activation 格式。
  - C 對 A 的 198 個改動裡，165 個在原低位元集合之外，B 也是 168／201。
  - 換句話說，在不刷新 error 的情況下，擴大範圍到全節點，就重現了 B 大部分的格式改動。但這些改動對 loss 的影響，本批資料解析不出來。

**3. C 相較 B 省時多少、保留多少品質？**
- **省時**（已有 static 快取與 q0_knee 時的額外時間，3 次中位數）：
  - C 68.8 秒、B 79.9 秒，**少 11.1 秒（−14%）**。
  - 省下的只有 activation 候選誤差（12.3 秒），另外多了快取讀取 0.37 秒和組表 0.8 秒。
  - 兩者的 weight sensitivity 都要 26 秒，固定成本也都約 26 秒。
- **保留的品質**：
  - 點估計上，C 相對 A 的改善約為 B 的 59%（−0.00029 對 −0.00049）。
  - 但兩者都未確認，C 對 B 的非劣也沒有成立，**實際保留多少品質，本輪無法判定**。

**4. 下一步值得研究選擇性更新，還是維持 static BC 預設？**
- **建議維持 static BC 預設，暫停這條 refresh／選擇性更新的研究線。**
- 理由一：在 knee，四個條件之間的差異都在 0.0005 nat/token 的量級。
  - 完整 refresh 在兩批文章上的點估計分別是 −0.00120（確認）和 −0.00049（未確認）。
  - 這個量級小於先前測得的跨後端實作差異（約 0.001–0.002）。
- 理由二：要可靠地分解這麼小的效果，需要數千篇文章，成本和可得的收益不成比例。
- 若要繼續，比研究選擇規則更優先的是兩件事，都需要另外預先登記：
  1. 先確認完整 refresh 在 knee 的效果在 native kernel 上存在；
  2. 以事先計算的樣本量，估計 B − A 的真實大小。

**不能由本輪宣稱**
- 低位元節點太少是 D 失效的唯一原因。
- 已分離 sensitivity 與 error 更新之間的所有交互作用。
- 不能把本批和上一輪的平均 NLL 直接相減比較；上表的 B − A 只是兩批各自的配對結果並列，見「和上一輪並列」。

## 條件與方法（`plan.json`）

**沿用上一輪的輸入**（hash 全部核對一致，`check.json` → `1_inputs`）
- q0_knee（= 凍結的 A）。
- `knee/knee.json` 的精確上限：weight bytes ≤ 4,524,359,631.556，BOPs ≤ 46,648,327,679,632.125。
- 凍結的 B、D 配置與成本表，以及低位元集合（169 個）。
- FP32 static 統計：static 表、ΔW 快取、activation 誤差快取。
- checkpoint 與 router、calibration、ranges、量化器、程式碼。

**C 的定義**
- **梯度**：全部 3051 個搜尋投影的 sensitivity 來自同一次 q0_knee calibration forward（q0_knee 自己的 dynamic route），反傳時：
  - 把記錄的路由當成常數；
  - activation QDQ 經 identity STE；
  - gate／up 共用輸入使用 total gradient。
- **weight 成本**：W_new(f) = Σ_c S_new,c × D_static,c(f)，和 B 的 weight 項逐位元相同。
- **activation 成本**：A_new(f) = Σ_slots,c [G2_new,c × var_static,c(f) + GS_new,c × μ_static,c(f)²]。
  - static 的平均與變異用 static routed rows 正規化；
  - 以原 BC 的運算順序計算。
- **範圍**：更新每個投影的所有候選格式。
- **搜尋**：在同一組上限下重新搜尋一次，所有節點都可以改格式。

**實作陷阱的處理**
- 上一輪的 `refresh()` 只有在 D 時才載入 activation 快取；`build()` 判斷是否只更新 sensitivity 的唯一依據，是有沒有傳入快取。
- 本輪的 C 明確傳入快取、不給目標集合。
- 每次執行都用計時器驗證快取確實被使用（`check.json` → `cache_use`）：
  - C 和 D：activation 誤差區塊為 0 秒，快取讀取與組表區塊大於 0；
  - B：相反。
- C 的 activation 列和 B 全部不同（3051 列），也和 static 全部不同。

**fallback**：C 的 3051 個投影都有 q0_knee calibration token，fallback 為 0。

## 控制（`check.json`、`controls.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 1. 輸入 | A／B／D 配置、B／D／static 表、knee、低位元集合、上一輪 check，以及 20 項重用依賴的 hash，全部和上一輪記錄一致 |
| 2. forward | 本輪重新跑的獨立 `dr.forward`，和上一輪的 CE、hidden state、路由逐位元相同；9 次 refresh 和端到端 C 的 forward 也逐位元相同 |
| 3. 梯度 | 9 次 refresh 和端到端 C 的 16 層輸入梯度 sha256，全部等於上一輪 B 的值；重算和 forward 的差為 0 |
| 4. 成本列 | C 的 weight 列 = B（全部 3051 列逐位元相同）；凍結的 D 的 169 個目標列 = C（逐位元相同） |
| 5. 獨立重算 | 用記錄的梯度、static 誤差快取和 ΔW，以 numpy 重算全部 3051 列：weight 最大相對差 0、activation 4.7e-16（門檻 1e-12）。改寫成 G2·var + GS·μ² 的形式差 5.6e-16；若誤用 q0_knee 的 rows 正規化，差異達 100%，表示這個檢查分辨得出錯誤的正規化 |
| 6. 有效性 | C 恰好涵蓋 3051 個搜尋投影；fallback 0；有限；格式合法 |
| 7. 上限 | C：4,421,058,560 bytes、46,648,180,867,072 BOPs，在精確上限內 |
| 8. 重複 | C 3 次的表和配置相同（主要配置 = 第 1 次）；本輪的 B、D 各 3 次和凍結的 B、D 逐位元相同；端到端 C 重現了 static 表、兩個快取、knee 和 C 的表與配置 |
| 9. 是否相同 | C 和 A、B、D 都不同，各自評估 |
| 評估 | 程式控制（上一輪的窗口 0–127）：FP32、A、B、D 的 CE 差 0；C 第一批重跑逐位元相同；router 未被替換 |

## 配置（`freeze.json`、`analysis.json`）

**實際資源**

| | weight bytes | BOPs | M_rel | B_rel（R） |
|---|---:|---:|---:|---:|
| A | 4,418,699,264 | 46,645,278,408,704 | 0.690593 | 0.7070606 |
| B | 4,420,796,416 | 46,647,774,019,584 | 0.690921 | 0.7070984 |
| C | 4,421,058,560 | 46,648,180,867,072 | 0.690962 | 0.7071046 |
| D | 4,422,893,568 | 46,647,753,048,064 | 0.691249 | 0.7070981 |

**格式變動**

| | 不同的投影 | weight | activation | 在原低位元集合內／外 |
|---|---:|---:|---:|---|
| C 對 A | 198 | 113 | 89 | 33／165 |
| B 對 A | 201 | 114 | 93 | 33／168 |
| D 對 A | 37 | 18 | 22 | 32／5 |
| C 對 B | 34 | 3 | 33 | |
| C 對 D | 166 | 95 | 73 | |

C 的 weight 改變：int6→int5 32、int5→int6 30、int4→int5 18、int5→int4 13、int6→int7 12、int7→int6 7、int7→int8 1。

**搜尋**：C 在自己的表上的目標值是 17.94290（LP 下界 17.94264），由 LP 接受；剪枝後 70,146 個選項；BOPs 用量 99.9997%。

**交叉目標值**（列：用哪張表；欄：哪個配置）。不同表的數字彼此不可比，也不代表品質。

| 表 | A | B | C | D |
|---|---:|---:|---:|---:|
| static | 18.29732 | 18.32288 | 18.32138 | 18.29394 |
| B | 18.01559 | 17.96180 | 17.98334 | 18.00874 |
| C | 17.98246 | 17.94847 | 17.94290 | 17.97559 |
| D | 18.30982 | 18.33187 | 18.33032 | 18.30320 |

**實際路由的計算成本**（評估資料，2400 個窗口）

| | 搜尋節點 BOPs，相對全 int8 | 含未搜尋 expert，相對全 int8 | 送到未搜尋 expert 的 token-slot | 搜尋 proxy BOPs（≤ 4.66483e13） |
|---|---:|---:|---:|---:|
| A | 0.67799 | 0.70815 | 161,727 | 4.66453e13 |
| B | 0.67805 | 0.70822 | 161,853 | 4.66478e13 |
| C | 0.67801 | 0.70817 | 161,745 | 4.66482e13 |
| D | 0.67828 | 0.70845 | 161,804 | 4.66478e13 |

- FP32 有 159,597 個 token-slot 送到未搜尋的 expert。
- 路由：和 FP32 的 expert 集合不同的比例約 23.4%；B／C／D 和 A 不同的比例分別是 16.7%、16.8%、15.1%。

**和上一輪並列**（不同的文章，各自配對；不合併，也不跨批相減）

| | 上一輪 knee（600 篇） | 本輪（600 篇） |
|---|---|---|
| B − A | −0.00120 [−0.00184, −0.00057] | −0.00049 [−0.00117, +0.00022] |
| D − A | +0.000001 [−0.00063, +0.00065] | −0.00037 [−0.00102, +0.00028] |

兩批的點估計大小不同，但 CI 互相重疊。[推論] 合起來看，knee 上完整 refresh 的效果大約落在「沒有效果」到「改善約 0.0018 nat/token」之間；以目前的樣本量無法再縮小。

## 計時（`analysis.json` → `timing`）

**方式**
- B／C／D 各 3 次，交錯執行：B1 C1 D1｜C2 D2 B2｜D3 B3 C3。
- 每次一個 process，前後都有 CUDA synchronization，沒有清除系統快取。
- 每次執行前，checkpoint、ΔW 快取、activation 快取都 100% 在 page cache 中。
- 控制區塊不計入方法時間。

**已有 static 快取與 q0_knee 時的額外更新＋搜尋**（中位數，3 次；秒）

| 類別 | B | C | D |
|---|---:|---:|---:|
| 準備與載入（checkpoint 6.3、q0 量化 7.1、量化器初始化 2.3、其他 1.1） | 16.8 | 16.8 | 17.0 |
| forward | 1.6 | 1.6 | 1.6 |
| backward | 2.4 | 2.4 | 2.4 |
| weight sensitivity | 26.3 | 26.2 | 0.8 |
| activation 梯度統計 | 3.2 | 3.2 | 0.15 |
| activation 候選誤差 | 12.3 | — | — |
| activation 快取讀取 | — | 0.37 | 0.03 |
| ΔW 快取讀取 | 11.4 | 11.2 | 0.8 |
| 組表 | 0.2 | 1.0 | 0.06 |
| 剪枝與搜尋 | 4.9 | 4.9 | 4.9 |
| 輸出、未分類 | 0.9 | 0.9 | 0.5 |
| **方法時間：中位數（範圍）** | **79.9（79.8–80.0）** | **68.8（68.1–68.8）** | **28.1（27.7–28.1）** |
| process 牆鐘：中位數 | 82.7 | 71.6 | 30.2 |
| 峰值 GPU／CPU（GB） | 4.21／14.5 | 4.21／15.7 | 4.14／14.6 |

**快取容量**：activation 誤差快取 1.17 GB（建立約 1 秒、寫入約 0.3 秒，只有 C／D 需要），ΔW 快取 0.34 GB（B／C／D 都需要）。

**從零到最終配置**

| | 秒 | 性質 |
|---|---:|---|
| **C：static（含兩個快取）→ auto knee → C → 搜尋** | **240.3（process 244.3）** | 本輪單次實測 |
| C：各階段中位數相加 | 241.7 | 估算（static 用 selective-sensitivity 的中位數、auto 用上一輪的中位數，加上本輪 C 的中位數） |
| A | 170.5 | 歷史值（上一輪單次實測） |
| B | 246.6 | 歷史值（上一輪單次實測） |
| D | 196.1 | 歷史值（上一輪單次實測） |

- 端到端 C 的分界時間：static 統計完成於 92.5 秒，快取寫完於 94.8 秒，auto knee 完成於 175.3 秒。
- 峰值 GPU／CPU 為 4.21／15.9 GB。

## 資料（`segments.json`、`dedup_checks.json`）

**來源**：WikiText-103 raw train，使用嚴格文章邊界。

**排除**

| 理由 | 篇數 |
|---|---:|
| 標題或內容和 WikiText-2 重疊 | 613 |
| WikiText-103 內部重複 | 464 |
| 所有歷史 OLMoE 用過的 WikiText-103 文章 | 3074 |

- 3074 篇包含上一輪 knee 的 600 篇。這是實際掃描所有 manifest 得到的，只依路徑排除本輪自己的目錄。
- calibration 屬於 WikiText-2，已隨 WikiText-2 一起排除。

**抽樣**
- 在凍結之後才抽。
- 依 seed 22601 的順序看了 630 篇：接受 600 篇；跳過 28 篇近重複、2 篇太短。窗口起點用 seed 22602。
- 和歷史文章的重疊為 0。

## 修訂（`amendments.json`）

兩次修訂都只改 check 的判定程式，發生在凍結和抽資料之前，不影響任何表、配置或計時。

1. **第一次 check 中止（KeyError）**：端到端的記錄沒有有限值欄位。
   - 有限值檢查改為只看 9 次 refresh；端到端 C 的表和 C1 逐位元相同，已由重複控制涵蓋。
   - 腳本 sha 從 `5f68b995…` 改為 `17a37112…`。
2. **第二次 check 的「快取使用」判定失敗**：
   - 原因：把 refresh 的判定規則（不得有 activation 誤差區塊）套用到端到端執行；而端到端裡 12.8 秒的 activation 誤差，正是建立 static 統計與快取的那一步。
   - 修正：端到端改為檢查「有快取讀取與組表區塊，且 C 表和 C1 逐位元相同」；9 次 refresh 維持原規則。
   - 失敗的輸出保留為 `check_failed_cacheuse_rule_20261022T1236.json`。
   - 腳本 sha 改為 `b3d1b362…`。

## 研究成本（不屬於方法成本）

| 項目 | 時間 |
|---|---|
| plan | 43.3 秒 |
| controls | 24.1 秒 |
| refresh 計時 9 次 | 558.7 秒 |
| 端到端 C | 245.0 秒 |
| check（最後一次） | 88.2 秒 |
| 抽資料 | 117.2 秒 |
| 評估 | 1300.1 秒（FP32 123.0 秒、A／B／C／D 各約 268 秒、程式控制約 84 秒、重跑 20.6 秒；峰值 GPU 7.7 GB） |
| 分析 | 17.0 秒 |

**磁碟**：`work/` 1.7 GB，主要是端到端 C 的快取副本，內容已確認和既有快取相同。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_knee_all_sensitivity_20261022      # 新實驗請改用新目錄
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_knee_all_sensitivity.py
$CPU $S --stage plan --out-dir $O
$GPU $S --stage controls --out-dir $O
$GPU $S --stage timing --phase refresh --out-dir $O    # 9 次，約 9 分鐘
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

- **預先登記與修訂**：`plan.json`（含上一輪 hash 核對）、`plan.md`、`amendments.json`、`input_hashes_before.json`。
- **各次執行**：`work/runs/{refresh_{B,C,D}_r*,e2e_C}/`（`run.json`、`table.json`、`config.json`；`refresh_C_r1/target_stats.npz` 用於控制 5）。
- **凍結**：`configs/{A,B,C,D}.json`、`tables/C.json`、`freeze.json`。
- **控制**：`controls.json`、`check.json`、`check_failed_cacheuse_rule_20261022T1236.json`、`controls_summary.json`。
- **資料**：`segments.json`、`tokens.npy`、`dedup_checks.json`、`data_freeze.json`。
- **結果**：`eval.json`、`routes_{fp32,A,B,C,D}.npz`、`per_article.csv`、`per_window.csv`、`analysis.json`。
- **時間**：`timing.json`、`timing.jsonl`。
