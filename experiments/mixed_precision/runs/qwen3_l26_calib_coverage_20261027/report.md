# L26 校準文章覆蓋：同樣 1280 tokens，單篇 10 窗口 vs 10 篇各 1 窗口

日期：2026-10-27。程式：`run_qwen3_l26_calib_coverage.py`。
規格：`reports/qwen3_l26_calib_coverage_prompt_20261027.md`，加上同日請求中多要求的項目：既有 proxy 分數、留一窗口、排序一致性、峰值記憶體。
計畫：`plan.json`／`plan.md`，03:16:16 寫入；本輪沒有 amendment。

範圍：Qwen3-0.6B 的 `/model/model/layers.26/mlp/down_proj/MatMul`。權重固定為 A 的 INT6，背景 F（其他 195 個投影依 A），activation 的 input 和 output 同格式。
選擇集為 f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16，fp32 只作數值控制。數值單位都是相對 f16 的 nat/token。

## 首頁：四個問題

**1. 原 C0 的 INT8 選擇是一般 single 組的常見現象，還是個案？**

在本輪的 pool 內是**個案**：
- 60 個新的單篇組中，只有 1 組選 INT8（`WT2.p2.single.a3`，C 上 INT8 = −0.00423）。INT8 在 C 上為負的單篇組只有 2/60。60 個多篇組全部沒有選 INT8，INT8 在 C 上全為正。
- 把每個來源的 300 個窗口合併後，INT8 在 C 上是 +0.0103（WT2）／+0.0096（WT103），接近新文章的 +0.0085。C0 的 −0.0009 落在單篇組分布的最負端。
- C0 是 WT2 validation 串接後的前 1280 tokens（從標題開始），新的單篇組則在文章內隨機起點取段。因此「這篇文章」和「這個起點」兩個因素本輪無法再拆開。

**2. 相同 1280-token 預算、相同文字總曝光下，多篇覆蓋有沒有降低新文章上的平均選擇傷害？**

- **主要比較達到預先登記標準**：WT2 的 L(multi) − L(single) = **−0.00037 [−0.00055, −0.00020]**（81／128 篇變好）。WT103 同向同量級：−0.00040 [−0.00062, −0.00019]。
- 但效果小，而且有四個限制：
  1. **兩種覆蓋相對 f16 都仍然有害**。WT2 單篇 +0.00175 [+0.00096, +0.00256]，多篇 +0.00138 [+0.00075, +0.00203]，所以是「減輕傷害」，不是勝過 f16。
  2. 改善來自少數 pick 的改變。WT2 30 組中：int8 1→0、p6e1 11→9、bf16 6→8、p8e1 0→1。
  3. WT2 三個 pool 中有一個反向：p0 為 +0.00053（多篇有 1 組選了 p8e1）。WT103 三個 pool 都是 −0.0004。
  4. **留一窗口的改選率在多篇組反而較高**：WT2 0.207 vs 0.173，WT103 0.193 vs 0.100。多篇組的 C margin 較小：WT2 0.00106 vs 0.00167。
- 依預先登記的判讀規則：「文章覆蓋對這個 L26 診斷有幫助」必須同時滿足新文章改善和改選率降低。後者沒有達到，所以**此結論不成立**。能說的只有：在這些 pool 中，平均選擇後果略好，但校準選擇沒有因此更穩定。

**3. WT2 與 WT103 的結果是否一致？來源差異與覆蓋差異能否分開描述？**

可以分開描述，兩者的方向一致：
- 同一覆蓋下，WT103 − WT2：單篇 −0.00002 [−0.00018, +0.00014]，多篇 −0.00005 [−0.00014, +0.00004]。兩者都是未定，不代表相等。
- 覆蓋效果在兩個來源都約為 −0.0004。
- 限制：新文章只來自 WT103 train。WT2 validation 是既有的開發 corpus，30 篇中有 10 篇與歷史 Qwen 窗口重疊（已記錄、未排除）。兩來源在 split、歷史使用和文章母體上都不同，所以不能把差異或「無差異」單獨歸因於 corpus。

**4. 原 ranges 固定時，這些資料選擇的改變是否足以避免 INT8 誤選？**

**INT8 誤選大致消失，但問題沒有解決**：
- 120 組中只有 1 組選 INT8。
- 但 p6e1 在 43/120 組被選中（實測 D_C 選擇器），而它在新文章上是 +0.0040 [+0.0018, +0.0062]，128 篇中 78 篇變差。
- 四格的平均後果都比 f16 差。新文章上集合內最好的仍是 f16，bf16 與之幾乎相同（+0.00003，CI 跨 0）。
- 在固定 ranges、只用 1280 tokens 的條件下，僅改變校準文字的組成，不足以避免有害的選擇。

## 1. 研究問題與界線

本輪先用**校準實測 loss** 選格式，以排除 proxy 近似這個因素。

- 規格檔要求「不算新 S_u／Khat」，而同日的請求要求同時報告既有 proxy。處理方式：proxy 在同一批窗口上另外計算，作為獨立標示的次要選擇器（第 7 節），完全不進入主要選擇或主要比較。此衝突已記錄在 `plan.json` 的 `spec_conflict`。
- 沒有修改正式 BC、sensitivity、量化誤差、A、量化器、ranges、MILP／knee 或 native。
- 沒有訓練、STE、HVP、JVP、margin 或 confidence 規則，也沒有擴到 L25。
- 本輪結果不代表 BC criterion、其他層或完整自動搜尋 pipeline 已被修正或驗證。

## 2. 資料來源與抽樣

| 項目 | WT2 validation | WT103 train |
|---|---|---|
| 檔案 | `gpt2_assets/data/validation-00000-of-00001.parquet` | `gpt2_assets/data_wt103/train-0000{0,1}-of-00002.parquet` |
| 文章切分 | 頂層標題、嚴格邊界（`split_articles`），共 60 篇 | 既有的已驗證 metadata |
| 排除規則 | 排除 C0（Homarus gammarus）；跳過 <1280 tokens、與 C0 或已選文章近重複 | 排除 9897 篇已用文章（`used_wt103_verified`）、WT2 同篇／近重複、歷史 Qwen 重疊、窗口 WT2 share ≥0.5、<1280 tokens |
| 順序 seed | 26102721 | 26102722 |
| 起點 seed | default_rng([26102723, 0]) | default_rng([26102723, 1]) |
| 抽取 | 走訪 31 篇，選入 30 篇（short 1） | 走訪 46 篇，選入 30 篇（歷史重疊 5、short 4、WT2 近重複 2、舊文章近重複 4、窗口 WT2 1） |
| 歷史重疊 | 10/30 篇與歷史 Qwen 窗口重疊（只記錄） | 無（已排除） |

- **共同規則**：文字 = `"\n\n".join(文章 rows)`；`add_special_tokens=False`，不加 BOS 或 chat template。起點在 [0, n−1280] 均勻抽取，取連續 1280 tokens，切成 10 個不重疊的 128-token 窗口 M[a,j]。每個窗口的位置都從 0 開始。
- **pool**：每個來源依選入順序切成 3 個 pool × 10 篇，pool 之間不重複。所有文章 id 互異，也不含 C0。
- **抽樣器驗證**：抽樣器先完整重現上一輪 L26 的新文章抽樣（`sampler_check.json`）：token、segments、跳過數都相同。之後才抽校準文章。
- **分組**：每個 pool 有 10 個單篇組（a = M[a,0..9]）和 10 個多篇組（第 j 列 = M[(j−r) mod 10, j]），組內順序不打亂。同一窗口在兩種覆蓋中位於同一 batch 列。
  - 兩來源 × 3 pool × 20 組 = 120 組，加上 C0 共 121 組。
  - 共 600 個不重複窗口（76,800 tokens），每組 1280 input tokens、1270 targets。
- **新文章（N）**：在全部 pick 凍結後才抽。
  - 128 篇 WT103 train 文章 × 2 窗口（每篇 254 targets），沿用既有 sampler 規則；順序 seed 26102724，起點 seed 26102725。
  - 排除了 9897 篇已用文章、本輪 30 篇 WT103 校準文章（同時加入近重複參考）、WT2 全部文章（含本輪 WT2 校準文章）及近重複。可選 17468 篇，走訪 152 篇，跳過 WT2 近重複 3、歷史重疊 8、舊／校準文章近重複 13。
  - 上一輪 L26 的 N 已經看過，本輪沒有使用。
- **未來稽核相容性**：校準 token 檔和 N token 檔都各自配有 `segments.json`。用未來稽核的規則自查後，300 個 WT2 窗口、300 個 WT103 窗口、256 個 N 窗口全部可解析（`calibration_data_freeze.json`、`data_freeze.json`）。

各 pool 的文章、起點、raw rows、字元範圍和 token hash 都記在 `calib/wt2/segments.json`、`calib/wt103/segments.json` 和 `groups.json`。

## 3. 固定條件

- **ranges**：`runs/native_eps_fix_20261014/qwen3/ranges_by_point.json`，sha256 `f93189ce…1371`。
  - LMReplay 內的 ranges 與檔案相等，雜湊為 `71302795…bce0`；每組做完後都重新核對，全程沒有改變。N 評估也核對過。
  - 目標的 x range 為 [−780.31, 119.28]，y range 為 [−246.57, 147.94]。
  - INT 的 scale／zero point：x int8 為 3.5278／93，y int8 為 1.5471／31（int6、int4 見 `controls_calib.json`）。
- **評估器**：`run_qwen3_l26_suffix_bridge.Env`。先做完整的 f16 forward（batch 8＋尾批 2），捕捉 x26、r26、block27 kwargs；每個格式再經目標 QDQ／GEMM、真實 block27、final norm、head 計算 CE。
- **其他設定**：A 配置、權重 INT6、Posit LUT（三種格式都驗證無誤）、FP32、eager、TF32 關閉、deterministic、`use_cache=False`。
- **環境**：RTX 3060、torch 2.14.0+cu130、CUDA 13.0。nvidia-smi 的 NVML 版本不符警告仍在，未處理，不影響 CUDA。

## 4. 控制（全部通過，`controls_calib.json`、`controls_eval.json`）

| 控制 | 結果 |
|---|---|
| C0 逐位置 CE（9 格式）vs 上一輪 F | 最大差 0（逐位元相同） |
| C0 的 S_u、Khat256、K2_actual vs 上一輪 | 全部 0 |
| C0 留一窗口 vs 事後診斷 | D 7 int8／3 f16、S_u+K2 7／3、P256 10／0，完全重現 |
| 每組 f16 重播 vs 完整 forward CE | 0 |
| 13 個控制組 × 9 格式：重播 vs 完整 forward（block27 輸入、final u、CE） | 全部 0 |
| kwargs | position_ids、cache_position = 0..127，因果 mask，無 cache，batch 為 8+2 |
| 分割 | 每個 pool 的 100 窗口在 single 和 multi 各用一次、同一列；每個 multi 組都有 10 篇不同文章 |
| 共享窗口（全部 9 格式、全部 600 窗口） | single vs multi 的 CE 最大差 0（逐位元相同）；proxy 也是 0 |
| 每個 pool 的 pooled d_C 表 | single 與 multi 的差 ≤1.1e−16，argmin 相同 |
| 其他 | f16 的 proxy 恆為 0、位置 127 的 g = 0、P0+Rest = All、最後重跑 C0 完全重現、hook／權重／activation map 與配置相同 |

pooled 表相同是數學必然。本輪檢查的是「每次只用 10 個窗口時」的選擇，而不是 pooled 表。

## 5. 校準結果（D_C 選擇器，`frozen_picks.json`、`scores_C.csv`、`windows_C.csv`）

| 格 | 選擇頻率（30 組） | pick 的 C 值平均 | 平均 margin | 留一窗口改選率 | 有任何改選的組 | INT8 在 C 的範圍 |
|---|---|---:|---:|---:|---:|---|
| WT2 單篇 | f16 12、p6e1 11、bf16 6、int8 1 | −0.00160 | 0.00167 | 0.173 | 20 | −0.0042 … +0.0200 |
| WT2 多篇 | f16 12、p6e1 9、bf16 8、p8e1 1 | −0.00097 | 0.00106 | 0.207 | 21 | +0.0002 … +0.0164 |
| WT103 單篇 | p6e1 13、f16 11、bf16 6 | −0.00235 | 0.00240 | 0.100 | 13 | −0.0009 … +0.0210 |
| WT103 多篇 | f16 16、p6e1 10、bf16 4 | −0.00127 | 0.00135 | 0.193 | 20 | +0.0020 … +0.0168 |
| C0 | int8 | −0.00090 | 0.00090 | 0.3（7 int8／3 f16） | — | −0.0009 |

- **多篇組的 C 值較溫和**：pick 的 C 值較不負、margin 較小，INT8 也不再有負值。
- **代價是改選率較高**：多篇組刪掉一個窗口就容易改選。
  - 事後描述：排除 f16↔bf16 這種對新文章幾乎無害的互換後，改選率為 WT2 0.120（單篇）vs 0.133（多篇），WT103 0.047 vs 0.130，多篇仍較高。
- **pooled 參考**：每個 pool 的 100 窗口合併後（等於 12,800 tokens，不是本輪預算），argmin 在 5/6 個 pool 是 f16，WT2.p1 是 p6e1。

## 6. 新文章（N，128 篇，F 背景，`N_truth.csv`、`stats.json`、`groups_summary.csv`）

各格式相對 f16（All）：

| 格式 | mean [95% CI] | 變差／變好篇數 |
|---|---|---|
| bf16 | +0.00003 [−0.00004, +0.00011] | 70／58 |
| p6e1 | +0.00398 [+0.00184, +0.00617] | 78／50 |
| p8e1 | +0.00528 [+0.00437, +0.00616] | 106／22 |
| int8 | +0.00853 [+0.00663, +0.01051] | 99／29 |
| int6 | +0.06673 | 127／1 |
| p4e1 | +0.06676 | 123／5 |
| int4 | +0.30888 | 128／0 |
| fp32（控制，不在選擇集） | −0.00006 [−0.00011, −0.00001] | — |

N 的平均最佳（描述性）為 f16，所以 regret 等於 L 本身。

各格的平均選擇後果 L（D_C 選擇器）：

| 格 | L vs f16 | L − C0 int8 | 組平均有害篇數 | Kendall τ_b（C vs N） | N 決定性配對 對／錯（平均） |
|---|---|---|---:|---:|---|
| WT2 單篇 | +0.00175 [+0.00096, +0.00256] | −0.00678 [−0.00882, −0.00474] | 45.9 | 0.824 | 23.77／1.23 |
| WT2 多篇 | +0.00138 [+0.00075, +0.00203] | −0.00715 [−0.00921, −0.00511] | 45.6 | 0.840 | 23.90／1.10 |
| WT103 單篇 | +0.00173 [+0.00081, +0.00268] | −0.00680 [−0.00894, −0.00464] | 47.8 | 0.807 | 23.60／1.40 |
| WT103 多篇 | +0.00133 [+0.00062, +0.00206] | −0.00720 [−0.00928, −0.00513] | 35.3 | 0.857 | 23.97／1.03 |
| C0（int8） | +0.00853 | 0 | 99 | — | — |

覆蓋與來源比較：

| 比較 | mean [95% CI] | 判讀 |
|---|---|---|
| **主要：WT2 多篇 − 單篇** | **−0.00037 [−0.00055, −0.00020]** | 達到預先登記的改善標準 |
| WT2.p0 | +0.00053 [+0.00044, +0.00062] | 反向（多篇有 1 組選 p8e1） |
| WT2.p1 | −0.00040 [−0.00062, −0.00018] | |
| WT2.p2 | −0.00125 [−0.00155, −0.00095] | 單篇的 int8 組消失 |
| WT103 多篇 − 單篇 | −0.00040 [−0.00062, −0.00019] | 三個 pool 各 −0.0004 |
| WT103 − WT2（單篇） | −0.00002 [−0.00018, +0.00014] | 未定 |
| WT103 − WT2（多篇） | −0.00005 [−0.00014, +0.00004] | 未定 |

- 排序一致性（描述性）：多篇組的 τ 略高、錯的配對略少，這只是描述，不作檢定。
- **統計說明**：bootstrap 只反映「固定這 6 個 pool 和這些 pick 時」新文章的不確定性。
  - 30 個組不是 30 個獨立的校準集，多篇組彼此共享文章。主要效果約等於少數幾個 pick 的改變。
  - 所以 CI 不能當作校準抽樣的不確定性，也不保證換一批 pool 仍然改善。

## 7. 既有 proxy（次要選擇器，不混為一法）

| 選擇器 | 格 | 選擇頻率 | L vs f16 | 多篇 − 單篇 | 改選率 |
|---|---|---|---|---|---:|
| P256 | WT2 單篇 | p6e1 19、f16 4、bf16 4、p8e1 2、int8 1 | +0.00316 | | 0.120 |
| P256 | WT2 多篇 | p6e1 21、f16 5、bf16 3、p8e1 1 | +0.00296 | −0.00020 [−0.00037, −0.00002] | 0.130 |
| P256 | WT103 單篇 | p6e1 17、f16 9、bf16 4 | +0.00226 | | 0.130 |
| P256 | WT103 多篇 | p6e1 19、f16 9、bf16 2 | +0.00252 | +0.00026 [+0.00012, +0.00041] | 0.200 |
| S_u+K2 | WT2 單篇 | p6e1 13、f16 10、bf16 6、int8 1 | +0.00201 | | 0.157 |
| S_u+K2 | WT2 多篇 | f16 11、p6e1 11、bf16 7、p8e1 1 | +0.00164 | −0.00037 [−0.00055, −0.00020] | 0.187 |
| S_u+K2 | WT103 單篇 | p6e1 13、f16 11、bf16 6 | +0.00173 | | 0.093 |
| S_u+K2 | WT103 多篇 | f16 14、p6e1 13、bf16 3 | +0.00173 | −0.00000 | 0.213 |

- P256 的樂觀偏差讓它更常選 p6e1，後果在四格都比實測 D_C 差。P256 的覆蓋效果在兩來源方向相反。
- S_u+K2 接近實測，但在 WT103 沒有覆蓋效果。
- proxy 結果不改變第 6 節的主要判讀。

## 8. 校準到新文章的位移（每來源 pooled 300 窗口，`stats.json` 的 `C_to_N_shift`）

| 格式 | C0 | WT2 pooled | WT103 pooled | N |
|---|---:|---:|---:|---:|
| int8 | −0.00090 | +0.01025 | +0.00959 | +0.00853 |
| p6e1 | +0.01018 | +0.00290 | +0.00214 | +0.00398 |
| p8e1 | +0.00345 | +0.00590 | +0.00627 | +0.00528 |
| bf16 | +0.00027 | +0.00011 | +0.00012 | +0.00003 |

- **p6e1 的 P0／Rest 結構相同**：p6e1 在 P0 為負（WT2 −0.0058、WT103 −0.0052、N −0.0052），在 Rest 為正（+0.0087／+0.0074／N +0.0092）。
- 只用 10 個窗口時，這兩項的抵銷容易偏向負值，因此常被選中。
- INT8 的 P0 很小，主要是 Rest 的傷害（N +0.0084）。C0 的 Rest 為 −0.0010，屬於異常。
- 這裡的 P0／Rest 只指最終 CE 的位置，不是源頭歸因。

## 9. 成本與規模（`timing.json`）

| 階段 | 時間 | 說明 |
|---|---:|---|
| 歷史稽核 | 37 s | 72 個檔案、7105 個窗口 |
| 校準抽樣 | 131 s | 語料載入 30 s、抽樣器重現檢查 84 s、WT2 0.5 s、WT103 15 s |
| GPU 校準 | 512 s | 121 組、1210 個組窗口 × 9 格式，明細見下表 |
| 新文章抽樣 | 111 s | |
| 新文章評估 | 52 s | 256 窗口 × 9 格式 |
| 分析 | 2 s | |
| 合計 | 約 14.5 分鐘 | |

GPU 校準 512 s 的明細：

| 部分 | 時間 | 每組 |
|---|---:|---:|
| 實測 CE 路徑：prefix 76 s、候選後綴 15 s、head CE 73 s | 164 s | 1.36 s |
| proxy：梯度 16 s、proxy 239 s | 255 s | 2.1 s |
| 重播 vs 完整 forward 控制（13 組） | 74 s | |
| 初始化與設定背景 | 15 s | |

- 峰值 VRAM（torch allocated）：校準 6.64 GB，評估 4.39 GB；峰值 RSS 5.3 GB。
- **規模**：不重複窗口 600＋C0 10；實際重播的組窗口 1210（每個新窗口在兩種覆蓋各算一次）。
- 這是實測 loss 的診斷成本，不代表未來 BC／P256 更新的成本。也不代表建議的 pipeline 每次要用 600 個窗口。

## 10. 限制

- **pool 數量**：只有 3 個 pool／來源，N 的 CI 不包含校準抽樣的變異。WT2 有 1/3 個 pool 反向。
- **因素混雜**：
  - C0 的特殊性同時包含「這篇文章」和「從標題開始」，本輪無法拆開。
  - 兩個來源在 split、歷史使用、長度分布上都不同。
  - 新文章只來自 WT103。
- **WT2 的使用狀態**：WT2 validation 是既有的開發 corpus，10/30 篇曾出現在歷史 Qwen 窗口。不宣稱校準 corpus 是全新的。
- **結論範圍**：結論只限於長度 ≥1280 tokens 的文章、這個節點（L26 Down、INT6 權重、F 背景）以及固定 ranges。
- **冒煙測試**：03:11–03:15 在 scratchpad 跑 plan、audit、calibdata 和只含 C0 的 calib。當時看到了抽中的校準文章名單和 C0 的已知結果，沒有看到任何新組的 loss 或新文章。冒煙後修了 audit 的 tokenizers 匯入順序，並用隨機數做了分組／控制／分析的離線測試。這些都在 plan 寫入之前，且已記錄在 plan。

## 11. 結論

在固定 ranges 和固定 1280 tokens 的條件下：

- **C0 的 INT8 選擇是個案**。在新抽的 60 個單篇組中只出現 1 次，多篇組中 0 次。
- **多篇覆蓋小幅減輕了平均選擇傷害**：WT2 −0.00037（主要標準成立），WT103 −0.00040。但它沒有讓留一窗口的選擇更穩定（反而較不穩），兩種覆蓋也都仍比 f16 差。
- 依預先登記的雙條件規則，本輪**不能說「文章覆蓋對這個 L26 診斷有幫助」**。
- 主要的殘餘問題是 **p6e1**：P0 為負、Rest 為正，10 個窗口時容易被抵銷成負值，在約 1/3 的組被選中，而它在新文章上有害。

下一步需要另外預先登記，本輪沒有執行，也不在本輪擴大。可以考慮兩個方向：
1. 在同一資料協定下，測試更大的校準預算（例如多個 pool 合併）的選擇後果。
2. 把 p6e1 的 P0 抵銷當作單獨的問題處理。

## 重現（在 `experiments/mixed_precision` 下執行）

```bash
O=runs/qwen3_l26_calib_coverage_20261027
../../venv/bin/python run_qwen3_l26_calib_coverage.py --stage plan --out-dir $O --smoke-note "<見 plan.json 的 smoke 欄位>"
../../venv/bin/python run_qwen3_l26_calib_coverage.py --stage audit --out-dir $O
../../venv/bin/python run_qwen3_l26_calib_coverage.py --stage calibdata --out-dir $O
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python run_qwen3_l26_calib_coverage.py --stage calib --out-dir $O
../../venv/bin/python run_qwen3_l26_calib_coverage.py --stage data --out-dir $O
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python run_qwen3_l26_calib_coverage.py --stage eval --out-dir $O
../../venv/bin/python run_qwen3_l26_calib_coverage.py --stage analyze --out-dir $O
```

腳本 sha256 為 `77cafa4c…`，每個階段都記錄在 `timing.jsonl`。

## 檔案

- **計畫**：`plan.json`／`plan.md`（含 ranges hash、seed、容差、冒煙說明）、`sampler_check.json`、`audit.json`。
- **校準資料**：`calib/{wt2,wt103}/{tokens.npy,segments.json}`、`groups.json`、`calibration_exclusions.json`、`calibration_data_freeze.json`。
- **校準結果**：`controls_calib.json`、`C_values.npz`（每組 [9 格式, 10 窗口, 127] 的 CE／S_u／Khat256／K2_actual）、`scores_C.csv`、`windows_C.csv`、`frozen_picks.json`（含留一窗口結果）、`freeze.json`。
- **新文章**：`tokens.npy`、`segments.json`、`exclusions.json`、`data_freeze.json`、`eval_ce_pos.npz`、`controls_eval.json`。
- **統計**：`stats.json`、`N_truth.csv`、`N_article_D.npz`、`groups_summary.csv`。
- **時間**：`timing.json`、`timing.jsonl`、`logs/`。
