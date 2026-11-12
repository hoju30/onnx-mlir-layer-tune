# OLMoE aggressive 資源點完整配置 A／B：原 BC vs Down-input 變異項逐位置配對

- 日期：2026-10-23
- 程式：`run_olmoe_pair_aggressive_ab.py`
  - plan 到 data 階段：sha256 `4cdcc5cd…`；
  - eval 起：`1cb5d3ab…`，只多一行建立 `work/` 目錄；見第 8 節。
- 預先登記：`plan.json`、`plan.md`（17:20:48），在搜尋之前。
- 凍結：
  - `freeze.json`（17:21:11）：A／B 配置與表的 hash；
  - `data_freeze.json`（17:24:50）：600 篇新文章。
- 正式 pipeline 預設不變；沒有 refresh、router fine-tune、spike-only 選點或新係數。

## 結論摘要

**唯一主要比較：Δ = NLL(B) − NLL(A) = −2.07e-4，95% CI [−1.01e-3, +6.09e-4]，CI 含 0。**

依預先登記的判讀：**改善未確認，保留原 BC，這條配對方向先收斂。** 「沒有顯著變差」不當作等效或採用的證據。

**問題 1：低位元比例實際提高多少？沒有提高，反而略降。**

| 項目 | A | B |
|---|---|---|
| Down activation ≤6 bit | 12.19%（124 / 1017） | 10.72%（109 / 1017） |
| Down activation ≤5 bit | 1.18%（12） | 1.08%（11） |
| Down activation ≤4 bit | 0.29%（3） | 0.29%（3） |
| 全部 activation ≤6 bit | 37.36% | 36.74% |
| 全部 weight ≤4 bit | 38.05% | 38.09%（+1 個投影） |

**問題 2：配對評分在此點是否改善完整模型？未確認。**
- Δ 的 CI 含 0；B 較好的文章 307 篇、較差的 293 篇。

  | 配置 | NLL | PPL |
  |---|---|---|
  | FP32 | 2.95455 | 19.193 |
  | A | 2.98870 | 19.860 |
  | B | 2.98849 | 19.856 |

- 相對量化損傷的改變是 −0.6%，只是點估計。

**問題 3：改變了哪些資源分配？**
- 37 個投影不同，分屬 37 個 expert：只改 activation 33 個、只改 weight 3 個、兩者都改 1 個。其中 Down 31 個、Gate 4 個、Up 2 個。
- B 把 Down activation 往高精度移：
  - p6e1 → p8e1：14 個；
  - p8e1 → int8：5 個；
  - p6e1 → int7：4 個；
  - int6 → int7：3 個。
- 代價由以下兩處支付：
  - 2 個極端 spike expert 的 f16 降為 p8e1：L1.E11、L6.E59；
  - 少數 weight 降位元：int5 → int4 共 2 個，例如 L10.E35 down、L4.E36 gate。
- 兩者的 size 都用滿（餘量 0）。BOPs 用量：A 99.988%，B 99.994%。

**問題 4：品質收益值得額外搜尋時間嗎？**
- 在此點，B 的搜尋沒有額外時間：A、B 都由 LP 解直接接受，求解約 0.9 s，加上剪枝約 4.1 s。
- 額外的 offline 成本只來自配對統計（上一輪實測：完整建表 +3.4 s，或增量版 17.7 s）。
- 但品質收益未確認，所以不值得接入。

**問題 5：是否有足夠理由進入自動 knee 驗證？沒有。**
- 依預先登記的規則，CI 含 0 時保留 BC，配對方向先收斂。
- 正常 knee 的上一輪也未確認改善。兩批是不同文章，不可相減比較資源點的交互作用。

## 1. A／B 與資源點

| 項目 | 內容 |
|---|---|
| A | `runs/olmoe_proxy_refresh_20261022/table_static.json`（sha `6d9c52ef…`）＋歷史 aggressive 配置 `configs/q0.json`（sha `2fe46627…`） |
| B | 重用 `runs/olmoe_pair_knee_ab_20261023/tables_B.json`（sha `0e77ca31…`，與上一輪 freeze 相同）。只改 1017 個 Down-input 的 activation 變異項：Σ_c G2_c/n·Σ_t r² 換成 Σ_c Σ_t g²r²；其他全部不變 |
| 資源上限 | 讀 `runs/olmoe_proxy_refresh_20261022/plan.json["budget"]`：max_bytes = 3804495872.0、max_bops = 39226380910592（R ≈ 0.594603） |

- 上限直接使用檔案中的精確值，不以配置的實際用量重新定義。
- 沒有建新的 frontier、沒有重選 knee、沒有換資源點。

## 2. 控制

**表與搜尋**

| 檢查 | 結果 |
|---|---|
| A、B 表與 q0 的 hash | 與 plan 及上一輪 freeze 相同 |
| B 表 weight 列 | 等於 A |
| B 表非 Down activation 列 | 等於 A |
| B 表 Down 列 | 1009 / 1017 改變；有限、格式完整 |
| A 表搜尋 | 重現歷史 q0（配置完全相同；目標值 42.20031180706998 與歷史紀錄相同） |
| A、B 配置 | 格式合法、完整，都在精確上限內 |

**評估**

| 檢查 | 結果 |
|---|---|
| 程式控制：q0 在 proxy_refresh_confirm 前 128 窗口 | 與舊紀錄 CE 差 0.0 |
| B 前 128 窗口重跑 | CE 差 0；hidden 128 / 128 逐位元相同；routes 相同 |
| 有限性、hook 審核 | 通過 |
| 輸入檔與模型檔 hash | 前後一致 |

## 3. 搜尋與下界

A、B 都用同一套 LP-first 規則：支配剪枝 → LP → 取整＋修復，若差距 ≤ max(1e-3 × 目標, 1e-6 × scale) 就接受。兩者都在 LP 階段被接受，**沒有使用 MILP 或 1-ppm 重試**。

| | A | B |
|---|---|---|
| 路徑 | LP（取整解被接受） | LP（取整解被接受） |
| 最終目標值（自身表） | 42.200312 | 42.603184 |
| 原始預算的 LP 下界 | 42.182707 | 42.594877 |
| 最終解 − 原始 LP 下界 | 0.0176（相對 4.2e-4） | 0.0083（相對 1.9e-4） |
| 接受容許值 | 0.0422 | 0.0426 |
| 是否證明原始預算最佳 | 否 | 否 |
| fractional 節點／修復交換 | 1／0 | 1／0 |
| 剪枝／LP 求解時間 | 4.1 s／0.9 s | 4.2 s／0.9 s |
| size 用量 | 100.000%（餘量 0 bytes） | 100.000%（餘量 0 bytes） |
| BOPs 用量 | 99.9876%（餘量 4.88e9） | 99.9941%（餘量 2.31e9） |

**交叉評分**（不同表的絕對值不可比較）

| 表 | A 配置 | B 配置 |
|---|---|---|
| A 表 | 42.2003 | 42.2369 |
| B 表 | 42.7844 | 42.6032 |

各表上都是自己的解較低，不屬於「搜尋受限」。

## 4. 配置差異（`config_diff.json`）

**改變的投影**

| 分類 | 細項 |
|---|---|
| 總數 | 37 個投影，分屬 37 個 expert |
| 改變內容 | 只改 activation 33、只改 weight 3、兩者都改 1 |
| 投影類型 | down_proj 31、gate_proj 4、up_proj 2 |
| 層別 | 前段（L0–L1）5、中段 23、後段（L14–L15）9 |
| 強 spike expert（R_token ≥ discovery p90） | 7 個 |

**格式轉換**

| 類型 | 轉換（數量） |
|---|---|
| activation | p6e1 → p8e1（14）、p8e1 → int8（5）、p6e1 → int7（4）、int6 → int7（3）、f16 → p8e1（2）、int7 → p8e1（2）、p8e1 → p6e1（2）、int5 → int6（1）、int7 → int8（1） |
| weight | int5 → int4（2）、int4 → int5（1）、int5 → int6（1） |

**A 原本為 f16 的 Down activation（13 個）**
- B 只降了 2 個：L1.E11（weight 也由 int5 → int6）、L6.E59。
- 其餘 11 個維持 f16：L1.E9、L1.E18、L2.E30、L0.E52、L4.E27、L10.E4、L11.E56、L12.E59、L13.E2、L14.E60、L15.E1。
- 全部 activation 中，f16 從 23 個變為 21 個。

**Down activation 分布**

| 格式 | A | B |
|---|---|---|
| p8e1 | 781 | 790 |
| p6e1 | 84 | 72 |
| int8 | 30 | 36 |
| int7 | 69 | 71 |
| int6 | 28 | 26 |
| int5 | 9 | 8 |
| f16 | 13 | 11 |
| int4 | 2 | 2 |
| p4e1 | 1 | 1 |

低位元比例見結論摘要第 1 點。

**全部 activation 的格式家族**：Posit 2703 → 2698、INT 325 → 332、FP 23 → 21。

**weight 分布幾乎不變**：int4 1161 → 1162、int5 1549 → 1547、int6 310 → 311。

這是在共用預算下對全模型的重新分配：只改 Down 的評分，也讓 4 個 Gate、2 個 Up 投影改變了。

## 5. 新資料評估（600 篇 × 4 窗口 × 128 token）

**資料**
- 使用 `wt_manifest_scan.used_wt103_verified`，在資料階段重新計算，排除 7121 篇歷史文章。這包括剛完成的正常 knee A／B，與 plan 時的計數一致，沒有硬編碼。
- 也排除 WikiText-2 與近似重複；跳過近似重複 64 篇、太短 5 篇。
- 與歷史文章的 content hash 交集為 0（已直接核對）。

**主要結果**（每篇先彙整 4 個窗口，再做文章 paired bootstrap，10000 次）

| 指標 | 值 |
|---|---|
| Δ = NLL(B) − NLL(A) | **−2.07e-4** |
| 95% CI | **[−1.01e-3, +6.09e-4]**，含 0 → **改善未確認** |
| 文章間標準差 | 1.02e-2 |
| B 較好／較差 | 307／293 篇 |

**各自的結果**

| 條件 | NLL（token 加權） | PPL | 相對 FP32 的損傷（文章平均，95% CI） |
|---|---|---|---|
| FP32 | 2.954553 | 19.1931 | – |
| A | 2.988700 | 19.8599 | 3.415e-2 [3.261e-2, 3.568e-2] |
| B | 2.988494 | 19.8558 | 3.394e-2 [3.237e-2, 3.548e-2] |

相對量化損傷的改變：−0.6%（點估計，未確認）。

**dynamic route 下的實際工作量**（診斷，不是主要比較）

| 項目 | A | B |
|---|---|---|
| BOPs proxy（相對 int8 × fp32 計數） | 0.56142 | 0.56139 |
| 路由到未搜尋 expert 的 token | 164,978 | 164,931 |

- FP32 下路由到未搜尋 expert 的 token 為 161,535。
- B 與 A 的路由集合在 18.7% 的（token, 層）上不同。

## 6. 成本

**本輪實測**

| 階段 | 時間 |
|---|---|
| 讀表、驗證與準備（prep） | 1.8 s（讀兩張表 0.06 s） |
| 剪枝＋搜尋（A 與 B） | A 4.1 + 0.9 s，B 4.2 + 0.9 s；階段總計 12.0 s |
| 抽資料 | 218 s |
| 最終評估（另列） | FP32 123.7 s、A 269.9 s、B 268.2 s、B 重跑 20.9 s、程式控制 21.1 s；GPU 峰值 7.7 GB |

**從頭成本估計**
- 標示：「歷史建表實測（正常 knee A／B，2 次交錯的中位數）＋本輪搜尋實測」，**不是本輪的端到端計時**。
- A：92.2 + 5.0 ≈ 97 s。
- B 從頭：95.7 + 5.1 ≈ 101 s。
- B 增量（重用 static 表與 cache）：17.7 + 5.1 ≈ 23 s。
- 本輪重用了既有的 B 表，但這不代表實際部署時建表成本為零。
- 也不宣稱 ONNX-MLIR native 整合或低位元 kernel 加速。

## 7. 判讀

**依預先登記的規則**
- CI 含 0：改善未確認，保留原 BC，這條配對方向先收斂。
- 不進入自動 knee 驗證，也不改預設。

**控制與搜尋**
- 都通過。兩者走同樣的 LP 接受路徑，都沒有證明最佳，資源都在上限內，所以不是「比較受限」。

**描述上**
- B 在此點的主要改變是讓 Down activation 更多用 p8e1／int8／int7，少用 p6e1。
- 只釋出 2 個 f16，並由少量 weight 降位元支付；低位元比例反而略降。
- 整體改變很小（37 / 3051 個投影），相對文章間的變異（標準差 1.0e-2），600 篇文章無法分辨。

**跨資源點**
- 不把本輪與正常 knee 的 NLL 相減來推論資源點的交互作用：兩批文章不同。

## 8. 修正紀錄（`amendments.json`）

- 時間：17:26，在任何評估 forward 之前。
- 原因：eval 在初始化量化器時停止。`dev_quant` 要寫入 `out/work`，而本輪沒有建立這個目錄。
- 影響：沒有任何 forward 執行，也沒有寫出評估結果；失敗的 log 保留為 `logs/eval_failed_missing_workdir.log`。
- 修正：eval 先建立 `work/`，再重跑。配置、資料與 plan 都沒有改變。
- script sha：`4cdcc5cd…` → `1cb5d3ab…`。

**另：上一輪報告的下界更正**
- 在 `runs/olmoe_pair_knee_ab_20261023/report.md` 開頭加註，並新增 `correction_lower_bound_20261023.json`（含原始 hash）。
- 正常 knee 的 B 不是「已證明最佳」：gap 0 屬於預算收緊 1 ppm 的重試問題。
- 相對原始預算的 LP 下界 18.784720，B 最終解的差距是 3.5e-5（相對 1.9e-6）。
- 原結論不變。

## 9. 重現

```
cd experiments/mixed_precision
O=runs/olmoe_pair_aggressive_ab_20261023
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_pair_aggressive_ab.py
$CPU $S --stage correct_prev --out-dir $O     # one-time note in the normal-knee report
$CPU $S --stage plan --out-dir $O
$CPU $S --stage prep --out-dir $O
$CPU $S --stage search --out-dir $O
$CPU $S --stage configs --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage eval --out-dir $O
$CPU $S --stage analyze --out-dir $O
$CPU $S --stage figures --out-dir $O
```

各階段都拒絕覆寫既有輸出。

### 主要 hash（sha256 前 16 位）

| 檔案 | hash |
|---|---|
| `plan.json` | `06f3443ff002bf76` |
| `prep.json` | `53ae5333e721e711` |
| `search.json` | `2b085f1e21aa551d` |
| `config_diff.json` | `20c1604c0a4270bf` |
| `configs/A.json` | `2fe46627745c06c4`（= 歷史 q0） |
| `configs/B.json` | `bc2312db6d4c1e30` |
| `freeze.json` | `473ca6f9f540ba94` |
| `tokens.npy` | `a8bdfa618f75e275` |
| `segments.json` | `efe4999bc2ab5dfa` |
| `eval.json` | `6505a755096f6150` |
| `analysis.json` | `46b673135c9c3501` |
| `per_article.csv` | `d2f72965c0932bca` |
| `amendments.json` | `9e0bad7edaec931f` |

### 輸出檔

| 類別 | 檔案 |
|---|---|
| 預先登記與凍結 | `plan.json`、`plan.md`、`freeze.json`、`data_freeze.json` |
| 表驗證 | `prep.json` |
| 配置與搜尋 | `configs/A.json`、`configs/B.json`、`search.json`（含下界拆分）、`logs/A_lpfirst_001_lp.log`、`logs/B_lpfirst_001_lp.log`、`config_diff.json` |
| 資料與評估 | `segments.json`、`tokens.npy`、`eval.json`、`routes_{fp32,A,B}.npz` |
| 分析 | `analysis.json`、`per_article.csv`、`per_window.csv` |
| 圖表 | `figures/transitions.png`、`figures/per_article_delta.png` |
| 計時與紀錄 | `timing.json`、`timing.jsonl`、`amendments.json`、`logs/` |
