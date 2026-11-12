# Qwen3-0.6B：branch 與 total 輸入敏感度的全模型對照

日期：2026-10-14　分支：`posit-work-continue`（HEAD `2a784020`，加上未提交的修改，見第 9 節）　腳本：`experiments/mixed_precision/run_qwen_branch_ablation.py`

## 結論

**研究問題**：在目前 per-consumer QDQ 的量化方式下，每個 consumer 用自己的 branch 輸入敏感度，能否在相同資源成本下，選出比 total 敏感度損傷更低的完整模型？

**答案：證據不足，不支持把 branch 設為預設。**

| 比較（branch − total，每 token mean NLL） | calibration（窗口 0–9，n=10） | tuning（窗口 10–29，n=20） |
|---|---|---|
| automatic knee（資源不同） | −0.0108 ± 0.0058 | −0.0004 ± 0.0068 |
| 相同上限：total knee 的 size／BOPs | **+0.0116** ± 0.0084 | **−0.0110** ± 0.0073 |
| 相同上限：branch knee 的 size／BOPs | −0.0107 ± 0.0129 | −0.0032 ± 0.0092 |
| 消融：只有 FFN 用 branch（knee） | +0.0066 ± 0.0105 | −0.0025 ± 0.0038 |
| 消融：只有 Attention 用 branch（knee） | −0.0086 ± 0.0090 | −0.0039 ± 0.0063 |

表中都是平均差 ± SE。量化損傷（對 FP32）約每 token 0.05–0.08 nats。

1. **全部 10 個 paired 95% CI 都包含 0。** 在 total knee 的上限下，calibration 與 tuning 的方向相反。tuning 上 5 組比較的平均都是負的（branch 略好），但幅度只有 0.0004–0.011，而且這 5 組共用部分配置，彼此不獨立。
2. **成本歸屬改變了大量選擇，但沒有對應的 loss 改善。**
   - branch 讓 34–58/196 個節點改選格式，主要在 q/k/v：q_proj 的激活從 f16／p8e1 改成 int8，省下的 BOPs 拿去提高 q/k/v 的權重位元。
   - 兩種 criterion 對彼此的選擇評價差很多。在 branch knee 的上限下，branch knee 用 total 成本算是 559.6，比 total 的最佳值 392.1 高 43%。但兩者實際的 loss 差在雜訊內。
3. **位置消融**：
   - 全模型配置的改變幾乎都來自 Attention：attention_branch 的 knee 改了 32 個節點，資源點和 all_branch 相同。
   - 只換 gate/up 時，knee 只改 2 個節點，對 loss 沒有可辨識的影響。
   - 先前 SwiGLU 研究聚焦的 gate/up，在完整配置中其實影響很小。
4. **建議**：
   - branch 歸屬是 per-consumer QDQ 語意下的正確敏感度，是一個語意修正；但本實驗沒有顯示它帶來搜尋效益。
   - 預設維持 total；branch 保留為選項（`--input-grad branch` 產生的 `table_branch.json`，以及本腳本）。
   - 若要分辨每 token 0.005 量級的差異，需要約 140–150 個 tuning 窗口（第 7 節）。
5. **held-out 結果**（決策凍結之後才執行，只用來報告）：test PPL 是 FP32 49.88、total knee 53.77、branch knee 53.60；branch − total 為 −0.0031 ± 0.0063 nats/token，95% CI 包含 0（第 8 節）。

## 1. 設定

| 項目 | 設定 |
|---|---|
| 模型 | Qwen3-0.6B，修正 epsilon 後的 onnx-mlir（RMSNorm eps 1e-6），196 個 projection |
| 成本表 | 重用 `runs/native_eps_fix_20261014/qwen3/table.json`（total）與 `table_branch.json`（branch），不重建 |
| B／C | B = weight criterion W（兩表相同）；C = activation criterion A |
| 搜尋 | `mp_auto.auto_select`，與 `run_native_auto.py --policy 2d` 相同（見下） |
| 編譯 | per-channel int 權重（`MIXEDP_WEIGHT_PER_CHANNEL=1`）、Krnl -O3／clang -O0、constants to file、bytecode；int 激活用成本表同一次 calibration 的 ranges（`ranges_by_point.json`） |
| loss | 每窗口 summed next-token CE：logits 第 0..126 個位置對 token 1..127，最後一個位置不計；mean NLL = summed CE ÷ 127 |

**搜尋設定**：
- 格式：GRID11 × GRID11。
- 目標值：B[w] + C_variant[a]。
- 資源：weight size 與 BOPs 都以 all-int8 為 1；共同 budget 套用在 max(size, BOPs)。
- 選點：normalized chord knee，samples 17、refinements 2。
- MILP time limit 10 秒，就是 `run_native_auto.py` 的 `--time-limit` 預設值。`auto_select` 本身的預設是 60 秒，但入口傳入 10 秒，所以這裡用 10 秒。

**評估資料**：
- **calibration（窗口 0–9）**：就是建表與 ranges 用的 10 個 validation 窗口，屬於樣本內。
- **tuning（窗口 10–29）**：同一個 validation split 往後 20 個不重疊的窗口，用與 `qwen3_prepare.py` 相同的 tokenizer 和切法重新產生；沒有用於建表、range 或搜尋。
- **held-out（test split 32 窗口）**：只在決策凍結後用一次。

**評估函式**：
- 沒有使用 `qwen3_spec.metrics`，它綁定的是 test tokens。
- 每個窗口明確用自己的 targets 計算：`cross_entropy(z[s, :-1], tokens[s, 1:]).sum()`。

## 2. 資產確認（`assets.json`）

| 檢查 | 結果 |
|---|---|
| 節點 | 兩表的 B、C 都是 196 個，和搜尋節點集合完全相同；q/k/v/o/gate/up/down 各 28 個 |
| 格式 | 每列都是 GRID11 的 11 種；fp32 的成本都是 0 |
| 值 | 8624 個值，非有限 0、負值 0，所以沒有被排除的候選 |
| W 一致 | `table.json` 的 B 與 `table_branch.json` 的 B 完全相同 |
| A 差異位置 | 只有 q/k/v/gate/up 共 140 個節點不同，而且這五類每個節點都不同；o/down 完全相同（單一 consumer） |
| ranges | 2870 個點，對應 2086 個 tensor；每個 tensor 只有一組 range；196 個節點的所有激活 tensor 都有 range，值有限且 lo ≤ hi |
| calibration tokens | 表的 `calib_inputs.npy` 等於 `validation_tokens.npy`；重新 tokenize 的前 10 窗口完全相同 |
| 模型 epsilon | ONNX 中 113 個 RMSNorm 的 Add 常數（28 × 4 + 1）都是 1e-6 |
| 編譯器 | `onnx-mlir-opt` sha256 `3059659f…`，建置時間晚於所有相關原始碼修改，也早於成本表；epsilon 修正在原始碼中 |
| 資源定義 | 本腳本讀取的節點、weight elements、FLOP proxy 與 `qwen3_spec` 完全相同 |
| 數值 | native FP32 − PyTorch FP32 的 summed CE：calibration +0.034、tuning +0.071；每窗口最大 0.004，約每 token 3e-5 |

各檔的 sha256 在 `assets.json`；程式與環境的 hash 在 `environment_*.json`。

## 3. 搜尋入口與驗證

**新入口**：`run_qwen_branch_ablation.py`
- 直接讀兩張表，建立四種 activation 成本變體：total、ffn_branch（只有 gate/up 用 branch 列）、attention_branch（只有 q/k/v）、all_branch（就是 `table_branch.json`）。
- 每個變體用同一個 B，各跑一次 `auto_select`。
- 變體的成本只在記憶體中建立，不寫回、也不修改任何既有表。

**目前其他入口仍然只用 total**：
- `run_native_auto.py` 沒有修改。
- `native_tables(..., input_grad="branch")` 仍回傳 total 的 B、C；docstring 已註明這點，branch 搜尋必須讀 `table_branch.json`。

**驗證（`selftest.json`，以及 search stage 內的檢查）**：
- **只替換指定列**：
  - all_branch 有 140 列來自 branch，ffn_branch 56 列，attention_branch 84 列，total 0 列；其餘列都和 total 相同。
  - all_branch 等於 `table_branch.json`；FFN 與 Attention 兩個變體的聯集等於 all_branch。
- **搜尋確實讀到指定的表**：
  - 玩具問題中，branch 表把某節點的低位元激活設得很貴。讀 branch 表的搜尋把該節點的激活從 8 位元提高到 16 位元；讀 total 表的不會。
  - 在真實資料上，每個候選與 knee 的 objective，都用該變體的表重新計算並比對（相對差 ≤ 1e-9）。
- **沒有漏節點或錯誤格式**：每個配置的 196 個節點都有 GRID11 格式且成本有限；編譯環境 `MIXEDP_NODE_FORMATS` 正好列出 196 個節點。
- **資源**：控制點的實際 weight bytes 與 BOPs 都 ≤ 上限，用整數值精確比較。

**`mp_milp.milp_select` 與 `mp_auto.auto_select`**：新增可選參數 `info` 與 `solver_log`，記錄每次 solve 的 status、mip_gap、dual bound 與秒數。沒有傳入時行為不變，已用同一個小問題驗證結果相同。

## 4. Automatic knee：total vs branch

| 成本 | knee size | knee BOPs | max(size, BOPs) | 目標值（自身成本） | 搜尋秒數 | MILP 次數 | status | knee 那次 gap |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| total | 0.75982 | 0.87798 | 0.87798 | 330.463 | 25.9 | 23 | 23 × 0 | 0 |
| all_branch | 0.76935 | 0.84077 | 0.84077 | 279.212 | 27.1 | 23 | 23 × 0 | 0 |
| ffn_branch | 0.76131 | 0.87798 | 0.87798 | 311.729 | 26.3 | 23 | 23 × 0 | 0 |
| attention_branch | 0.76786 | 0.84077 | 0.84077 | 297.779 | 29.6 | 23 | 23 × 0 | 0 |

**MILP 狀態**
- 所有 MILP 都是 status 0（最佳）；knee 附近（budget ≤ 2 倍 knee）最大的相對 gap 是 1.9e-5。
- `search.json` 的最大相對 gap（0.6%–2.7%）都出現在 budget ≥ 2.8、目標值約 0.003 的 solve，那裡落在 HiGHS 的絕對容差內。
- HiGHS 在 stderr 印出兩行 `HighsMipSolverData::transformNewIntegerFeasibleSolution tmpSolver.run();`，但 status 都是 0。

**搜尋時間**：「搜尋秒數」只算 `auto_select` 的時間，不含建表或模型執行。branch 不增加搜尋成本：27.1 秒 vs 25.9 秒。

**兩個主要 knee 在 native 下的結果**

| 配置 | size／BOPs | calibration mean NLL（PPL） | tuning mean NLL（PPL） | Δ vs FP32 calibration | Δ vs FP32 tuning |
|---|---|---:|---:|---:|---:|
| FP32 | — | 3.49909（33.085） | 3.65568（38.694） | — | — |
| total knee | 0.7598／0.8780 | 3.56007（35.166） | 3.73529（41.900） | +0.0610 ± 0.0125 | +0.0796 ± 0.0085 |
| branch knee | 0.7693／0.8408 | 3.54930（34.789） | 3.73492（41.884） | +0.0502 ± 0.0120 | +0.0792 ± 0.0093 |

- 所有配置的 logits 都是有限值，編譯 warnings 都是 0。
- **branch − total**：calibration −0.0108 ± 0.0058（95% CI [−0.0238, +0.0023]，10 窗口中 7 個較低）；tuning −0.0004 ± 0.0068（CI [−0.0145, +0.0138]，20 窗口中 13 個較低）。
- 兩個 knee 的資源不同：branch 的 size 多 1.3%，BOPs 少 4.2%。所以這組比較不能單獨歸因於 branch。
- **改選的 34 個節點**（按類別 q/k/v/o/gate/up/down）：14/7/8/2/2/1/0。
  - 23 個節點改了權重，27 個改了激活。
  - 主要的移動：
    - q_proj 激活 f16／p8e1 → int8：14 個；
    - k_proj 激活 f16 → int8 或 p8e1；
    - v_proj 激活 f16 → p8e1；
    - q/k/v 權重 int5 → int6、int6 → int7。
  - 完整清單在 `analysis.json` 的 `comparisons[...].node_diff.list`。

## 5. 相同資源限制下的比較

**做法**
- 以兩個 knee 的實際 weight bytes 與 BOPs 當作兩個上限點，用整數精確值，沒有經過顯示用的四捨五入。
- 每個上限點分別用 total 與 all_branch 成本各跑一次 MILP。
- 4 次 solve 都是 status 0、gap 0，實際資源都 ≤ 上限，沒有不可行的情況。

| 控制點 | 成本 | 上限 size／BOPs | 實際 size／BOPs | 目標值 | 該點 knee 在此成本下的目標值 | 與 knee 相同？ |
|---|---|---|---|---:|---:|---|
| total knee | total | 0.759821／0.877976 | 0.759821／0.877976 | 330.463 | 330.463 | 是（同一個 hash） |
| total knee | all_branch | 0.759821／0.877976 | 0.759821／0.877976 | 247.389 | 278.169 | 否，57 個節點不同 |
| branch knee | total | 0.769345／0.840774 | **0.743750**／0.840774 | 392.083 | 559.604 | 否，58 個節點不同 |
| branch knee | all_branch | 0.769345／0.840774 | 0.769345／0.840774 | 279.212 | 279.212 | 是（同一個 hash） |

在 branch knee 的上限下，total 成本只用了 0.7438 的 size：BOPs 是綁定的限制，total 不認為多用 bytes 划算。所以兩者面對的上限相同，但實際 size 不同。

**同上限的 paired 結果（branch − total）**

| 上限點 | 改選節點 q/k/v/o/gate/up/down | calibration | tuning | 全部 30 |
|---|---|---|---|---|
| total knee | 57：15/9/11/11/4/2/5 | +0.0116 ± 0.0084 [−0.0074, +0.0306]（5/10） | −0.0110 ± 0.0073 [−0.0263, +0.0043]（11/20） | −0.0035 ± 0.0059 [−0.0155, +0.0085]（16/30） |
| branch knee | 58：19/11/9/3/6/6/4 | −0.0107 ± 0.0129 [−0.0400, +0.0186]（7/10） | −0.0032 ± 0.0092 [−0.0223, +0.0160]（13/20） | −0.0057 ± 0.0074 [−0.0207, +0.0094]（20/30） |

格式：平均 ± SE [95% CI]（branch 較低的窗口數）。

**判讀**
- 同資源下，兩個上限點都看不出穩定的差異。
- 在 total knee 上限，樣本內（calibration）branch 較差，樣本外（tuning）branch 較好，方向不一致。
- 在 branch knee 上限，兩者都略好，但 CI 很寬。

## 6. FFN／Attention 位置消融

| 變體 | 改用 branch 的列 | knee size／BOPs | 相對 total knee 改選節點 q/k/v/o/gate/up/down | calibration Δ | tuning Δ |
|---|---|---|---|---|---|
| ffn_branch | gate、up（56 列） | 0.7613／0.8780 | 2：0/0/0/1/1/0/0 | +0.0066 ± 0.0105（3/10） | −0.0025 ± 0.0038（10/20） |
| attention_branch | q、k、v（84 列） | 0.7679／0.8408 | 32：14/7/8/1/1/1/0 | −0.0086 ± 0.0090（7/10） | −0.0039 ± 0.0063（10/20） |
| all_branch | 140 列 | 0.7693／0.8408 | 34：14/7/8/2/2/1/0 | −0.0108 ± 0.0058（7/10） | −0.0004 ± 0.0068（13/20） |

- **選擇的改變來自 Attention。**
  - attention_branch 的 knee 已經包含 all_branch 34 個改選中的 32 個，並落在相同的 BOPs 點（0.8408）。
  - q/k/v 有三個 consumer 共用輸入：總梯度相對 q 的分支梯度高估最多（成本表上 q 的 total 與 branch 相對差中位數 0.78），所以 branch 讓 q 的激活大幅變便宜。
- **FFN 的影響很小。** ffn_branch 只改 2 個節點，knee 的 BOPs 不變。這 2 個節點的改變在 calibration 與 tuning 上也造成 +0.0066 和 −0.0025 的差，可以當作「換掉少數節點」的雜訊參考。
- **loss 都在雜訊內。** 三種變體的差異都包含 0，無法說哪個位置帶來改善。所以不能用局部（FFN 或 Attention）的結果宣稱全模型 branch 有效。

## 7. 不確定性與穩定性

**雜訊大小**
- 每窗口 Δ 的標準差約 0.017–0.041 nats/token（由 SE × √n 換算）；10 或 20 個窗口的 SE 約 0.004–0.013。
- 本實驗看到的 branch 效應都 ≤ 0.012，與 SE 同量級。
- 若要用 2 個 SE 分辨 0.005 的差，大約需要 n ≈ (2 × 0.03 / 0.005)² ≈ 144 個 tuning 窗口。目前 validation split 有 262,337 個 token，可以切出約 2000 個 128-token 窗口，資料量足夠，只是評估成本約為目前 30 個窗口的 5 倍（第 10 節）。

**不穩定的跡象**
- **方向不一致**：total knee 上限點的 calibration 與 tuning 方向相反；ffn_branch 也是。
- **criterion 差異沒有轉成 loss 差異**：兩種 criterion 對同一配置的評價差很大（相對自身最佳值 +12%～+43%），但實際 loss 差在雜訊內。這和 SwiGLU 研究的結論一致：criterion 對「接近 frontier 的配置之間」的細部差異沒有預測力，只能預測粗排序。
- **小改動的雜訊參考**：只差 2 個節點的 ffn_branch knee，和 total knee 的 loss 差也達 ±0.007。

**資料角色**
- calibration windows 同時用來建表，是樣本內。
- tuning windows 與 calibration 同屬 validation split，但沒有參與任何建表、range 或搜尋。
- 本實驗只以這兩組資料做決策；決策在 `decision.json`，時間 2026-10-14T03:42:05。

## 8. Held-out 結果（決策凍結後才執行，只用來報告）

決策（`decision.json`，2026-10-14T03:42:05）凍結了 total knee 與 branch knee 的 config hash 之後，才在 test split 的 32 個窗口上評估這兩組與 FP32。`heldout` stage 會檢查：
- `analysis.json` 在決策後沒有變動；
- 配置 hash 沒有變；
- 執行的是 eval 時保留的同一個 `.so`（sha256 相同）。

| 配置 | test PPL | mean NLL | Δ vs FP32（每窗口 mean NLL） | finite |
|---|---:|---:|---|---|
| FP32 | 49.884 | 3.90969 | — | True |
| total knee | 53.768 | 3.98469 | +0.0750 ± 0.0050 [+0.0647, +0.0853]（0/32） | True |
| branch knee | 53.602 | 3.98158 | +0.0719 ± 0.0063 [+0.0591, +0.0847]（0/32） | True |

- **branch − total**：−0.0031 ± 0.0063，95% CI [−0.0160, +0.0097]，32 個窗口中 16 個較低。和 calibration／tuning 的結論相同：差異在雜訊內。
- **FP32 對照**：FP32 的 test PPL 49.884，等於 `qwen3_prepare.py` 的 PyTorch FP32（49.88）。這也確認評估用的編譯結果套用了修正後的 epsilon。
- **和先前結果的關係**：先前在 epsilon bug 下，自動流程的 2D knee 是 PPL 53.5。這次的成本表與編譯器都不同，不能直接比較。
- 這些數字沒有用來選格式、調 knee 或 criterion，也沒有回頭修改決策。

## 9. 程式修改（全部未提交）

| 檔案 | 修改 |
|---|---|
| `experiments/mixed_precision/run_qwen_branch_ablation.py`（新增） | 本實驗的入口：check、selftest、search、eval、analyze、decide、heldout、tables 八個 stage |
| `experiments/mixed_precision/mp_milp.py` | `milp_select(..., info=None)`：可選的求解紀錄；預設行為不變 |
| `experiments/mixed_precision/mp_auto.py` | `auto_select(..., solver_log=None)`：可選的每次 solve 紀錄；預設行為不變 |
| `experiments/mixed_precision/run_native_bc_pipeline.py` | 只改 `native_tables` 的 docstring，註明 `input_grad="branch"` 時回傳值仍是 total 表 |

沒有改動 `run_native_auto.py` 的預設，也沒有改動任何既有的成本表或 run 目錄。

## 10. 耗時

| 階段 | 內容 | 時間 |
|---|---|---:|
| check | 資產檢查（含 `qwen3_spec` 對照、重新 tokenize）9 秒；PyTorch FP32 30 窗口 17 秒 | 26 秒 |
| selftest | 變體列與玩具搜尋 | < 1 秒 |
| search | 4 次 automatic search（各 25.9–29.6 秒）+ 4 次控制點 MILP（各 0.9–1.2 秒） | 116 秒 |
| eval 第 1 批 | FP32 + 2 個 knee，同時 2 組 | 731 秒 |
| eval 第 2 批 | 2 個控制點 + 2 個消融，同時 2 組 | 1090 秒 |
| 每個配置 | build 30–112 秒；30 窗口執行 187–476 秒（5 個 process） | — |
| analyze／tables | — | < 1 秒 |
| held-out | 3 組 × 32 窗口（同時 2 組；每組 241–572 秒） | 781 秒 |

**搜尋延遲與模型執行要分開看**
- **搜尋延遲**只算 MILP 搜尋：每個變體約 26–30 秒，branch 並不更慢。
- **模型編譯與執行**是驗證成本，和搜尋延遲無關。

**為什麼同時跑兩組沒有加速**：單獨執行一組時，30 個窗口只要 187 秒（922 CPU 秒）；兩組同時跑時，每組 400–476 秒（1850–2300 CPU 秒）。CPU 時間加倍，推測是記憶體頻寬受限。

**branch 的額外成本**
- 建表：`table_branch.json` 和 `table.json` 來自同一次 grads build，只多兩個統計陣列。先前 Qwen 全流程是 1074 秒，舊版是 1123 秒，沒有明顯差別。
- 搜尋：沒有額外成本。

## 11. 限制

1. **窗口數**：只有 10 個 calibration 窗口與 20 個 tuning 窗口，對 0.005 量級的差異檢定力不足（第 7 節）。
2. **比較不獨立**：各比較共用部分配置，例如 total knee 同時出現在多組比較中。
3. **只有一個資源區域**：兩個上限點都在 size 約 0.74–0.77、BOPs 約 0.84–0.88。更低位元的區域沒有測。
4. **只有一種選點政策**：只看了 2D knee；沒有測 3D policy 或其他 budget。
5. **選擇沒有重複**：MILP 是確定性的，但成本表只有一份（10 個 calibration 窗口）；表本身的抽樣變異沒有估計。
6. **與 SwiGLU 研究的差異**：先前的 SwiGLU 研究只看 gate/up；本實驗顯示全模型的選擇改變主要在 q/k/v，那裡沒有做過逐節點的 ΔL 驗證。

## 12. 對後續 interaction-aware 研究的意義

1. **需要更多評估窗口。** 在這個資源區域，全模型配置之間的實際 loss 差異約 ±0.01 nats/token，與 10–20 窗口的 SE 同量級。任何 interaction-aware 的修改（例如共享輸入的成對項、per-tensor 的激活決策），都必須證明比這個雜訊大的改善。建議先把 tuning 窗口增加到約 150 個，再比較方法。
2. **研究重點應放在 Attention 的輸入群組，不是 FFN。** 共享輸入的歸屬對全模型選擇的影響集中在 q/k/v：三個 consumer、量級差異最大。gate/up 幾乎不影響 knee。
3. **加總型 criterion 只能可靠排序粗略差異。** 兩種 criterion 對同一配置的評價可以相差 43%，實際 loss 卻相同。只改歸屬或加交互項，未必能把這種差異轉成可量測的精度改善。若要研究交互作用，應先找到「criterion 差異確實對應 loss 差異」的區域，例如更低位元的配置，再設計對應的項。
4. **語意修正與搜尋效益要分開。** branch 歸屬讓 criterion 與 per-consumer QDQ 的語意一致（量級比值 1.68 → 0.91），這點已經成立；但目前沒有證據顯示它改善最終精度。

## 13. 重跑

在 `experiments/mixed_precision/` 下執行：

```bash
O=runs/qwen3_branch_ablation_20261014        # 新實驗請換一個新的空目錄
PY=../../venv/bin/python
$PY run_qwen_branch_ablation.py --stage check    --out-dir $O
$PY run_qwen_branch_ablation.py --stage selftest --out-dir $O
$PY run_qwen_branch_ablation.py --stage search   --out-dir $O
$PY run_qwen_branch_ablation.py --stage eval     --out-dir $O --labels fp32,knee_total,knee_all_branch
$PY run_qwen_branch_ablation.py --stage eval     --out-dir $O \
    --labels ctrl_total_all_branch,ctrl_all_branch_total,knee_ffn_branch,knee_attention_branch
$PY run_qwen_branch_ablation.py --stage analyze  --out-dir $O
$PY run_qwen_branch_ablation.py --stage decide   --out-dir $O --decision "…"   # 先凍結決策
$PY run_qwen_branch_ablation.py --stage heldout  --out-dir $O                  # 之後才評估 held-out
$PY run_qwen_branch_ablation.py --stage tables   --out-dir $O
```

**輸出檔案**

| 檔案 | 內容 |
|---|---|
| `assets.json` | 資產檢查 |
| `selftest.json` | 自我測試 |
| `selection_<variant>.json` | 嘗試的 budgets、solver log、frontier、knee、formats |
| `search.json` | 搜尋摘要、控制點、`config_labels` |
| `configs.json` | 每個配置的 formats、hash、資源，以及各變體成本下的目標值 |
| `eval/<hash>/result.json` | 每窗口 summed CE、finite、warnings、build／run 時間、編譯環境 |
| `eval/<hash>/compile_env.json` | 每個配置的編譯環境 |
| `analysis.json`、`per_window.csv`、`comparisons.csv`、`configs.csv` | 分析結果與原始數據 |
| `decision.json` | 凍結的決策 |
| `heldout.json` | held-out 結果 |
| `tables.md` | 報告用的表格 |
| `environment_*.json` | 環境紀錄 |
