# 修正整合：量化器裝置修正、`mp_milp` 最小修正、報告更正

日期：2026-10-16

三項修改都經過使用者同意，理由分別是：

- **量化器修正**：確保不同執行環境的結果一致。
- **最小修正**：避免漏掉可行解。
- **報告更正**：避免沿用過時的結論。

沒有修改 BC criterion、BC 表、range、候選格式、資源模型、knee 規則，也沒有覆寫既有實驗的數據。

## 1. 共用量化器 `qwen3_swiglu_common.int_act_qdq`

**修改**：scale 和 zero-point 改成建立在輸入所在的裝置上（`device=x.device`）。

**原因**：在 CUDA 上，「除以 CPU 的 0 維張量」會被改成「乘以倒數」，和 CPU 及 QuantizeLinear 的真除法在最後一個位元上不同（`runs/olmoe_search_final_20261016` 實驗 A）。

**驗證**（`quantizer_check.json`，腳本為 `check_quant_fix.py`；測試範圍：int4–int8，25 組 range，即 5 組人工設定加 20 組 OLMoE 的真實 range，共 1.25 億個元素）：

| 檢查 | 不一致的元素數 |
|---|---:|
| 修正後的 CPU vs 修正前的 CPU | 0（CPU 行為不變） |
| 修正後的 CUDA vs CPU | 0 |
| 修正前的 CUDA vs CPU | 55（舊路徑的差異） |
| `run_olmoe_multipoint.DevQuant.act` 完整路徑（int、posit、f16、bf16；30 組 range × 9 種格式，5400 萬個元素），CUDA vs CPU | 0 |

**影響範圍**：只有在 GPU 上使用 int 活化量化的程式，之後的結果會改變。

- 相關腳本：`run_olmoe_fixed_route.py`、`run_olmoe_multipoint.py`（`DevQuant`）、`run_qwen35_bc.py`、`run_qwen35_knee_gateup.py`、`run_qwen35_weight_elementwise.py` 的 `--device cuda` 路徑。
- CPU 上的計算和 native 路徑不受影響。
- 已完成的實驗沒有重算：
  - OLMoE 系列報告都已加上更正段落。
  - Qwen3.5 系列以 native 為主要證據，GPU 只用來交叉檢查，所以沒有加更正段落。若要引用其中的 GPU 數字，請記得它們用的是舊量化器。
- 其他用 Python 浮點數當 scale 的舊腳本（`run_gpt2_*`、`run_mbv2_*`、`check_shared_fix.py`）都沒有 CUDA 路徑，不受影響，也沒有修改。

## 2. `mp_milp.milp_select` 最小修正

**修改內容**：

- 保留原本的流程：先在精確預算上求解；如果答案超出精確檢查，就把預算縮 1 ppm 重試。
- 新增的只有一件事：當重試沒有解，或重試的答案仍然超出預算，而第一次求解有答案時，用逐步換選項的方式把第一次的答案修復進精確預算。修復的演算法和 `run_solver_regression.repair` 相同。
- 修復後的答案回傳 status 1（可行、但未證明最佳）。`info` 會多出 `repaired` 和 `repair_swaps` 兩欄。
- 第一次求解本身沒有解時，仍和原本一樣直接失敗；不會改用最便宜的配置代替。
- `mp_auto` 只更新了文件說明：status 1 也包含修復後的答案。

**驗證**（`milp_check.json`，腳本為 `check_milp_fix.py`）：

- **8 個 dense 問題**：用新的 `mp_auto.auto_select` 完整重跑，和 `runs/lp_pipeline_regression_20261016` 的「原規則＋MILP」相比：
  - 嘗試過的預算完全相同；
  - 所有候選的預算、BC、配置雜湊完全相同；
  - 最終 knee 完全相同；
  - 修復路徑一次也沒有觸發（每個問題各有 0–3 次照常使用 1 ppm 重試）。
- **OLMoE R 0.5**：
  - 修改前，這一點會丟出「MILP found no solution」，該預算從 frontier 消失。這正是 multipoint 實驗 knee 被誤選的原因。
  - 修改後，修復 1 次交換即成功，status 1，BC 802.6066939。
  - 配置和 `runs/olmoe_search_final_20261016` 的修正版 MILP 逐一相同，耗時 114 s。

**影響範圍**：只有「重試失敗」的預算結果會改變。在目前驗證過的 9 個問題中，只有 OLMoE 的 R 0.5 會發生。

## 3. 報告更正

以下 7 份報告在第一個章節前，加入「更正與後續（2026-10-16）」段落，原文保留不改：

| 報告 | 更正內容 |
|---|---|
| `olmoe_multipoint_20261015` | knee 應為 R 0.7071（完整重跑為 R 0.7071056），不是 R 0.569；R 0.5 失敗的原因；GPU 量化器；CPU 交叉檢查不可重現 |
| `olmoe_fixed_route_20261015` | GPU 量化器，對單一 expert 損傷的影響未量化；CPU 交叉檢查 |
| `olmoe_milp_speed_20261016` | 本報告的建議後來的回歸結果；`mp_milp` 只加了最小修正；knee 的說明 |
| `solver_regression_20261016` | OLMoE 配對 loss 已由 512 窗口確認等價；這裡的參考 knee 不是完整重跑；最小修正已整合；dense 的延伸結論 |
| `olmoe_knee_check_20261016` | 「實作雜訊」推論的依據不可重現；`bc_p125` 不是完整重跑的 knee；等價已確認 |
| `olmoe_search_final_20261016` | 量化器和最小修正都已整合 |
| `lp_pipeline_regression_20261016` | 「沒有修改 `mp_milp`」的說法已過時；結果不受影響 |

## 檔案

`check_quant_fix.py`、`quantizer_check.json`、`check_milp_fix.py`、`milp_check.json`；`posit_oracle*` 和 `spec_work/` 是驗證時產生的工作檔。
