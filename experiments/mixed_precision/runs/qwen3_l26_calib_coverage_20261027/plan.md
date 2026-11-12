# 計畫：L26 校準文章覆蓋與來源（固定 1280 tokens、固定 ranges）

時間：2026-10-27T03:16:14+08:00。以 plan.json 為準。

- 目標 /model/model/layers.26/mlp/down_proj/MatMul，權重 A 的 int6，背景 F；activation input／output 同格式；選擇集 f16, p8e1, p6e1, p4e1, int8, int6, int4, bf16（fp32 僅數值控制）。
- ranges 固定為 runs/native_eps_fix_20261014/qwen3/ranges_by_point.json（sha256 f93189ce6f8a…），所有組與新文章共用，不重算。
- 來源 WT2 validation／WT103 train，各 3 pool × 10 篇（pool 間不重複），每篇一段連續 1280 tokens → 10 窗口 M[a,j]。
- 單篇組 a = M[a,0..9]；多篇組 r 的第 j 列 = M[(j−r) mod 10, j]；兩種覆蓋用完全相同的 100 窗口、同 batch 列。
- 種子：{'wt2_order': 26102721, 'wt103_order': 26102722, 'span_start': 26102723, 'n_order': 26102724, 'n_start': 26102725, 'bootstrap': 26102726}。WT2 排除 C0（Homarus gammarus），歷史重疊只記錄；WT103 沿用既有 sampler 規則並排除所有已用文章（9897）。
- 主要選擇：校準實測 d_C argmin（平手 f16 優先）；proxy（P256、S_u+K2_actual）另列，不混為一法。留一窗口只重聚合已存數值。
- 全部 121 組 pick 凍結後才抽 128 篇新 WT103 文章 × 2 窗口（排除所有校準文章與舊 N），只量 F 背景真值。
- 主要比較：delta_n = L_n(WT2,multi) − L_n(WT2,single)，20000 次文章 bootstrap（seed 26102726）；mean<0 且 CI 上界<0 才算改善。
- 「覆蓋有幫助」需主要改善成立且 WT2 multi 的留一窗口改選率低於 single；只更穩不等於更準。
- 冒煙測試：2026-10-27 03:11-03:15 in the scratchpad (outside runs/, invisible to the manifest scan and the token audit) with this script: plan, audit, calibdata (the sampler check reproduced runs/qwen3_l26_suffix_bridge_20261027 N exactly; the 60 calibration articles were drawn and their titles / token counts / skips were visible; no model loss), calib --smoke-c0-only (C0 only: every control passed, bitwise C0 reproduction of CE and proxies, C0 picks int8 for D_C / P256 / S_u+K2 and leave-one-window-out 7/3, 10/0, 7/3 = known results). Fixes before this plan: tokenizers import order in the audit; a random-number offline test of grouping / pool controls / analysis. No new-group loss and no new article seen.

