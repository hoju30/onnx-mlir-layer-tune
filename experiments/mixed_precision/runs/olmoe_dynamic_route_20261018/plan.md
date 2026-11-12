# 預先登記：OLMoE 固定 BC 配置的 fixed-route／dynamic-route 對照

時間：2026-10-18T02:28:09+0800。以 plan.json 為準。

## 配置

- `runs/olmoe_search_final_20261016/C/configs/milp.json`（sha256 `8ebcc86eea94…`）：search_final B 修正後 MILP knee，R 0.7071056。
- 理由：OLMoE 的預設 pipeline 仍是 `mp_auto.auto_select` + `mp_milp`；2026-10-16 整合的最小修正就是 search_final B 的修正後 MILP，其 knee 即此檔，並已在 512 個新窗口驗證。不用 bc_p125（修正前路徑）與 LP knee（研究求解器）。在任何新評估前記錄。
- 量化：DevQuant + 修正後 int activation（與 search_final C 相同）；ranges 為 multipoint 的 ranges.json；21 個未搜尋投影為 bf16 權重 + FP32 activation；router、attention、norm、LM head 皆 FP32，不量化 router。

## 條件

- fp32_dyn：原模型自行路由（記錄 ids、weights、router top-16 logits）。
- fp32_fix：原模型，固定 fp32_dyn 的 ids 與 weights（控制組）。
- q_fix：BC 配置，固定 fp32_dyn 的 ids 與 weights（與歷史 fixed-route 相同語意：ids 與 weights 都固定）。
- q_dyn：BC 配置，FP32 router 在量化模型自身 hidden state 上路由。
- 四者共用同一函式、同窗口、每批 128 窗口、GPU FP32、TF32 關閉；每窗口 127 個預測 token。

## 檢查

- pilot: fp32_dyn (capture batch shape, 64 windows) reproduces search_final C route hashes and dynamic CE bitwise; q_fix and fp32_fix with C routes (chunk 128) reproduce C truth (milp, fp32) bitwise
- fp32_fix == fp32_dyn (CE and final hidden state bitwise); dynamic path without quantization == fp32_dyn
- q_dyn rerun bitwise; q_fix fed with q_dyn own routes == q_dyn bitwise (route source is the only difference)
- config: 3051 nodes, supported formats only (explicit list), ranges for every quantized activation, posit LUTs present; router weight object untouched; activation hooks only on expert projections
- route records: tokens, layer, model revision and hashes stored; weights finite; all CE finite; 127 tokens/window
- any failure: fix and record before interpreting q_dyn; if bitwise equality cannot hold, quantify FP32 control and same-path rerun differences first

## Pilot 與樣本數

- Pilot：search_final C 前 128 個窗口（已看過的資料），做歷史重現與所有數值控制，估計 Δ_route 的 4 窗口區塊 SD。
- 規則：n = ceil((1.645 s / (ε/2))²)，限制在 [32, 128] 篇。

## 資料

- 嚴格文章邊界：標題行前後都必須是空行（得到 600／60／60 篇），表格或公式中的類標題行併入所在文章。
- 排除先前所有 OLMoE 文本所在的整篇文章（含 D、V、H、J、C1–C8）。計數（不含 route、loss）：validation 10、test 18、train 556 篇可用（≥512 token）。
- 原評估來源優先：validation 與 test 的可用文章全部納入（共 28 篇，非挑選），其餘 n−28 篇自 train 均勻抽樣。分層：validation／test／train，分層 bootstrap 並分開報告。
- 每篇 4 個連續窗口，起點均勻；seed 18051；評估前凍結。

## 終點與統計

- 每篇文章：各條件 CE = 4 窗口 summed CE / 508。D_fixed = CE(q_fix) − CE(fp32_dyn)；D_dynamic = CE(q_dyn) − CE(fp32_dyn)。
- 主要終點：Δ_route = CE(q_dyn) − CE(q_fix) 的文章等權平均（每篇 508 token）。正值 = dynamic 額外損傷。
- 10000 次分層 paired article bootstrap（seed 18061），95% 與 90% 區間；PPL 由平均 CE 取 exp。
- ε = 0.002 nat/token，沿用 search_final 與 solver regression 的 OLMoE 配置等價界線；為研究容忍界線，非通用標準。
- 判定：90% CI ⊂ [−ε, +ε] → 容忍範圍內等價；90% CI 下界 > +ε → dynamic 額外損傷超過容忍；90% CI 上界 < −ε → dynamic 明顯較好；其餘 → 不確定。95% CI 含 0 不等於等價。

## 路由統計（描述）

- q_dyn 對 fp32_dyn，逐層，位置 0–126：top-8 集合完全相同比例；重疊率；各 expert dispatch 變化（TVD）；依 expert 對齊的權重 L1；FP32 第 8／9 名 logit margin 分位與變動率；進入 top-8 的 expert 在 FP32 的名次；每篇文章路由變動與 Δ_route 的關係。
- token 不視為獨立樣本；路由改變多不代表造成損傷。

## 不做

- 不加入一階項、不重搜 knee、不依新 loss 改配置、不篩文章、不改 calibration／range／量化器、不做逐層干預或格式消融。
