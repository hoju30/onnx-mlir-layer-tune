# 預先登記：固定 q0、只 fine-tune router

時間：2026-10-22T03:54:10+0800。以 plan.json 為準。

- 條件：A FP32 dynamic；B q0；C q0 + router FT（第 20 步）；D q1（完整 refresh 對照）。全部用自己的 dynamic route。
- 只訓練 16 個 router 矩陣（mlp.gate.weight，不是 expert 的 gate_proj）；q0 格式、量化後 expert 權重、ranges、attention／norm／embedding／LM head、未搜尋投影全部固定。
- AdamW，lr 1e-4（固定），betas (0.9, 0.999)，eps 1e-8，weight decay 0，router 全體 global norm clip 1.0，20 次 update，每次用全部 10 個 calibration 窗口，loss = next-token mean CE，無 distillation／load-balancing。第 20 步為唯一主要結果。
- 梯度：每步以當前 router 做 dynamic forward；逐層重算 backward，routing weights 保留梯度（傳到 router 與較前層 hidden state），top-k 為離散選擇；activation QDQ 用 identity STE（surrogate）；expert 權重固定但輸入梯度保留；所有層梯度算完才 update。
- 控制：零更新與 dr.forward 逐位元一致；2 層子堆疊完整 autograd 對逐層 backward；初始梯度算兩次逐位元一致；optimizer 只含 router；checkpoint 存讀重現；評估 forward 重現前次結果。
- 評估：訓練與凍結後才抽 600 篇新 WikiText-103 文章 × 4 窗口；排除 WikiText-2、WT103 內部重複、所有歷史 OLMoE 文章（含前次 128 與 600 篇）、近重複；seed 22301／22302／22311。
- 主要終點：ΔFT = NLL(q0+FT) − NLL(q0)；95% CI 上界 < 0 → 確認改善，下界 > 0 → 確認變差，其他 → 未確認；整段 CI 落在 ±0.002 才算等價。
- 次要（探索）：Δrefresh = NLL(q1) − NLL(q0)、ΔFT_vs_refresh = NLL(q0+FT) − NLL(q1)。
- 成本：同步計時各段；評估與控制另列；實際路由 BOPs proxy、routed token counts 作描述。

## 修訂（2026-10-22T03:56:07+0800，抽評估資料之前）

- 歷史文章排除數由 674 更正為 1274：沿用的 manifest 掃描函式會略過確認實驗（600 篇）的目錄；規則文字本來就要求排除，實作已修正。可用文章由 26721 更正為 26121。原 plan 保存為 plan_original_20261022T0354.json，詳見 amendments.json。
