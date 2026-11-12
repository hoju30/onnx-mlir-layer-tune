# 預先登記：OLMoE 完整更新一次 BC proxy 是否有收益

時間：2026-10-22T02:35:30+0800。以 plan.json 為準。

- 資源上限：aggressive 配置的精確 weight bytes = 3804495872、BOPs = 39226380910592（R 0.5946031），BOPs 用 FP32 calibration routed rows，兩次搜尋相同。
- 固定：3051 個搜尋投影、GRID11×GRID11、原 10 個 calibration 窗口、固定 ranges、修正後量化器、router／attention／norm FP32。
- A（static）：本輪在 FP32 reference 上重建 BC 表 → LP 優先搜尋 → q0；與歷史表交叉核對。
- B（refresh）：q0 的 calibration dynamic forward（自身路由）→ 路由視為常數、activation QDQ 以 identity STE 反傳 → 依原 BC 形式重算 → 同一搜尋 → q1；只更新一次，不依 loss 回退。
- weight 項：GY = 對 y-QDQ 前輸出的梯度、X = linear 實際吃到的（x-QDQ 後）輸入；ΔW = Q_f(W) − W（原始 W，自 static 精確重用）。
- activation 項：x 槽 gate／up 用共享的 QDQ 前輸入與 total gradient，down 用 QDQ 前 SiLU(g)·u；y 槽用 QDQ 前 linear 輸出；err = QDQ_f(t) − t（固定 ranges），含 variance 與 bias 兩部分；fp32 選項成本為 0。
- q0 無 calibration token 的原搜尋 expert 回退 static 表（記錄）；搜尋集合不擴張；n 為 q0 的 routed rows。
- 搜尋：LP → 取整 → 修復，差距門檻未過才退回修正後 MILP；兩表各自精確支配剪枝。
- 評估：128 篇未用過的 WikiText-103 文章 × 4 窗口；seed 22101／22102／22111；FP32、q0、q1 皆正常 dynamic。
- 主要終點：Δ = NLL(q1) − NLL(q0)；95% CI 上界 < 0 → 改善，下界 > 0 → 較差，其他 → 無法確認；整段 95% CI 落在 ±0.002 才算等價。
- 不做：部分更新消融、改 BC、router 微調、改粒度、一階項、整合預設；不宣稱重現 GEMQ。
