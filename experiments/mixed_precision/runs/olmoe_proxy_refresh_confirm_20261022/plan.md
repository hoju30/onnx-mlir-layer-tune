# 預先登記：一次 proxy refresh 收益的新文本確認

時間：2026-10-22T03:19:22+0800。以 plan.json 為準。

- 固定配置：q0 `2fe46627745c…`、q1 `346fd805dfcb…`（runs/olmoe_proxy_refresh_20261022），ranges 與修正後量化器同前；不重建、不重搜、不改任何設定。
- 資料：600 篇新的 WikiText-103 train 文章 × 4 窗口（2400 窗口）；排除 WikiText-2、WT103 內部重複、所有歷史 OLMoE WT103 文章（674 篇，含上一輪 128 篇）、近重複（≥10% 5-gram）；seed 22201／22202；可用 26721 篇。
- 掃描的歷史 manifest：runs/olmoe_calibration_stability_20261018/segments.json、runs/olmoe_crosstext_20261017/segments.json、runs/olmoe_dynamic_multipoint_20261018/segments.json、runs/olmoe_dynamic_route_20261018/segments.json、runs/olmoe_first_order_validation_20261017/segments.json、runs/olmoe_proxy_refresh_20261022/segments.json、runs/olmoe_route_decomposition_20261018/segments.json、runs/olmoe_route_dynamic_prop_20261018/segments.json、runs/olmoe_route_mass_restore_20261018/segments.json、runs/olmoe_route_precision_protection_20261019/final_segments.json、runs/olmoe_route_precision_protection_20261019/selection_segments.json；以及 WikiText-2 窗口檔 runs/olmoe_fixed_route_20261015/plan.json、runs/olmoe_multipoint_20261015/plan.json、runs/solver_regression_20261016/plan.json、runs/olmoe_search_final_20261016/C/windows.json、runs/olmoe_bc_numerics_20261016/B/windows.json（整份 WikiText-2 排除即涵蓋）。
- 評估：FP32、q0、q1 皆正常 dynamic；同一 forward、chunk 128、每窗口 127 個預測 token。
- 程式控制：上一輪看過的窗口 0–127 須逐位元重現上一輪 CE（只作控制）。
- 主要終點：Δ = NLL(q1) − NLL(q0)（文章配對）；10000 次 bootstrap（seed 22211）。
- A：95% CI 上界 < 0 → 確認 refresh 較好；下界 > 0 → 確認較差；其他 → 方向未確認。
- B：整段 95% CI 落在 ±0.002 → 確認等價。A、B 分開判定，可同時成立；不顯著不等於等價。
- 樣本數固定 600（預期半寬約 0.001，僅估計）；不加樣、不挑子集、不重選配置；上一輪 128 篇只並列為歷史結果，不合併。
