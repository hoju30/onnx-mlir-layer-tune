# 計畫：L27 情境效應的位置與 FFN 二分（續）

時間：2026-10-26T15:10:34+08:00。以 plan.json 為準。

- 背景：U 全 FP32、F 全依 A、ffn（L0–26 FFN）、ffn_early（L0–13 FFN）、ffn_late（L14–26 FFN）、ffn_gu（L0–26 gate+up）、ffn_down（L0–26 down）。
- 目標：L27 Down w int8；a = f16／p8e1（x 與 y）；U、F 另測 p8e1 只在 position 0、只在 1–127。
- 步驟 1：U、F 中 D_pos0、D_rest 與交互；位置的情境改變（F − U）兩個比較 Holm。步驟 2：4 個子組 D_g − D_U（Holm）；FFN 內加成性描述。
- 重現：新文章上重量 C1 與 ffn − U。資料：128 篇新 WT103 × 2；seed {'order': 26102641, 'start': 26102642, 'bootstrap': 26102643}。GPU LMReplay，非 native。
- 冒煙測試：scratch smoke 15:09-15:10 of plan and controls only (already-seen l27-confirm articles and calibration windows; controls passed); no new article seen; nothing changed afterwards

