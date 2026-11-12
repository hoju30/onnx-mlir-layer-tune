# 計畫：L27 Down f16→p8e1 效果的情境依賴與來源歸因

時間：2026-10-26T14:55:51+08:00。以 plan.json 為準；寫入後修改須記 amendment。

- 問題：同一批文章上，L27 Down 的 a 由 f16 改 p8e1（x 與 y）的效果是否隨「其他投影是否量化」改變？是哪一組造成？
- 配置：A = 原 BC knee（hash 36205a06…）；L27 Down 的 w 一律 int8；背景決定其他 195 個投影哪些依 A（w 與 a），其餘 FP32。
- 背景：U 全 FP32｜own L27 自己的 q/k/v/o/gate/up｜early L0–13｜late L14–26｜attn L0–26 的 q/k/v/o｜ffn L0–26 的 gate/up/down｜P = L0–26｜F = 全部（F + f16 即 A）。
- 條件：8 背景 × (f16, p8e1) + U、F 的 p8e1 只在 x／只在 y = 20 個；GPU LMReplay（非 native）。
- 資料：128 篇新 WT103 × 2 窗口；排除 9129 篇已用文章；seed {'order': 26102631, 'start': 26102632, 'bootstrap': 26102633}。
- 主要：C1 = D_F − D_U（D = p8e1 − f16，負值 p8e1 較好），95% CI 排除 0 才算確認情境依賴；歸因：D_g − D_U（5 組，Holm）；加成殘差與 x／y 槽位拆分為描述。
- 控制：在 L27 confirm 的文章上重現前輪 PyTorch A／C（逐位元）；目標 hook 與 LMReplay 原路徑相同；U 且目標 (w, a) 皆 fp32 = 全 FP32；TF32 關閉、deterministic。
- 冒煙測試：scratch smoke 2026-10-26 14:54-14:55 (outside runs/) of plan and controls only: the controls ran on the already-seen articles of runs/qwen3_l27_native_confirm_20261024 (reproduced that round's PyTorch A / C bitwise, C - A -0.00834) and on the 10 calibration windows (own hooks = LMReplay path; U with target fp32 = all FP32; the int8 target weight alone -0.0019 on calibration); no new article seen; nothing changed afterwards

