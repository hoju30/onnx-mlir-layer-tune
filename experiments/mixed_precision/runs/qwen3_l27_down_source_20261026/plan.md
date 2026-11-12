# 計畫：前層 Down 的權重／activation 拆分與 position 0 背景效應的 native 確認

時間：2026-10-26T15:33:53+08:00。以 plan.json 為準。

- GPU：背景 U（全 FP32）、dn（L0–26 Down w 與 a 依 A）、dn_w（只權重）、dn_a（只 activation，x 與 y）、dn_ax（只 input）、dn_ay（只 output）、F（全 A）；L27 Down w int8、a = f16／p8e1 整節點；保存逐位置 CE，position 0 的變化即「只在 position 0 用 p8e1」的效果（L27 之後只有逐位置運算）。
- GPU 主要：各拆分的 position 0 情境改變（D_b,pos0 − D_U,pos0），4 個 Holm；加成性描述；dn、F 重現。
- native：U_f16、U_p8e1（新編譯）、F_f16、F_p8e1（重用 L27 confirm 的 A、C .so）；每篇第 1 個窗口（128 個）的逐位置 CE；主要：position 0 背景效應 D_F,pos0 − D_U,pos0 與 95% CI；同窗口 GPU 對照。
- 資料：128 篇新 WT103 × 2；seed {'order': 26102651, 'start': 26102652, 'bootstrap': 26102653}。
- 冒煙測試：scratch smoke 15:28-15:33 of plan, gpu_controls, build and native_controls only (already-seen follow-up articles, calibration windows 0-1); all controls passed; no new article seen; nothing changed afterwards

