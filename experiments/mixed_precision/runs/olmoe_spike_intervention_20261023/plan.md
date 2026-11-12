# 預先登記：OLMoE spike rows 量化誤差的因果干預

時間：2026-10-23T04:15:01+0800。以 plan.json 為準。

## 問題
- BC 對 spike rows 給出高成本；只消除這些 rows 的量化誤差，最終 NLL 是否相應改變？
- 不改 BC criterion、量化器、搜尋、knee；不 fine-tune；不加 spike 懲罰；不把干預放進正式 pipeline；不加一階項或逐 token 變體。

## 資料隔離（已修正）
- 改用 `wt_manifest_scan.used_wt103_verified`（內容 hash 判定；沒有 hash 的 manifest 必須證明是 WikiText-2，否則報錯）：排除 6233 篇歷史 WT103 文章，掃描 22 個 manifest。
- 另排除 WikiText-2 文章與近似重複；本輪 discovery／選點用到的文章（token-spike discovery 與確認資料）都在排除清單中。更正紀錄見 correction_summary.json。

## 選點與格式
- 重點：L1.E9（known_case）, L1.E11（position0）, L1.E18（position0）, L2.E30（position0）, L6.E59（int4_improvement）。L1.E9 是已知案例重現，其餘是推廣檢查，分開報告。
- 對照：每層（1、2、6）一個弱 spike tap，同層、R_token ≤ p50、最接近 {1: 'L1.E9', 2: 'L2.E30', 6: 'L6.E59'} 的 log rows、優先同 q0 位元類別；允許前幾輪用過的 tap；若該層沒有 ≤ p50 的候選，取該層 R_token 最低者並記錄。只用 discovery 統計，不用任何 loss。
- 主要格式：int4, p6e1；不跑 int6／p4e1（成本控制）。

## spike 定義與條件
- spike row：s_t ≥ 6.0 × 該 expert 在 token-spike 輪 FP32 discovery 的 median s_t。門檻：{'L1.E9': 1.7154, 'L1.E11': 1.3535, 'L1.E18': 1.7508, 'L2.E30': 4.9503, 'L6.E59': 4.6118}。
- 同一輸入、同一 tap 的所有格式與條件使用完全相同的 mask（以 row-id checksum 檢查）。
- Z：tap 完全 FP32；F：全部量化；P：spike rows 保留 FP32；S：只量化 spike rows；R：同數量的隨機一般 rows 保留 FP32。
- R 的 seeds：[23341, 23342, 23343]；在同一窗口、同一 expert 內抽，數量等於該窗口的 spike rows；不足時用同一 chunk 的其他一般 rows 補足並報告；各 seed 先按窗口平均再做文章統計，不挑 seed。
- 「保護」是把 Q(h) 換回 h；不刪 token、不設為 0。所有條件用原 range，P 不會改善其他 rows 的間距：本輪測的是 spike rows 自身誤差，不是 range 被撐大。對照 tap 只跑 F、P、S。

## 測量
- FP32 背景、一次只干預一個 expert 的 Down 輸入、dynamic route；96 篇新 WT103 文章 × 3 窗口，凍結後抽取。
- 記錄 rows、spike rows、被保護 rows、補足數、mask checksum、殘餘誤差、range、下游 route 變化、覆蓋文章數。

## BC 反事實
- 在 tap 的 BC calibration rows 上，把被保護 rows 的誤差設為 0，重算 S1、S2、mean、variance、bias；n 不變；用原 G2／GS 與原正規化。不從原分數直接減舊占比。T_F 與 cache 的 BC 成本比對。

## 主要對比與多重比較
- C1 = NLL(P) − NLL(F)；C2 = NLL(P) − NLL(R̄)；C3 = D_F − D_P − D_S（整體非加性診斷，不等於 Hessian 交叉項）。
- 重現族（L1.E9：C1、C2 × 2 格式 = 4 檢定）與推廣族（其餘 4 個 tap：16 檢定）各自 Holm，FWER 0.05；p 值為雙尾文章 bootstrap。C3 與對照 tap 只報 95% CI。

## 判讀類別（預先固定）
- A：C1 < 0 且 Holm 顯著。D：C1 > 0 且 Holm 顯著。C：C3 的 95% CI 排除 0。
- B：以同一 tap 的 int4 為錨，k = D_F(int4) ÷ T_F(int4)（只有 D_F(int4) 的 95% CI > 0 才定義），E_BC = k·(T_P − T_F)，|E_BC| ≥ 0.0005；C1 的 95% CI 完全落在 ±0.25·|E_BC| 內 → BC 預測大變化而實測小且 CI 窄。
- E：A、B、D 都不成立 → 無法判定（不顯著不等於效果小）。BC 與 NLL 尺度不同，不要求數值相等。

## 統計與預算
- 10000 次文章 bootstrap（seed 23311）；窗口與候選不當成獨立樣本。
- 預算（分鐘）：{'prep': 5, 'data': 5, 'measure': 25, 'analyze': 5, 'figures': 3, 'total_gpu': 35}；測量超時就在下一個層組前停止；看到結果後不加資料、不換 seed、不改門檻。
