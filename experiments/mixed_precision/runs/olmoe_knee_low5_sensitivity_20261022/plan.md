# 預先登記：OLMoE 正常 knee 的 ≤5-bit selective sensitivity refresh（D5）

時間：2026-10-22T15:58:35+0800。以 plan.json 為準。

- 背景：≤5-bit 門檻是在看過 D4（≤4-bit，knee 無收益）與 C（全節點）結果後提出的新假說，本輪以新評估文章檢驗一次；成功也只支持本模型／knee／calibration，不測 ≤6-bit。
- D5 目標：凍結的 A 中 weight 或 activation ≤5 bit 的投影，1512 個（int4 169＋int5 1343）；D4 是子集合；其外 1539 個。
- 更新：目標投影所有候選格式的 weight／activation sensitivity（q0_knee 工作狀態），static 誤差統計（明確傳入快取、static rows 正規化）；非目標沿用 static；在 knee 精確上限重新搜尋一次。
- A／B／C／D4 沿用凍結配置（hash 核對）；D5 計時只收集目標統計，不用 C 的統計遮罩。
- 計時：B／C／D4／D5 各 3 次交錯（B1 C1 D41 D51 C2 D42 D52 B2 D43 D53 B3 C3）；D5 端到端 1 次。
- 評估：600 篇未用過的 WikiText-103 文章 × 4（排除 3674 篇歷史文章）；FP32、A、B、C、D4、D5；seed 22701／22702／22711。
- 主要：D5 − A、D5 − D4（校正上界 < 0 才確認改善）、D5 − B（校正上界 < +0.0005 才確認非劣；下界 > +0.0005 超過界線）；Bonferroni 98.33%。
- 次要（95%）：D5 − C、B − A、C − A、D4 − A。
