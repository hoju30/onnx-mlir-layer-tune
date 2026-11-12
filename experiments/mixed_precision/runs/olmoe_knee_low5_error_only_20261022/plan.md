# 預先登記：OLMoE 正常 knee 的 ≤5-bit activation error-only refresh（E5）

時間：2026-10-22T20:43:12+0800。以 plan.json 為準。

- 背景：看過全節點 error-only E（未確認改善、更新約 25 秒）後，依使用者要求提出的新假說；以新文章檢驗一次；不測 ≤6-bit 或其他規則。
- E5 目標：A 中 weight 或 activation ≤5 bit 的投影 1512 個（= D5 的目標集合）；其外 1539 個。
- E5 成本：weight 全部 = A；目標投影的 activation（所有候選格式）= Σ(G2_static·var_new + GS_static·μ_new²)，新誤差在 E 的 tap、q0 rows 正規化；非目標 activation 列 = A（逐位元）。
- 實作：forward-only（inference_mode、無 retain_grad、計數 autograd）；完整 q0 forward；只計算目標投影所需的 slot 誤差（不先算 E 再遮罩）。
- 控制：輸入與快取 hash、weight 列 = A、目標列 = E、非目標列 = A、numpy 獨立重算、forward 一致、無梯度計算、合法與上限、重複一致、端到端重現。
- 計時：B／E／E5 各 3 次交錯（B1 E1 E51 E2 E52 B2 E53 B3 E3）；E5 端到端 1 次。
- 評估：600 篇未用過的 WikiText-103 文章 × 4（排除 4874 篇歷史文章）；FP32、A、B、E、D5、E5；seed 22901／22902／22911。
- 主要：E5 − A、E5 − E（校正上界 < 0 才確認改善）、E5 − B（校正上界 < +0.0005 非劣；下界 > +0.0005 超過界線）；Bonferroni 98.33%。
- 次要（95%）：E5 − D5、B − A、E − A、D5 − A。
