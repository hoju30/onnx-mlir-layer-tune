# 預先登記：OLMoE 正常 knee 的 activation error-only refresh（E）

時間：2026-10-22T18:36:16+0800。以 plan.json 為準。

- A／B／C 沿用凍結配置（D4／D5 只用於交叉評分診斷）；E 用同一個 q0_knee、knee/knee.json 的精確上限重新搜尋一次，所有節點可改格式。
- E：weight 成本整張沿用 A；activation 成本 = Σ(G2_static·var_new + GS_static·μ_new²)，新誤差在 B 相同的 tap 取得、以 q0 routed rows 正規化，G2／GS 維持 static 尺度。
- G2_static／GS_static：在本輪的 static 重建中補存（不從成本表反推），重建表須和 A 逐位元相同；建立／寫入／讀取成本只算給 E。
- E 的更新為 forward-only：inference_mode、無 retain_grad 的觀測路徑、計數 autograd 呼叫（須為 0）；首次建立原始 BC 仍需要 backward。
- 控制：static 重建、E weight 列 = A、forward 一致、numpy 獨立重算 E、以 C 存的 q0 梯度＋新誤差重現 B、無梯度計算、合法與上限、重複一致。
- 計時：B／C／E 各 3 次交錯（B1 C1 E1 C2 E2 B2 E3 B3 C3）；static＋梯度快取單次；E 端到端單次。
- 評估：600 篇未用過的 WikiText-103 文章 × 4（排除 4274 篇歷史文章）；FP32、A、B、C、E；seed 22801／22802／22811。
- 主要：E − A、E − C（校正上界 < 0 才確認較好）、E − B（校正上界 < +0.0005 非劣；下界 > +0.0005 超過界線）；Bonferroni 98.33%。次要：B − A、C − A（95%）。
