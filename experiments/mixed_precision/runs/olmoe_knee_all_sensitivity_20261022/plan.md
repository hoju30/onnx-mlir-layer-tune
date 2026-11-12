# 預先登記：OLMoE 正常 knee 的全節點 sensitivity-only refresh（C）

時間：2026-10-22T12:19:25+0800。以 plan.json 為準。

- A／B／D 沿用上一輪凍結的配置（hash 核對）；C 用同一個 q0_knee、knee/knee.json 的精確上限重新搜尋一次，所有節點可改格式。
- C：全部 3051 個搜尋投影、所有候選格式更新 sensitivity；weight／activation 誤差沿用 static，activation 以 static routed rows 正規化。
- 實作：上一輪的 build，明確傳入 static activation 快取、不給 targets；每次執行以計時器驗證快取確實被使用（不能只把 variant 叫做 C）。
- 控制：輸入 hash、forward、input gradient、C weight 列 = B、D 目標列 = C、numpy 獨立重算全部 C 列、合法與上限、重複一致、端到端重現。
- 計時：B／C／D 各 3 次交錯（B1 C1 D1 C2 D2 B2 D3 B3 C3）；C 端到端 1 次（單次實測）。
- 評估：600 篇未用過的 WikiText-103 文章 × 4（排除 3074 篇歷史文章）；FP32、A、B、C、D；seed 22601／22602／22611。
- 主要比較：C − A、C − D、B − C（負值代表前者較好）；95% CI 與 Bonferroni 98.33% CI，以校正區間確認方向。
- 另報 B − A、D − A；C − B 以 B − C 校正區間反號判讀非劣（+0.0005）。
