# 預先登記：BC 的 channel 彙整是否忽略了 spike token 上梯度與誤差的位置對應（診斷）

時間：2026-10-23T14:07:17+0800。以 plan.json 為準。

## 界線
- 不改正式 BC criterion、量化器、搜尋、knee；不 fine-tune、不加 spike 懲罰、不加一階項；只新增逐位置配對的診斷分數，pipeline 不使用它。
- 不用 INT4 實測 NLL 換算 BC 尺度；比較排序與干預效果的相對重要性，不要求 proxy 等於 NLL。

## tap、格式、條件（沿用上一輪，不調整）
- tap：L1.E9（known_case）, L1.E11（position0）, L1.E18（position0）, L2.E30（position0）, L6.E59（int4_improvement）, L1.E37（control_L1）, L2.E15（control_L2）, L6.E44（control_L6）。關鍵 channel：{'L1.E9': 915, 'L1.E11': 808, 'L1.E18': 627, 'L2.E30': 742, 'L6.E59': 750}。L6.E59 保留並分開解讀。
- 格式：int4, p6e1；spike 門檻沿用 taps.json；條件 F、P、S、R（3 seeds）、Z。

## 梯度與誤差（BC calibration，FP32）
- loss：所有窗口、所有預測 token 的 CE 加總（與 BC cache 相同的 sum-loss）；route 與權重在 no-grad forward 記錄、backward 中固定（無 router／top-k 梯度）。
- G2_c = Σ_t g²，GS_c = Σ_窗口 (Σ_t g)²；g 與 h（因此 e）來自同一次 forward；只含被路由到 expert 的 rows。
- 控制：梯度路徑的 h 與上一輪擷取路徑逐位元相同、CE 與 dr.forward 相同；G2／GS 重現 cache；S1／S2 與 T_old(F) 重現 cache；各條件的 T_old 重現上一輪 BC 反事實。不一致時先解釋差異。

## 公式（只改位置配對）
- r = e − μ_c；V_old = Σ_c (Σ_t g²)/n · Σ_t r²（= BC 的 G2/n·(S2 − nμ²)）；V_pair = Σ_c Σ_t g² r²；B = Σ_c GS_c μ_c²（不變）。
- T_old = V_old + B；T_pair = V_pair + B；逐 channel 驗證 V_pair,c − V_old,c = n·Cov_t(g², r²)（母體共變異）。
- 各條件：被保護 rows 的誤差設為 0，重算 μ、r、S1、S2、B；梯度固定；R 取 3 seeds 平均。

## 機制分析（既有資料，只作 discovery）
- 每個 tap × 格式（F）：spike／一般 rows 的 g²、r² 占比與中位數、共變異貢獻、對 V_old／V_pair 的貢獻；關鍵 channel vs 其餘；各窗口對 V_pair − V_old 的貢獻與 leave-one-window-out；同 channel 內打亂 g² 的對照（200 次，seed 23401）。
- 單格「位置錯配」：V_pair/V_old 落在 [1/1.5, 1.5] 之外，且 V_pair 落在打亂分布的中間 95% 之外。「明顯錯配」：[('L1.E9', 'p6e1'), ('L1.E18', 'p6e1'), ('L2.E30', 'int4')] 中至少 2 格成立。
- 與上一輪實測比較只是機制分析，不算驗證。

## 確認（凍結後的新文章）
- 96 篇新 WT103 文章 × 3 窗口；verified 掃描排除 6329 篇（含上一輪：True）、WikiText-2、近似重複；測量沿用上一輪 measure 階段，不改。
- 主要比較：P1 L1.E9 F 下 int4 vs p6e1；P2 L1.E18 F 下 int4 vs p6e1；P3 L2.E30／int4 P vs R；P4 L2.E30／p6e1 P vs F。
- 主要比較用 Holm（4 個，FWER 0.05），文章 bootstrap 雙尾 p；次要比較只報 95% CI。不可判定的比較不算 proxy 排對；同一 tap 的比較不是獨立成功。

## 判讀（依序 C → D → A → B）
- C：沒有明顯錯配。D：T_pair 修正至少 1 個主要比較，但破壞 ≥1 個主要或 ≥2 個決定性次要比較。A：明顯錯配且 T_pair 在決定性主要比較上對得比 T_old 多、沒有破壞主要比較、次要破壞 ≤ 1。B：其餘（有錯配但排序沒改善）。
- 不論結果：不宣稱算到 Hessian、不宣稱 knee 改善、正式 pipeline 不變；記錄逐 token 統計的時間、記憶體、儲存量。

## 預算（分鐘）：{'grad': 5, 'mech': 5, 'data': 5, 'measure': 25, 'analyze': 5, 'figures': 3, 'total_gpu': 35}；不因結果加資料、換 seed 或改門檻。
