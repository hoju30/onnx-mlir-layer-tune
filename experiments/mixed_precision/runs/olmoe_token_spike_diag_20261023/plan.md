# 預先登記：OLMoE token-level activation spikes 與 BC 排序（診斷）

時間：2026-10-23T02:58:51+0800。以 plan.json 為準。

## 目的與限制
- 參考 arXiv 2405.14428 的 token-wise 尺度指標，只借用量測方式，不實作 QFeM／QFeP。
- 分開回答：token spike 是否存在；是否增加量化誤差與實測損傷；BC 是否因此更容易排錯。
- 不修改 criterion、量化器、搜尋、knee 或 pipeline；不 fine-tune；不加入 SmoothQuant 等方法；本輪沒有干預。

## 資料用途
- discovery：BC calibration（10 × 128，WikiText-2 validation）＋上一輪確認資料（64 篇 WikiText-103 × 3 窗口，結果已看過）＝ 202 窗口；FP32 與 q0。
- BC token 分解只用 calibration 窗口（BC 實際使用的 rows）。
- 確認：凍結後才抽 96 篇新 WikiText-103 文章 × 3 窗口；排除 6074 篇歷史文章，並排除重複與近似重複。不因結果增加資料。

## 指標（不可混稱）
- s_t = max_c |h[t,c]|；R_token = max_t s_t ÷ median_t s_t（每個 expert）。
- R_channel = max_c(max_t |h[t,c]|) ÷ median_c(max_t |h[t,c]|)；R_channel_cal 只用 calibration（＝上一輪 R_cross）。
- 兩者分子相同；log R_token − log R_channel = log median_c(max_t) − log median_t(max_c)。
- 覆蓋量：每個 expert 的 rows（全部／calibration／上一輪資料）與窗口數；R_token@256＝200 次 256-row 子樣本的平均；spike 率；rows 與各指標的 Spearman。
- 窗口內 R_token（該窗口 rows ≥ 8）的中位數與最大值。
- spike token：s_t ≥ 6.0 × median_t s_t（主要）；強 spike ≥ 20.0×；極端 ≥ 100.0×。
- 層級 MoE 類比（次要）：s_t^layer = token 的 8 個 expert 中最大的 s_t；R_token^layer = 該層所有 token 的 max ÷ median。
- 身分：來源、文章、偏移、窗口位置、token ID 與文字、類別、窗口內首次出現、expert、top-k 位置與權重、c*、SiLU(g) 與 up。
- OLMoE 沒有 BOS，窗口不含特殊 token；窗口位置 0 不是文章開頭，也不是 BOS。token ID、位置、上下文分開報告，並附 base rate。
- 層分布：layer × expert 的 R_token、R_channel、最大 |h|、rows；每層畫分布，不只畫最大值；前段 (0, 1)、後段 (14, 15) 與中段都保留。

## 誤差與 BC 分解
- 候選：int4, int6, int8, p4e1, p6e1, p8e1；固定量化器與 range。spike 與一般 rows 的 MSE、誤差能量占比；INT clipping（依來源分開）；spike × outlier channel 的 2×2 占比。
- BC 變異項可精確拆到 row：v_t = Σ_c G2_c/n·(e_tc − μ_c)²。偏差項 Σ_c GS_c μ_c² 有跨 row 項，只能「分攤」成 b_t = Σ_c GS_c μ_c e_tc/n（加總精確、可為負）。兩者都不是每個 token 的真實 BC 成本。重算的 S1／S2／n 與 static cache 比對。

## 選點（不使用任何 BC 誤排資訊）
- 合格：searched expert 的 Down 輸入，calibration rows ≥ 64、discovery rows ≥ 512；排除上一輪用過的 24 個 expert。
- 配對：隨機（seed 23221）抽 4 個 token 強 tap（R_token ≥ p90，每層最多 2 個），各配一個 token 弱 tap（≤ p50）：同層優先，否則最近層；再同 q0 Down activation 位元類別；再最近 log rows。不匹配處列出。
- 不一致案例：2 個 token 主導（R_token ≥ p90 中 R_channel 百分位最低者）、2 個 channel 主導（R_channel ≥ p90 中 R_token 百分位最低者）。
- 已知案例 L1.E9 單獨列出，不併入新案例的統計。

## 確認測量
- FP32 背景，一次只量化一個 Down 輸入 tap，dynamic route，6 個候選；共 78 個 variant；依上一輪耗時估計 8–23 分鐘。
- 在 tap 上記錄確認資料的 R_token、凍結門檻下的 spike rows、spike／一般 rows 的誤差能量與 clipping。

## 主要比較
- P1 同位元 INT／Posit 排序；P2 全部候選（INT-INT、Posit-Posit、跨家族）；P3 決定性／不可判定／BC 平手的數量。
- P4 token 強 vs 弱（4 組配對）：(a) int4 與 p4e1 的平均配對差 ΔNLL(強) − ΔNLL(弱)，文章 bootstrap，Bonferroni 2 個檢定 → 97.5% CI，結論只限選中的 tap；(b) 誤排率差（tap bootstrap，描述性）。
- P5 R_token vs R_channel：12 個新 tap 上與 int4／p4e1 ΔNLL、誤排率的 Spearman（描述性）；4 個不一致案例逐一報告。
- L1.E9：上一輪 4 個決定性誤排是否在新資料上重現（逐對）。

## 統計
- 決定性配對 = 95% 文章 bootstrap CI 排除 0（分類規則，不是檢定）；99% 作為敏感度分析。BC 平手 = 相對差 < 0.02，不計入準確率。
- 不可判定的配對不算排對；同一 tap 的配對不是獨立案例；不使用只取 ΔNLL 為正的強弱比值。
- 10000 次文章 bootstrap（seed 23211）；tap bootstrap seed 23212。
- 預算（分鐘）：{'discovery': 10, 'select': 4, 'data': 5, 'measure': 25, 'analyze': 5, 'figures': 3, 'total_gpu': 40}；測量超過預算則在下一個層組前停止，只分析已完成的 tap。看到確認結果後不改 tap、候選、門檻、seed 或資料。
