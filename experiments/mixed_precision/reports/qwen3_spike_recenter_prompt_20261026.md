# Claude 執行 prompt：Qwen3 Down-input 的分組平均誤差與 spike 同位置評分

請實作並完成本輪實驗。研究問題是：排除 spike 對一般 token 平均量化誤差的影響，再以 spike 自己的梯度評分，能否改善完整 Down-input QDQ 的格式排序，同時保持低建表成本？

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增腳本：`run_qwen3_spike_recenter.py`。
輸出：`runs/qwen3_spike_recenter_<實際日期>/`，存在則另開目錄，不覆寫。
本輪只新增實驗程式與結果，不修改正式 criterion、共用量化器、搜尋或 native pipeline。

## 1. 先讀與已知事實

- `runs/qwen3_down_spike_diag_20261023/report.md`、`scores.csv`、`per_article_losses.csv`、`controls.json`、`plan.json`、`work/`。
- `runs/qwen3_posit_variance_diagnostic_20261025/report.md`、`summary_table.csv`、`pairs.csv`、`positions.csv`。
- `run_qwen_down_spike_diag.py`、`qwen3_swiglu_common.py`、`wt_manifest_scan.py`。

之前已測過：

- 原 BC、spike 分組敏感度、全元素同位置配對，在新文章上均為 74/78 可判定同位元配對排對。
- 舊分組只改各組的 mean(g²)，仍共用全域 μ、r=e−μ 與 U；舊同位置配對也保留全域 U。
- 四個共同誤排：L2 int6/p6e1、L2 int8/p8e1、L27 int4/p4e1、L27 int6/p6e1。
- 最新診斷在原 10 窗口發現：L2 p6e1 的 mean-only 損傷 2.647、r-only 3.859，完整量化卻只有 0.052 nat/token；有限幅度交互差約 −6.454。這支持檢查全域平均分解，但不是已驗證修正。
- L2 的 all/Rest 局部曲率尚未確認；不能聲稱已排除所有數值因素。L27 4-bit 的 r 方向顯示一階與有限幅度效應，本輪不保證能修正它。

不要把本輪稱為第一次做 spike 分組，也不要預設 Posit 一定應比 INT 好。

## 2. 固定範圍

- Qwen3-0.6B，config 的全部 28 層 Down input：h = SiLU(gate) × up。
- 一次只量化一層的 h；所有權重、其他 activation 與 Down output 維持 FP32。
- 格式固定 int4、p4e1、int6、p6e1、int8、p8e1，另有 FP32 identity。
- 原 10×128 calibration、原 INT range/scale/zero-point、原 Universal Posit LUT、原 loss 不變。
- g 是每個窗口 summed next-token CE 對 h 的 FP32 梯度，127 targets/window。position 127 的 g=0，但仍參與原 1280 rows 與各組樣本數。
- 分數用 float64 累加；不加一階、HVP、FD、額外擬合係數，不更新 proxy 的模型背景。
- 本輪驗證真值始終是直接 Q_f(h) 的完整量化 ΔNLL，不是只施加平均偏差或剩餘誤差。

## 3. 固定遮罩與公式

每層先用 FP32 calibration h 建立一個所有格式共用的 row mask：

    a_sp = max_c |h_spc|
    threshold = 6 × median_(s,p)(a_sp)
    S = {a_sp >= threshold}; N = 其餘 rows

沿用舊程式的 median 與邊界語意，與舊 mask 核對。不掃門檻，不硬編 position 0，不使用 g、e、格式或 diagnostic loss 決定 mask。

對每個格式，e=Q_f(h)−h，定義：

    μ_all,c = mean_all(e_c)
    r_all = e−μ_all
    μ_N,c = mean_N(e_c)

對非空集合 G 與固定 channel 向量 m，定義：

    V_G(m) = Σ_c mean_G(g_c²) × Σ_(s,p)∈G (e_spc−m_c)²
    U_G(m) = Σ_(s,c) m_c² × [Σ_(p:(s,p)∈G) g_spc]²
    P_G(m) = Σ_((s,p)∈G,c) g_spc² × (e_spc−m_c)²

mean_G 的分母為該組實際 row 數，包括組內 g=0 的最後位置。U_N 只對 N 的位置加總梯度；不可只改 μ 而仍用全位置的梯度和。不把 U 改成 token-level。

需要固定報告六種分數：

    A   = V_all(μ_all) + U_all(μ_all)                  # 原 BC
    P   = P_all(μ_all) + U_all(μ_all)                  # 舊全元素同位置配對
    G00 = V_N(μ_all) + V_S(μ_all) + U_all(μ_all)       # 舊 spike 分組
    G01 = V_N(μ_all) + P_S(μ_all) + U_all(μ_all)       # 保留全域分解，只細化 spike 配對
    G10 = V_N(μ_N) + U_N(μ_N) + V_S(0)                # 新分解，spike 用組平均 g² 乘原 e²
    D = G11 = V_N(μ_N) + U_N(μ_N) + P_S(0)           # 唯一預先指定的主要候選

核心改動是一般 token 用自己的 μ_N；spike 以原誤差 e、自己的梯度評分，不再讓 spike 的誤差進入 μ_N。這會同時改變平均／剩餘分解與平均項的適用位置，不能簡稱「只換一個 μ 數值」。

G00/G01/G10/G11 是以舊分組為共同起點的四格對照：

- G01−G00：固定全域分解時，spike 同位置配對的影響。
- G11−G10：固定新分解時，同一配對修改的影響。
- G10−G00、G11−G01：固定 spike 聚合方式時，新分解的影響。
- A、P 是歷史基準。不可把 A 與 G00 的差異混稱為重估 μ 的效果。

邊界：

- S 空、N 空，或門檻非正／非有限：G00/G01/G10/G11 明確回退 A，逐位元相同並記原因；不要用 ε 補分母。P 仍按舊全元素公式，不能強迫它等於 A。
- 任一非空組少於 8 rows：報低覆蓋旗標，仍按固定公式計算，不依結果更換回退規則。
- 本輪以上都是未校準 proxy；不把分數直接命名為 nat/token。可為顯示而除以 1270，但必須註明仍是 proxy；實測 ΔNLL 另列。

## 4. 先做快取重播，減少不必要推論

優先讀取舊 `work/h_L*.npy`、`g_L*.npy`，核對模型、calibration、range、量化器、cache hashes 與舊分數。缺少或不相容時，只重做原 calibration 的一次 forward/backward；不可以刷新後或 diagnostic 的 g 替代。

先凍結本輪公式、六方法、mask 規則、數值容差、資料量與決策，再算分數。可在已看過的 calibration 冒煙測試，揭露看過哪些輸出。

保存全部分項、μ_all/μ_N、mask、各方法分數、84 個同位元預測及 leave-one-window-out 結果。已看過的舊 per_article_losses 可作探索性重播，明確與新資料分開；不得據此挑方法、門檻或修改主要候選 D。

省時停止條件：若 D 與 A 的 84 個同位元實際選擇完全相同，完成快取分析與報告後停止，不抽新文章或重跑 evaluation。此時單點選格式的損傷差依定義為 0；它不代表跨層資源分配、其他位元配對或完整配置必然不變。

若選擇有改變，完成下面的新文章驗證。不能因舊資料看起來沒有改善就跳過已指定的驗證，也不能只挑已知四個失敗點。

## 5. 數值控制

- A、P、G00 應重現舊 scores.csv 的 T_old、T_pair、T_group，明確對齊 raw 分數、是否含 U、是否除以 1270。沿用舊 CPU FP32 cache 優先，凍結 abs+rel 容差；換後端須另報差異與選擇是否改變。
- 零誤差全部分數為 0；N 內 e−μ_N 的和接近 0；空組回退可重現。
- 檢查：

      G01−G00 = Σ_(S,c) [g_spc²−mean_S(g_c²)] × r_all,spc²
      G11−G10 = Σ_(S,c) [g_spc²−mean_S(g_c²)] × e_spc²

  這是分數恆等式，不是真實 loss 的交互項。
- 不沿用「常數梯度時所有新舊分數相等」這個不一定成立的控制；不同分解的 U 本來就可能不同。
- Posit LUT 對 Universal；INT scale/zp 裝置正確；CPU/GPU QDQ 在 calibration 逐元素比對。
- TF32 off、deterministic、FP32、eager attention，RMSNorm epsilon 沿用已修正 1e-6。
- identity 零損傷、full forward 與完整 suffix replay 一致；只改 Down input、不改權重或 output，最後位置梯度為 0。沿用已驗證 evaluator 的相同數值路徑。
- 控制失敗先修程式並記 amendment；不靠看過新資料後放寬容差。保留非有限值與失敗記錄。

## 6. 新文章與評估

在所有分數、預測、程式 hashes 與 plan 凍結後才抽資料：

- 128 篇全新 WikiText-103 train 文章，每篇 3 個不重疊 128-token 窗口；Qwen3 tokenizer。
- 用 `wt_manifest_scan.used_wt103_verified`、舊 runner 的歷史 Qwen token audit 與文章／內容／近重複規則，排除本輪前所有已用文章、WT2 與重疊窗口。不只排除 20261023 那一輪。
- 凍結全新 seeds，例如 order=26102601、start=26102602、bootstrap=26102603。
- 同一份新文章與完整 QDQ loss 供全部六方法使用；不需要為六種評分方法各跑一次模型。
- 28 層 ×（identity＋六格式），每次只量化一層。按文章聚合 3 個窗口的 summed CE/381，ΔNLL=量化−同路徑 identity，保留負號。
- 不在新文章重算 calibration μ、g、range 或 spike mask 來改預測。

## 7. 排序、主要終點與決策

主要配對固定 28×3=84：每層 int4/p4e1、int6/p6e1、int8/p8e1。

- 每方法在每對中挑較小 raw 分數；精確相等挑 INT。
- 沿用相對差 <2% 的 proxy tie 旗標，tie 不算排對，但選擇仍依較小分數。全零分數明定為 exact tie，避免 0/0。
- 全部方法共用同一實測可判定集合：文章 bootstrap 10000 次的 INT−Posit ΔNLL 95% CI 排除 0。CI 含 0 報未定。
- 報全 84 對、修正數、新增誤排數、proxy ties、改變的層與位元；特別逐一報 L2 4/6/8-bit、L27 4/6/8-bit，不把當初已知四例當新發現。
- 特定配對的「確認修正」用既有 84 對 simultaneous 95% CI 方法；未校正結果另列。不要以 84 對當 84 個獨立統計樣本。

唯一主要方法比較：D−A。定義每篇文章：

    R_M(article) = mean_84 pairs [ΔNLL(article, layer, method M 所選格式)]
    G(article) = R_D(article)−R_A(article)

所有層、格式與方法共同重抽文章，報 G 的平均與雙側 95% bootstrap CI。這是多個單層條件的平均，不是同時量化 28 層的整體模型損傷。

其他方法的 D−G10、G01−G00、G10−G00、D−G01 等消融比較，預先完整列出比較家族並作 Holm 校正；列為次要機制分析，不可事後從六方法挑最佳者取代 D 的主要結論。

Leave-one-calibration-window-out：固定完整 calibration 的 threshold，只在剩餘 rows 重算各方法需要的 μ、梯度和與分數，報 10 次選擇翻轉率。它是穩定性描述，不是新文本驗證。

下一輪完整配置研究的候選資格，需同時滿足：

1. D−A 的主要 G 之 95% CI 上界 <0。
2. 在 simultaneous 可判定配對上，D 相對 A 的修正數大於新增誤排數，且 L2 6/8-bit 至少一對獲確認修正。
3. 若新資料可判定 L2 4-bit，D 維持其正確選擇；不靠一律降低 Posit 分數取得收益。
4. 上述 L2 確認修正的選擇在 10 次 leave-one-window-out 至少 8 次維持。

達標只支持另立完整配置與 native 成本驗證，不代表本輪已改善 pipeline。未達標則保留原 BC；報告是無收益、不穩定、增加誤排或證據不足，不追加門檻搜尋、其他係數或新資料補樣。

## 8. 成本、輸出與停止

分別量測：cache 讀取、原 calibration forward/backward（若重跑）、QDQ/e 建立、mask、A 與 D 的分數統計、額外統計記憶體、LOO、exploratory replay、新文章 evaluation、峰值 RAM/VRAM。

若快取允許，A/D 統計 kernel 做相同輸入與同步條件的 warmup 後重複計時，報中位數與範圍，不重跑模型。分開共用成本與增量成本，避免將分析所有消融的總時間當成 D 的必要時間。

理論上可在既有 forward/backward 取得 h、g 後增加統計；本輪仍需量實際快取與資料流成本。不能用 Python 快取重播時間宣稱 native 不增加 latency，或宣稱已省掉一個 pass。

輸出 plan.json/md、hashes、controls、cache provenance、scores.csv、components.csv、masks、mu、frozen_predictions、LOO、old_replay（探索性）、新 segments/tokens/per_article_losses（若有）、same_bit_pairs.csv、selection.csv、timing 與 report.md。

報告首頁回答：

1. 新 μ_N 是否讓 L2 Posit 的一般 token 平均項／剩餘項減少？哪些分項改變？分數變小本身不當作成功。
2. L2 6/8-bit 是否真正排對、4-bit 是否維持？L27 與其他層有沒有新增錯誤？
3. 相對舊分組／舊配對，改善來自哪個固定修改？有哪些仍無法區分的因素？
4. 依主要終點是否值得進入完整配置驗證，額外建表成本是多少？

到此停止，不自動修改正式 BC、量化 scale、搜尋、knee、native lowering，不做 fine-tuning、HVP／FD 或追加模型。
