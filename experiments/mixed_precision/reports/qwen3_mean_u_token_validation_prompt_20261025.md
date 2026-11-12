# Claude 執行 prompt：U_token 平均項的跨層與新文本驗證

請直接在 repository 實作並完成以下實驗。這輪驗證一個已固定的平均項候選，不修改正式 criterion 或完整 pipeline。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議腳本：`run_qwen3_mean_u_token_validation.py`。
輸出：`runs/qwen3_mean_u_token_validation_<實際日期>/`；若已存在另開新目錄，不覆寫舊實驗。

## 1. 背景與目標

先讀：
- `runs/qwen3_posit_mean_bias_20261025/report.md`、`summary_table.csv`、`controls.json`。
- `run_qwen3_posit_mean_bias.py`。
- `runs/qwen3_posit_score_geometry_allbits_20261025/report.md`。
- `run_qwen_prefix_rank_shift.py`、`wt_manifest_scan.py`、`qwen3_swiglu_common.py`。

已知 L27 的 10 個舊 calibration 窗口顯示：
- 4-bit：原 U 對 mean-only 的方向正確，但對 Posit 的相對懲罰較重。
- 6-bit：原 U 的平均方向曲率排序錯；U_token 與 HVP 同序。
- 8-bit：U_token 與 HVP 同序，但一階項會改變正向 mean-only loss 排序。
- 只有單層舊資料，不能證明新文本／其他層有效。
- 即使把 U 換成 U_token，原變異項不動，L27 4-bit 完整 input 分數仍排錯。這輪不處理 position 0 的變異項，也不宣稱修好完整 BC。

研究問題：保留同 token 內跨 channel 的平均誤差方向、避免原 U 的跨位置聚合，能否在新文本及其他層改善平均項的排序？新增建表成本是多少？

結構限制：只有 L27 後段逐位置獨立；L0–26 後面仍有 attention，真實跨位置交互可能存在。不能預設原 U 的跨位置項在所有層都應移除。本輪是在檢驗這個候選能否轉移，而不是先認定它正確。

## 2. 固定範圍

- Qwen3-0.6B，全部 28 層 Down input；層數從 config/mapping 核對，不只挑舊失敗層。
- 每次只介入一層；所有權重（含 target Down）與其餘 activation 全部 FP32。
- Down output 不另外 QDQ；其後層使用原始完整模型。
- 六格式：int4、p4e1、int6、p6e1、int8、p8e1。
- 主配對：每層三組同位元 INT/Posit，共 84 對；沒有跨位元選擇。
- calibration 固定原 10×128 窗口，原 range、量化器、Posit LUT 不變。
- 本輪只測固定 calibration 平均偏移，不測完整量化、完整配置、weight-only、fine-tune 或 native 編譯；不新增 HVP。
- 所有格式使用同一候選規則；不擬合 Posit 專用係數、不掃門檻、不選層例外。

## 3. 新文章前凍結 calibration 量與選擇

對每個 layer、format，在原 calibration FP32 graph 計算：

    e_f = Q_f(x)−x
    μ_f,c = mean_(s,p) e_f,spc，n=10×128
    b_f,spc = μ_f,c
    g = 每窗口 summed CE 對該 Down input 的梯度

    U_old = Σ_(s,c) μ_c²(Σ_p g_spc)²
    U_token = Σ_(s,p)(Σ_c g_spc μ_c)²
    F_b = Σ_(s,p,c) g_spc μ_c

均值分母維持 1280；三種分數最後除以 10×127，表示每 target 尺度：

    S_old = U_old / (2×1270)
    S_new = U_token / (2×1270)
    F = F_b / 1270

凍結兩組 selector：

    q_old_even = argmin(S_old)
    q_new_even = argmin(S_new)

    q_old_signed = argmin(F+S_old)
    q_new_signed = argmin(F+S_new)

signed 比較兩方法都加入同一 F，是為了隔離 U 的變化；不能只對新方法加一階。

以上只在各同位元 pair 內選擇。數值 tie：|S1−S2|≤1e−12×max(1,|S1|,|S2|)，固定選 INT，並另列 tie。signed selectors 同樣處理。不得用新文章改分數／μ／選擇。

保存每 channel μ、原 U／U_token／F、完整配對分數、選擇、hashes。L27 應重現前輪對應統計，先確認控制再凍結。

## 4. 新文本與資料隔離

- 128 篇全新 WT103 文章，每篇兩個不重疊的 128-token 窗口，共 256 窗口。
- 重用 verified manifests、歷史 Qwen token audit 與前輪文章／窗口去重規則。
- 排除所有歷史文章，不只最近一輪；含 prefix_rank_shift、l27_native_confirm、down_spike 與所有後續已有 segments 的實驗。
- 任一窗口 WT2 word-fivegram overlap≥50% 即拒收；其餘 near-duplicate 門檻沿用前輪。未解析歷史來源先處理，不靜默跳過。
- seeds：文章順序 32101、起點 32102、bootstrap 32103。先查是否已被使用；若衝突，在看新 loss 前改成未用的一組並寫入 plan。
- freeze 程式、計畫、selectors 與 calibration 後才抽取正式資料；manifest/tokens hashes 再凍結後才評估新 loss。
- 不因結果擴樣本、換窗口或追加候選。

## 5. 介入與主要量測

使用固定的 calibration μ，在新文章上量測：

    ΔL_(l,f,k)(α) = NLL_k(x_l+αμ_l,f)−NLL_k(FP32)
    α ∈ {+1/8, −1/8, +1, −1}

每次只改一層的 Down input。這是人工平均偏移介入，不是新文章完整 Posit QDQ。

每層都必須執行介入後的真實完整 suffix，包含後續 attention；不能對 L0–26 套用 L27 的 norm/head-only 重播。

每個 layer：FP32 baseline＋6格式×4幅度，共25條件；28層共700個target條件。baseline 可安全共用，但要驗證一致性。

對每個 α>0：

    E(α) = [ΔL(+α)+ΔL(−α)]/2
    O(α) = [ΔL(+α)−ΔL(−α)]/2

E 是固定幅度的偶次反應，包含高階項；不能稱真正 Hessian。O 同樣不是純一階。

主目標 A：q_old_even／q_new_even 對 α0=1/8 的 E(α0) 選擇是否改善。
次目標 B：q_old_signed／q_new_signed 對 α=+1 的實測 ΔL 選擇是否改善。

原 F 在新文本可能失準。另報 O(1/8)/(1/8) 與凍結 F 的差異，標為有限幅度奇次反應，不用它重選格式；signed 次目標失敗不能一律歸因於 U。

報告必須區分曲率 proxy 與正向 loss；不能因一階造成 ΔL 翻序就把曲率 proxy 判錯。

如果保存新文章誤差均值作附加診斷，只能在 selectors 已凍結後計算並獨立標記，不可替換 calibration μ 或重新選格式。本輪主要介入始終使用原 calibration μ。

## 6. 統計與公平比較

文章是統計單位：先平均每篇兩窗口，每篇254 targets。所有條件使用相同文章。

L0–26 的81配對作主要泛化集合 P（L27 是已知案例，另外報告）；每文章平均選擇損傷：

    R_even,m(k) = mean_(pairs in P) E_(k,l,q_m_even)(1/8)
    G_even(k) = R_even,new(k)−R_even,old(k)

    R_signed,m(k) = mean_(pairs in P) ΔL_(k,l,q_m_signed)(+1)
    G_signed(k) = R_signed,new(k)−R_signed,old(k)

負值代表新方法選擇較好。兩者都是單點介入的平均，不是完整配置損傷，禁止相加成 pipeline PPL 收益。

另外完整報告全84對、L27的3對、排除L2與L27的78對；這些均預先指定為補充分析，不能事後挑最好的集合取代主要結果。

文章 paired bootstrap 20000次（同一組 indices 用於全部條件）。G_even 是主要指標，報95% CI；G_signed 是預先指定次要指標，報95% CI。不得把只有 G_even 改善寫成實際正向損傷也改善。若對兩指標作共同的『任一有效』結論，另報 Bonferroni97.5% CI。

G_even 同時報未縮放的實際 E 損傷（nat/token），與除以 α0² 的偶次係數尺度；不要將後者寫成部署時真實 NLL 改善。

逐配對報 INT−Posit 的 E(1/8)、E(1)、ΔL(+1) 與95% CI。E(1/8)的84對另以零中心化 bootstrap 雙尾p做 Holm family；raw95% CI 與Holm結果皆保留。

報告：
- 84對全部結果、可判定／未定數、兩方法排對數與新增誤排。
- 兩方法選擇不同的配對，及其實測差距。
- 4/6/8-bit分別統計、各layer分布。
- 除去已知 L2／L27 的預先指定敏感度分析；不得事後刪掉不利層。
- 校準 score tie 的完整清單與處置；未定配對不從 R 指標剔除。

## 7. 建表成本與實作控制

U_token 需要 μ 已知後計算逐 token 的 g·μ；不能只報最後 dot-product 耗時就稱沒有新增成本。

先檢查目前 native 統計的真實 pass 順序。這輪可採其中一種明確路徑，執行前記錄：
1. 保存原 calibration 的 g，μ 完成後重讀計算 U_token；或
2. 先取得 μ，再跑梯度 pass。

共同前向／梯度、原 U、新 U_token、F、QDQ／μ、快取寫讀、額外 pass、峰值 RAM/GPU與快取大小分開記錄。不必為了比較多做所有實作；未實作的替代路徑只能列成本推估。此為Python診斷，不稱 native latency。

控制：
- GPU FP32、TF32 off、deterministic、既有 INT device-scale/zp、Posit Universal LUT 驗證。
- 正確 layer/slot、權重 FP32、無 output QDQ；mean shift 使用固定 μ，所有條件切換後可重現。
- summed CE 梯度與每target正規化不可混用；μ分母1280不變。
- replay可快取未受介入的prefix，但校準上全部條件需和完整forward的同一介入核對；凍結容差，不以結果放寬。
- 不可直接以 y+αWμ 取代 W(x+αμ)，除非獨立驗證浮點路徑符合控制；優先保持與完整forward相同線性層計算。
- α=0對照、FP32 baseline一致；完整矩陣非有限值保留失敗，不刪掉條件。
- 報α=1/8的張量實際改變比例及rounding殘差；本輪沒有以更小α證明收斂，不能把E(1/8)/(1/8)²稱為真正局部Hessian。
- checkpoints包含程式／資料／μ／selector hashes，續跑先驗證。
- 舊calibration可smoke，不能以新文章loss選batch/chunk或修改計畫。記錄smoke見過的資料。

## 8. 判讀與交付

- G_even 的CI支持改善，且收益不限於少數舊失敗點、沒有清楚新增誤排：支持 U_token 作平均曲率項的下一輪候選；不直接換pipeline。
- 只有G_even改善、G_signed沒有改善：只支持曲率評分進步，保留一階與有限幅度問題。
- 收益集中L2／L27：只作局部結論。
- 新增明確誤排或選擇收益未確認：不採用、不掃新係數；依資料報失敗／未定。
- 即使兩指標都改善，也還沒有驗證完整量化、變異項或完整配置。

輸出計畫／hashes、calibration scores與μ、selectors/freeze、controls、歷史audit、資料manifest/tokens、逐窗口／文章loss、全部配對表、bootstrap統計、建表與評估時間、記憶體、report.md。

報告首頁先回答：新方法改了幾個選擇、修正幾對／新增幾對、G_even與G_signed及CI、改善分布、額外建表成本。不要只列準確率而省略分母／未定數。

完成後停止，不自動修改criterion、追加HVP、處理變異項、進行全模型配置或native實驗。後續是否修變異項與如何整合，依本輪結果另定。
