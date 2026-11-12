# Claude 執行 prompt：L2 Posit4 的原始量化誤差分區與有限幅度診斷

請直接實作並完成以下實驗。目的：找出「分組平均誤差修正 D 為何修正 L2 6/8-bit，卻嚴重低估 L2 Posit4」的具體來源。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增腳本：`run_qwen3_raw_error_spike_diagnostic.py`。
輸出：`runs/qwen3_raw_error_spike_diagnostic_<實際日期>/`，存在則另開目錄，不覆寫。
本輪完成機制診斷與報告即可，不修改正式 criterion、量化器、搜尋或 native pipeline。

## 1. 先讀與已知事實

- `runs/qwen3_spike_recenter_20261026/report.md`、`components.csv`、`scores.csv`、`masks.npz`、`freeze.json`、`per_article_losses.csv`、`segments.json`、`tokens.npy`、`controls_scores.json`、`controls_eval.json`。
- `runs/qwen3_posit_variance_diagnostic_20261025/report.md`、`summary_table.csv`、`positions.csv`、`controls.json`。
- `runs/qwen3_down_spike_diag_20261023/report.md`、`work/h_L2.npy`、`work/g_L2.npy`。
- `run_qwen3_spike_recenter.py`、`run_qwen3_posit_variance_diagnostic.py`、`run_qwen_down_spike_diag.py`、`qwen3_swiglu_common.py`。

最新結果：

- D = V_N(μ_N)+U_N(μ_N)+Σ_(S,c)g²e²。
- 在新文章上，L2 的 int4/p4e1 完整量化 ΔNLL 為 0.286/3.824；int6/p6e1 為 0.286/0.064；int8/p8e1 為 0.273/0.001。
- D 對三個配對全部改選 Posit，修正 6/8-bit，新增代價極大的 4-bit 誤排。整體單層配對平均損傷增加 0.03624 nat/token，因此已決定保留原 BC。
- D 與不做 spike 同位置配對的 G10 選擇完全相同。
- 前輪 p4e1 的 position-0 **中心化剩餘誤差 r** 損傷很大，但 D 評分的是原始 **e**；r_S 與 e_S 不同，不能直接拿前輪結果當作本輪原因已證明。

本輪不預設損傷一定來自 spike。允許 spike 自身、一般 token、兩者交互，或數值未定等結果。

## 2. 固定範圍與資料

- 只做 Qwen3-0.6B **L2 Down input**，不再掃 28 層。
- 六格式全部報告：int4、p4e1、int6、p6e1、int8、p8e1，加 FP32 identity。
- 一次只改 L2 Down input。Gate/Up/Down 與其他權重、其他 activation、Down output 都維持 FP32。
- 原 calibration、INT range/scale/zero-point、Universal Posit LUT、RMSNorm epsilon、loss 不變。Posit 不新增外部 scaling。
- A：原 10×128 calibration，做全部分區、幅度曲線與原 proxy 分項重現。
- B：直接使用 `qwen3_spike_recenter_20261026` 已有 128 篇×3窗口的 tokens/segments，只重播 α=1 的分區，不抽新文章，不更換文章。
- B 的完整量化結果已看過，本輪是事後機制延伸；不能稱為獨立確認、盲測或新的泛化檢定。
- 開始前凍結 plan、公式、遮罩、幅度 grid、程式與輸入 hashes、數值容差。冒煙只用 calibration；揭露看過的輸出，不能據此修改研究問題或挑格式。

## 3. 遮罩與誤差方向必須固定

從 recenter 輪的 calibration 分項取得 L2 的 frozen threshold，核對等於原 FP32 activation 的：

    a_sp = max_c |x_spc|
    threshold = 6 × median_calibration(a)
    S = {a_sp >= threshold}; N = complement(S)

calibration mask 必須與舊 masks.npz 逐位元相同；不要硬編 position 0。

對 B 的每個窗口，用原 FP32 x 與同一 frozen threshold 建立 S/N；不在 B 重算 median 或調門檻。這只是診斷方向的定義，不更新原校準分數。

每個格式先在原 FP32 x 上計算一次：

    q = Q_f(x)
    e = q−x
    e_S = M_S e
    e_N = M_N e
    e_all = e_S+e_N = e

e、q、S/N 固定且 detach，所有 α 共用。不得減去任何 μ、重新中心化、重新量化 x+αd，或依擾動後 activation 重找 spike。

每份資料完整報 S/N 數量、每窗口與位置覆蓋、空組與低覆蓋。S 空時 e_S=0，N 空時 e_N=0；不刪除窗口或改門檻。

## 4. 干預方式與幅度曲線

對 d∈{e_S,e_N,e_all}，測：

    Δ_f,G(α) = L(x+αd)−L(x)

calibration 固定 grid：α=0，及 ±2^(−k)，k=0,...,10（到 ±1/1024）。全部報告，不事後挑最有利的 α。

α=1 的主要評估使用實際 masked QDQ 路徑：

    S-only: where(S, q, x)
    N-only: where(N, q, x)
    all:    q

α=0 直接使用 x。其他 α 用 x+αd。另報 α=1 的 x+e_G 重建與 masked QDQ 的輸入/loss 差異，避免浮點重建掩蓋原 evaluator 的語意。

對 B，只測 baseline、S-only、N-only、all（α=1），六格式全做；同一 prefix/suffix cache 共用。all 應重現上一輪同文章的 L2 完整 QDQ loss，對齊窗口順序、batch 路徑與正規化。

L2 的 loss closure 必須包含真實 Down、residual、L3–L27 的全部 attention/FFN、final norm、LM head、next-token CE。不可改成最後一層的 head-only closure，或只測 Down 輸出 MSE。

## 5. 直接回答格式誤排來源

每個格式計算有限擾動交互差：

    I_SN,f(α) = Δ_f,all(α)−Δ_f,S(α)−Δ_f,N(α)

它可正可負，不裁切。L2 後續有 attention，不強迫 S/N 的損傷可加；I_SN 非零不是控制失敗，也不是已求得 Hessian cross term。

對三個同位元配對，統一採用「Posit−INT」方向，正值代表 Posit 較傷：

    D_all = Δ_P,all−Δ_I,all
    D_S   = Δ_P,S−Δ_I,S
    D_N   = Δ_P,N−Δ_I,N
    D_int = I_SN,P−I_SN,I
    D_all = D_S+D_N+D_int

完整報各 α 及 α=1 的這個恆等分解。它是指定干預下的差值分解，不是唯一的因果損傷分攤。

主表必須回答：L2 p4e1 相對 int4 多出的約 3.54 nat/token，主要呈現在哪個分項？6/8-bit 為何沒有同樣結果？同時報各格式絕對 Δ 值，不能只看 Posit4。

若分項互相抵消，保留其正負號；不要用可能超過 100% 的比例包裝成唯一歸因。若新分區顯示一般 token 或交互主導，照實報告，不追加另一種 mask。

## 6. 對照 D 的 proxy 與一階資訊

用原 CPU calibration h/g/e 重現 D 的分項與總分：

    P_S = Σ_(S,c) g_spc² e_spc²
    C_N = V_N(μ_N)+U_N(μ_N)
    D = C_N+P_S

分別對照 S-only、N-only、all 的實測損傷。raw proxy 不當作已校準 NLL；若顯示 P_S/2 或 C_N/2，明確註記 1/2 是曲率比較慣例，不代表兩者就是實際 Hessian。C_N 是分數，與第 5 節的損傷差 D_N 不同。

另在 calibration 的同一 GPU loss closure 求一次 FP32 baseline 梯度，以固定 d 計算：

    F_G = g_GPU · e_G
    O_G(α) = [Δ_G(+α)−Δ_G(−α)]/2
    E_G(α) = [Δ_G(+α)+Δ_G(−α)]/2

報 O/α 對 F 的收斂，以及 E/α² 隨尺度變化，對照 P_S/2 等 proxy。保留原 CPU g 用於重現 D，GPU g/F 另標後端；核對 CPU/GPU 差異與梯度鏈 g_x=g_y W_down，不把兩者混作同一精確數值。

本輪不求 HVP、完整 Hessian 或額外有限差分梯度。E(α) 是有限幅度偶次反應，O(α) 含高階奇次；不能稱為純二階／純一階，也不能直接宣稱真實曲率已被驗證。

特別注意：spike 約 3000，α=1/64 仍可能是很大的絕對擾動。逐方向報 α||d||、最大擾動、實際改變比例與重建 rounding。小 α 可能受 FP32 消去誤差影響；未收斂即報未定，不挑一個符合故事的尺度。baseline 重播誤差只能作噪聲參考，不能證明排除所有數值因素。

## 7. 簡單量化統計與控制

- 在 S/N 分別報 |x|、|e| 的分位數、最大值、Σe²、輸出為零的比例，以及 Posit 超過 ±maxpos 的元素數/比例。maxpos 從實際候選值集取得，不能僅靠名稱猜。
- INT 一般 token 是否量化為零，要分 calibration/B 實際量測，不能把前輪 calibration 結論直接套到所有文章。
- 以上統計只用於解釋，不新增 clipping、scale 搜尋、bias correction 或候選篩選。
- 模型/range/LUT/token/cache hashes、TF32 off、deterministic、FP32、eager、RMSNorm epsilon 1e-6。
- Posit LUT 對 Universal；INT scale/zp 在正確裝置。calibration QDQ 與既有結果逐元素比對。
- baseline、直接 full QDQ 與真實 suffix replay 一致；只改 L2 Down input、權重不變、無 output QDQ。
- 原 D 分項/總分、calibration mask 與 B 的上一輪 L2 完整 loss 均重現。
- e_S+e_N=e、α=0、零方向、空組、同位元格式差的恆等分解均核對。最後一個位置沒有 CE target，但仍保留在 mask 與資料中。
- F_all=F_S+F_N 應成立；不要求 Δ 或 E 可加。
- 所有固定數值容差於新分區 loss 前凍結。修程式記 amendment，非有限值或未通過控制保留；不得重跑到出現期望結果才選取。

## 8. 輸出、判讀與停止

calibration：每窗口、每格式、分區、全部 α 的 loss；F、O、E、proxy、實際擾動與 rounding。總量用 summed CE，再除以 1270 targets 顯示 nat/token。

B：每文章（3窗口/381 targets）各格式 S/N/all/I_SN 與三個同位元差值分解；報平均、中位數、文章符號比例與覆蓋。不得把 384 個窗口當獨立文章，或對原 10 calibration 窗口作文章 bootstrap。若另報文章 bootstrap CI，標記為事後描述的不確定性，不能稱新的獨立確認。

保存 plan.json/md、hashes、controls、mask/threshold、固定誤差摘要、curves.csv、per_window.csv、per_article.csv、paired_decomposition.csv、proxy_components.csv、統計、曲線圖、timing 與 report.md。

報告首頁至少三張小表：

1. 六格式 α=1：S-only、N-only、I_SN、完整損傷，以及 C_N、P_S 的 raw proxy。
2. 三個同位元對：D_S、D_N、D_int、D_all 四個實測損傷差；calibration 與 B 分開。
3. 三個同位元對的有限幅度排序、F 的作用、偶次反應有無可靠尺度趨勢，及數值未定處。

分清楚「觀察到完整幅度損傷集中在哪個干預」「局部 proxy 與有限反應不一致」「唯一根因或通用修正尚未證明」。不要寫任何局部 proxy 永遠不可能捕捉 spike 損傷。

分開記錄 prefix/cache、一次 baseline backward、QDQ/統計、calibration 曲線、B 的分區重播、CPU/GPU 峰值記憶體。這是診斷成本，不是 native 建表成本。

完成後只依結果提出下一個具體假說，然後停止。不修改正式 BC 或 D、不新增格式/門檻/係數、不追加層或新文章、不搜尋完整配置、不改 Posit scaling、不做 fine-tuning 或 native 整合。
