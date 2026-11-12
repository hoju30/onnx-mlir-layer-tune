# Claude 執行 prompt：BC 變異項的 Posit 誤排機制診斷

請直接實作並完成以下診斷。使用者已將焦點從平均項轉到變異／剩餘誤差項；本輪先定位原因，不修改正式 criterion。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議腳本：`run_qwen3_posit_variance_diagnostic.py`。
輸出：`runs/qwen3_posit_variance_diagnostic_<實際日期>/`。存在則另開目錄，不覆寫。

## 1. 背景與先讀檔案

- `runs/qwen3_u_token_full_qdq_replay_20261025/report.md`、`scores.csv`、`same_bit_pairs.csv`。
- `runs/qwen3_posit_mean_bias_20261025/report.md`、`summary_table.csv`、`raw.json`、`controls.json`。
- `runs/qwen3_mean_u_token_validation_20261025/report.md`。
- `runs/qwen3_down_spike_diag_20261023/report.md`、`scores.csv`。
- `run_qwen3_posit_mean_bias.py`、`run_qwen_l27_reversal_diag.py`、`run_qwen_prefix_rank_shift.py`、`qwen3_swiglu_common.py`。

已有結果：只將 U_old 換成 U_token、保留原 V，完整 Down-input QDQ 的84個同位元選擇及336個層內跨位元排序全部不變；仍有L2的6/8-bit、L27的4/6-bit四個明確誤排。

Posit原分數中V的占比約為：L2 6/8-bit均99.4%、L27 4-bit為50.7%、L27 6-bit為91.8%。這是proxy數值構成，不是實際損傷歸屬。

先前spike分組、同位置中心化配對沒有修正主要誤排，不要重做同樣改法後直接稱新修正。

## 2. 研究問題與固定範圍

要區分：
1. 原變異項的位置平均／方向近似，是否在局部曲率階段就失配？
2. 是否是signed first-order使真實損傷排序不同？
3. 是否小幅度合理，但完整r擾動產生高階非線性？

範圍：
- 模型Qwen3-0.6B。
- L27 Down input為主要診斷；L2 Down input為預先指定的另一類失敗點對照。
- 兩層均完整報int4、p4e1、int6、p6e1、int8、p8e1；不得只挑Posit較好的配對。L2的4-bit也是必要對照。
- 每次只介入一層，所有weight與其他activation均FP32，Down output不QDQ。
- 原10×128 calibration，原range、LUT、模型與loss不變。
- 只作這批已反覆看過的calibration機制診斷；無新文章／泛化宣稱。
- 不改U、V、量化器、search、knee或pipeline；不做fine-tune、native編譯或完整配置。

## 3. 原誤差分解必須固定

對每個layer、format：

    e = Q_f(x)−x
    μ_c = mean_(s,p)e_spc，n=10×128
    b_spc = μ_c
    r = e−b

μ用float64算一次並固定。e、μ、b、r在autograd外建立並detach，不穿過QDQ／不使用STE。

本輪研究的是r，不是原full e，也不將r命名為新的量化器。r-only對應人工偏移x+r；數學上等價Q(x)−μ，但須報浮點路徑差異。

固定位置方向：

    r_all=r，r_P0=M_P0 r，r_Rest=M_Rest r。

不得在P0/Rest重新估μ或重新中心化r。mask後的r不一定零均值，但它仍是原r方向的拆分。

即使完整r的channel均值為零，F_r=gᵀr也不一定為零；不能把變異分量的一階項直接設零。

## 4. 變異proxy與方向對齊

g為原FP32模型每窗口summed CE對目標Down input的梯度。

    V_old = Σ_c mean_(s,p)(g_spc²) × Σ_(s,p)r_spc²
    V_elem = Σ_(s,p,c)(g_spc r_spc)²
    V_token = Σ_(s,p)[Σ_c g_spc r_spc]²
    V_window = Σ_s[Σ_(p,c)g_spc r_spc]²

這四種全不加U；V_elem不同於舊T_pair=V_elem+U。各proxy仍不是真正Hessian。

比較V_old與V_elem：只改g²與r²的位置配對；可用協方差恆等式檢查。比較V_elem與V_token：加入同token內跨channel項。V_window再引入跨位置的經驗聚合，不能直接稱真實損傷交互。

對位置拆分，原V的分項定義為：

    V_old,m = Σ_c mean_all(g²)_c × Σ_(s,p)(M_m r)_spc²。

保留原mean_all(g²)、原μ與n；驗證V_old,all=V_old,P0+V_old,Rest。不把mask後r重新丟回Var()並重算均值。

另保存原U作背景資料，但本輪比較r曲率／r-only損傷時，不加U或mean方向F_b。

這不表示masked r重新套用完整BC時的均值項為零；本輪的mask分項是固定原r的方向分解，並非重算一套完整BC。

## 5. 真正r方向的導數與HVP

對每個layer、format、mask，以固定d=M_m r計算：

    F_r = gᵀd
    H_r = ∇_x(gᵀd)，g須create_graph=True
    K_r = ½dᵀH_r

HVP由autograd求得，不以差分冒稱HVP、不建立完整Hessian。F_r/K_r可負，不裁切。

L27的loss closure：真實Down(x′)、原residual、final RMSNorm、LM head與CE。
L2則必須包含介入後的全部真實suffix，包括後續attention及FFN；不能使用L27的norm/head-only closure。

可先無梯度快取未受介入的prefix、target Down input、該層MLP residual與必要position/mask參數，再將目標x detach為leaf以求導；所有後續運算必須保留正確graph。權重不求梯度但不可detach後續activation。

優先microbatch=1／小batch確保二階graph記憶體可控；不同batch的重播loss、g數值差異另核對。若L2二階graph不支援或OOM，先嘗試相同FP32/eager語意的小batch與prefix cache；仍失敗就保留失敗狀態，不改成另一模型／低精度或把FD當HVP。

原loss與導數先用summed CE計算；表格最後除以1270 targets。proxy曲率比較為V_old/2、V_elem/2、V_token/2、V_window/2，對照K_r。

若已取得H_r，可額外以零額外HVP的dot product記錄bᵀH_r，標為平均／剩餘方向的二階混合項；不把它當成完整有限擾動的I_br，也不因此另開mean sweep。

## 6. r-only幅度曲線與完整量化參考

量測所有layer/format/mask：

    ΔL_r,m(α)=L(x+αM_m r)−L(x)
    α∈{0, ±1, ±1/2, ±1/4, ±1/8, ±1/16, ±1/32, ±1/64}。

全部報告，不依結果挑幅度。

    O_r(α)=[ΔL_r(+α)−ΔL_r(−α)]/2
    E_r(α)=[ΔL_r(+α)+ΔL_r(−α)]/2
    R_r(α)=ΔL_r(α)−αF_r−α²K_r。

檢查O_r(α)/α與F_r、E_r(α)/α²與K_r的收斂。α=1的奇偶反應含高階項，不稱純一階／二階。

比較：αF_r、α²K_r、兩者合計、各proxy/2×α²與實測曲線。proxy先和K_r／偶次反應比，不因純二階proxy與含一階的總ΔL不同就直接說曲率錯。

每層/format另量baseline、直接Q_f(x)、x+b（不另掃幅度），保存ΔL_e、ΔL_b、ΔL_r(1)及I_br=ΔL_e−ΔL_b−ΔL_r(1)。只作與前輪一致的參考，不預設可加。L27應重現前輪；L2無舊mean/residual值時保留新增量測標記。

## 7. L27與L2的位置控制不能混用

- F對固定方向是線性的，兩層都應有F_all=F_P0+F_Rest。
- 原V分項兩層都應可加，因為使用固定g平均與固定r。
- L27後段逐位置獨立：K_all=K_P0+K_Rest，且r-only ΔL_all(α)=ΔL_P0(α)+ΔL_Rest(α)，在凍結的數值容差內驗證。
- L2後面有attention，不強迫K與實測ΔL可加。報：

      K_position_cross=K_all−K_P0−K_Rest
      I_position(α)=ΔL_all(α)−ΔL_P0(α)−ΔL_Rest(α)。

  它們可正可負，不能因非零就判控制失敗。
- 原g在position127應為零；causal suffix下H_r的position127也應為零。
- 不直接將position0命名為已證實attention-sink根因，也不根據結果刪掉它。
- 本輪固定原μ後研究r；不能據此排除position0透過μ影響Rest的可能。

## 8. 數值控制與凍結

先用合成方向與舊資料smoke，凍結計畫、程式hashes、容差後正式跑。記錄smoke看過的結果；同一批calibration即使凍結後重跑，也不是獨立驗證。

- 完整forward與真實suffix replay在FP32及介入條件核對；不可默默換用y+Wαr而改浮點語意。
- L27 full、mean-only、r-only loss與前輪核對；原V/U與既有分數比對，標明來源與後端差異。
- V_elem應對應舊表的V_pair，不是T_pair；r-only α=1應等於x+r，不是直接Q(x)。直接Q(x)是另外的full e參考。
- μ分母1280不變，r平均接近零、b+r=e與QDQ重建差異全報。
- g_x=g_y W_down、α=0、position127、TF32 off、deterministic、Posit LUT驗證與裝置端INT scale/zp。
- 使用eager attention以保持二階autograd語意可檢查；不默默改用低精度／不支援二階的fused路徑。
- HVP用梯度中心差分驗證：

      FD_h=[g(x+h d)−g(x−h d)]/(2h)
      h∈{1/32,1/64,1/128}。

  全部步長報向量誤差及½dᵀFD_h對K_r的誤差；使用預先凍結的絕對＋相對容差、檢查相鄰步長收斂。若固定步長尚未夠小或rounding主導，報未定，不挑最吻合的一步。
- 報小α實際改變比例與rounding殘差。若float64參考，確認RMSNorm內部cast並另標記；不得僅model.double()就宣稱完整double參考。
- 非有限值、未支援導數與失敗條件保留，不刪候選或改成正值。
- 不以10個重複使用的窗口作獨立文章bootstrap，不做泛化顯著性宣稱。

## 9. 判讀規則

1. V_old/2與K_r的成對方向相反，且K_r經FD與小α支持：支持變異proxy的局部方向曲率失配。比較V_elem/V_token提供位置配對／channel耦合線索，但不能直接認定唯一根因。
2. V_old與K_r同序，但F_r改變正向損傷排序：記錄signed first-order作用；仍需其他資料驗證，不能直接修改criterion。
3. K_r描述小α，但完整α=1的E／O反應改變排序：支持有限擾動高階效應。區分高階奇次／偶次，不將O(1)叫純一階。
4. 某格式或位置無局部收斂／數值控制不過：保留未定，不能硬作曲率根因判斷。
5. 各層/位元可能有不同原因，L27結論不直接推廣L2。
6. 即使某個proxy在r-only排對，也不代表full e或完整BC已修好；需保留mean/residual補償與完整量化參考。

## 10. 輸出與停止範圍

輸出計畫／hashes、controls、固定全域μ與逐窗口r統計、每channel原V／paired項、全部α loss、F_r/K_r、HVP-FD收斂、P0/Rest貢獻與位置交互、INT−Posit對比、full/mean/residual/I_br參考、曲線圖、timing與report.md。

首頁分L27與L2各一表，欄位：V_old/2、V_elem/2、V_token/2、V_window/2、F_r、K_r、ΔL_r(1)、原full ΔL_e；逐位元成對判斷與數值可信狀態。所有分數／loss明確標單位與是否含1/2，排序為主，不靠數字碰巧接近宣稱已校準。

先回答四個完整誤排在r方向上各屬哪類證據；再列其他兩個同位元對照。報告「已證明／僅線索／未定」的界線。

分開記錄prefix／FP32前向、梯度、每方向HVP、FD控制、QDQ／proxy、曲線評估、峰值記憶體。這是小型診斷成本，不是已實作native建表時間。

完成後停止：不自動改criterion，不追加新文章、跨層全掃、RMSNorm替代、Posit scaling／bias correction、完整配置搜尋或native實驗。依本輪結果再提出一個具體低成本修正假說。
