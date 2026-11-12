# Claude 執行 prompt：先定位 Posit 的 activation 平均偏差項 U

請依本規格實作並完成小型機制診斷。本規格將先前的整體方向曲率實驗縮小為「原 BC 的平均偏差項 U」；先前文件若有較廣範圍，以本規格為準。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議腳本：`run_qwen3_posit_mean_bias.py`。
輸出：`runs/qwen3_posit_mean_bias_<實際日期>/`。存在則另開新目錄，不覆寫舊實驗。

## 1. 研究問題

目標是理解並改善 BC 對 Posit 的平均偏差項評分。先不修改正式 criterion、變異項、量化器、range、搜尋或 pipeline。

先閱讀：
- `runs/qwen3_l27_reversal_diag_20261024/report.md`、`stats.json`。
- `runs/qwen3_posit_score_geometry_allbits_20261025/report.md`、`scores.csv`。
- `run_qwen_l27_reversal_diag.py`、`qwen3_swiglu_common.py`。
- `reports/qwen3_posit_hvp_diagnostic_prompt_20261025.md` 的數值控制部分。

線索：在前輪新文章 L27 Down input 的 Rest 分解中，變異項偏好 Posit4，但 U_INT4≈0.235、U_Posit4≈1.235，使總分錯選 INT4。這只定位了分數來源，尚未證明 U 所代表的平均偏移本身真的被高估。

這輪要分辨：
1. 原 U 對平均偏移方向的曲率近似失配。
2. 平均偏移有 signed first-order 或有限擾動效應，純非負 U 無法反映。
3. 平均分量自身可能有損傷，但與剩餘誤差有補償；獨立相加會誤判。

## 2. 固定範圍

- Qwen3-0.6B，L27 Down input，所有權重與其餘 activation 都 FP32。
- 格式固定 int4、p4e1、int6、p6e1、int8、p8e1。
- 4-bit 主分析；6/8-bit 預先指定的對照，全部報告。
- 使用原 10×128 calibration、原 range、既有驗證過的量化函式/LUT。
- 只在這批已反覆觀察的 calibration 定位機制，不新增文章，不宣稱泛化或 pipeline 收益。
- 不擬合 Posit 專用係數、不依 loss 改格式／channel／mask、不刪掉 U 作為正式修正。

## 3. 使用原 μ，拆出平均與剩餘誤差

令 x_spc 為原 FP32 Down input，s=校準窗口、p=位置、c=channel。

    e_f = Q_f(x)−x
    μ_f,c = mean_(s,p) e_f,spc，分母 n=10×128
    b_f,spc = μ_f,c
    r_f = e_f−b_f

μ 只從原完整 e 算一次，之後固定。b 是對每個 token 的同一 channel 施加相同偏移。b/r 是人工誤差分解介入，不是原量化器各自獨立產生的兩種誤差，也不是唯一可加的真實損傷份額。

所有 e、μ、b、r 在 autograd 圖外建立並 detach；量化函式不可用 STE。

主實測條件：
- baseline：x。
- mean-only：x+b。
- residual-only：x+r。
- full：直接 Q_f(x)，另核對 x+e 與 x+b+r 的浮點重建差異。

全部經真實 FP32 Down、原 residual、最終 RMSNorm、LM head 與 summed next-token CE。

逐窗口記錄 ΔL_b、ΔL_r、ΔL_e，以及

    I_br = ΔL_e−ΔL_b−ΔL_r。

不要假定平均分量和剩餘誤差對 loss 可加，也不要將 I_br 省略後把 full 損傷都歸給 U。

## 4. 對齊原 U 的方向，而不是 full e 的方向

原 g 為每窗口 summed CE 對 x 的梯度：

    U = Σ_(s,c) [Σ_p g_spc]² μ_c²
    U_elem = Σ_(s,p,c) (g_spc μ_c)²
    U_token = Σ_(s,p) [Σ_c g_spc μ_c]²

- U 是原 BC 的平均偏差項。
- U−U_elem 隔離跨位置的經驗聚合項。
- U_token−U_elem 隔離同 token 內跨 channel 項。
- U_token 與 U 直接比較會同時改兩種聚合，必須保留 U_elem 橋樑。
- 這些仍是梯度構成的 proxy，不是真正 Hessian；最後 Down 後真實 loss 逐位置獨立，不能將 U 的跨位置項稱為真實 token 損傷交互。

只對固定平均方向 b，用 autograd 算：

    F_b = gᵀb
    H_b = ∇_x(gᵀb)
    K_b = ½bᵀH_b

g 須 create_graph=True；b detach；HVP 由 autograd 計算，不建立完整 Hessian。K_b、F_b 的正負號都保留。

比較 U/2、U_elem/2、U_token/2 與同一 b 方向的 K_b。不要拿 U 去比較 full e 的 HVP 或 full ΔL，就宣稱 U 高估幾倍。

所有導數先以 summed CE 計算；彙整展示最後才除以 10×127 targets。BC 的 μ 內部分母仍是 10×128。

## 5. 平均方向的有限擾動與位置對照

只對 b 掃描：

    ΔL_b(α)=L(x+αb)−L(x)
    α∈{0, ±1, ±1/2, ±1/4, ±1/8, ±1/16}

全數報告，與 αF_b+α²K_b 比較，另列剩餘項。正反方向可拆奇次 O(α)、偶次 E(α)，檢查 O(α)/α 與 F_b、E(α)/α² 與 K_b 的一致性；α=1 的 O/E 含高階項。

P0/Rest 是次要機制對照，固定同一原 μ：

    b_P0=M_P0 b，b_Rest=M_Rest b。

這裡只測固定 b 的位置效應，不重新計算 Rest 的 μ，也不拿它直接解釋前輪「mask 原 e 後重新中心化」的數值。

令 A_sc=g_s,0,c，B_sc=Σ_(p>0)g_spc：

    U0=Σ μ²A²，UR=Σ μ²B²，Ucross=2Σ μ²AB
    U=U0+UR+Ucross。

逐值驗證，不裁負的 cross。這是原 U 按梯度位置的代數拆分，不是重新把 mask 誤差丟回原 BC 的分數。

對固定 b 驗證 F_all=F_P0+F_Rest、K_all=K_P0+K_Rest，以及 mean-only 實測損傷逐位置加總。HVP 與上述 α 曲線均以 all 為主；位置對照至少保留 F/K、α=1 與 α=−1。

## 6. 控制與紀錄

先 smoke、凍結計畫／程式 hash／數值容差，再跑正式量測。完整記錄已看過的舊資料與 smoke；修改須 amendment，不依 desired ranking 調步長或容差。

- 沿用 float32、TF32 關閉、deterministic、裝置端 INT scale/zp 與 Universal 驗證過的 LUT。
- 完整 forward 與 suffix replay 一致；只介入正確 x slot，Down output 不另量化，權重 FP32。
- 重現舊 calibration full loss 與原 U；比對時注意未正規化／每 token 的分母。
- μ 與 r 的平均、b+r=e、x+b+r 與直接 Q(x) 的誤差全部保存；以固定容差處理浮點差異。
- g_x=g_y W_down，position 127 的 g/H_b 應為零；α=0 的 ΔL=0。
- HVP 用梯度中心差分作驗證，h 固定 1/32、1/64、1/128，所有 h 與收斂情況都報告，不挑最吻合的一步；採凍結的絕對＋相對容差。無法收斂時記數值未定。
- 報小 α 的實際改變比例與 rounding 殘差。若加 float64 參考，檢查 RMSNorm 內部 float32 cast，另標記 backend。
- 不刪除非有限值／失敗候選，不裁成正值。
- 列每 channel 的 μ、GS、U_c 及 INT−Posit 差；全表保存，可描述集中程度，不依 channel loss 設新 mask。
- 原 10 個窗口只作描述性結果，不 bootstrap 假裝獨立新文章。

## 7. 判讀

1. U/2 與 K_b 排序相反，且 K_b 經數值控制及小 α 曲線支持：原 U 的平均方向曲率 proxy 失配。再看 U_elem/U_token 哪類聚合差異提供線索，不能直接宣稱某項已是唯一根因。
2. U/2 與 K_b 方向合理，但 signed F_b 或高階反應使 ΔL_b 排序不同：區分缺一階與有限擾動；不能只調 U 比例。
3. mean-only 的損傷確實較大，但 full 損傷較小且 I_br 顯著改變差距／方向：要處理平均與剩餘分量的補償，不能說平均項本身必須打折。
4. 不同格式有不同機制時分開保留；數值不足以區分時報未定。
5. 即使本輪找到 U 的候選修正，也不代表 position 0 的變異項或完整 BC 已修好。

## 8. 交付與停止範圍

輸出計畫／hashes、controls、μ/channel 表、逐窗口 mean/residual/full loss、I_br、mean-only α 曲線、F_b/K_b/HVP-FD 控制、U/U_elem/U_token 成對排序表、位置代數拆分、耗時與峰值記憶體、report.md。

首頁先用一張表並排 INT4 與 Posit4 的 U/2、U_elem/2、U_token/2、F_b、K_b、ΔL_b、ΔL_r、ΔL_e、I_br，再列 6/8-bit。不要將這些不同 proxy 的數值接近當成已校準的證據，排序為主。

報告回答「U 對平均偏移本身是否排錯」「是不是聚合／一階／有限擾動／與 residual 補償」「下一個低成本修正要保留什麼資訊」。分開記錄前向／梯度、HVP、QDQ／統計、介入評估與分析成本。

完成後停止，不自動修改 criterion 或追加新文章／全模型配置。候選修正要另凍結計畫，再用新文章、其他層與原本排對的 INT/Posit 對照驗證。
