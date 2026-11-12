# Claude 執行 prompt：定位 L27 Down 的 INT4／Posit4 誤排

本文件是待執行設計，不是已完成結果。請在 `/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision` 實作並跑完以下小型診斷，先凍結計畫再評估新文章。
建議腳本：`run_qwen_l27_slot_attribution.py`。
輸出：`runs/qwen3_l27_slot_attribution_<實際日期>/`。若已存在，使用新目錄，不覆寫舊實驗。

## 1. 要回答的問題

上一輪前綴量化實驗，81 個同位元比較未觀察到排名反轉；L27 的 INT4／Posit4 在前層 FP32 時 BC 就已排錯。
本輪定位這個局部誤排，依序回答：

1. Down input 的 BC 項本身排錯，還是 output 項排錯？
2. 若 input/output 單獨的評分不足以解釋一起量化的排名，串聯量化的交互作用有多大？
3. 原 BC 來自 FP32 權重，但前輪固定 target weight 為 INT8：這個差異是否影響結論？
4. 若 position 0 在分數中重要，其實測任務損傷是否也重要？這是用介入驗證，不能由誤差能量直接推論。

不改正式 criterion、range、量化器、搜尋器、knee；不 fine-tune，不啟動全模型刷新或搜尋，不新增一階、FD、Hessian 或 per-element criterion。
本輪是 GPU FP32 + QDQ 診斷，不能稱為 native 收益或正式 pipeline 改善。
主要格式是 INT4／Posit4，不等同前輪完整配置的 FP16→Posit8 問題；不要把本輪結果直接當成後者的機制證明。

## 2. 先核對來源與已知線索

請讀：

- `runs/qwen3_prefix_rank_shift_20261024/report.md`、`pairwise_results.csv`、`calibration_scores.json`
- `runs/qwen3_down_spike_diag_20261023/report.md`、`scores.csv`、`native_consistency.csv`
- `runs/qwen3_l27_native_confirm_20261024/report.md`
- `runs/qwen3_error_decomposition_20261014/report.md`：先前是 Gate／Up，不能當成本輪 Down 的現成結論。
- `run_qwen_prefix_rank_shift.py`、`run_qwen_down_spike_diag.py`、`run_lp_pipeline_eval.py`、`qwen3_swiglu_common.py`
- `wt_manifest_scan.py`

模型：`qwen3_assets/model` 的 Qwen3-0.6B，28 blocks；target 是 L27 `mlp.down_proj`。
沿用 `runs/native_eps_fix_20261014/qwen3/` 的原 calibration、ranges、BC 表，以及 `runs/qwen3_prepare_20261011/node_mapping.json`。
原配置 A：`runs/qwen3_down_fp16_protection_20261024/configs/A.json`；L27 w=int8，a=f16。核對檔案 SHA256 為 `ac10afa96c7833cacc6f7886857a97c9ba9b4e2cc151712194373f0e9490c06b`。

已看過的舊 calibration PyTorch 統計拆分（不是新實驗結果）：

| 格式 | C_x | C_y | C_x+C_y | native 總 C |
|---|---:|---:|---:|---:|
| int4 | 677.1215 | 5793.0695 | 6470.1910 | 6482.3095 |
| p4e1 | 2772.5191 | 4386.0967 | 7158.6158 | 7176.5869 |

來源是 scores.csv 的 T_old，以及 native_consistency.csv 的 pytorch_x_plus_y；C_y 由總分減 C_x 得到。
它顯示 input 項偏好 INT4、output 項偏好 Posit4，總分偏好 INT4。這只是 proxy 的分解；還沒知道在相同設定下，input-only/output-only 的實測排名如何。
舊 Down-input-only 的 loss 採全 FP32 權重；新 prefix 實驗的 target weight 是 INT8 且 x/y 都量化。不能直接跨實驗相減作因果歸因。

## 3. 主實驗：只變目標 x/y 的量化

除 L27 Down 權重與指定的 activation 介入外，其他投影、前後運算全 FP32。沒有量化前綴。
主背景 W8：L27 Down 權重按 A 量化成 INT8，與 prefix 實驗的 U 背景相同。
次背景 W0：L27 Down 權重維持原 FP32，作為原 BC 統計背景與舊 input-only 研究的對照。
兩背景的 activation range 完全相同；權重在各背景內固定，不因候選或 intervention 反覆量化。

在每個 W 背景，取 x 為未量化的 Down input、W 為該背景固定權重，測試：

| 條件 | Down 實際輸出 |
|---|---|
| baseline | W x |
| X：只量化 input | W Q_x(x) |
| Y：只量化 output | Q_y(W x) |
| XY：兩者都量化 | Q_y(W Q_x(x)) |

以上線性記法假設 target 無 bias；先核對 config/模組，若有 bias，所有條件保留同一 bias，使用實際線性層定義。
不能用 baseline 的 Q_y(Wx) 誤差加到 X 輸出上假裝 XY；XY 的 output 量化必須作用在已改變的實際輸入結果。

格式固定為 `int4`, `p4e1`, `int6`, `p6e1`。4-bit 是主要誤排案例，6-bit 是同層原 BC 排對的對照，不依新結果更換。
條件數：每背景 1 baseline + 4 格式 × 3 介入 = 13，兩背景共 26。
量化前後輸出都加回相同 residual，再經真實最終 RMSNorm、LM head 計算 next-token CE。

## 4. 預先指定的位置介入：W8、4-bit

僅 W8 的 int4／p4e1，對 X、Y、XY 各加以下兩種位置遮罩：

- P0：只有每個窗口 position 0 施加指定 QDQ。
- Rest：只有 position 1–127 施加指定 QDQ。

遮罩在指定 x/y slot 生效，未選中的 row 維持當下張量原值；XY 的 x/y 使用相同 mask。
新增 2 格式 × 3 介入 × 2 遮罩 = 12 條件，加主實驗總共 38。
不新增其他 spike 門檻、不根據 loss 選 token/channel。

所有損傷都除以完整窗口的 127 個 targets，不能 P0 除以 1、Rest 除以 126 後直接相加。
position 0 預測的是 input token 1；position 127 沒有 next-token target，但仍沿用 native 統計的 128 個位置。
L27 Down 之後只有逐位置運算，固定 range 下不應有跨位置的損傷交互：

    ΔL_all ≈ ΔL_P0 + ΔL_Rest

將此作控制；它不同於 X 與 Y 兩個 slot 的交互作用，後者可以非零。不得把「position 0」直接命名成已證實的 attention sink 機制。

## 5. 原 BC 拆分：公式不變，僅診斷各項

使用原完整 FP32 graph、原 10 個 calibration 窗口重現原 BC，不能拿 W8 模型的統計假裝原表。
對 x/y 各保存原張量、g、每格式誤差 e=Q(t)−t；g 是每窗口 summed CE 的梯度。
對 slot t 和 channel c：

    n = 10 × 128
    μ_c = sum_(s,p) e_spc / n
    G2_c = sum_(s,p) g_spc²
    GS_c = sum_s (sum_p g_spc)²
    V_t = sum_c G2_c [sum_(s,p) e_spc²/n − μ_c²]
    B_t = sum_c GS_c μ_c²
    C_t = V_t + B_t
    C_X = C_x, C_Y = C_y, C_XY = C_x+C_y

這裡 B_t 是 activation 的 bias 項，報告中寫作 activation_bias，避免與權重成本 B(w) 混淆。
保存每 channel 的 V、bias、總分與 INT−Posit 分數差，找出哪部分推動誤排。BC 是排序 proxy，不能直接把其數值當作 NLL 或與 NLL 的量級相除。

先核對 C_x+C_y 重現 native 表：沿用既有 1% 最大相對差控制，並額外確認本輪四格式的候選排序一致；報告實際誤差，不調係數對齊。
原表應同時與舊 calibration 拆分核對；若不一致，先找數值/slot/梯度錯誤，不進入正式損傷歸因。

對 W8 的位置介入，可額外用「原 FP32 g 固定」與遮罩後誤差 e_P0、e_Rest 計算原公式的反事實分數，明確標為原 proxy 的行為診斷，不是 W8 的真實曲率。
保留 n=1280 與全部 g，遮罩外 e=0；均值依遮罩後 e 重算。由於全域均值與 bias 項存在，分數不一定可直接相加：

    C_all = C_P0 + C_Rest + C_cross
    C_cross = 2 sum_c (GS_c−G2_c) μ_P0,c μ_Rest,c

逐值驗證此代數恆等式；C_cross 可以為負，不能裁掉。不得稱某個位置的 C_P0/C_all 是唯一、可加的「貢獻比例」。
位置 counterfactual 分數只用來對照遮罩介入的方向，不藉此設計新 criterion 或刪掉 position 0。

## 6. 資料與凍結

Calibration 固定為原 10×128 窗口；不用它挑測試層、位元、mask 或擬合係數。
主要診斷固定 128 篇全新 WT103 文章 × 2 個不重疊 128-token 窗口，每篇 254 targets；所有 38 條件用同一批文章。
重用 verified manifest 與歷史 Qwen token audit，排除所有歷史文章，特別包括 prefix_rank_shift 的 128 篇、native confirm、FP16 protection、Down spike 實驗。
沿用 prefix_rank_shift 的文字/窗口去重規則，包含任一窗口 WT2 word 5-gram overlap ≥50% 即拒收；處理方式在看 loss 前凍結。
seeds 建議文章順序 31101、起點 31102、bootstrap 31103；先檢查沒有衝突。
計畫需記錄已看過的舊結果、所有條件、控制容差、分析規則、版本與 hashes；計畫後修改留 amendment。
正式資料與程式 hash 凍結後才計算新 loss，不依顯著性擴樣本或換窗口。

## 7. 實測統計與因果拆分

文章 k、權重背景 w、介入 S∈{X,Y,XY}、格式 f 的損傷：

    d_(w,S,f,k) = NLL_(w,S,f,k) − NLL_(w,baseline,k)

主要同位元排名差（正值 Posit4 較好）：

    D_S(k) = d_(W8,S,int4,k) − d_(W8,S,p4e1,k)

每格式的 input/output 交互作用與其對排名的影響：

    I_f(k) = d_(W8,XY,f,k) − d_(W8,X,f,k) − d_(W8,Y,f,k)
    J(k)   = I_int4(k) − I_p4e1(k)
    D_XY   = D_X + D_Y + J

主要統計 family 為 D_X、D_Y、D_XY、J 四項：文章 paired bootstrap 20000 次，所有條件共用 indices，報 95% CI 與 Bonferroni 98.75% CI。
先在文章內平均兩窗口，不把 token/window/channel 當獨立文章。
凍結 ε=0.001 nat/token 作為描述實際影響的參考門檻；它不是 BC 分數門檻。
W0 的 4-bit、W8/W0 的 6-bit、權重背景差、位置遮罩全部另列次要分析，報完整效應與 95% CI，不挑顯著結果。

位置分析報每個 S 的 D_all、D_P0、D_Rest，以及精確加總殘差。若要聲稱新的特定位置效應獲多重比較支持，對 3 slots × 2 masks 的 6 個 4-bit D 另列 family 並用 Holm 校正。
除了 NLL，可報每位置/slot 的誤差平方和、clipping 比例、範圍與 channel 集中程度，但不能用它們直接代替任務損傷或宣稱因果。

## 8. 控制與成本

- 使用既有真實模型/量化函式；INT scale/zp 同 device、Posit LUT 對 Universal 驗證、TF32 關閉、deterministic float32。
- 只保留必要的 target hook；X 不能偷偷量化 y，Y 不能偷偷量化 x，XY 不重複套 hook；baseline 不得改變該背景的 target weight。
- W8 量化後權重與前輪相同；每次切換由原始權重/cache 取得，禁止重複量化。
- 在舊 calibration 上，W8 的 XY 必須重現 prefix_rank_shift 的 U 對應 loss；W0 的 X 應對照舊 input-only 路徑，將任何 backend 差異明列。
- 重播 L27 後段應對照完整 forward。先以舊 calibration 確認控制，沿用逐窗口 summed CE 最大差 ≤1e−6 的容差；若不同 CPU/GPU 路徑做原表重現，另列原有相對誤差容差，不混用兩者。
- P0/Rest 的損傷加總、D_XY=D_X+D_Y+J、C 分數的 mask 恆等式均作數值控制。超過預先容差時先定位，不事後放寬。
- g 在 position 127 應為零；報原 g 與 g_y W 的局部鏈式法則檢查。
- 完整矩陣須有限。非有限值保留失敗狀態，不能裁成有限值或刪候選後作主要決策。
- 若使用 cache，需區分 W0/W8、slot、mask、格式、資料與程式 hash；新 loss 不得和舊文章拼接成主要樣本。
- 記錄 calibration backward/分數拆分、資料準備、38 條件評估、分析的 wall time 與峰值記憶體。這是診斷成本，不是 pipeline latency。

## 9. 判讀規則與交付

請逐項回答，允許多個因素同時存在：

1. 若 X 單獨就「BC 偏好 INT4、實測 Posit4 較好」，則 input 項本身已不足；不需要假設只有 XY 交互才產生錯誤。
2. 若 Y 單獨也排錯，分開記錄；不能因 C_y 比較大就推論它是主要誤排來源。應看 INT−Posit 的分數差與實測差。
3. 若 D_X+D_Y 與 D_XY 方向不同，且有足夠 CI 證據，支持串聯介入的非加性改變排名。若 J 雖非零但沒翻轉方向，只能說改變差距。
4. 即使 X/Y 各自排序都對，兩者相加仍可能因相對權重錯配而排錯；此時不可直接把總誤排都歸給交互作用。列出代理兩項差與實測兩項差。
5. W0 與 W8 都誤排：target weight 格式差異不足以單獨解釋問題。只有 W8 誤排：保留當地 weight/activation 交互假說，不能改稱前綴造成。
6. 若 P0 的原 proxy 判斷與 P0 實測損傷不一致，可定位為位置相關的估計失配；但尚不能區分 g² 曲率近似、一階項、幅度非線性或其他因素。不能宣稱刪 P0 就能修好正式 pipeline。
7. 只有在本輪定位 slot/位置後，另行預登記下一輪機制驗證；不要自動追加 FD/HVP、敏感度刷新、格式搜尋或 native 全模型試驗。

輸出 plan.md/json、hashes、controls.json、資料 manifest/tokens、逐條件配置、原 BC x/y 的 V/bias/channel 拆分、每窗口與每文章 loss、位置遮罩結果、interaction.csv、stats.json、timing.json、report.md。
報告首頁用一張表並排：input-only、output-only、兩者一起，各自的 BC 偏好、W8 實測偏好與差距/CI，再加上 W0 對照。
最後回答「目前定位到哪裡」「哪些原因還沒證明」「下一輪最小且可區分假說的測試」，不把局部診斷成功寫成 criterion 或 pipeline 已改善。


## This execution

Frozen 2026-10-24T15:26:50+08:00; script hashes in plan.json. 38 conditions; seeds {'order': 31101, 'start': 31102, 'bootstrap': 31103}.
