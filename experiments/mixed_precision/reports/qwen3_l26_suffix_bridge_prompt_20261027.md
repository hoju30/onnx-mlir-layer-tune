# Claude 執行 prompt：L26 Down 的後綴重播與 top-256 評分

請直接實作、完成以下實驗並寫報告。前輪在 L27，固定的 `S_u + Khat256` 成功選到 Posit8，並在新文章上確認有益；本輪把目標移到 L26 Down，先確認相同的 head 評分近似是否仍保留候選排名，以及把後綴重播算進去後是否還省時。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增：`run_qwen3_l26_suffix_bridge.py`。
輸出：`runs/qwen3_l26_suffix_bridge_<實際日期>/`；存在則另開目錄，不覆寫。

只新增實驗與報告，不修改正式 BC、原 A 配置、量化器、MILP／knee 或 native。不增加訓練、STE、HVP、JVP、低秩方法、可調係數或其他目標層。

## 1. 先讀與本輪定位

- `runs/qwen3_topk_ce_curvature_20261026/` 的 report、plan、frozen_predictions、C_terms、controls、bench、stats、freeze、data_freeze。
- `run_qwen3_topk_ce_curvature.py` 與 `reports/qwen3_topk_ce_curvature_prompt_20261026.md`。
- `run_qwen_prefix_rank_shift.py` 的 `forward_capture`、`replay_ce`、kwargs 保存、sampler／audit。
- `run_qwen3_conditional_signed_score.py`、`run_qwen3_norm_ce_decomposition.py` 的 raw x／residual 捕捉、batch 語意及數值控制。
- `runs/qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json`、原 calibration／ranges／LUT。

已知 L27：新資料 F 的 Posit8 相對 f16，D=−0.0093343；只有 S_u 選 INT8，D=+0.3762678。完整計時含 baseline：近似 0.456 s、完整 K2 0.815 s、直接 CE 0.558 s，範圍僅 C 的 L27 評分。

**本輪每個候選都會實際走過 block27。**因此它驗證的是「非最後層目標經真實後綴傳播後，head 二次近似是否仍可壓縮候選評分，以及實際成本」，並沒有取得免重播的 L26 局部曲率／criterion。即使成功，也不能把這一步寫成已解決低成本跨層 proxy。

前輪結果不能直接當作 L26 的正確性或加速證據；本輪也不使用尚未證明的『L27 形式是其他層上界』假說。

## 2. 固定目標、背景與資料

目標：`/model/model/layers.26/mlp/down_proj/MatMul`。
原 A 的該節點是 **weight=int6、activation=f16**。本輪所有條件的目標 weight 固定 INT6，從原始 FP32 權重以既有量化器得到，不反覆量化已量化權重。

只做兩背景：

- F（主要）：其他 195 個投影完全沿用原 A；包含 block27 原本的 weights 與 activation QDQ。
- U（次要）：其他投影的 weight／activation 全 FP32；目標 weight 仍 INT6。

目標的 activation 格式同時套用 input／output：f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16、fp32。selector 為前八個（a bits≤16），fp32 只作控制。

- C：原 10×128 calibration；建立並凍結分數、排序與 selector。
- N：全新 128 篇文章×2 個 128-token 窗口，只量真值。
- 不用舊 top-k N 做冒煙、挑方法或替代新文章。
- k 固定 256；這輪不加 k=1024、不選 k、不調任何係數。

先寫 plan，固定公式、背景、格式、數值容差、主要比較、計時規則與 hash。冒煙只用 C，揭露看過什麼；看見新結果後不得改設計。控制失敗只修實作，保留紀錄與 amendment。

## 3. 真實後綴重播，不能跳過 QDQ

在相同背景、baseline target a=f16 下捕捉：

- x26：目標 input-QDQ 前的 raw Down input。
- r26：layer26 的 post_attention_layernorm 輸入，即 FFN residual。
- block27 的實際 forward kwargs：attention mask、position ids／embeddings、cache position 等，以目前 transformers 介面為準。
- 目標實際 INT6 權重 W26q；固定 range／LUT。

候選：

    z26_f = Q_y,f(linear(Q_x,f(x26), W26q))
    h27_f = r26 + z26_f
    h_final_f = actual_block27(h27_f, original_kwargs)
    u_f = final_RMSNorm(h_final_f)
    logits_f = LM_head(u_f)

block27 的 attention、MLP、residual、Norm 與背景 QDQ 全部使用實際路徑。不要取代成 identity／線性層、關閉後綴 QDQ、凍結中間 activation，或以 baseline 的中間結果覆蓋候選結果。

原 L27 Env／background helper 有硬編碼 target、INT8 與 layer27，不能直接假定套用後已變成 L26。新增明確參數化的實驗 wrapper，檢查原始 hook 與自己 hook 不重複 QDQ，不修改共用正式工具以影響舊實驗。

使用 `use_cache=False`，不在候選之間共享或累積 KV cache。kwargs／mask 的 batch 與序列形狀要對應該批；不能拿前一批的可變 cache 物件給下一批。

target GEMM、block27 及真值 head 維持完整 forward 的 batch8／尾批形狀（C 是8+2）。保留相同數值設定、dtype、TF32 off、eager、deterministic。粗 QDQ 會放大末位差異，不能靠放寬容差掩蓋 batch／layout 錯誤。

final norm 只做一次。保存每候選的 u，用於 C 診斷；benchmark 不能把預先算好的候選 u 當免費輸入。

## 4. baseline 梯度只經 head

把 baseline 的 u0=u_f16 detach 為 leaf，只經原未量化 affine LM head 求：

    g_u = d(summed next-token CE) / du at u0
    delta_u_f = u_f - u0
    S_u(f) = g_u · delta_u_f

每窗口一次 baseline head backward，沒有每候選 backward；不對 block27 或 QDQ 反傳。這裡的 g_u 在最終 Norm 後，不是 L26 的局部梯度。不得把末端 delta_u 配上目標層梯度。

head、baseline 機率 p0、top-256、tail 均值與 Khat256 完全沿用前輪已驗證的實作與精度：

    a_v = W_head,v · delta_u
    q = sum_tail p0
    q*a_tail = w_tail · delta_u
    w_tail = W_head^T(p0 * 1_tail)          # baseline 一次，共用候選
    m = sum_top p0*a_v + q*a_tail
    Khat256 = 1/2 [sum_top p0*(a_v-m)^2 + q*(a_tail-m)^2]
    P256 = S_u + Khat256

top-k 只由該背景／窗口／位置的 baseline p0 決定，機率不重新正規化、平手用 token id；沿用前輪 q≈0 的穩定分支。這輪不順便改成 FP32 softmax、重用梯度算均值或加 tail variance 補償，避免把跨層測試和數值優化混在一起。

## 5. C 上的參考與控制

在 C 的全部背景／格式量：S_u、Khat256、P256、完整 K2_actual、K2_affine、KL 與真實 D，定義及 head rounding 分解沿用前輪。

    D = CE_f - CE_f16
    K2_actual = 1/2 Var_p0(actual_logits_f - actual_logits_0)
    K2_affine = 1/2 Var_p0(W_head*delta_u)

另報 K2_affine−Khat256（tail 方差）、KL−K2_actual（二次近似餘項）、actual／affine 與 baseline probability 的數值差。它們不能混成『跨層傳播誤差』；取得 delta_u 的後綴在本輪是實際重播。

恆等分解仍適用於 final u 與 affine head，須同時報浮點殘差。Khat 只在實數下是 affine K2 的下界，不是 KL 或 D 的下界。

控制至少包括：

- C 全部2背景×9格式：完整 forward 與 replay 的 block27 輸入、final u、逐位置 CE 一致；先固定容差（CE max abs≤1e−6，u 的 abs+rel 容差亦先登記），失敗先定位，不換更寬容差繼續。
- baseline F 就是原 A；C 的逐位置 CE 應重現上一輪 F／f16。U 的目標權重不同於舊 L27 U，不能要求它重現舊 U。
- r26+z26 等於完整 forward 進入 block27 的張量；所有候選都有原後綴 QDQ。
- 目標 weight=原 A INT6，背景 weight pointers／activation map 正確，切換後可重現，hook 沒有倍增。
- baseline f16 的 delta_u／S／K／D 都為0；fp32 只關目標 activation QDQ，不強迫為0。
- head 為未量化 affine，無 softcap／temperature；g_u 與解析 W^T(p−onehot) 的比較，沿用適當精度控制。
- 末端位置127沒有target，head gradient／score貢獻為0；synthetic top-k、tail 恆等式與數值精度控制沿用前輪。
- score／D 的 P0+Rest=All。

**P0／Rest 在本輪只指最終 CE 的位置。**L26 的源位置擾動可以經 block27 attention 影響後續位置，不能把『最終P0收益』直接歸因成『只量化源P0的收益』。本輪不新增源位置 mask／注入實驗，不宣稱源P0與Rest擾動的損傷可加。

## 6. C 凍結、N 驗證

以下 selector／對照只由 C 決定，平手 f16 優先、其餘沿用固定格式順序：

    primary = argmin P256,All
    quadratic_reference = argmin (S_u+K2_actual),All
    measured_C_reference = argmin D_C,All
    first_order = argmin S_u,All
    original_config_control = f16

報全部格式的方向、28對排序、三個同位元INT／Posit對、選擇與margin。將預測、selector、原始A、程式及輸入hash寫入freeze，才產生／查看N結果。

N 使用既有 sampler／manifest／tokenizer／shingle audit，排除所有舊 Qwen 文章、C、前輪 top-k 的128篇以及它們的歷史重用資料。保存排除清單、近重複控制、tokens／segments hash與data freeze；未解決的歷史 tokens 不算 audit 通過。

建議 seed：26102711=文章順序、26102712=窗口抽樣、26102713=bootstrap；先確認未用，若衝突則在看新結果前更換並記錄。

N 的2背景×9格式只量真值，不算新梯度／proxy，不更新range、k或selector。文章為統計單位，兩窗口先合成254-target平均；paired article bootstrap20000次、95%CI，同一resampling共用全部比較。

唯一主要品質比較：F背景的 frozen primary 相對原配置f16的平均D。mean<0且CI上界<0才稱確認改善；若選f16，差值恆0，寫『沒有新增品質收益』。

次要完整報告：

- primary相對 first_order、quadratic_reference、measured_C_reference 的差值與CI。
- 若兩selector選同格式，差恆0；若格式不同但CI跨0，只能說未定，不能稱等價或非劣。
- U結果，以及frozen C分數對N平均損傷的方向／28對排序／同位元三對，報正確／錯誤／未定，成對CI僅作描述。
- N集合內平均最佳只作descriptive oracle，不據此換selector。
- 若所有合理selector都選f16，不強求Posit收益；分開判斷『保留直接CE的選擇』『沒有新增品質收益』。
- 若 primary mean>0.01且CI下界>0，標示嚴重有害。不能以平均排序改善掩蓋最終錯選。

本輪不把次要比較中某個顯著值升格成主要成功，不追加文章或層直到得到有利結果。

## 7. 真正的成本比較必須包含 block27

F背景、C、相同8個selector候選（f16重用baseline，其餘7個實際計算），比較：

1. suffix replay + S_u。
2. suffix replay + S_u + Khat256。
3. suffix replay + S_u + 完整K2（可部署FP32 GEMM＋穩定累加）。
4. suffix replay + 直接head／CE。

每個方法都包含候選target QDQ／GEMM、**每候選 block27、final norm**與它的head評分。不能先算好所有u_f後只比head部分，再稱為L26評分加速。

另分開列head-only診斷時間，用來解釋與L27的差別，但主比較是包含後綴的時間。float64高精度控制不是速度對照的唯一完整K2版本。

計時層次：init與背景準備；取得raw x／r／kwargs的prefix；baseline target／block27／Norm／head／backward／top-k與w_tail；候選target；候選block27；候選Norm；候選head近似／完整K2／CE；gather與記憶體。若prefix捕捉順便完成整個baseline forward，交代已算出的u0／CE及重用方式，不能把這段工作當免費，四方法成本範圍一致。

CUDA同步、5次warmup、20次計時，報median／p90；熱增量、包含baseline的總評分時間、含prefix／初始化的範圍分開。共用初始化組合出的冷時間明確標為重建值，不冒充多個獨立冷進程實測。

top-k W_T快取／gather要納入。前輪10窗口快取約1.33GB，這輪交代chunk與cache策略、peak VRAM／RSS；不得只報候選少量dot的時間。

## 8. 報告與停止

首頁回答：

1. 把目標移到L26後，P256是否保留完整K2／直接CE的候選順序與選擇？C與N分開。
2. 新文章有沒有比原f16配置改善？若沒有，是本來無可確認收益、近似選錯、完整二次也選錯，還是calibration選擇未轉移？
3. 把真實block27重播加進去後，比直接CE到底省多少時間／增加多少顯存？
4. 結果支持繼續往前一層，先研究免逐候選重播的方法，還是停止這個近似？

判讀分開：完整K2對但Khat錯先看tail；完整K2也錯先看CE二次餘項；同C實測selector到N也有落差時另記資料轉移，不把它全歸因近似。排對但不省時，寫『機制有效、時間收益未成立』。

輸出plan／hash／amendments、C控制與分項、frozen predictions、N audit／data freeze／逐位置CE與文章分析、timing／memory、report.md。

完成後停止。不改正式BC、不混加raw BC、不擴到L25或其他節點、不改k、不重新MILP／knee／native、不fine-tune。即使成功，本輪仍包含每候選真實後綴重播，只能支持這個橋接步驟，不能宣稱已建立全模型的低成本局部criterion。
