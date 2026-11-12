# Claude 執行 prompt：L27 一階評分失敗，來自 RMSNorm 還是 CE 非線性？

請直接實作並完成以下機制診斷。前輪 signed 分數能描述 Posit8 的背景效應，卻把 INT8 選成最佳；本輪先精確拆解失敗餘項，再決定要改 criterion 的哪個部分。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增腳本：`run_qwen3_norm_ce_decomposition.py`。
輸出：`runs/qwen3_norm_ce_decomposition_<實際日期>/`，存在則另開目錄，不覆寫。

只新增實驗與報告，不修改正式BC、搜尋、量化器、配置或native pipeline。不新增文章、層、係數、HVP、STE或fine-tuning。

## 1. 先讀與問題

- `runs/qwen3_conditional_signed_score_20261026/report.md`、`scores.csv`、`frozen_predictions.json`、`stats.json`、`eval_ce_pos.npz`、`controls_calib.json`、`controls_eval.json`、`segments.json`、`tokens.npy`、`freeze.json`。
- `run_qwen3_conditional_signed_score.py`，特別是 Env 的 full/zf/closure/grad、量化前x捕捉、相同batch形狀與前輪重現方式。
- `runs/qwen3_l27_down_source_20261026/report.md`。
- 原 `native_eps_fix_20261014/qwen3` calibration/ranges 與 `qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json`。

已知 F 背景、相對 f16：

| 格式 | 舊一階 S_z | calibration 真值 D | D−S_z |
|---|---:|---:|---:|
| p8e1 | −0.01980 | −0.00453 | +0.01527 |
| p6e1 | −0.08748 | +0.07974 | +0.16721 |
| p4e1 | −0.02261 | +0.25136 | +0.27397 |
| int8 | −0.14508 | +0.37778 | +0.52286 |

分數在 calibration 本身就失配。本輪要查這些有限幅度餘項中，RMSNorm 線性化與輸出 CE 非線性分別呈現多少；不預設一定需要新 Hessian。

## 2. 固定範圍與資料

- Qwen3-0.6B L27 Down，目標權重固定原 A 的 INT8。
- 背景 U/F/W 與前輪完全相同：其他投影FP32／全部依A量化／只量化L0–26 Down權重。
- 九種格式沿用原順序：f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16、fp32。
- f16 是參考與零方向；fp32 activation 仍保留INT8目標權重，不是零方向。
- P0／Rest／All，position127無target，仍保留128位置。
- C：原10×128 calibration。
- H：直接重用前輪128篇×2窗口，token/manifest/hash與順序不變。它已看過，是事後機制延伸，不能稱新文章確認。

開始前固定 plan、公式、資料、數值容差、比較項與selector規則，保存程式/輸入hash。冒煙只用 C，揭露看過的結果，不依結果更換方法。

## 3. 保持前輪完整 input/output QDQ 語意

同背景擷取 input-QDQ 前的 x_b、原 FFN residual r_b、實際 INT8 權重 Wq：

    z_f = Q_y,f(linear(Q_x,f(x_b), Wq))
    z_0 = z_f16
    h_f = r_b+z_f; h_0 = r_b+z_0
    u_f = final_RMSNorm(h_f); u_0 = final_RMSNorm(h_0)
    logits_f = LM_head(u_f); logits_0 = LM_head(u_0)

檢查 LM head 是未量化的 affine 投影（linear，bias如有則保留），其後只有標準next-token CE。若有額外非線性／溫度／其他處理，先報告並定位，不硬套下一節恆等式。

候選z的matmul必須沿用完整forward的相同batch形狀。前輪已發現末位差異會被粗QDQ放大，不能把batch8/2改成一次10而不重驗。

固定同背景 r，z_0 detach為leaf。用同一 baseline loss 的一個 autograd 呼叫取得：

    g_z = ∂L/∂z at z_0
    g_u = ∂L/∂u at u_0

可以 `autograd.grad(loss, (z_leaf,u_0))` 或等價方式，不需每格式backward；確認 g_z 等於透過RMSNorm Jacobian作用到g_u的結果。只經norm/head，無STE、不對QDQ反傳、不更新權重。

## 4. 核心分解

沿用舊一階分數，新增「候選先通過實際Norm」的分數：

    S_z(f) = g_z · (z_f−z_0)
    S_u(f) = g_u · (u_f−u_0)
    R_norm(f) = S_u(f)−S_z(f)

逐窗口、逐預測位置計算，再按P0/Rest/All聚合。

標準affine-head CE在實數運算下滿足：

    D(f) = CE(logits_f,y)−CE(logits_0,y)
         = S_u(f)+KL(p_0 || p_f)

因此：

    D(f)−S_z(f) = R_norm(f)+R_CE(f)
    R_CE(f) = KL(p_0 || p_f)

KL方向是baseline p_0到candidate p_f，不能反過來。p是每個預測位置的完整vocabulary分布；target為下一token。

- R_norm保留正負，表示這個分解下的Norm線性化校正，不能稱為唯一因果歸屬。
- KL在實數下非負，包含CE全部高階餘項，不是只含Hessian二階。
- R_norm可能和KL互相抵消；不能預設S_u一定更準。
- 負的S_u仍可能對應正損傷，不能把負分直接當作安全候選。

## 5. 浮點殘差單獨列出

不要把FP32 GEMM、softmax或CE rounding歸成模型非線性。使用同一份**實際FP32 logits**，轉float64做診斷：

    D_64 = CE64(logits_f,y)−CE64(logits_0,y)
    S_logit,64 = (p_0,64−onehot(y)) · (logits_f−logits_0)
    KL_64 = Σ_v p_0,64(v)[logp_0,64(v)−logp_f,64(v)]
    eps_head = S_logit,64−S_u
    eps_CE = D_actual_FP32CE−D_64

核對兩個恆等式：

    D_64 = S_logit,64+KL_64
    D_actual = S_z+R_norm+KL_64+eps_head+eps_CE

使用log_softmax/logsumexp穩定計算，不用log(已underflow的softmax)。對極小負KL保留數值與容差，不直接clip成0；若大幅負值或重建不過，先修實作。

float64只用於相同FP32 logits的參考運算，不能宣稱整個模型已用float64重跑。

避免儲存全部128篇×vocab的float64 logits。head依前輪相同batch形狀計算，之後可逐窗口／位置分塊轉float64，保存分項、最大誤差與摘要即可；不得為了省記憶體改變前面QDQ的GEMM形狀而破壞重現。

## 6. C與H上如何分析

C：建立 S_z/S_u，重現前輪 S_z 與D。保存候選排序、P0/Rest/All分項，凍結新的 S_u selector。

H：為了逐文章的恆等分解，允許求每篇自己背景下的 baseline g_z/g_u，取得 **local S_z/local S_u**。它們僅作事後機制分析，不是由校準資料凍結後對新文本的預測。

**不能用原10窗口的平均S_z/S_u，去計算H逐文章的R_norm或KL恆等式。** 所有分解項都必須來自同背景、同文章、同位置。

主要先報 F背景 All 的int8、p6e1、p4e1、p8e1：前輪大餘項由哪部分構成？再報P0/Rest與U/W，不能只挑成功格式。

每項報D、S_z、S_u、R_norm、KL、eps_head、eps_CE；C與H分開。H另報平均、中位數、正負號數、數值範圍與取消情形。分項可能超過總餘項或互相抵消，不以百分比宣稱唯一貢獻。

如果RMSNorm校正本身就讓int8／p6等候選恢復合理方向，表示實際Norm變換值得作為低成本評分的一部分；若KL仍主導錯誤，表示輸出CE非線性的代價不可忽略。允許兩者同時重要或部分格式不同，不強迫單一答案。

如報H文章bootstrap，20000次、同一文章重抽涵蓋全部背景/格式，預先固定未使用seed（例如26102671）。因H已看過，區間只作事後不確定性描述，不稱新獨立確認。C不做文章bootstrap。

## 7. S_u是否值得下一輪獨立驗證？

選擇集合與前輪相同：f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16；a bits≤16，目標weight不變。fp32是對照，不參與selector。

只在C按最小S_u,All決定每背景的候選，平手保留f16再按前輪固定順序。不得用H的local分數或真值挑格式。

在H上做探索性重播：

- frozen S_u selector與舊 frozen S_z selector、原BC（f16）的所選格式與實測D。
- 在F真值下，比較 frozen argmin(S_u,F) 與 frozen argmin(S_u,U)，區分條件化是否實際改善選擇。
- 同位元INT／Posit排序、相對f16的好壞方向、是否選到新的嚴重有害格式。

S_u若有改善，只能說值得另用全新文章確認；本輪不追加文章，不根據結果換selector、係數或門檻。若仍選錯，保留失敗，不直接把KL或真實D當新proxy。

## 8. 成本與研究界線

必須分開量：模型初始化/量化權重準備、每背景prefix、一次baseline取兩組梯度、九候選z、九候選RMSNorm、S內積、候選LM head/CE、KL64診斷、控制與分析、峰值VRAM/RAM。

**S_u評分本身不需要每候選LM head／CE，也不需要每候選backward。** 但本輪精確KL需要候選logits，這是診斷成本；不能把取得KL稱為免費proxy，或說將來把精確KL加回criterion不增加推論成本。

不使用前輪「每背景<1秒」當完整成本，清楚區分warm準備後增量與冷啟動/背景切換。L27是特殊的最後一層，不能據此推廣其他層或native成本。

## 9. 控制、輸出與停止

- 三背景、目標INT8、raw x／residual／Wq、原range/LUT、TF32 off、deterministic、FP32/eager/epsilon完全不變。
- 前輪calibration S_z與真值、H全部逐位置CE重現；完整forward與closure一致；同batch語意。
- baseline f16的所有差值與KL為0；fp32不強迫為0；position127梯度與loss貢獻為0。
- S_z/S_u、D/KL、R_norm與數值殘差的P0+Rest=All，逐位置與總量都核對。
- 一次baseline梯度鏈與兩項重建恆等式，全部預先固定abs+rel容差，失敗保留，不挑有利重跑。
- 不把g_z與g_u混用座標，不把Norm後誤差配Norm前梯度。

輸出plan/hashes/controls、逐窗口/位置分解、C/H摘要、S_z/S_u排序、frozen S_u selector、H探索性selector重播、KL與rounding摘要、timing與report.md。

首頁回答：

1. INT8被錯判為有利，主要在哪個位置、哪個分解項出現？
2. Posit6/4的誤判與Posit8成功，是否呈現同一規律？
3. 實際通過RMSNorm再評分是否真的改善方向/排序/selector，還是仍缺CE代價？
4. 下一個值得驗證的是Norm-aware評分、CE餘項的低成本近似、或仍無可用規律？哪些只是待驗假說？

完成後停止：不改正式BC、不混入raw BC、不加新係數/二階/端點分數、不新增層/文章、不跑MILP/knee、不做native新統計或fine-tuning。只依結果提出下一個最小實驗。
