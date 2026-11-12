# Claude 執行 prompt：低成本 CE 二次項能否修正 L27 候選排名？

請直接實作、跑完以下實驗並寫報告。這輪不是改正式 BC，而是確認前輪找到的 CE 非線性代價，能不能以低成本近似補進新的 signed 評分，並在全新文章上改善選擇。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增：`run_qwen3_topk_ce_curvature.py`。
輸出：`runs/qwen3_topk_ce_curvature_<實際日期>/`；存在則另開目錄，不覆寫。

只新增實驗程式、輸出與報告。不改正式 BC、量化器、原配置、MILP／knee 或 native pipeline；不增加訓練、STE、HVP、新層或可調係數。不要做到一半只交計畫，請完成預定實驗後停止。

## 1. 先讀與已知結果

- `runs/qwen3_norm_ce_decomposition_20261026/` 的 `report.md`、`plan.json`、`scores_C.csv`、`C_decomp.npz`、`frozen_predictions.json`、控制與時間檔。
- `run_qwen3_norm_ce_decomposition.py`、`run_qwen3_conditional_signed_score.py`，以及後者沿用的新文章抽樣／排重程式。
- `reports/qwen3_norm_ce_decomposition_prompt_20261026.md`。
- 原 calibration／ranges／LUT 與 `runs/qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json`。

前輪 F 背景、C 的關鍵結果（nat/token，相對相同背景的 f16）：

| 格式 | S_u | 精確 KL | 真值 D |
|---|---:|---:|---:|
| p8e1 | −0.016190 | +0.011660 | −0.004530 |
| p6e1 | −0.035332 | +0.115066 | +0.079736 |
| p4e1 | +0.020120 | +0.231260 | +0.251359 |
| int8 | −0.162250 | +0.540027 | +0.377777 |

`S_u` 仍選 INT8。事後用 `S_u + K2`，三背景的八個候選在 C 的完整排序都與真值相同、均選 p8e1；但那是樣本內事後結果，且完整 K2 要做每個候選的 vocabulary head。

本輪問題是：固定的便宜近似是否保留這個優勢，以及實際省多少時間。不要把本輪結果解讀成原 BC 的所有誤差已被定位。

## 2. 固定範圍與計畫

- Qwen3-0.6B，僅 L27 Down，GPU LMReplay。
- 目標 weight 在所有條件都固定 A 的 INT8；本輪格式名稱指 activation，完整 input／output QDQ 一起變。
- 背景 U／F／W 完全沿用前輪：其他投影 FP32／其他投影依 A 量化／只量化 L0–26 Down 權重。
- 九格式：f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16、fp32。
- selector 集合為前八個（a bits≤16）；fp32 僅作控制，不能當成全模型 FP32。
- P0、Rest、All 定義不變；128 位置、127 個 next-token targets。
- C：原 10×128 calibration，只用它建立分數、排序及凍結 selector。
- 新資料 N：全新 128 篇文章，每篇 2 個 128-token 窗口，只量真值；不得用 N 計算新梯度／proxy、調 k 或重選配置。

先寫 plan，固定公式、k、背景、比較、抽樣規則、容差、成本計法與程式／輸入 hash。主方法 k=256，不能看結果改成其他 k。另算固定 k=1024，僅作 C 上的次要診斷，不進本輪新文章 selector、不作備用贏家。

冒煙只用 C 並揭露看過的結果。控制失敗可修 bug，但須保存失敗原因、amendment 與 hash；不可因結果不好改方法。控制通過後，無論 C 結果好壞都完成已登記的新文章驗證。

## 3. 保持候選與前輪語意

沿用同背景 raw x、residual r、INT8 Wq、range、LUT、rounding、epsilon 與 eager／FP32／TF32 off／deterministic：

    z_f = Q_y,f(linear(Q_x,f(x), Wq))
    u_f = final_RMSNorm(r + z_f)
    delta_u = u_f - u_f16
    S_u(f) = g_u · delta_u

baseline autograd 仍只經 final norm／head，沒有對 QDQ 反傳；每個窗口一次，取 summed CE 的 g_u，不用 mean loss 或額外縮放。

候選 z 的 GEMM 必須保持前輪完整 forward 的 batch 8／尾批形狀；不能把 C 的 8+2 改成 10。候選 head 的真值與精確參考也沿用既有批次。baseline 梯度的 microbatch 1 與完整 head 批次若有浮點差異，要單列，不強迫它們在位元上相同。

先在 C 重現前輪 S_z、S_u、D、KL、K2 與 selector。不要因新近似而換量化器或候選輸出。

## 4. 主方法：top-256 加上合併 tail 平均值

以下全部逐窗口、逐預測位置計算；W 是未量化的 LM head 矩陣，W_v 是第 v 個詞彙的 row。bias 如存在，在 delta logits 中相消。

令 p_v 為 baseline f16 的完整機率，T_k 為該位置按 p_v 排序的前 k 個詞彙（機率相同時以 token id 固定次序）：

    a_v = W_v · delta_u
    w_bar = sum_v p_v W_v = W^T p
    m = w_bar · delta_u
    q = sum_{v not in T_k} p_v
    a_tail = (m - sum_{v in T_k} p_v a_v) / q

主要近似：

    Khat_k = 1/2 [sum_{v in T_k} p_v (a_v - m)^2
                  + q (a_tail - m)^2]
    P_k(f) = S_u(f) + Khat_k(f)

只採 k=256 作主 selector。k=1024 在 C 上以相同規則計算、報告，不另選 k，不加 multiplier、clipping penalty、tail variance 補償或最小位元門檻。

重要：

- top-k 只根據 baseline p，不能依候選結果或標籤挑詞彙。
- 不把 top-k 機率重新正規化；tail 機率 q 必須保留。
- w_bar 是 baseline 的一次性計算，所有候選共用。主定義是 W^T p；可以驗證後重用 summed-CE 梯度的等價式 `g_u + W_y`，但要確認相同 baseline p、dtype、head 與 loss scaling。若不一致，使用明確 W^T p 版本並把成本全部計入。
- 不把 FP32 CE 的 g_u 和另一份 p64 無檢查地混用，再宣稱平均值精確。選定實作精度與概率來源後寫入 plan；累加盡量 float64，便宜 GEMM 可以 FP32，數值誤差另外核對。
- q 接近 0 時採穩定分支並固定容差；q=0 的 tail 項為 0。不能任意 clip 明顯負的 mass 或掩蓋平均值不一致。
- 只儲存所需 baseline top-k ids／機率／w_bar，head rows 按位置分塊 gather；不得一次配置全部窗口×127×k×hidden 的巨大 tensor。

實數下，這個近似滿足：

    K2_affine = 1/2 Var_p(W delta_u)
    K2_affine - Khat_k = 1/2 q Var_{p(.|tail)}(a_v) >= 0

這是對 affine Fisher 二次項的下界，**不是 KL 或實際 NLL 損傷的下界**。省掉的是 tail 內部的方差，不能預設一定足以阻止有害 INT8。

## 5. 精確參考與數值控制

只在 C 計算下列昂貴參考，與 cheap score 的計時路徑分開：

1. 前輪 `K2_actual = 1/2 Var_p(logits_f - logits_0)`：使用實際 FP32 head logits 的差。
2. `K2_affine = 1/2 Var_p(W delta_u)`：與主近似同一 baseline p 的 affine 參考。
3. 前輪 KL 與真值 D。

把 `K2_actual - K2_affine`（head／減法與概率來源的數值差）、`K2_affine - Khat`（tail 方差遺漏）、`KL - K2_actual`（二次近似餘項）分開。不要把它們統稱為 tail 誤差。

可對 C 的固定少數窗口／位置做 float64 affine 數值參考，不必把全模型或所有 head 計算改成 float64；參考範圍事先固定。對所有 C 的 FP32 實作保存誤差摘要與容差，超限須定位。

控制至少包括：

- f16 的 delta_u、S、K、D 都為 0；fp32 不強迫為 0。
- position 127 沒有 target，所有 score／loss 貢獻為 0；其 baseline 機率或 w_bar 不必為 0。
- P0 + Rest = All，以及逐窗口／位置的聚合一致。
- 對小型 synthetic vocabulary：k=V 時回到完整 Fisher 方差；q=0 分支；加入共同 logit shift 不改變 K。
- C 的 nested top-k 滿足 `Khat_256 <= Khat_1024 <= K2_affine`，使用預先固定 abs+rel 容差；不把超限直接裁成相等。
- top-k 實作、tail 均值、w_bar 的抽點參考，以及若使用 g_u+W_y 的等價控制。
- head 仍為未量化 affine，無 softcap／temperature；full forward 與 closure、梯度鏈及舊結果重現沿用前輪控制。

## 6. C 上先回答近似有沒有保留排名

三背景、全部格式分別報 S_z、S_u、Khat256、Khat1024、K2_affine、K2_actual、KL、P256、D；P0／Rest／All 都保留。

主要先看 F 的 p8e1／p6e1／p4e1／int8，再完整報告，不能只挑成功例子。

報告：

- Khat／K2 的絕對遺漏、tail mass、tail 方差遺漏量；接近 0 的項不要用爆炸的相對誤差。
- 對 D 的好壞方向、八候選 28 對的排序、同位元 INT／Posit 三對、argmin 與 margin。
- 是避開 INT8、卻錯選其他有害格式，還是真的選到有利候選？
- k=1024 相比 k=256 是否改善只是 C 上的診斷，不能用來替換主方法。

只由 C 凍結每背景的：

    primary pick = argmin P256,All
    reference pick = argmin (S_u + K2_actual),All
    old pick = argmin S_u,All
    BC control = f16

精確相等時優先 f16，再按既有固定格式順序。所有 prediction／排序／selector／程式 hash 寫入 freeze 檔後，才能產生與查看 N 的結果。

## 7. 全新文章驗證：只看凍結分數能否選對

使用既有 corpus／sampler／近重複排除流程，新抽 128 篇×2 窗口。排除所有舊 Qwen 文章、C、前輪 signed 的 128 篇與其 Norm／CE 重用資料，納入現有 manifest／tokens／shingle audit。沿用已修正的 subset resolution，不把未解決的資料稽核當通過。

plan 中先固定三個未使用 seed（可先查 26102681／82／83 是否未用，衝突則在看新結果之前另選並記錄）。保存文章 id、窗口、tokens、hash、排除清單、近重複控制與 data freeze。

N 對三背景、九格式量實際逐位置 CE。不得更新 range、gradient、proxy、k 或 selector；本輪不算 N local proxy。文章為統計單位，兩窗口先合成該文章的 254-target 平均，不能當 256 篇獨立樣本。

唯一主要品質問題：F 背景下，C 凍結的 P256 pick 相對 BC／f16 的平均 D 是否改善。使用預先固定的 paired article bootstrap 20000 次、95% CI；只有 mean<0 且 CI 上界<0 才稱該主要比較確認改善。若選 f16，差值恆為 0，不稱改善。

其他結果作預先登記的次要效應估計，附 CI，不憑多次比較中某一個顯著值升格為主要成功：

- primary pick 相對 old S_u pick 與 full-K2 reference pick 的實際損傷。
- U／W 的 selector 結果，以及 F 真值下使用 U selector 的差異。
- frozen C 分數的好壞方向、28 對排序與三個同位元對；報正確／錯誤／不確定，成對 CI 判讀僅為描述。
- 集合內 N 平均實測最佳格式僅作 descriptive oracle，不用它回頭改 selector。
- P0／Rest、每篇正負號數及有害尾端；若 mean>0.01 且 CI 下界>0，明確標為嚴重有害，不能用整體排序略有改善淡化。

同一次文章 resampling 共用於所有背景／格式／selector 差值。N 無論成功失敗都完整報告，不追加資料直到得到有利結果。

## 8. 實際成本：不能只報 MAC 減少

top-k 每個位置不同，gather 可能比 dense head 更不划算，所以必須實測。至少分開：

- init、背景量化／切換、prefix capture、一次 baseline head／backward。
- baseline p／top-k 排序、w_bar、top-k row gather／快取的時間與記憶體。
- 八個 selector 候選的 z／Norm／S_u，以及 Khat256 的候選運算與 reduction。
- 相同候選的完整 K2 可部署實作（FP32 GEMM，穩定累加）與直接 FP32 head+CE 的時間。
- float64 高精度診斷、控制、N 評估，分開列；不能拿慢的 float64 診斷當唯一速度基準。

比較 Su、Su+Khat256、Su+完整K2、直接候選head+CE。共同 prefix 與 baseline 不能在一邊計入、一邊排除；baseline 前處理只算一次但必須算。f16 可重用 baseline、K=0，兩邊都採同樣重用，不故意多跑完整 head。

報三種成本：冷的 C 全流程、已準備 prefix 後含 baseline 前處理的評分總時間、baseline 已快取後的候選熱增量。另報 GPU peak VRAM／CPU RSS、k／vocab／hidden 與快取策略。

固定相同工作量與候選數，CUDA synchronize，至少 5 次 warmup、20 次計時，報 median／p90；不在每個內積內過度同步而造成不公平開銷。若 CPU／CUDA 或資料搬運瓶頸，照實列出。

不能從 L27 的 suffix 成本外推全模型 offline latency。便宜近似若能排對卻沒省時間，就明確寫「品質有線索，省時未成立」。

## 9. 交付與結論

輸出 plan／hash／amendments、C 數值與控制、baseline mass／tail 摘要、frozen predictions、N data freeze 與真值、逐文章分析、timing、`report.md`。

報告首頁回答：

1. 固定 top-256 是否補回 INT8／Posit6／4 遺漏的代價？剩餘錯誤是 tail 方差、二次近似餘項，還是其他？
2. C 凍結 selector 選了什麼？N 上是否比 BC／f16 與舊 signed score 好？與完整 K2 差多少？
3. top-k／w_bar／gather 全算進去後，是否真的比完整 K2／直接 CE 省時？
4. 結果支持繼續做哪個最小步驟，或應先停止這個近似？

即使成功，也只支持 L27 GPU 的候選評分研究；原 BC 自帶類二階項，本輪不是證明它少了一個任意可相加的 K2。不要混加 raw BC、宣布正式整合、擴到其他層、重跑 MILP／native 或做 fine-tuning。若 k=256 失敗而 k=1024 只在 C 成功，提出下一輪獨立確認，不能本輪換方法再稱確認成功。
