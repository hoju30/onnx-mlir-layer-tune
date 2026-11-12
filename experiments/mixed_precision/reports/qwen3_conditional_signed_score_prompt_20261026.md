# Claude 執行 prompt：L27 的條件化、帶正負號一階評分

請直接實作並完成本輪 criterion 可行性實驗。目標是從已確認的 L27 背景依賴，驗證一個具體評分假說：在目前量化背景下，使用投影輸出的梯度與候選格式造成的完整輸出變化，能否預測有利／有害方向、position 0 與其餘位置的差異，以及候選排序。

這輪不是先用實測 loss 選候選的局部搜尋，也不是重新修改 BC 的平方梯度係數。先單獨檢驗新分數，不修改正式 BC、搜尋、配置、量化器或 native pipeline。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增腳本：`run_qwen3_conditional_signed_score.py`。
輸出：`runs/qwen3_conditional_signed_score_<實際日期>/`，存在則另開目錄，不覆寫。

## 1. 必讀與已知證據

- `runs/qwen3_l27_down_source_20261026/report.md`、`stats.json`、`native_ce_pos.npz`、`gpu_ce_pos.npz`、`configs/`、`gpu_controls.json`、`native_controls.json`。
- `runs/qwen3_l27_context_followup_20261026/report.md` 與 `runs/qwen3_l27_context_attribution_20261026/report.md`。
- `runs/qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json`、`table_A.json`。
- `runs/native_eps_fix_20261014/qwen3/`。
- `run_qwen3_l27_down_source.py`、`run_qwen3_l27_context_attribution.py`、`run_qwen3_l27_context_followup.py`、`run_qwen3_posit_variance_diagnostic.py`、`run_lp_pipeline_eval.py`、`qwen3_swiglu_common.py`、`wt_manifest_scan.py`。

已知：目標 L27 Down 的權重固定 INT8，activation 在 f16 與 p8e1 間替換。native 已確認 position 0 的 p8e1 收益在量化背景下較大；其背景改變約 −0.00784 nat/token。上游 Down 權重單獨量化，也在 GPU 產生部分效應約 −0.00267。具體誤差補償路徑仍未證明。

前輪端點平方梯度 E 不等於本輪方法。E 會把誤差懲罰放大，沒有直接表示負損傷；本輪固定測 baseline 梯度的 signed dot product，不使用候選端點梯度。

## 2. 範圍與三個固定背景

只做 Qwen3-0.6B **L27 Down**。L27 之後只有 residual、final RMSNorm、LM head 與 CE，可以直接求 post-output-QDQ leaf 的梯度，不需要對量化器做 STE。

固定原 A 為 `qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json`，核對與歷史原 BC knee 的 hash 相同。

- U：其他 195 個投影的 w/a 全部 FP32。
- F：其他 195 個投影全部依 A 的 w/a，是真正完整量化配置背景。
- W：只讓 L0–26 的 Down 權重依 A 量化，其 activation 與其他投影均 FP32；沿用 down_source 的 dn_w 語意。

三個背景下，目標 L27 Down 權重一律使用 A 的 INT8 權重，基準 activation 一律 f16（x/y 都含）。同一背景的所有候選保持相同的目標權重與非目標設定。

本輪不擴展 L25/L26：其完整量化 suffix 包含其他 QDQ，非零梯度需要另外定義 surrogate/STE。不可悄悄將後綴改為 FP32，再稱為相同的完整配置驗證。跨層泛化留待本輪結果之後。

## 3. 固定候選與原始量化語意

候選：p8e1 為主要已知案例；p6e1、p4e1、int8、int6、int4 為固定格式對照／大擾動壓力測試；bf16、fp32 為次要精度對照。另有 f16 baseline，共九個格式設定。

- 原 range/INT scale/zero-point、Universal Posit LUT、weight per-channel 量化與 epsilon 不變。
- Posit 不加外部 scaling；不 clip、不改候選、不使用 diagnostic 資料重估參數。
- f16 才是零方向。fp32 activation 相對 f16 不一定是零，也不表示目標權重切回 FP32。

## 4. 新分數必須在完整輸出槽位建立

每背景 b，用真實量化 forward 擷取：

    x_b：L27 Down 的 input-QDQ 前原始輸入
    r_b：L27 FFN residual 相加前的原 residual
    Wq：目標實際 INT8 量化再表示為FP32的權重

不能把已 f16 量化的 input 當 x_b 再量化一次。hook 順序、三個背景、所有設定都要核對。

對候選 f：

    z_b,f = Q_y,f(linear(Q_x,f(x_b), Wq))
    z_b,0 = z_b,f16
    δz_b,f = z_b,f−z_b,0

所有候選從相同背景的 x_b 與 Wq 建立；完整包含 input/output QDQ，不能只用 x 的 error 或未量化的 linear output 代替。

固定同背景 r_b，將 z_b,0 detach 為需要梯度的 leaf：

    L_b(z) = summed_next_token_CE(LM_head(final_norm(r_b+z)))
    g_b = ∂L_b/∂z evaluated at z_b,0
    S_b(f) = Σ_(s,p,c) g_b,spc × δz_b,f,spc

g_b 用真實 autograd，不能對 QDQ backward；不使用 STE、HVP、Fisher 或候選端點梯度。權重不更新，候選 δz 固定並 detach。

每背景每 calibration 窗口只取一次 baseline 梯度，重用來評全部格式。用 microbatch 1 或預先固定小batch控制記憶體。

S 可正可負，不平方、不取絕對值、不截斷為零。負值代表預測此候選比同背景 f16 更有利。

計算三個固定位置分項：P0（位置0）、Rest（位置1–127）、All。梯度最後位置127應為零，資料仍保留128位置。驗證 S_All=S_P0+S_Rest。

## 5. calibration：分數、近似誤差與數值檢查

沿用原 10×128 calibration，每窗口 summed CE（127 targets）；分數與實際損傷最後除以1270，表明是相對 f16 的一階預測與實測 nat/token 貢獻。

每背景／格式取得：

    S_b,G(f)
    D_b,G(f) = L_b(z_b,0+M_Gδz_b,f)−L_b(z_b,0)

當 G=All，α=1 的 z_b,f 必須重現真實完整 forward 的該背景 x/y 候選 QDQ loss。P0/Rest 的 loss 分解亦應重現前輪逐位置效應與 position-aware hooks。

實測 calibration 真值只用於分析，不依它改公式或挑主要候選；也不依結果刪掉 p4e1 等壓力測試。

對 g 的 directional 控制，在 post-output-QDQ 空間插值：

    z(α)=z_b,0+αM_Gδz_b,f
    α∈{0, ±1, ±1/2, ±1/4, ±1/8, ±1/16, ±1/32, ±1/64}
    O(α)=[ΔL(+α)−ΔL(−α)]/2

全 grid 報 O/α 對 S、實際改變比例與 rounding，使用預先凍結的 abs+rel 容差。大擾動可有高階效應；小尺度未收斂則報未定，不挑最吻合的一點。這是梯度檢查，不建立Hessian，也不把有限幅度odd/even反應冒稱精確一/二階導數。

另外保存只作機制分析的四個 calibration 分數：

    <g_U, δz_U>, <g_U, δz_F>, <g_F, δz_U>, <g_F, δz_F>

相同 token、位置與 hidden channel 對齊，P0/Rest/All 分開。這可觀察條件化梯度與候選擾動各自改變時分數如何變，不是實際模型配置或唯一因果歸因；不使用混搭分數來選最好的方法。

## 6. 分數凍結後，用新文章驗證

在完整 plan、程式hash、數值控制、calibration 分數與預測凍結後，抽128篇全新 WT103 train文章，每篇2個不重疊128-token窗口。

- 沿用 verified manifest、歷史 Qwen token audit、WT2／近重複排除，包含前面所有 L27 與其他研究文章；不能僅排除最近一輪。
- 建議新 seeds order/start/bootstrap=26102661/62/63。計畫前核對是否已使用；若占用則在看任何新loss之前另定未用組並記錄。
- 新文章只作真值：三背景×九格式，固定量化語意，保存逐位置 CE。每篇用兩窗口共254 targets正規化。
- **不在新文章重算 g 或 S，不用新資料更新預測或重估 range。** 使用 frozen calibration scores 預測各背景、各格式在新文章的平均方向與排序。
- 可用相同背景的 prefix/原 residual 快取與最後 norm/head 重播，全部候選共用，不為每個候選重新跑前綴。

本輪主要後端GPU，因目的是先驗證新評分。重用已知 native 窗口核對真實 f16/p8e1 forward 與先前GPU路徑，不據此宣稱本輪新文章的新 criterion 已 native 確認。此輪不再編譯全部候選跑一小時 native。

## 7. 預先固定的主要判讀

研究的是排序與方向；數值量級／相對誤差提供診斷，不因數字碰巧接近就說criterion已校準。

主要案例固定為 p8e1，相對各背景的 f16：

1. 背景改變：frozen S_F−S_U 是否與新文章 D_F−D_U 同方向？W−U是否也捕捉到來源效應？
2. 位置差異：S_F,P0 是否預測有利，S_F,Rest 是否預測有害？All是否預測完整背景下的收益？
3. 公式還是轉移問題：若calibration本身就失配，優先判一階近似不足；若calibration可描述而新文章失配，判文本轉移/代表性問題待查。不能把兩者混稱同一原因。

對新文章用paired article bootstrap 20000次，各背景/格式共同重抽。主比較家族固定六項：F−U的P0/Rest/All、W−U的P0、F背景p8e1的P0/Rest；報95%CI與Holm。背景本身 D_U CI跨零時保留未定，不能因S_U任意正負就判為已確認誤排。

其他格式全部報每背景相對f16方向、同位元INT/Posit排序、跨位元探索性排序。不得只報p8e1成功案例。大擾動（p4e1/int4）失敗需保留，不能將L27 p8成功宣稱為全位元通用。

同時比較原BC的凍結格式排序，清楚標其raw尺度與背景不變的限制。不得把S直接加到原BC表或以係數調到對；本輪不搜尋係數、不加二階項。

## 8. 額外檢查：此分數實際會選到什麼？

每背景在calibration上按最小S_All凍結一個候選，保留f16；選擇集合限定a bits≤16（六INT/Posit、bf16與f16）。fp32僅作精度對照，不參與此同資源上限選擇。目標weight不變，因此候選不增加weight bytes，activation bits不高於基準。

精確平手保留f16，再按預先固定格式順序；另報margin，不事後加門檻。

在新文章上報所選格式的真實D、文章不確定性與資源變化，與原BC同集合選擇/保留f16對照。這是單一投影的selector結果，不是新完整搜尋或pipeline收益。

若p8e1的機制分數合理，但argmin卻挑了嚴重有害的另一格式，要明確判為「局部機制有支持，但不能直接替代候選評分」，不能只稱成功。

## 9. 控制、成本與停止

- 模型/配置/量化器/range/token/hash，TF32 off、deterministic、FP32、eager、epsilon1e−6。
- 三背景與目標INT8權重精確核對；W只含上游Down權重量化，非activation。
- 擷取input-QDQ前x，候選z從同一x/W建立，無doubleQDQ；full forward与最後suffix replay一致。
- baseline f16 δz=0、S=0、Δloss=0；fp32不強迫為0。
- 目標post-QDQ leaf梯度、位置127為0、S位置加總與真值位置加總、α0/α1、quantizer與非有限值均檢查。
- 前輪U/F/W的f16/p8e1結果可在相同歷史文章重現；明確區分重現資料與新文章。
- 新分數梯度只對finalnorm/head傳播，沒有STE；如程式意外需要對上游QDQ反傳，視為實作錯誤先修正。
- 不對原10個反覆使用的calibration窗口做獨立文章bootstrap。所有冒煙輸出與修正揭露；看過新loss後不改方法或擴樣。

分開記錄：每背景前綴/quant準備、baseline backward次數、九候選局部z/QDQ、dot product、calibration曲線/真值控制、新文章真值評估、快取與峰值記憶體。給出只建立S的必要增量成本，不把整個診斷時間算成將來pipeline成本，也不宣稱已量compiler-native建表。

輸出plan/hash/controls、背景/候選定義、calibration g/δz摘要、scores.csv、frozen_predictions/selector、calibration truth、曲線、cross-pair分數、segments/tokens、逐文章與逐位置truth、排名/背景差統計、timing與report.md。

report首頁必須列：

1. U/F/W各背景p8e1的P0/Rest/All：frozen S、calibration真值、新文章真值與CI。
2. F−U、W−U：分數是否描述正確方向；g/δz混搭只提供哪些線索。
3. 六格式/bf16/fp32的成功、失敗、未定，與實際selector是否選到好格式。
4. 本輪支持「條件化有用」「signed項有用」「只適用小擾動」中的哪些部分，哪一部分尚未支持。

到此停止：不改正式BC、不重新跑MILP/knee、不加二階/端點/係數、不新增層或文章、不做fine-tune或STE。跨層或native新評分整合另依本輪結果設計。
