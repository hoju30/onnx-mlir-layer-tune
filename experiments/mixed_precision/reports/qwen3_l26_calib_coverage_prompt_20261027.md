# Claude 執行 prompt：L26 的校準選擇反轉，與文章覆蓋或資料來源有關嗎？

請直接實作、跑完以下實驗並寫報告。這輪保持每次校準的 token 數與原量化器不變，用匹配文字的分組實驗，檢查「一篇的10窗口」與「10篇各1窗口」是否影響選擇穩定性，並分開觀察資料來源。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增：`run_qwen3_l26_calib_coverage.py`。
輸出：`runs/qwen3_l26_calib_coverage_<實際日期>/`，存在則另開目錄，不覆寫。

只新增實驗、輸出與報告。不修改正式 BC、原配置、量化器、ranges、MILP／knee 或 native；不更新 sensitive，不算新 S_u／Khat，不訓練、不做 STE／HVP／JVP，不加 margin 或 confidence 選擇規則。

## 1. 先讀與已知事實

- `runs/qwen3_l26_calib_diagnostic_20261027/`：report、source_audit、summary、calibration_windows、leave_one_window_out。
- `run_qwen3_l26_calib_diagnostic.py`。
- `runs/qwen3_l26_suffix_bridge_20261027/`：report、plan、frozen_predictions、controls、C_terms、eval_ce_pos、data_freeze。
- `run_qwen3_l26_suffix_bridge.py` 的 Env、完整 forward、後綴重播與 batch 控制。
- `run_qwen_prefix_rank_shift.py` 及後續修正過的 manifest／tokenizer／near-duplicate audit／新文章 sampler。
- `qwen3_prepare.py:148–155`、`run_native_bc_pipeline.py` 的 calibration／range 準備流程。

原 C0 的10×128窗口全來自 WT2 validation 開頭〈Homarus gammarus〉的前1280 tokens。L26 F 的 INT8 activation 在 C0 實測 D=−0.00089985，但新文章 D=+0.00959477；直接以 C0 的精確 loss 選也會選 INT8。原範圍固定，P256 另有約−0.00138328的樂觀偏差。

本輪先用**校準實測 loss**選格式，暫時移除 proxy 近似這個因素。即使改善，也不能宣稱 BC 或 P256 已改善。

## 2. 固定模型條件與主要問題

- Qwen3-0.6B，僅 `/model/model/layers.26/mlp/down_proj/MatMul`。
- weight 固定原 A 的 INT6；其他195個投影全部沿用 `qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json`，即 F 背景。
- 量化器、所有 ranges、LUT、epsilon、FP32／eager／TF32 off／deterministic、use_cache=False 不變。
- 目標 activation input／output 一起換格式。selector：f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16。fp32可保留為數值控制，不進選擇。
- 每組校準嚴格為10×128=1280 input tokens、1270 next-token targets。窗口內位置各自從0開始。
- C0：原10窗口，作歷史基準，不改內容、不重新校準range。

研究兩個軸：

| 來源 | 單篇覆蓋 | 多篇覆蓋 |
|---|---|---|
| WikiText-2 validation | 一篇的10個連續窗口 | 10篇各1窗口 |
| WikiText-103 train | 一篇的10個連續窗口 | 10篇各1窗口 |

**唯一主要問題**：在 WT2 validation 來源固定、每次1280-token預算固定時，多篇覆蓋是否降低新文章上的平均選擇損傷？WT103 作預先登記的來源重現與次要比較。

## 3. 先固定計畫，再抽校準文章

先寫 plan，固定下面的 pool 數、分組公式、抽樣與去重、seed、數值容差、selector、主要比較與統計。校準與N結果都不能拿來改這些規則。只允許修實作bug並保存amendment／hash。

每來源固定3個 pool，每pool10篇互異文章；同一來源的3個pool文章不重複。每篇至少1280 tokens。先用metadata／token長度檢查可行性，不看模型loss。

已做純CPU粗查：WT2 validation有60篇，獨立tokenize後58篇長度≥1280；這不是去重後資格保證。若排除後不足30篇，保留失敗並停止，不看loss後改門檻／來源／pool數。

建議未使用seed（執行前再查）：26102721=WT2文章順序；26102722=WT103文章順序；26102723=span起點；26102724=N文章順序；26102725=N窗口起點；26102726=bootstrap。衝突只能在查看新loss之前更換並記錄。

文章資格與來源：

- WT2只用現有validation parquet，按頂層文章標題切分，保留raw row／char範圍；排除原 Homarus gammarus，以免將C0直接塞入新pool。
- WT2是既有開發corpus，不宣稱校準corpus本身全新；保存可辨識的歷史重疊，不按舊loss挑文章。
- WT103只用現有train，沿用已驗證的文章metadata、排除WT2同篇／近重複及歷史Qwen文章的規則。不能拿已看過的N文章當新校準。
- 兩來源各自與跨來源都排除選入文章間的exact／near duplicates。先固定既有near-duplicate門檻，不為湊數改門檻。
- 所有條件使用相同長度門檻；本輪校準結論限於符合長度門檻的文章。

每篇以原tokenizer、add_special_tokens=False tokenize。均勻抽一個能容納1280 tokens的起點，取連續1280 tokens，切為10個非重疊128-token窗口。起點規則在兩來源與兩種覆蓋間完全相同，不額外注入BOS或改chat template。

先把文章id／標題／來源／raw rows、原始文字hash、token數、起點與offsets、token hash、排除紀錄及完整校準group manifest寫入 `calibration_data_freeze`，再開始模型計算。

## 4. 同一批文字重新分組，並保留相同 GEMM 位置

每pool建立矩陣 `M[a,j]`：a=0..9是文章，j=0..9是該篇連續span內的窗口。

單篇組：共10組，組a為 `M[a,0],...,M[a,9]`，按j排序。

多篇組：共10組，組r（r=0..9）的第j個窗口為：

    a = (j - r) mod 10
    multi[r,j] = M[a,j], j = 0..9

所以每個多篇組有10篇各1窗口；每個pool的100個窗口，在10個單篇組和10個多篇組中**各恰好使用一次**。兩種覆蓋的總文字、token邊際與窗口內位置完全相同，只改每次10窗口的組成。

**不要shuffle組內順序。**同一 `M[a,j]` 在兩種覆蓋都位於第j格，因此實際batch8＋尾2的shape及batch row都相同。這是避免把已知的QDQ／GEMM末位差異混入覆蓋效果。

每來源3pool×20組=60組，兩來源共120組；每組仍只有1280 tokens。總共有600個unique窗口，各在兩種覆蓋中評估一次。實驗總計算量不等於建議pipeline每次增加到600窗口。

不能把所有組的資料合併成一個大校準集來挑格式，再稱同樣1280-token預算；也不能拿N挑最好的一組。

## 5. 直接實測校準損傷與選擇

每個10窗口組都實際以batch8＋2執行，用既有且已驗證的 L26 wrapper：baseline完整forward捕捉x26／r26／同批kwargs，再對候選重播真實block27、final norm、head／CE。保留F背景QDQ、固定ranges與無KV cache。

不先把600窗口以任意batch算完，再當作各10窗口組的實際結果。可以快取已驗證的共同資料以省成本，但需要證明其數值等價，不能省略下面的共享窗口控制。

每組每格式：

    d_C(f) = mean over 1270 targets [CE(f) - CE(f16)]
    pick = argmin_f d_C(f)

精確平手優先f16，再按固定格式順序。保存所有逐窗口／逐位置CE、D、格式排序、margin、INT8正負窗口數與選擇；不加收益門檻、uncertainty penalty、保留FP16 guard或近似分數。

C0用同一evaluator重現原8＋2的逐位置CE與INT8選擇，不因這輪新組而重新抓range。C0僅作參考，不影響新校準組抽樣或主要比較。

## 6. 分組與數值控制必須先通過

- 原A／INT6 weight／ranges／LUT的hash與pointer；目標原LMReplay hook是no-op，自訂QDQ不重複；其他投影F配置一致。
- C0與前輪逐位置CE重現；完整forward與suffix replay控制沿用前輪，CE max abs≤1e−6，其他tensor abs+rel容差先登記。
- 所有窗口固定128長度、位置重設、mask／position kwargs正確、use_cache=False，沒有跨窗口／batch統計或動態重抓scale。
- 每pool驗證single與multi各完整使用同樣100個窗口；multi每組恰有10篇；shared窗口的batch shape／row保持相同。
- **全部格式、全部共享窗口**比較single與multi的逐位置CE：max abs≤1e−6。若失敗，先定位batch／layout／hook等數值差，不能把差異解讀成覆蓋效果或放寬門檻。
- 每pool把10個single組與10個multi組各自平均，得到的pooled d_C表必須相同（預先固定容差），pooled argmin一致。這是數學控制，不是待驗收益。
- f16的D恆0、fp32只關目標activation、目標weight仍INT6；P0＋Rest＝All。

若兩種覆蓋最後的格式頻率也相同，平均N效果必然相同，照實報零。不要把pooled表必然相同誤寫成『文章覆蓋沒有影響』：本輪檢查的是每次只用10窗口時的選擇。

## 7. 全部選擇凍結後，再抽全新N

保存120個frozen picks、各組排序／margin、C0 pick、程式／資料／配置hash與freeze。然後才抽全新的128篇WT103 train文章，每篇兩個128-token窗口，沿用既有sampler的≥256-token規則及起點抽樣。

N排除所有校準pool文章、C0、歷史Qwen文章與舊N，以及exact／near duplicates。新校準資料的source manifest需明確納入audit／排除；不能因舊manifest scanner尚不認識新格式，就跳過或誤判未解決的token檔。

保存N tokens／segments／hash、排除清單與data freeze。N只量F背景所有候選格式的真值，不更新range、proxy、calibration、group或pick。

N每篇兩窗口先合成254-target平均，文章為統計單位；P0／Rest仍為最終CE位置，不作源位置歸因。

## 8. 主要統計與穩定性

令 `D_n(f)` 為N第n篇文章中格式f相對f16的損傷。對來源s、覆蓋c，定義：

    L_n(s,c) = mean over its 3 pools × 10 frozen picks [D_n(pick)]

這是這些已凍結校準組的**平均選擇後果**，不是一個ensemble配置，也不是用全部600窗口得到的新selector。

唯一主要比較：

    delta_n = L_n(WT2, multi) - L_n(WT2, single)

20000次paired article bootstrap、固定seed、95%CI。mean<0且CI上界<0，才稱「這三個WT2 pool中，多篇覆蓋的平均選擇後果在新N改善」；CI跨0只說未定，不能說等價或普遍無效。

次要結果完整報效應與描述性CI，不升格為主要成功：

- WT103內multi−single，以及各來源每個pool的效果，避免平均值掩蓋反向。
- 相同覆蓋下WT103−WT2的來源間差異；N只來自WT103，不能宣稱某來源普遍最佳。
- 四格L與f16、C0 frozen INT8的差值；若多篇只讓選擇回到f16，寫『避免傷害』，不寫『勝過f16』。
- 每組選什麼、INT8／f16頻率、C margin、平均N損傷、N有害篇數；完整選擇頻率，不只挑表現好的組。
- 相對N集合內平均最佳格式的regret僅作描述；不能用它回頭挑來源／pool／組。
- P0／Rest及各格式C／N位移；所有格式與pool都列出。

同一次N article resampling共用全部比較。30個校準組不是30個獨立校準資料集，尤其multi組共享文章；不要對這些組直接做獨立樣本t檢定或把N的CI當成校準抽樣不確定性。

文章bootstrap只反映**固定這六個pool與120個pick下**的N不確定性；三個pool分別列出一致性，仍不能據此保證所有校準抽樣或模型都改善。

## 9. 成本、交付與停止

記錄metadata／tokenize／audit／sampling、unique窗口數、實際重播窗口數、prefix／candidate suffix／head CE、N評估與控制時間，報peak RAM／VRAM。

本輪是實測loss的診斷；不能把它當成未來BC／P256更新成本，或因新校準選擇較好就宣稱低成本pipeline已成立。

報告首頁回答：

1. 原C0的INT8選擇，是一般single組常見現象，還是這篇／這個起點的個案？只能在本輪pool範圍內回答。
2. 同樣1280-token預算、相同文字總曝光，多篇覆蓋有沒有降低新N的平均選擇傷害？
3. WT2與WT103結果是否一致？來源差異與覆蓋差異能否分開描述？
4. 原ranges固定時，這些資料選擇改變是否已足以避免INT8誤選，或仍未解決？

輸出plan／hash／amendments、校準pool與分組manifest／freeze、C0重現與shared-window控制、全部C真值／picks、N audit／data freeze／真值、統計與timing、report.md。

完成後停止。不改正式BC、不更新sensitive或quantization error、不加P256、不修改ranges、不挑最好pool部署、不擴到L25、不重跑MILP／knee／native、不fine-tune。若得到正結果，下一輪才單獨預登記評分方法在這個資料協定下是否仍成立。
