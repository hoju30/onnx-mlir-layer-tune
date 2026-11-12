# Claude 執行 prompt：Qwen3 前層量化是否改變後層 activation 候選排名

這是待執行的實驗設計，不是已完成的實驗，也尚未取代實際執行時的 preregistration。請在下列 repository 實作、跑完控制與實驗、寫報告；不得把設計中的假說寫成結果。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增腳本：`run_qwen_prefix_rank_shift.py`。
建議輸出：`runs/qwen3_prefix_rank_shift_<實際日期>/`；若已存在，使用新目錄，不覆寫舊實驗。

## 1. 目的與邊界

使用 Qwen3-0.6B，回答三個分開的問題：

1. 只把前面的 Transformer blocks 量化，是否讓後面 Down activation 候選的「實測相對排名」改變？
2. 原始 BC 排名在兩種背景下的正確率有何不同？區分「原本已經排錯」與「背景改變後才排錯」。
3. 在 calibration 上直接依新背景重選候選，在未見文章上是否比沿用舊背景的選擇更好？收益是否大到值得下一輪研究低成本重新評分？

不修改正式 BC、搜尋器、knee、量化 range、共享量化函式或模型權重補償方法。不 fine-tune，不新增 spike/一階/FD/Hessian criterion，不跑新的全模型搜尋。
本輪不重算 BC 梯度或統計。新背景的「直接實測評分」只是診斷參照，不能稱為 refreshed BC。

本輪主要是 GPU FP32 + QDQ 診斷；不宣稱 native 或真正低位元 kernel 的收益。結論限於本模型的 Down activation、以下候選與背景，不泛稱所有模型、所有投影或完整 pipeline 都成立。

## 2. 先讀與核對

- `runs/qwen3_l27_native_confirm_20261024/report.md`
- `runs/qwen3_down_fp16_protection_20261024/report.md`
- `runs/qwen3_down_spike_diag_20261023/report.md`
- `run_lp_pipeline_eval.py` 的 `LMReplay`
- `run_qwen_down_fp16_protection.py`
- `run_qwen_down_spike_diag.py`
- `qwen3_swiglu_common.py`
- `wt_manifest_scan.py`

沿用：

- 真實模型：`qwen3_assets/model`，不是 tiny-check checkpoint。
- Node mapping：`runs/qwen3_prepare_20261011/node_mapping.json`。
- 固定 ranges、原 BC 表、原 calibration：`runs/native_eps_fix_20261014/qwen3/`。
- 原始 knee A：`runs/qwen3_down_fp16_protection_20261024/configs/A.json`，核對與 `runs/lp_pipeline_regression_20261016/C/qwen3/configs/ref.json` 相同。
- A 檔案 SHA256 應為 `ac10afa96c7833cacc6f7886857a97c9ba9b4e2cc151712194373f0e9490c06b`。

核對 config 的 28 blocks、196 個投影與 mapping；用 mapping 解析層號，禁止以字串排序判斷前後（例如 L10 與 L2）。不要改動 repository 中其他未提交工作。

## 3. 固定介入設計

目標涵蓋全部 L0–L27 的 `mlp.down_proj`，不是只挑已知錯誤層。

每個目標層 i 建立兩個背景：

| 背景 | blocks 0…i−1 | block i 中其他投影 | blocks i+1…27 |
|---|---|---|---|
| U：前層 FP32 | 全部 FP32 | FP32 | FP32 |
| Q：前層依 A 量化 | 所有搜尋投影的 w/a 沿用 A | FP32 | FP32 |

兩個背景的目標 Down 權重都固定為 A 在該投影所指定的格式，並確保量化後權重逐位元相同。因此 U 不應稱為「整個模型純 FP32」。target 權重不隨 activation 候選改變。

Q 的前層包含完整 blocks 中的 attention 與 FFN 搜尋投影，不只 Down；norm、embedding、LM head 等非搜尋項維持原設定。當前 block 的 attention、Gate、Up 維持 FP32。這樣 U→Q 唯一改動的是 i 之前的搜尋投影。

L0 的前層集合為空，U 與 Q 必須相同，作為負控制；L0 不計入主要跨層統計。

每個背景的目標 Down activation 測試：

- `fp32`：背景內 baseline，只停用目標 activation QDQ，目標權重仍保持 A。
- `int4`, `p4e1`, `int6`, `p6e1`, `int8`, `p8e1`, `f16`。

`a` 必須同時作用於 Down input x 與 output y，與 native pass / LMReplay 語意一致。不能誤用上一輪 Down-input-only 實驗的語意。所有 INT 的 x/y ranges、scale/zp 與 Posit LUT 固定，不因背景或評估文章重新校準。

主矩陣：28 層 × 2 背景 × 8 設定 = 448 個條件；L0 重複條件可在驗證相同後去重。每條件跑原 calibration 和正式診斷資料。

主要候選比較：同位元 INT/Posit，27 個非控制層 × 3 對 = 81 對。
`f16` vs `p8e1` 為另列的 27 個次要比較，包含 L27，但資源不同，不能稱為等成本排名。其他跨位元比較只作描述。

## 4. 先固定計畫、資料與選擇，再看正式結果

先寫 plan.md/json，記錄所有背景、候選、公式、門檻、資料規則、seeds、控制容差、分析規則、計算時間預估、版本和 hashes。
允許先用舊 calibration 冒煙測試；記錄看過的資訊。禁止根據新文章結果挑層、換候選、改門檻或增加樣本。

Calibration 固定為原 10 × 128-token 窗口；不增加或更換。其用途是候選選擇與數值控制，不把這 10 個窗口當成 10 篇獨立文章來聲稱泛化。

正式診斷：128 篇新的 WT103 文章，每篇 2 個不重疊 128-token 窗口；每篇 254 個 next-token targets。所有條件使用相同窗口。
使用現有 verified manifest / 歷史 token overlap 檢查，排除已使用文章、WT2 與近重複文本，包括先前 spike、FP16 protection、native confirm 等實驗。先核對剩餘資料足夠。
沿用並明確寫出既有文章/窗口去重規則；本輪還要在抽樣階段拒收任一窗口與 WT2 word 5-gram overlap ≥50% 的文章。不得看 loss 後才排除。

固定 seeds：文章順序 30101、窗口起點 30102、bootstrap 30103；若與已有實驗衝突，在讀正式 loss 前另定並記錄。
凍結文章清單、tokens、程式、config、ranges、BC 表與 selector hashes 後才跑正式診斷。排程可分批、可安全續跑，但不得依效應大小提前停樣本或擴樣本。

## 5. 區分原 BC 排名、舊背景實測排名、新背景實測排名

對每個 i 與同位元候選集合 F_b={INT_b, Posit_b}，在正式評估前凍結三個 selector：

1. q_BC：最小化原 native BC `C[node][a]`。目標 w 固定，B(w) 是相同常數，不影響候選排序。
2. q_U：在原 calibration 上，最小化背景 U 的實測 NLL。
3. q_Q：在同一原 calibration 上，最小化背景 Q 的實測 NLL。

q_U 與 q_Q 比較才能隔離「使用舊背景資訊」與「使用新背景資訊」的差異。q_BC 與 q_Q 的差異同時含 proxy 誤差，不能全部歸因於背景。
q_Q 不是已知真實最佳候選，更不是新 criterion：它仍可能受原 calibration 代表性與噪音限制。

統一凍結 tie 規則：

- BC 若 |C1−C2| ≤ 0.02 max(|C1|,|C2|,1e−30)，標記 proxy tie。
- Calibration 實測 NLL 差 ≤1e−8 nat/token，標記數值 tie。
- tie 時若 A 原本的 a 在此候選對中，選 A 原 a；否則固定選 INT。不得參考新文章。
- Calibration 實測差 <0.001 的選擇另標記 low margin，但不得事後刪除；保留原始完整排名與分數。

## 6. 評估公式與問題拆分

令 E_s(i,f,k) 是背景 s∈{U,Q}、目標層 i、格式 f 在文章 k 的平均 NLL。
背景內額外損傷：d_s(i,f,k)=E_s(i,f,k)−E_s(i,fp32,k)。保留正負號。

同位元成對差異：

    D_s(i,b,k) = E_s(i,INT_b,k) − E_s(i,Posit_b,k)

正值表示 Posit 較好，負值表示 INT 較好。這裡背景 baseline 會抵消，不可使用 `E_Q(candidate)−E_U(fp32)` 比較候選額外損傷。

背景交互作用：

    J(i,b,k) = D_Q(i,b,k) − D_U(i,b,k)

J 不為零代表相對影響改變；但只要 D_U、D_Q 同號，就不等於排名反轉。

分別回答：

- D_U vs D_Q：真實排名是否改變？
- 原 BC 預測 vs D_U：原先已經排錯多少？
- 原 BC 預測 vs D_Q：前層量化後還排對多少？
- 每一對標記「兩邊都對、原本對後來錯、原本錯後來對、兩邊都錯、proxy tie、實測未定」。
- 正確率報告分母、未定數與覆蓋率；比較兩背景時另報共同可判定集合，不以兩個不同難度子集的百分比斷言準確率下降。

不要只報 Spearman；全候選相關可被位元數差異主導，掩蓋同位元錯選。逐層同位元結果與選錯代價是重點。

## 7. 主要實用指標：凍結 selector 的未見資料收益

對 81 個非控制層/位元對，每篇文章計算：

    G_shift(k) = mean_(i,b) [E_Q(i,q_U(i,b),k) − E_Q(i,q_Q(i,b),k)]
    G_BC(k)    = mean_(i,b) [E_Q(i,q_BC(i,b),k) − E_Q(i,q_Q(i,b),k)]

正值代表新背景的 calibration 選擇在未見文章上較好。

G_shift 是主問題：重新考慮背景是否有實際價值。
G_BC 是另一個主要參照：與原 BC 相比是否有可改善空間。
兩個主要指標都報平均、95% CI，以及 Bonferroni 97.5% CI（控制兩指標的整體 5% 錯誤率）。

這些量是「不同單點介入的平均收益」，不是把 81 個替換一起套上模型的收益。禁止加總成完整配置 NLL 改善，禁止稱為新 knee 的收益。

預先以 ε=0.001 nat/token 作為本輪平均單點收益的實用門檻；這是沿用前輪量級的研究決策門檻，不是由結果調出，也不是全模型加總保證。

附帶報告：

- q_U≠q_Q 的個數、分布層與位元；這個集合由 calibration 決定，可以另報其平均收益與 CI，但列為次要指標。
- 在 U 背景也評估 q_U 與 q_Q，呈現選擇優勢是否隨背景變化。
- G_shift/G_BC 按 4/6/8 bit 分層，按 L1–8、L9–17、L18–27 分組；只作探索性定位。
- 列出所有層，另報排除歷史已知診斷層 L2/L14/L27 後的次要摘要，避免全由舊案例主導。
- 新資料上逐對 observed-best 的 gap 可以作描述性潛力估計，但因在同批資料取 min 有樂觀偏差，不當作 selector 泛化成績。
- f16/p8e1 的品質與 BOPs 另報，不混入等成本 G_shift/G_BC，也不只看品質直接決定保護取消。

## 8. 統計與反轉證據

文章為抽樣單位，先在文章內平均兩窗口。所有背景、格式與 selector 使用同一組 paired article bootstrap indices，20000 次；不得把窗口、token、層或 81 個比較當成獨立資料樣本。

對所有 D_U/D_Q/J 報原始效應與 95% CI。逐對圖中 CI 包含 0 一律標為未定，不能以點估計正負號宣稱確證反轉。

另外提供多重比較控制的嚴格反轉清單：

- 81 對 × 2 背景 = 162 個 D 的檢驗，作一個 Holm family，α=.05。
- 使用文章配對、以零假設中心化的 bootstrap 雙尾檢驗並記錄實作；精確全零差值令 p=1。保留原始 p、Holm 調整 p。
- 一對只有兩背景 D 的符號相反、兩項都通過此 family 的 Holm 檢驗，才標為「多重比較控制後支持排名反轉」。
- J 的 81 個檢驗另作 Holm family，作為相對效應改變的次要結果；J 顯著不等於反轉。
- L0 是負控制，不加入上述 81 對。f16/p8e1 是次要 family，與主要等成本結果分開。

報告區分「觀察到反轉」「未校正 CI 支持」「多重比較控制後支持」，不要只挑顯著案例。

## 9. 必要控制與執行成本

1. 對每個背景輸出實際 config、每層量化投影數與 hash，確認 Q 只量化 block id < i。
2. 目標權重在 U/Q、全部候選中一致；其他非前綴權重未被上一次 config 污染。每次由原始權重或正確 cache 設定，避免反覆量化已量化權重。
3. L0 的 U/Q 逐窗口 CE 應一致；全部 FP32 identity 與無 hook 路徑一致；切回相同 config 可重現。
4. 各背景 baseline 只停用目標 a，不能意外恢復目標 w 或前層量化。
5. 目標 x/y QDQ 不重複，INT scale/zp 在 tensor 所在 device；TF32 關閉、float32、deterministic；Posit LUT 對 Universal 的既有控制通過。
6. 固定原 ranges；不要用新背景或 held-out range 重校準。輸出 clipping/overflow 統計可以是描述，不可回饋量化器。
7. 任一條件非有限值須完整保留失敗狀態；不裁切成有限值、不默默刪除比較。若主要矩陣不完整，停止主要決策並報未完成，另登記修正。
8. 可重用前綴結果或重播 suffix 節省時間，但 cache 必須區分背景、層、資料與 config hash，並在舊 calibration 上與完整 forward 對照通過。不能沿用 U activation cache 去假裝 Q。
9. 原 BC 表 C 是 input+output，不能拿上一輪 input-only score 代替；若重現原表需原完整 FP32 統計路徑，不能拿固定 target weight 已量化的 U 當作原表的精確控制。
10. 數值控制先通過再跑正式資料。容差在計畫中寫死，失敗不得為了通過而事後放寬。

先用舊 calibration 量測少量條件，提供全矩陣 wall-time/峰值記憶體預估，再繼續既定實驗；記錄 setup、calibration 候選 forward、正式評估、統計的時間，不把這次研究掃描時間稱為正式 pipeline 的 latency。
已有 GPU/CPU 環境可參考歷史腳本；不要任意安裝或升級套件。
提供可續跑 checkpoint，包含程式、資料、config、ranges、候選 hash，防止新舊結果混用。

## 10. 預先決策與報告

報告須直接回答三個問題，並依以下規則判斷：

- 若 G_shift 的 97.5% CI 下界 > ε：支持在本測試範圍進一步研究背景相關重新評分。若 G_BC 也通過同一門檻，才同時支持相對舊 BC 有實用改善空間。
- 若只 G_BC 明顯改善、G_shift 沒有證據：可能是原 proxy 排名誤差，不能把收益歸因於前層量化造成的排名改變。
- 若有少數可信反轉，但 G_shift 上界 < ε：平均收益低於本輪門檻，保留局部案例，暫不做全模型重評；不能說沒有交互作用。
- 若 CI 跨過 ε，或原 calibration 的 q_Q 無法在未見資料改善：結論未定/這個 calibration selector 沒有驗證收益；不能推出所有重新評分方法都無效。
- 若效應集中在 L27 或少數歷史案例：只支持局部追查，不支持普遍增加 offline 成本。

即使通過，也只進入下一輪「維持 BC 公式、在新背景更新統計是否能抓到這些排序」的預先登記。不得在本輪自行啟動刷新、全模型搜尋或將多個選擇組成新配置後宣稱效果。

輸出至少：plan.md/json、hashes、controls.json、完整 configs、資料 manifest/tokens、calibration_scores/choices.json、逐窗口與逐文章 CE、pairwise_results.csv、selection_gain.json、timing.json、report.md。
圖表：U/Q 的候選差異散點圖（四象限清楚區分同向與反轉，標不確定性）、各層/位元的 BC 對錯與排名變化表、凍結 selector 的未見資料收益及 CI。
最後用數據說明：多少排名反轉、哪些不是原本 BC 就排錯、錯選代價多大、是否值得下一輪重新評分、證據還缺什麼。
