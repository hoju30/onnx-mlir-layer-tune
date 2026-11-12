# Claude 執行 prompt：確認 spike 機制，並完成 Qwen3 完整配置的 native pipeline A/B

請直接完成兩階段工作：先確認 L2 Posit4 的有限幅度損傷，再用一個預先固定的端點敏感度候選跑完整搜尋、編譯與新文章評估。使用者已明確授權接著跑完整 pipeline；數值控制通過後，不因單點結果不漂亮就只停在診斷，也不另外等待確認。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增腳本：`run_qwen3_spike_endpoint_pipeline.py`。
輸出：`runs/qwen3_spike_endpoint_pipeline_<實際日期>/`，存在則另開目錄，不覆寫。
正式預設、共用量化器、MILP 與 knee 不變；候選成本表與程式只放本輪實驗。

## 1. 必讀與背景

- `runs/qwen3_raw_error_spike_diagnostic_20261026/report.md`、`curves.csv`、`proxy_components.csv`、`per_article.csv`。
- `runs/qwen3_spike_recenter_20261026/report.md`、`components.csv`、`masks.npz`。
- `runs/lp_pipeline_regression_20261016/report.md`，特別是 Qwen3 保留 MILP／原 knee 的結論。
- `runs/qwen3_down_fp16_protection_20261024/report.md`、`runs/qwen3_l27_native_confirm_20261024/report.md`。
- `runs/native_eps_fix_20261014/qwen3/`、`runs/qwen3_prepare_20261011/`。
- `run_qwen3_raw_error_spike_diagnostic.py`、`run_qwen3_spike_recenter.py`、`run_native_bc_pipeline.py`、`run_solver_regression.py`、`mp_auto.py`、`mp_milp.py`、`run_lp_pipeline_eval.py`、`run_qwen_l27_native_confirm.py`、`wt_manifest_scan.py`。

已知：L2 p4e1 的原始誤差 S-only 損傷，在 calibration 的 α=1/2 約 0.032、α=1 約 3.969；既有 128 篇上 S-only 約 3.804，N-only 僅 0.021。前輪 D 修正 6/8-bit，卻新增代價巨大的 4-bit 誤排，因此正式 BC 保留。

目前定位到的是「整個 spike token 行」，不是已證明單一最大 channel 或 clipping 是唯一原因。本輪不要先下結論。

重要完整配置事實：正式 activation 格式 a 同時作用在投影 input x 與 output y。L2 p8e1 的 x-only 很安全，既有完整 a 替換卻曾損失約 +0.305 nat/token；不能忽略 y 或拿 x-only 結果直接代替整個 C(a)。

## 2. 預先凍結與兩階段邊界

開始前寫 plan.json/md，固定本文件的候選公式、格式、範圍、α、機制遮罩、搜尋方式、配置比較、統計與數值容差，記錄程式及輸入 hashes。

- 第一階段：原 10×128 calibration 上作機制診斷與 L2 端點評分預覽，不抽新文章、不改候選。
- 第二階段：自動套用相同公式到所有有 spike 的 Down input、全部原候選格式，搜尋完整配置、native 編譯，在未使用過的新文章上比較。
- 不根據第一階段選擇改用另一個公式、係數、channel 遮罩或層清單。單點沒有修好也照原規格測完整配置，保留負面結果。
- 若數值控制失敗，先定位修程式並記 amendment；不要把未通過控制的表丟給 solver，或降低精度繞過。

## 3. 第一階段：確認幅度與 channel 分區

模型 Qwen3-0.6B，只改 L2 Down input，其餘 weight/activation、Down output 維持 FP32。沿用原 calibration、INT qparams、Universal LUT、FP32/eager/RMSNorm epsilon=1e-6。

固定原 FP32 x 的 row mask：S={max_c|x| >= 6×median_calibration(max_c|x|)}；核對與前輪相同。q=Q_f(x)、e=q−x 固定，不重新中心化或重新量化擾動後 x。

### 幅度

對全部六格式 int4/p4e1、int6/p6e1、int8/p8e1，固定掃描 S-only 的 α∈{0,0.5,0.6,0.7,0.8,0.85,0.9,0.925,0.95,0.975,0.99,1}。α=1 直接 masked QDQ，其餘 x+αM_Se；與舊 α=0.5、1 結果核對。

每點報逐窗口 loss、平均 ΔNLL、實際擾動、rounding；不將離散 grid 的最大變化區間稱為數學不連續，也不事後加點來找最漂亮的門檻。

### channel/元素分區（只作機制，不決定候選）

對 p4e1/p6e1/p8e1，從實際候選值集取得 maxpos，定義：

    H = S 行中每行 |x| 最大的元素（相同最大值全部納入）
    O_f = S ∩ {|x|>maxpos_f}                    # 超出格式範圍的元素
    R_f = S \ O_f                              # 同行其餘元素

α=1 分別只對 H、O_f、R_f、完整 S 套實際 q，其他保持 x。O/R 是元素遮罩，不是固定 channel 名單；同一格式內固定且互補。H 不一定是 O 的子集，不能把 H/O/R 當互斥三組相加。

計算 I_OR=Δ_S−Δ_O−Δ_R，保留正負；報最大值元素 H 是否足以重現主要損傷、其他超範圍元素與同行其餘元素的作用。不強迫 H 或 O 的損傷等於 S，不宣稱唯一架構原因。

完整 suffix 必須包括真實 Down/residual、L3–L27 全部 attention/FFN、final norm/head/CE。凍結原規則後，如本輪判讀仍未定，照實記錄，不為強迫機制答案而調 mask。

## 4. 唯一主要候選 E：spike 的端點平方梯度更新

這是新的、尚未驗證的 proxy 假說。前輪報告提議的 ½(g0+g1)·e 是 loss-like 的路徑積分近似，不能直接加到原 BC 的平方梯度分數。本輪固定使用相同 raw proxy 尺度的平方梯度平均，不宣稱它是梯形 loss 積分或真正二階導數。

每個 Down input，在原 calibration 的 FP32 背景下建立相同的 S/N row mask，所有格式共用。不得硬編 L2/L27，或只處理 Posit4。

對每個原候選格式 f：

    q_f = Q_f(x); e_f = q_f−x                  # 固定原量化器與 range
    μ_N,c = mean_N(e_f,c)
    C_N(f) = Σ_c mean_N(g0,c²) Σ_N(e_f−μ_N)²
             + Σ_(s,c) μ_N,c² [Σ_(p∈N_s)g0,spc]²
    x1 = where(S,q_f,x)
    g0 = d(summed CE)/dx at x
    g1,f = d(summed CE)/dx1 at x1
    P0(f) = Σ_(S,c) g0,spc² e_f,spc²
    Pend(f) = ½ Σ_(S,c) [g0,spc²+g1,f,spc²] e_f,spc²
    C_x,D(f) = C_N(f)+P0(f)
    C_x,E(f) = C_N(f)+Pend(f)

g0/g1 由同一 GPU FP32 loss closure、相同 summed CE 求得；x1 detach 成 leaf，e/S 固定，不穿過量化器，不用 STE。g1 的模型權重與其他 activation 仍 FP32，只有此 Down input 的 S 行改變。保存全部格式，不以端點 loss 淘汰或挑選候選。

μ_N、mean_N 的分母是 N 的實際 row 數，仍含梯度為零的最後位置；U_N 的梯度和只能取 N。統計 float64，0.5 係數固定，不掃係數、不加入真實 ΔNLL 作成本。

S 空、N 空、門檻無效的回退按前輪固定規則回原 BC；fp32/identity 誤差為零不需端點 backward，分數為零。有 S 但 e_S=0 的格式可跳過相同端點，記錄原因與次數。

不要聲稱只保留 S 的梯度就能只算 S 的 backward：suffix 有 attention，完整 forward/backward 與必要中間量的成本都須計入。

第一階段可額外報 ½(g0+g1)·e_S 對真實 S-only ΔNLL 的對照，僅供診斷，不寫入搜尋成本。E 無論對 L2 是否改善，均不得換成這個量取代。

## 5. 完整成本表：保留 x/y 語意與後端對照

模型 196 個可搜尋投影與原 GRID11 不變：

    int4,int5,int6,int7,int8,p4e1,p6e1,p8e1,f16,bf16,fp32

所有有 S 的 Down input 都計算全部 11 格式，不把六格式診斷表冒充完整搜尋表。其他投影不改，所有 weight 成本 B 不改。

取得原 native 每個 Down 的 C_x,old 與 C_y,old，先驗證：

    C_old(node,f)=C_x,old(node,f)+C_y,old(node,f)

優先重用 hashes 相容的原 native 統計與 layout；若缺少重建 slot 的資料，直接執行原 native 統計補齊。不可以用 PyTorch C_x 去減 native C_total 而假稱剩下的是精確 native C_y。

固定四份實驗表：

    A：原 native BC 表。
    B：只有有 S 的 Down x，換成同一 GPU g0/x/e 路徑重算的「原 BC C_x」。
    D：同樣 Down x 換成 C_x,D（新分解、無端點更新）。
    E：同樣 Down x 換成 C_x,E（新分解＋端點平方梯度）。

所有替換都用：

    C_new_total = C_y,old + C_x,new

Down y 與其他 C entries、所有 B、格式、qparams 與資源成本維持相同。S 空的 Down 在 B/D/E 直接使用 A 的數值，逐位元相同。

B−A 用於觀察後端橋接影響；D−B 觀察新分解；E−D 隔離端點更新。E−A 才是主要完整方法比較。若 B 已造成顯著配置或品質改變，必須呈現，不能全部歸因端點更新。

本輪 E 若由 PyTorch 建表而模型由 onnx-mlir 編譯推論，要明確稱「混合建表＋完整配置 native 評估」，不能稱新統計已完全 compiler-native。C++/dialect 的完整統計整合屬看到收益後的後續工作。

## 6. 完整搜尋與資源公平比較

Qwen3 固定沿用目前 **MILP＋mp_auto 原 normalized-chord knee**，17 個粗掃點、2 輪細化、原 time limit/gap/重試修復與格式順序。不要切成 LP-first，或同時改 knee；既有 dense 回歸已指出兩者會改變 Qwen3 的品質。

先重現 A 的自動搜尋與資源；與既有原 BC knee 核對 config hash、size、BOPs、R（歷史約 0.759821/0.877976/0.877976）。若程式或工具鏈版本造成不同，記錄與定位，不手動改回舊配置來掩蓋。

凍結下列完整配置：

1. A_auto：A 原 BC 自動 knee。
2. E_auto：E 用相同自動流程獨立求 knee，不輸入人為預算。
3. B_cap、D_cap、E_cap：分別以 B/D/E 表，在 A_auto 的**兩個實際上限** weight bytes 與 BOPs 下求解。
4. A_aggr、E_aggr：預先固定的次要激進點。只依 A 的 frontier，從 R< R_Aknee 的點中選最接近 sqrt(R_min×R_Aknee) 的 R（log-distance；平手選較低 R，再按既有 config hash），作 A_aggr。E_aggr 以它的實際 bytes/BOPs 為上限求解。若 A knee 已是最低 R，報無此對照，不從 loss 另挑點。

固定預算只用於實驗公平比較；正式自動選點仍不需使用者先給預算。用實際兩個資源 cap，不只用 max(size,bops) 的單一 R，避免任一資源偷偷增加。

每個求解保存 status、gap、可行性、重試/修復、實際資源與成本；status 1 不能稱已證最優。相同 table+budget 的重算要區分冷啟動成本與快取。

跨表的 raw proxy 總值不是可直接比較的品質指標；在相同表內用於求解，品質結論看新文章 NLL。

先對 config hash 去重，只對獨特完整配置編譯/評估一次。若 E_cap/A_auto 或 E_auto/A_auto 完全相同，該比較按配置恆等報零，不為追求不同結果調公式或預算；若所有主要配置都相同，完整輸出搜尋／成本／配置恆等結論即可，無需重複推論。

## 7. Native 編譯與評估資料

所有配置與 cost table/search trace 完整凍結後，才抽 **128 篇未使用過的 WT103 train 文章，每篇 1 個 128-token 窗口**。沿用 verified manifests、歷史 Qwen token audit、WT2 與近重複排除；排除本輪前所有研究文章。固定新 seeds，例如 order=26102611、start=26102612、bootstrap=26102613；核對未被使用，不事後換 seed。

新文章僅用於最終評估，不用來挑方法、層、格式、knee、α 或更新任何 proxy。

主要後端：onnx-mlir 完整配置 native 模型。沿用 `run_qwen_l27_native_confirm.py`／`run_lp_pipeline_eval.stage_c_native` 的編譯及 `evaluate_native_search_chunked`：bytecode、constants to file、Krnl O3、clang O0、per-channel weight。保留 .so、constants、compile_env、工具鏈 hashes 與可續跑結果。

完整 a 同時對 x/y QDQ，全部 196 個 w/a 與配置逐項核對。模型執行不可混入單點診斷 hook，或手動保護 L2/L27。weight bits 與 activation bits 都由新搜尋決定。

編譯 FP32 與各獨特配置；先在固定 calibration 控制窗口檢查 identity、配置 env、範圍、量化、可重現性，不用 calibration 的模型 NLL 再挑一個最佳配置。主評估每篇 summed CE/127，PPL=exp(整批 NLL)，不平均每篇 PPL。

同批新窗口另外用 LMReplay 的 GPU QDQ 作後端對照；native 是主要證據，不能用 GPU 收益替代 native 結果。不要要求兩後端逐位元相同，但完整報逐文章效應差、相關、符號與數值範圍。native 若確實無法執行，要保存錯誤並明確標本輪尚未完成 native 驗證，不能寫成成功。

## 8. 主要終點與判讀

對同一文章的完整模型 NLL 作成對比較。預先固定兩個主要比較：

    P1 = native NLL(E_cap)−NLL(A_auto)      # 同資源上限：criterion 是否幫助選配置
    P2 = native NLL(E_auto)−NLL(A_auto)     # 真正無預算自動 pipeline 的結果

文章 paired bootstrap 20000 次，兩比較用 Bonferroni 97.5% 雙側 CI，另報一般 95% CI。所有配置共同重抽文章。相同配置的差值按恆等處理。

- P1 CI 上界<0：支持在相同資源上限下品質改善。必須同時驗證實際兩個資源皆不超 cap。
- P2 的品質與兩資源分開報；E_auto 若用了更多資源，不能說同成本勝出。若資源不增加且 CI 上界<0，才支持自動 pipeline 的品質/資源共同改善。
- 可額外以預先固定 δ=0.001 nat/token 報品質非劣，但必須稱非劣，不能稱 accuracy/PPL 更好；CI 含零且未落在容許區間時報未定。
- 激進點、B_cap−A_auto、D_cap−B_cap、E_cap−D_cap 為次要比較，固定家族並用 Holm 校正；不以它們取代主要結果。
- 不因 E 的單點排序排對更多就宣稱 pipeline 改善；不因 PPL 改善就宣稱 downstream accuracy 改善。
- 抽樣後不擴樣、不換文章、不調候選；無論改善、變差、未定或配置相同都完整報告。

## 9. 成本是研究結果的一部分

分開量：

- 原 native 建表/重用的來源與重建成本；原始單次 baseline forward/backward。
- GPU g0/h 收集、mask/μ/誤差統計。
- 每個 (layer,format) 的端點 forward/backward 次數與時間、未執行的零誤差/空組、同步與峰值 VRAM/RAM。
- B/D/E 各自必要的增量建表成本；不要把第一階段診斷曲線或所有消融分析時間當成 E 的必要成本。
- MILP/Pareto/knee 搜尋、編譯、最終品質評估、總 wall time。保留冷/暖、CPU/GPU、cache 與平行度差異。
- 完整 offline 成本主要報「建表＋搜尋＋選定模型編譯」，最終研究評估時間另列；所有候選研究編譯的額外成本也要揭露。

更新敏感度可能顯著增加建表成本；不要把它形容成免費，也不能用「只取 spike 行梯度」假稱只運算那些元素。

## 10. 輸出與完成條件

保留兩階段的 plan/hashes/controls、dense α 與 channel 分區、g0/g1 誤差與摘要、11 格式 cost components、native Cx/Cy 重建、A/B/D/E 表、search traces、完整 configs/config hashes/resource、.so/compile_env、segments/tokens、native/GPU 逐文章 NLL、統計與 timing。

report.md 首頁回答：

1. L2 的大損傷在哪段幅度、哪些固定 channel/元素分區出現？仍有哪些無法唯一歸因？
2. 端點平方梯度 E 是否修正 L2 4-bit、保住 6/8-bit？這只是 calibration 診斷，不能當最終品質。
3. 原 BC、新候選的自動 knee 與同資源配置各改了哪些 w/a？L2/L27 的 x/y 成本如何影響選擇？
4. 新文章 native 的 NLL/PPL 與資源是否改善？與 GPU 效應是否一致？
5. 每多花多少建表時間，換到多少品質或資源收益？是否符合低 offline 成本目標？

若機制仍未定但表與數值控制通過，依授權完成完整 pipeline；若配置相同，報配置相同與增量成本，勿捏造改善。正式預設到此仍不更動，最後依品質/資源/成本證據提出是否值得做 compiler-native 新統計整合的建議。
