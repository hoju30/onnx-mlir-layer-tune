# Claude 執行 prompt：完整配置的七個資源交換，哪些造成退步？

請直接實作並完成本輪小型 native 歸因實驗。現在先停止增加 spike／端點評分公式，查清楚上一輪同資源配置為何退步，再決定真正需要修改的部分。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議新增腳本：`run_qwen3_config_swap_attribution.py`。
輸出：`runs/qwen3_config_swap_attribution_<實際日期>/`，存在則另開目錄，不覆寫。

## 1. 先讀與研究對象

- `runs/qwen3_spike_endpoint_pipeline_20261026/report.md`、`stats.json`、`search_summary.json`。
- 同目錄 `configs/A_auto.json`、`configs/D_cap.json`、`configs/E_cap.json`、`table_A/B/D/E.json`、`builds.json`、`native_ce.npz`、`pytorch_ce.npz`、`segments.json`、`tokens.npy`、`controls_native.json`。
- `run_qwen3_spike_endpoint_pipeline.py`、`run_qwen_l27_native_confirm.py`、`run_lp_pipeline_eval.py`、`run_solver_regression.py`、`run_native_bc_pipeline.py`。
- `runs/qwen3_prefix_rank_shift_20261024/report.md`：之前是前綴量化、後綴 FP32 的背景比較；本輪使用真正完整混合精度背景與實際的 w/a 資源交換，不能稱為重做該實驗。

上一輪主要結果：A_auto native PPL 56.20；E_cap 56.60。同資源上限的 ΔNLL 約 +0.00710。B_cap 與 A_auto 完全相同，D_cap 與 E_cap 完全相同。

本輪固定：A = A_auto；T = E_cap（核對與 D_cap 完全相同）。不得誤用 E_auto 或激進配置。

目的：區分投影層級的 proxy 評分問題、跨層資源取捨、聯合配置的交互，以及後端數值路徑差異。不預設其中任何一項是根因。

## 2. 固定九個完整配置

從檔案自動解析 A→T 的差異，必須恰為以下七個投影；不手寫整份配置：

| 單位 | 投影 | A 的 w/a | T 的 w/a |
|---|---|---|---|
| 1 | L1 Gate | int7 / p8e1 | int6 / f16 |
| 2 | L6 K | int7 / p8e1 | int6 / p8e1 |
| 3 | L8 V | int7 / p8e1 | int7 / f16 |
| 4 | L11 Down | int6 / p8e1 | int6 / p6e1 |
| 5 | L21 O | int5 / p6e1 | int6 / p6e1 |
| 6 | L22 Gate | int6 / p8e1 | int5 / p8e1 |
| 7 | L26 Down | int6 / f16 | int7 / p8e1 |

建立 A_1…A_7：從 A 複製，每次只把該投影的整個 w/a 換成 T 的設定。加 A 與 T，共九個配置。七個單位按投影處理，同一投影的 w/a 同時變更時先不拆開。

所有其他投影均維持 A 的混合精度設定，包括所有後層；不是把非目標層切回 FP32。activation a 同時量化 x 與 y，weight 使用原 per-output-channel 量化語意。

逐一保存 config/hash/diff。九個配置都必須有完整 196 個投影；核對 A_i 只差指定投影，T 等於七個變更合併。

單項配置可能超過 A 的資源上限，這些只作歸因，不作部署候選。另列每項的 Δweight bytes、ΔBOPs、正規化 size/BOPs/R。不能因某項增加 NLL 就認定它是錯誤，它也可能是省資源需要支付的代價。

## 3. 固定資料：先用既有資料，不抽新文章

兩份資料分開報告：

- C：原 native calibration 10×128 tokens，沿用 `runs/native_eps_fix_20261014/qwen3/calib_inputs.npy`。
- H：上一輪 128 篇文章中，按原 manifest 順序的前 16 篇，每篇沿用原來那個 128-token 窗口。先按索引凍結，不根據 NLL、文章內容或配置差選取。

本輪是已知退步後的事後機制診斷；H 不是新的獨立確認資料。不得要求 H 重現完整 128 篇的方向或效應大小，也不能因子集不一致就換另外 16 篇。

固定九個條件與資料索引、計畫/程式/輸入 hashes、數值容差後才跑新的單項 loss。冒煙只用 calibration，揭露看過哪些結果。

## 4. Native 為主，GPU 作同條件對照

重用上一輪 A、T 的 .so/常數與設定，前提是 config、模型、ranges、量化器、工具鏈與產物 hashes 全相容。T 的已編譯代表可能命名為 D_cap，必須按 hash 辨識，不靠名稱猜。

只編譯 A_1…A_7；相同 config hash 去重。沿用 bytecode、constants to file、Krnl O3/clang O0、posit runtime 與逐元素量化路徑，不修改正式 pass。

A/T 在 H 的舊結果可直接按相同 token 索引重用，先在固定一個窗口與既有 .so 重播核對。C 中缺少的 A/T 結果補跑，不將別份 calibration 或截斷窗口結果冒充重現。

九個配置在 C/H 上都取得完整 native per-window summed CE。H 每篇一窗口除以127；C 用全部1270 targets彙總。保留逐窗口資料與可續跑檔。

用 LMReplay 在完全相同 C/H 上測全部九個配置，TF32 off、deterministic、FP32、eager、RMSNorm epsilon 1e-6。核對 activation map、量化後 weight hash，避免切換配置時殘留 hook 或重複量化權重。

native 是主要結果，GPU 只供來源定位；兩後端不要求逐位元相同。若關鍵介入的成對效應不一致，報未定與後端差，不直接判定任一後端有錯。

## 5. 每個改動的預測與真值

對原 A/B/D/E 四份凍結表，各自定義：

    cost_M(n,w,a) = table_M['B'][n][w]+table_M['C'][n][a]
    predicted_delta_M,i = objective_M(A_i)−objective_M(A)
    predicted_delta_M,T = objective_M(T)−objective_M(A)

報所有七項各表的符號、數值與相關 w/x/y（如可從既有成本分項取得），但不重算或改表。這些 raw proxy 沒有 NLL 校準，不能要求與 ΔNLL 數值相等。

原分數可分離，因此必須核對：

    predicted_delta_M,T = Σ_i predicted_delta_M,i

真實每窗口/文章的損傷：

    d_i = L(A_i)−L(A)
    d_T = L(T)−L(A)
    I = d_T−Σ_i d_i

**I 必須在完全相同的窗口/文章上逐一計算。** C/H 分別計算，native/GPU 分別計算；不可拿舊128篇平均 +0.0071 減本輪16篇的單項總和。I 是指定完整配置變更下的有限交互殘差，不是 Hessian cross term 或唯一的損傷分攤。

每個單項報兩種後端的 d_i、資源差與各表 predicted_delta，另報 d_T、Σd_i、I。特別確認：退步是否在原 calibration 上已出現，還是只在 H 呈現；小樣本無法分辨時保留未定。

不要僅用七項的排序相關係數下結論，也不能只按 d_i 大小挑最大者就稱它是主因。增加精度的變更通常耗資源、降低精度通常省資源；需要連同 Δbytes/ΔBOPs 看資源交換是否划算，不能把兩資源任意加成一個新分數。

可額外把 L11 Down、L26 Down 的變更合成一個固定「Down 組」，其餘五項為「其他組」，只作現有單項數值的加總描述，不新增組合推論。若某投影同時改 w/a，本輪結果只能歸因該投影整體，不能直接稱為 weight 或 activation 的問題。

## 6. 數值控制與統計界線

- A/T 與原配置、compiled env、資源重現，A_i 的 196 項逐項核對。
- 固定 A 的一個窗口在另一個 fresh process 重播，記錄誤差；舊 loss 重用需通過同 token/同產物控制。
- μ/range/scale/LUT/模型 epsilon/候選語意都不更新，無 fine-tune、無新 backward/HVP。
- H 上 A/T 對舊逐文章結果重現；已知完整128篇平均僅列背景，不混入本輪子集統計。
- 分數差加總恆等式、配置差、完整結果數量與非有限值檢查。
- 對 C 只作描述，不做文章 bootstrap。
- H 報逐文章、平均、中位數、正負號數；如報 paired bootstrap，固定10000次、seed=26102631並檢查未使用，九個條件共用同一組文章重抽。因只有16篇且已重複使用，CI僅作探索性不確定性描述；不宣稱獨立確認或小差異已被排除。
- 若比較多個單項的顯著性，明列同一比較家族與 Holm 校正；不得從七項挑最大值後用未校正區間包裝成已確認。
- baseline重播誤差、原數值容差凍結；失敗先修程式記amendment，不重跑挑選有利結果。

## 7. 判讀與下一步分流

按以下證據提出下一個具體實驗，但本輪不自動執行：

1. 若某投影的單項得失已和分數方向明顯相反：下一步只拆該投影的 w/x/y，確認公式或量化誤差分項；先不推廣全模型。
2. 若單項加總能解釋完整退步，但搜尋相信整組變更有利：優先檢查跨層預測的相對收益與資源代價，不能因同層格式排對就認定跨層取捨合理。
3. 若 I 明顯、且單項不足以解釋整體：下一輪再做少數具體組合或反向恢復，驗證交互；不要由非零 I 直接設計全模型二階方法。
4. 若關鍵結論受 native/GPU差異主導：下一輪先定位同配置同資料的數值路徑，不把結果全歸 criterion。
5. 若16篇與calibration都不足以定位：如實報未定，不追加文章、配置、係數或門檻來追求結論。

本輪不預先假定上述分流互斥；可以同時存在單項誤排與交互。不要把 I 以外的部分稱為唯一因果歸屬，也不以16篇取代上一輪128篇對整體退步的結論。

## 8. 輸出與停止

保存 plan/hashes/資料索引、九個configs/diffs/resources、編譯與重用記錄、controls、native/GPU逐窗口CE、每表predicted_delta、各介入d_i、d_T/Σd_i/I、backend效應差、統計與timing。

report.md 首頁：

- 七個投影一表：w/a變更、Δbytes、ΔBOPs、A/D/E預測差、native與GPU實測差、數值/抽樣是否未定。
- C/H各一表：完整變更損傷、單項加總、聯合殘差I。
- 回答「現有證據指向下一個應修改哪個投影/分項、跨層可比性、交互，或後端路徑？」明列支持與不足。

分開量編譯、native/GPU評估、快取重用與總研究時間。控制通過後完成固定九個條件，不因超過日常批次目標就以部分結果取代計畫。

完成後停止：不重新搜尋、不改正式BC或D/E、不刷新proxy、不追加反向恢復或新文章，不做新criterion/native統計整合。依本輪結果另立下一個最小修正實驗。
