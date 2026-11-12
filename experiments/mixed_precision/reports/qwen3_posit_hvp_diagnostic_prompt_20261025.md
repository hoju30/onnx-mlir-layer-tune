# Claude 執行 prompt：定位 Posit 的 BC 誤排，區分局部曲率與有限擾動

請直接在 repository 實作並完成以下診斷，先完成控制與凍結計畫再跑正式量測。這輪不修改正式 criterion；先確認哪一種修正有證據支持。

工作目錄：`/home/hoju/test/onnx-mlir-layer-tune/experiments/mixed_precision`。
建議腳本：`run_qwen3_posit_directional_curvature.py`。
輸出：`runs/qwen3_posit_directional_curvature_<實際日期>/`；存在則另開新目錄，不覆寫舊實驗。

## 1. 目的與已知資訊

目標是改善 BC 對 Posit 相對 INT 的損傷排序；不是修改 Posit 編碼，也不是擬合 BC 數值到 NLL。

先閱讀：
- `runs/qwen3_l27_reversal_diag_20261024/report.md`、`stats.json`。
- `runs/qwen3_posit_score_geometry_allbits_20261025/report.md`、`scores.csv`、`pairs.csv`。
- `reports/qwen3_posit_scoring_next_step_20261025.md`。
- `run_qwen_l27_reversal_diag.py`、`analyze_qwen3_posit_score_geometry.py`、`qwen3_swiglu_common.py`。

已知：L27 Down input 的 BC 在 4/6-bit 偏好 INT，實測 Posit 較好；output 的方向正確。原 calibration 也排錯。spike 分組、同位置配對與逐 token empirical-Fisher 尚未修正 4/6-bit。逐窗口 Fisher 在 calibration 排對，但在新文章的 6-bit 排錯。

前輪 α=±1 的奇次／偶次反應含高階項，不能當成已量得真正一階／Hessian 二階。BC 與 ΔNLL 沒有完成數值尺度校準，禁止因 INT 的數字巧合接近實測就認定 INT 已估準。

這輪要區分：
1. BC 對無窮小擾動的方向曲率就排錯。
2. 局部方向曲率合理，但完整量化誤差產生高階非線性。
3. signed first-order 改變總排序，或以上因素同時存在。

## 2. 固定範圍

- 模型 Qwen3-0.6B，L27 `mlp.down_proj`。
- 除目標 activation 介入外，所有權重、activation、前綴都 FP32；target weight 也 FP32。
- input x 為 SiLU(gate)·up；output y 為 Down 輸出。保留真實 residual、最終 RMSNorm、LM head 與 next-token CE。
- 主分析是 input 的 INT4/Posit4、INT6/Posit6；INT8/Posit8 與 output 為預先指定的對照。六格式全部報告。
- masks 固定 all、P0、Rest（位置 1–127）。所有格式使用同一 mask；不依新結果選位置／channel。
- 原 10×128 calibration 窗口、range、LUT 不變。本輪只做這批已觀察資料的機制診斷，不新增文章，不作泛化收益宣稱。
- 不啟動完整配置搜尋／native 編譯／fine-tune／統計刷新；不加 Posit 專用折扣係數，不根據 loss 換候選。

## 3. 真正的方向導數與 HVP

對 slot t、格式 f、mask m，先在 autograd 外固定：

    e = detach[(Q_f(t) − t) · m]

令 ℓ(t) 為同一窗口 summed CE（127 targets），保持其他張量固定。用 autograd 計算：

    g = ∇_t ℓ
    F = gᵀe
    H_e = ∇_t(gᵀe)，create_graph=True 後再求導
    K = ½eᵀH_e

e 與 mask 必須 detach；禁止穿過 QDQ 求導或使用 STE。不建立完整 Hessian。K 可以是負值，不裁切。

input 的實際 loss closure 必須包含 Down 線性層；output closure 則從 y 加 residual 開始。梯度／HVP 要對指定 slot 計算，不能把 x 與 y 混用。

先逐窗口計算，再加總並除以 10×127；不要把 mean CE 與 summed CE 的梯度混用。

## 4. 固定擾動曲線與比較

全部條件量測：

    ΔL_f,m(α) = L(t+αe) − L(t)
    α ∈ {0, ±1, ±1/2, ±1/4, ±1/8, ±1/16, ±1/32, ±1/64}

另用真正 QDQ 張量量一次 α=1；核對 t+e 與 Q(t) 的浮點差異，保留兩條路徑，不把重建誤差誤認成機制。

對每個 α>0：

    O_f(α) = [ΔL_f(+α) − ΔL_f(−α)]/2
    E_f(α) = [ΔL_f(+α) + ΔL_f(−α)]/2

比較 O(α)/α 與 F、E(α)/α² 與 K 的收斂；α=1 的 O/E 只能稱奇次／偶次反應。

比較以下原始預測，不擬合各格式係數：
- signed first-order：αF。
- autograd 方向二階：α²K，以及合計 αF+α²K。
- 原 BC 曲率 proxy：α²C/2。
- 逐 token empirical-Fisher：α²/2 × Σ_(s,p)[Σ_c g_spc e_spc]²。

另列 R(α)=ΔL(α)−αF−α²K；比較完整預測與剩餘項，不能只看 K 的排序判定總損傷原因。

BC 沿用原公式、n=10×128；每個 mask 將遮罩外誤差設為零並重算 μ，全部 g 保留。分數最後與其他量一致除以 1270 targets。另列 BC 的變異與平均偏差項。

曲率 proxy 應先與 E(α)／K 比；不能直接拿純二階預測與含一階的 ΔL 排名不同，就宣稱曲率有錯。

每種方法都列 INT−Posit 差；正值代表偏好 Posit。重點是排序方向與縮小擾動後的行為，幅度差作輔助。

## 5. 數值與實作控制

正式量測前用合成方向及舊窗口作 smoke，凍結程式 hash、計畫、控制容差；記錄已看過哪些資料與結果。正式後改動需 amendment，不依想要的排名調整容差或步長。

- 沿用 GPU float32、TF32 關閉、deterministic、既有裝置端 INT scale/zp 與驗證過的 Posit LUT。
- 完整 FP32 forward 與最後 Down 後段重播對照；檢查 hook、slot、權重與 residual。
- α=0 的 ΔL=0；真正 QDQ 路徑重現舊 calibration loss；原 BC x+y 與 native 表的相對差沿用 1% 控制。
- g_x=g_y W_down；position 127 的 g/H_e 應為零。
- HVP 必須來自 autograd。用梯度中心差分交叉檢查：

      H_e,FD(h) = [g(t+h e) − g(t−h e)]/(2h)
      h ∈ {1/32, 1/64, 1/128}

  全部步長與方向投影誤差都報告，不能挑最接近 HVP 的一步。小步長不保證已進入局部區間；使用凍結的絕對＋相對容差，檢查相鄰步長的收斂。沒有收斂就標記數值未定，不能硬作根因判斷。
- 報小 α 下 t+αe 實際改變的元素比例與相對理想 αe 的殘差，辨別 FP32 rounding。
- 如需 float64 參考，另標記並驗證 RMSNorm 是否內部強制 float32；不能只呼叫 model.double() 就宣稱是完整 double 參考。
- L27 Down 之後逐位置獨立，檢查實測 P0+Rest=all。BC 的 mask 分數含均值修正，不強迫其直接可加。
- F 與 HVP 的 P0/Rest 方向曲率應可加；每個 α 的實測損傷也應可加。保留所有誤差及失敗條件，不刪除或裁成有限值。
- 這是 10 個既有 calibration 窗口；逐窗口全列。不要把它們假裝成 10 篇獨立文章做泛化顯著性檢定。

## 6. 判讀規則

- BC 與 K 的候選方向相反，且 K 與小 α 的 E(α)/α² 相符：支持 BC 局部曲率 proxy 失配。但還沒單獨證明是 g²、channel 平均或 bias 哪一項。
- BC 與 K 同方向，而 E(1) 反方向：支持高階偶次反應影響完整擾動。
- O(α)/α 在小 α 接近 F，但 O(1) 明顯不同：支持高階奇次反應；不能將 O(1) 全稱一階。
- 加入 F 才能解釋總 ΔL 排序：記錄 signed first-order 的作用，同時檢查其他格式／位置，不能推廣成所有誤排都只缺一階。
- 只有 8-bit 局部近似有效而 4/6-bit 失效：保留幅度依賴的結論；不能直接據此設新 low-bit 門檻。
- 允許混合原因。控制／收斂不足則報未定，不必強行選唯一根因。

## 7. 交付與停止範圍

輸出 plan.md/json、程式／輸入 hashes、controls.json、逐窗口 F/K、BC 分項、全部 α 的 loss、奇偶拆分、INT−Posit 成對表、HVP-FD 誤差／收斂表、曲線圖、timing.json、report.md。

成本分開列：FP32 前向／梯度、每方向 HVP、BC／token 分數、QDQ、曲線評估、數值控制與峰值記憶體。這些是診斷成本，不是已實作的 native 建表 latency。

報告首頁回答：
1. 哪些 Posit 誤排在局部曲率階段就已發生？
2. 哪些是在放大擾動後才發生？
3. 一階項對哪些配對有決定性影響？
4. 下一個值得測的低成本修正應保留什麼資訊，以及目前不能下什麼結論？

完成後停止，不自動追加新文章、跨層完整搜尋或修改正式 criterion。若有候選修正，下一輪另凍結方案，以新文章和其他層驗證排序、選格式收益及建表成本。
