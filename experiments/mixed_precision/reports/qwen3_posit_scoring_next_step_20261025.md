# 優先研究 BC 對 Posit 的損傷評分

狀態：研究決策與下一輪設計；以下 HVP／擾動幅度驗證尚未執行。這不是已完成的修正，也不是預先登記完成的新樣本實驗。

## 目標

修正 BC 對 Posit 相對 INT 的候選排序，先聚焦 Qwen3-0.6B L27 Down input。成功標準是修正 INT4/Posit4 與 INT6/Posit6，並在新文章、其他位元與層上驗證，不以讓 BC 數值逼近 NLL 為唯一目標。

範圍不含修改 Posit 編碼／量化規則、擬合 Posit 專用折扣係數或根據 diagnostic loss 選格式。所有候選應使用同一評分規則。

## 已有證據與本次新增探索

- `runs/qwen3_l27_reversal_diag_20261024/report.md`：input 的 BC 排錯，output 的方向正確；原 calibration 也排錯。position 0 不是全部原因。
- 舊 spike 分組與同位置中心化配對未修正 4/6-bit 主要誤排。
- 本次 `runs/qwen3_posit_score_geometry_allbits_20261025/` 使用既有原 10 個 calibration 窗口 x/y/g 張量，無新模型 inference、無新 loss；是事後探索，不是新資料驗證。
- 未中心化的逐元素、逐 token、逐 channel、逐窗口 empirical-Fisher 全部列出，另作交叉項恆等式與 BC 均值修正拆分。

| L27 input，依舊 calibration 實測方向判斷 | 4-bit | 6-bit | 8-bit |
|---|---|---|---|
| 原 BC | 錯 | 錯 | 錯 |
| 逐 token empirical-Fisher | 錯 | 錯 | 對 |

逐 token 方法不足以修復低位元問題。逐窗口方法在這批 calibration 的三對都排對，但它在前輪新文章的 6-bit 已排錯；不能選它作正式解法。

最後 Down 之後的 norm/head/CE 逐位置獨立，逐窗口 Fisher 的跨位置項是經驗聚合項，不能當成已發現真實跨 token 損傷交互。

## 下一個最小問題

要區分：

1. 在非常小的誤差幅度下，BC 就已排錯，代表局部曲率的近似／方向表示不足。
2. 小幅度時能排對，直到完整 Posit 誤差才排錯，代表有限擾動的非線性不可忽略。
3. 局部曲率排序本身合理，但 signed first-order 會改變總排序。

前輪 α=±1 的奇次／偶次拆分含高階項，不能單憑它宣稱已測到真正 Hessian。

## 執行設計

先做有限的小型機制實驗，先凍結程式與計畫，再取得未觀察的量測；與新的文章驗證分階段記錄。

- 全 FP32 背景，僅 L27 Down input。4/6/8-bit INT/Posit；output 作同格式對照。range、LUT、calibration 不變。
- 原 10 個 calibration 窗口為機制定位資料，明確標記已看過它們的既有 loss，不把它們稱為獨立泛化驗證。
- 每個格式固定誤差方向 e=Q(x)−x，QDQ 與 e 的產生在 autograd 圖外完成並 detach。
- 用 autograd 計算 F=gᵀe，以及 H_dir=eᵀHe（HVP）；不建立完整 Hessian。
- 比較同一方向上的 L(x+αe)−L(x)，α 固定為 ±1、±1/2、±1/4、±1/8；原 α=1 真正 QDQ 使用 Q(x)，另核對浮點 x+e 與 Q(x) 的差異。
- 比較 Fα+½H_dir α²、BC/2×α²、逐 token empirical-Fisher×α² 與實測，不替各格式擬合比例或門檻。
- 位置固定 all、position 0、rest；全數報告，不根據結果新增 mask。
- HVP 是診斷參考，不能因此直接把它納入 native pipeline 或宣稱其成本適合全候選建表。

控制：FP32 loss 路徑一致、梯度鏈式法則、position 127 梯度為零、QDQ/誤差方向一致、非有限值檢查；HVP 需用固定小步長的中心差分作數值交叉檢查。具體誤差容差在執行前依 smoke 的數值精度凍結，不看正式效應調整。

## 判讀與修正方向

- 真實方向曲率已能排對，但 BC/Fisher 不能：優先找保留曲率／方向結構的低成本代理；再做跨層驗證。
- 真實方向曲率與 BC 同樣排錯，但完整幅度的實測排序反轉：優先研究有限擾動或量化背景下的低成本更新。
- F+½H_dir 才能排對：測試 signed first-order 的穩定性，不能只因一層改善就整合。
- 有混合結果時逐格式、位置保留，不能宣稱只有單一根因。

只有定位到具體可修正機制後，才凍結一個候選修正，先以 calibration 產生預測，再抽取全新且排除所有歷史文章的 diagnostic 資料，檢查 L2/L27 舊失敗點及其餘 Down 層。記錄新增誤排、選格式的實際 NLL、建表時間與記憶體，通過後才評估完整配置。
