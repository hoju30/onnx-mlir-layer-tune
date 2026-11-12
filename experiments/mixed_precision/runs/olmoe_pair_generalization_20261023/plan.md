# 預先登記：BC 與逐位置配對評分在非誤排挑選的 expert／格式上的排序差異

時間：2026-10-23T15:21:32+0800。以 plan.json 為準。

## 範圍與方法
- 只限 expert 的 Down 輸入 activation；不含 Gate／Up 共享輸入。結論只適用於此範圍。
- 只比較 T_old（BC：Σ_c G2_c/n·Σ_t r² + 偏差項）與 T_pair（Σ_c Σ_t g²r² + 相同偏差項）。
- 固定 BC calibration、sum-loss、梯度正規化、router 語義（route 與權重在 backward 中固定）、量化器、range、6 個候選；不刷新、不調參、不加 spike 懲罰或一階項、不用 NLL 換算尺度；不改搜尋、knee、pipeline。

## 排除與合格
- 前幾輪做過損傷或干預測量的 49 個 expert 不作新案例（清單見 plan.json）。
- 合格：有 Down 輸入 cache；calibration rows ≥ 64；discovery rows ≥ 512；R_token、range、cache 統計有限。不用任何 loss、BC 誤排或 T_pair 資訊。

## 選點（select 階段依規則產生，之後才收集梯度）
- 一般樣本 16 個：每層從合格新 expert 隨機抽 1 個（seed 23601）；某層沒有就從其餘合格池隨機補足並記錄。
- spike 樣本：強 4 個（R_token ≥ 合格集合 p90）、弱 4 個（≤ p50），每個強 tap 配同層、log rows 最近的弱 tap（seed 23602）；與一般樣本不重複；匹配不足列出。
- 兩組分開報告；合併數字只作描述。

## 格式與比較
- 格式：int4, int6, int8, p4e1, p6e1, p8e1；主要：int4 vs p4e1、int6 vs p6e1、int8 vs p8e1；其餘 12 個配對為次要。全部結果保留。

## 控制
- 梯度路徑 CE 與 dr.forward 相同；G2／GS、S1／S2、T_old 重現 cache；恆等式成立；streaming 累加（Σg²e²、Σg²e、Σg²）與逐元素計算一致。不一致時先解釋。

## 確認資料與測量
- 96 篇新 WT103 文章 × 3 窗口，凍結後抽取；verified 掃描排除 6425 篇歷史文章；排除 WikiText-2 與近似重複。
- FP32 背景、一次只量化一個 expert 的 Down 輸入、6 個格式、dynamic route；ΔNLL 相對 FP32。不做搜尋或 knee。

## 評估
- 決定性配對：文章 bootstrap 95% CI 排除 0；敏感度：99% CI 與組內 Holm。proxy 平手（相對差 < 0.02）算「不對」；兩個 proxy 用同一組配對。
- 類別：兩者皆對、修正、弄壞、兩者皆不對；不可判定配對不算排對，另報數量與覆蓋率；同時報 expert 等權結果。
- 主要指標：各組決定性同位元配對的 修正 − 弄壞，expert bootstrap 97.5% CI（兩組 Bonferroni）。
- 組別狀態：決定性同位元配對 < 8 → 不足；淨值 > 0 且 CI 下界 > 0 → 改善；淨值 < 0 且上界 < 0 → 變差；其餘 → 不變。ΔNLL 為負的案例保留並分開解讀。

## 結論規則
- D：兩組都不足。A：兩組都改善，且全部 15 個配對中各組 弄壞 ≤ 修正/2。B：spike 組改善、一般組沒有。C：其餘（至少一組足夠）。只有一般組改善：照實報告（規則未涵蓋）。
- 不論結果：不宣稱 Hessian、不宣稱 knee 改善、不改 pipeline。

## 預算（分鐘）：{'select': 2, 'grad': 8, 'mech': 10, 'data': 5, 'measure': 32, 'analyze': 5, 'figures': 3, 'total_gpu': 42}；測量超時在下一個層組前停止；不追加資料、不換 seed、不改門檻。
