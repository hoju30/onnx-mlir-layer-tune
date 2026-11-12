# 預先登記：OLMoE 完整配置 A／B — 原 BC vs 只替換 Down 輸入 activation 的逐位置配對評分

時間：2026-10-23T16:15:21+0800。以 plan.json 為準。

## 定義
- A：正常 knee 配置（runs/olmoe_knee_selective_sensitivity_20261022/configs/A.json）與原 static BC 表。
- B：所有 1017 個搜尋 expert 的 Down 輸入（down_proj x）activation 變異項由 Σ_c G2_c/n·Σ_t r² 改為 Σ_c Σ_t g²r²；偏差項、Down 輸出項、Gate／Up 項與共享輸入歸屬、全部 weight 列、量化器、range、calibration、資源模型、合法性、搜尋節點都不變；完整候選網格；FP32 成本 0。統一套用，不選 spike。
- streaming：Σg²e²、Σg²e 與 G2、μ（S1/n）同 rows；V_pair,c = Σg²e² − 2μΣg²e + μ²G2；float64；不做任何截斷；負值只計數報告，總和為負則停止。預先指定直接逐元素驗證：L1.E9, L2.E30, L9.E8, L0.E28, L15.E36。

## 搜尋
- A 的精確上限：max_bytes = 4524359631.556117，max_bops = 46648327679632.125（直接讀檔）。
- 兩張表用同一個 ss.search_table（支配剪枝 → LP → 四捨五入＋修復，誤差 ≤ max(0.001·目標, 1e-06·scale) 才接受，否則修正版 MILP，時限 300.0 s）；只搜一次，不建 frontier、不重選 knee。
- A 控制：重建的 A 表必須逐位元等於 static 表；A 表的搜尋必須重現 q0_knee。
- 近似處理：若 A 配置在 B 表上的目標值低於 B 的解，B 用修正版 MILP 重解一次，取 B 表上較低者並同時報告；仍高於 A 則標示「搜尋受限」。永不以 A 配置代替 B；不使用任何評估 NLL。

## 控制
- G2／GS、S1／S2、n 與 cache 一致；B 的 weight 列與非 Down activation 列逐位元等於 A；Down 列拆成輸入／輸出項分別記錄；有限、合法、覆蓋全部；增量版 Binc 等於完整 B。

## 評估（配置凍結後才抽資料）
- 600 篇新 WT103 文章 × 4 窗口；verified 掃描排除 6521 篇；排除 WikiText-2 與近似重複；核對 content hash 交集為 0。
- FP32、A、B 同一批文章、原 router、dynamic route、同一量化後端。
- 主要：Δ = NLL(B) − NLL(A)，文章 bootstrap 95% CI；唯一的主要比較。次要：NLL、PPL、相對 FP32 損傷、改善／變差文章數、相對損傷改善、路由 BOPs 與未搜尋 expert 用量。
- 決策：CI < 0 → 此資源點品質收益確認（權衡額外 offline 成本，下一輪再驗其他點與自動 knee，不改預設）；CI 含 0 → 未確認，保留 BC；CI > 0 → 不接入；資源或搜尋不公平 → 比較受限。

## 成本
- 分開量原 BC 統計、配對統計、組表與儲存、剪枝與搜尋、輸入到配置的總時間、最終評估；建表交錯 A1 B1 Binc1 A2 B2 Binc2；Binc 是重用 cache 的增量成本，B 是從頭建表成本。不宣稱 ONNX-MLIR native 或低位元 kernel 加速。

## 預算（分鐘）：{'builds': 12, 'search': 20, 'data': 6, 'eval': 20, 'analyze': 5, 'total_gpu': 50}；看到評估結果後不改預算、候選、配置或資料。
