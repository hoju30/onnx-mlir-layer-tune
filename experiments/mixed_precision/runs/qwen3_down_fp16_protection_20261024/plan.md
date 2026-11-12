# 計畫：Qwen3-0.6B Down activation FP16 保護是否必要

時間：2026-10-24T00:55:37+0800。以 plan.json 為準；寫入後不改配置、資料量、seed、δ 或決策規則。

- 性質：固定配置的局部介入；不改 criterion、不搜尋、不重選 knee、不 fine-tune、不換 calibration。真值是 GPU PyTorch QDQ（LMReplay）；native 只在 10 個 calibration 窗口交叉驗證。
- A = `runs/lp_pipeline_regression_20261016/C/qwen3/configs/ref.json`（= MILP 原規則 knee，hash 36205a0624a33d55…）；核對 L2、L27 Down 皆為 w=int8、a=f16。
- B：L2 Down a f16→p8e1；C：L27 Down a f16→p8e1；D：兩者。a 同時套用 projection input（x range）與 output（y range），與 native pass 相同；w 全部不變。
- 配置 hash：A 36205a0624a3、B 01438058b6b9、C d83946b8ec55、D 0c411b669dab。另評估 FP32（只作描述）。
- 資源（原 pipeline 成本函式，A 重現 knee 的 size／BOPs／R／BC）：A size 0.759821、BOPs 0.877976、R 0.877976、BC 330.463；B size 0.759821、BOPs 0.870833、R 0.870833、BC 873346.936；C size 0.759821、BOPs 0.870833、R 0.870833、BC 490.430；D size 0.759821、BOPs 0.863690、R 0.863690、BC 873506.903。BOPs 只是 proxy。
- 控制：config diff 與量化權重 hash；裝置上的 INT scale／zp、TF32 關閉、deterministic；Posit LUT 驗證；L2／L27 Down 的 input／output 實際格式；hook 不累積、切換不殘留；FP32 identity = 無量化模型；舊窗口重複 A 與交錯順序逐位元一致；原 epsilon／tokenizer／長度／loss／range；native（FP32／A／B／C／D，A 跑兩次）vs PyTorch 在 10 個 calibration 窗口的絕對 NLL 與 B−A、C−A、D−A paired 效應。控制失敗先定位，不進正式比較。
- 後端規則：任一比較的「native − PyTorch」paired 效應差 |平均| + 2 SE ≥ δ → 全部結論限定為 PyTorch QDQ 診斷；即使低於 δ，10 個 calibration 窗口也不是 native 對新文章結果的確認。
- 資料：512 篇新 WT103 文章 × 3 個不重疊 128-token 窗口（Qwen3 tokenizer）；verified 掃描排除 7849 篇（含 spike 診斷的 128 篇）；排除 WT2 重複／近重複；重跑歷史 Qwen token audit 並排除重疊；seed 順序 28101、起點 28102；凍結後才抽；不依新 loss 增減文章。
- 統計：文章 NLL = 3 窗口 summed CE / 381；d_B、d_C、d_D = NLL(X) − NLL(A)；paired 文章 bootstrap 20000 次（seed 28103），同一重抽；95% CI 與 Bonferroni 98.333% CI。
- δ = 0.001 nat/token（本輪選擇，非由新 loss 推得，不沿用 0.0075）。校正上界 < 0 → 支持改善；< δ → 新增損傷在容許內（CI 全正時寫「小幅下降但在容許內」）；校正下界 > δ → 超過容許；其餘未定。CI 跨 0 不等於等效。
- 次要：I = NLL(D) − NLL(B) − NLL(C) + NLL(A) 的平均與 95% CI；CI 跨 0 只表示未確認交互作用；不歸因 Gate/Up、position 0 或二階近似。
- 決策：在容許內且降低資源 → 只說「這些已選定位置存在降低保護的機會」，不硬編碼進 pipeline；超標 → 保留 f16；未定 → 保留原配置。不改預設、不啟動搜尋、不把 GPU QDQ 結果稱為 native 收益。
- 計畫前冒煙測試（scratchpad，不在 runs/，只用舊資料）：GPU 控制在 10 個 calibration 窗口上跑過，已看到 FP32／A／B／C／D 的 PyTorch NLL（B、D 約比 A 高 0.3 nat/token，C 約低 0.005）；native 只建 FP32／A、2 個窗口；資料階段用其他 seed（999101／999102）抽 6 篇未評估；eval／analyze 只用 9 個 calibration 窗口。修正 1 個程式錯誤。沒有評估任何新文章；配置、δ、資料量與規則來自需求，未因此更改。

## 預算（分鐘）：{'plan': 5, 'controls': 10, 'native': 30, 'freeze': 1, 'audit': 5, 'data': 15, 'eval': 20, 'analyze': 5}

