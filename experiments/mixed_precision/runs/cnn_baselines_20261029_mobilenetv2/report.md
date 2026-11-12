# MobileNetV2 weight mixed-precision baseline 比較：未完成（凍結）

日期：2026-10-29 執行，2026-11-12 凍結。程式：`run_cnn_baselines.py --model mobilenetv2`。計畫：`plan.json`／`plan.md`（10:40:16），修訂見 `amendments.json`（3 筆）。

**狀態：未完成，沒有準確度結果，不得引用為結論。**

- 依序執行的正式流程在 `score_infoq`（第 1 次）期間中斷：2026-10-29 的 Claude session 結束，該程序隨之停止，沒有留下 InfoQ 輸出。
- 之後的 `caps`、`search`、`evaluate`、`timing`（第 2、3 次）、`analyze` 都沒有執行。held-out shard 的預處理輸入已建立，但從未用於評估。
- 2026-11-12 研究方向改為 `agentic_mpq_spec_v0.2`，使用者決定凍結本 run，不再補跑（修訂 3）。

## 已完成的部分

設定與 ResNet18／EfficientNet-B0 run 相同（見 `../cnn_baselines_20261029/report.md`）。
- 模型為 timm `mobilenetv2_100.ra_in1k`，53 個 weight node。
- 搜尋範圍 51 個節點；`node_Conv_809`（conv_stem）與 `node_linear`（classifier）固定 INT8。

**對照（全部通過）**：
- checkpoint 折 BN 與 ONNX 比對：weight 最大差 5.3e-7、bias 2.9e-6。
- torch 與 onnxruntime 的 FP32 logits：最大差 1.19e-5，top-1 10/10。
- 前處理一致。
- BC 重建既有 B 表：最大相對差 0。

**評分表（只有第 1 次，calibration-only）**：

| 方法 | 裝置 | 張數 | 表格時間 (s) | 相對 BC | 與 BC 的 Spearman ρ（int4 / int6，51 個節點） | 備註 |
|---|---|---|---|---|---|---|
| BC | CPU，native | 10 | 21.0 | 1 | — | grads .so build 16.2 s（冷啟動）、grads runs 4.6 s |
| SQNR | CPU，ORT | 10 | 9.4 | 0.45 | 0.96 / 0.83 | 160 次 ORT |
| HTQ | GPU | 32 | 1.0 | 0.05 | 0.39 / 0.38 | — |
| HAWQ | GPU | 64 | 1358.7 | 64.7 | 0.97 / 0.97 | 53 層全收斂，每層 3–58 個 probe（共 1168），沒有負 trace |
| InfoQ | GPU | 1000 + 320 | — | — | — | 中斷，無輸出；冒煙測試估計約 94 s／（層, bit），全程約 2.7 h |

- 計時只有單次，沒有重複，也沒有範圍。
- Spearman ρ 是凍結時事後計算的描述值，不代表選出的配置是否相同。ResNet18 上 ρ 0.97 對應三個上限都選出同配置，但這不能直接推到本模型。

## 為什麼不能從這些表下結論

- 沒有評估任何配置，看不出排序相關的差異是否會轉成 accuracy 差異。
- A 組的資源上限（由 BC 的自動 frontier 決定）沒有正式寫入 `caps.json`。冒煙測試的估計值為 R_min 0.685、knee 0.705、寬鬆 0.853、激進 0.695，未用於任何正式比較。

若日後要補完，從 `score_infoq` 開始依序執行即可。也可以依新的修訂把 InfoQ 標示為未執行，從 `caps` 開始。現有的四張評分表與 plan 都保留。本 run 目錄 2.7 GB。
