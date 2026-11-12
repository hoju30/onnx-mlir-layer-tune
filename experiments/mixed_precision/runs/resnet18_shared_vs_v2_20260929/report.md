# ResNet18：shared 與 shared_v2

**ORT 模擬與 ONNX-MLIR 編譯評測均已完成。** 下方原表為 ORT 模擬；文末另外列出編譯後結果，兩種後端分開比較。

| 設定 | 內容 |
|---|---|
| 模型 | ResNet18，torchvision_v1 |
| 資料 | ImageNet-1K validation 抽樣子集 |
| 校準／held-out | 10／1,000 張，互不重疊 |
| FP32 top-1 | 70.6% |
| W/A 各自候選 | int4、int5、int6、int7、int8、f16、bf16、fp32 |
| 搜尋 | MILP；全部配置 solver status=0，資源上限檢查通過 |
| shared | 權重敏感度 ×（weight raw L2＋真實 output activation raw L2） |
| shared_v2 | 相同權重敏感度 ×（weight、input、output 的相對平方誤差總和） |
| 量化範圍 | 21 個有權重節點，各自同時選 W/A；其他節點 FP32 |

| 預算 f | shared top-1 | shared_v2 top-1 | 差異（百分點） |
|---|---:|---:|---:|
| 0.60 | 0.6% | 21.3% | +20.7 |
| 0.65 | 1.0% | 56.5% | +55.5 |
| 0.70 | 1.7% | 65.6% | +63.9 |
| 0.75 | 8.8% | 68.1% | +59.3 |
| 0.80 | 13.8% | 69.1% | +55.3 |

這組模擬中，shared_v2 在五個預算下均較好，但最低預算下仍明顯低於 FP32。此處的 shared 不是「以權重誤差代理 activation」的版本。兩組共用相同敏感度與格式選單，因此比較的是 perturbation 定義的改變，不能單獨據此判定哪種敏感度較準。

每組均通過有限數值檢查。實際使用的 weight size 與 BOPs proxy 詳見 `simulation_results.json`；相同的是資源上限，而非相同的實際用量。

- [模擬原始結果](simulation_results.json)
- [表格 CSV](simulation_comparison.csv)
- [全部配置與 MILP 狀態](configs.json)
- [設定、圖片清單、程式雜湊](experiment.json)
- [背景編譯紀錄](../resnet18_shared_vs_v2_20260929.log)

## 已完成的 ONNX-MLIR 編譯後結果

| 預算 f | shared top-1 | shared_v2 top-1 | 差異（百分點） |
|---|---:|---:|---:|
| 0.60 | 0.7% | 20.9% | +20.2 |
| 0.65 | 1.0% | 56.0% | +55.0 |
| 0.70 | 1.8% | 64.8% | +63.0 |
| 0.75 | 8.8% | 67.8% | +59.0 |
| 0.80 | 13.5% | 68.7% | +55.2 |

來源：[編譯結果摘要](summary.json)、[完整編譯結果](compiled/results.json)。全部配置通過量化位置、有限數值及警告檢查。
