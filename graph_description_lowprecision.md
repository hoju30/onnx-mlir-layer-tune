# 低精度混合格式配置最佳化研究架構(Low-Precision 版本)

> 本文件是 `graph_description.md`(Posit 版本)的低精度格式對照版本。Posit
> 版本維持不動、繼續保留；這份文件描述改用 `docs/LowPrecisionFormats.md`
> 中已實作的 `bf16 / f16 / int8 / fp8e4m3 / fp8e5m2 / FP32` 作為候選格式時,
> 研究架構需要調整的部分。整體 Configuration Generator → Accuracy
> Predictor → Cost Calculator → Pareto Selection 的迴圈概念不變,主要差異
> 集中在「格式如何在 IR 中實現」這一層,見第三節。

## 一、整體架構

```text
            Configuration Generator
                        ↓
            Candidate Configurations
                        ↓
    ┌───────────────────┴───────────────────┐
    ↓                                       ↓
Accuracy Predictor                      Cost Calculator
ML Prediction                           Static Analysis
    ↓                                       ↓
Predicted Accuracy                      Model Size
    ↓                                       ↓
Accuracy Loss =                         Peak Activation Memory
FP32 Accuracy − Predicted Accuracy
    └───────────────────┬───────────────────┘
                        ↓
                 Pareto Selection
                        ↓
            Selected Configurations
                        ↓
           Real Low-Precision Evaluator
                        ↓
              True Accuracy / Loss
                        ↓
           Dataset Update + Fine-tune
                        ↓
           Updated Accuracy Predictor
```

最佳化目標為：

$$
\min \mathrm{AccuracyLoss}
$$

$$
\min \mathrm{ModelSize}
$$

$$
\min \mathrm{PeakActivationMemory}
$$

與 Posit 版本相同,差異只在候選格式集合與其如何被實現(見二、三節)。

---

## 二、Configuration Generator

### 2.1 第一階段：Predictor 尚未建立

目前採用：

> **分層隨機抽樣＋少量人工代表配置**

分層條件包含：

* 低精度 node 比例，例如 25%、50%、75%、100%
* 不同格式組合（bf16／f16／int8／fp8e4m3／fp8e5m2／FP32）
* Conv/Gemm/MatMul（6 種格式皆可）與 elementwise op（僅 bf16/f16/FP32）
  的分佈比例——因為兩類 op 的候選格式集合大小不同（見 `docs/LowPrecisionFormats.md`
  的 per-op 支援表），分層抽樣需要分別考慮
* 模型前段、中段、後段
* 不同平均 bit-width 區間
* 高敏感度與低敏感度 node

人工代表配置包含：

* 全 FP32
* 全 bf16、全 f16
* 全 int8（僅套用在支援的 Conv/Gemm/MatMul node）
* 全 fp8e4m3、全 fp8e5m2（同樣僅套用在支援的 node）
* 單一 node 量化
* 僅前段或後段量化
* 相同格式的連續 region
* sensitivity-guided configuration

這些配置經由 Real Low-Precision Evaluator 取得真實 Accuracy，建立初始訓練
資料。

**校準（calibration）備註**：int8／fp8e4m3／fp8e5m2 需要外部提供的
activation scale／zero-point（`--convert-onnx-to-lowprecision` pass 本身不做
校準）。校準只需針對每個 (node, format) 組合在 calibration subset 上算一次
並快取，不需要每個抽樣的完整配置各算一次，因為 scale/zero-point 只跟該
node 的 activation 分佈與目標格式有關，跟其他 node 選了什麼格式無關。

### 2.2 第二階段：Predictor 建立後

改用：

> **Surrogate-assisted NSGA-II**

NSGA-II 使用 Accuracy Predictor 與 Cost Calculator 評估大量
configurations，再選出 Pareto configurations 進行真實評估。Mutation／
crossover 必須遵守 per-op 格式遮罩：elementwise node 的候選集合只有
{bf16, f16, FP32}，不能突變成 int8 或 fp8。

---

## 三、Accuracy Predictor

### 3.1 Graph 抽取來源

這是與 Posit 版本差異最大的地方。Posit 是把選中的 node 真正 lower 成
Posit Dialect 的新 op type（帶 N/ES），所以必須在 lowering 之後才能抽取
graph。低精度格式不是這樣運作：

* bf16／f16：只在原 op 前後包一層 `onnx.Cast`，**原本的 ONNX op type本身不變**
* int8：改寫成 `QuantizeLinear -> QLinearConv/QLinearMatMul -> DequantizeLinear`
  的小子圖，但這個改寫只在真正編譯執行時才會發生
* fp8e4m3／fp8e5m2：同樣是量化/反量化邊界 + `QLinearMatMul`，也只在真正
  編譯時才展開

因此低精度格式**沒有獨立的 dialect**，ONNX Dialect 的 graph 拓樸在所有
configuration 下都是同一個（只有每個 node 選中的格式不同）。Graph 抽取
應該在：

> **完成 shape inference 的 ONNX Dialect**

只做一次即可，不需要為每個 configuration 重新 lowering 再抽 graph；
selective lowering（`--convert-onnx-to-lowprecision`）留到 Real
Low-Precision Evaluator 真正編譯執行時才做。

### 3.2 Graph 定義

* **Node**：每個 ONNX operation（用 `onnx_node_name` 作為身分,不需要
  lowering 後才能還原）
* **Forward edge**：producer output tensor → consumer operation
* **Reverse edge**：提供 GNN 雙向 message passing
* 第一版只處理 tensor data dependency
* Graph 拓樸對所有 configuration 皆固定——與 Posit 版本不同（Posit 的
  graph 拓樸會因為哪些 node 被 lower 而改變）

Reverse edge 不代表真實執行方向，只用來讓前段 node 接收下游結構資訊。

### 3.3 Order 資訊

保留：

* `topological_index`：operation 執行順序
* `operand_index`：tensor 是 consumer 的第幾個輸入
* `producer_output_index`：tensor 是 producer 的第幾個輸出

因為 graph 結構在每個 configuration 下都相同，這些資訊只需在 ONNX 階段
算一次，之後每個 configuration 樣本共用同一份 `edge_index`。

---

## 四、Graph Feature

### 4.1 Node Feature

#### Static Feature

與 Posit 版本相同：

* op_type embedding
* input/output rank
* input/output log element count
* log weight element count
* log FLOPs
* kernel_h/w
* stride_h/w
* pad_top, pad_bottom, pad_left, pad_right
* dilation_h/w
* group
* log bias_element_count
* topological position
* in-degree／out-degree
* weight statistics(weight_mean,weight_std,weight_p99_abs,weight_zero_ratio,weight_dynamic_range)
* activation statistics(activation_mean, activation_std, activation_p99_abs,
  activation_zero_ratio, activation_dynamic_range)

#### Low-Precision Configuration Feature（取代 Posit 的 N/ES）

低精度候選集合固定且小（6 種），格式主要用類別特徵表示：

| 格式 | is_fp32 | bitwidth | is_float_format | requires_calibration |
|---|---|---|---|---|
| FP32 | 1 | 32 | 1 | 0 |
| bf16 | 0 | 16 | 1 | 0 |
| f16 | 0 | 16 | 1 | 0 |
| int8 | 0 | 8 | 0 | 1 |
| fp8e4m3 | 0 | 8 | 1 | 1 |
| fp8e5m2 | 0 | 8 | 1 | 1 |

`is_float_format` 用來區分 int8（線性定點,靠 scale/zero-point）跟其餘浮點
形狀的格式（即使 bitwidth 一樣也是完全不同的數值行為）。
`requires_calibration` 標記需要外部校準參數的格式。可再疊加一個可學習的
格式 embedding（one-hot 或 trainable embedding，over 這 6 種格式）。

#### 後續加入

* fake-quant error（`experiments/imagenet100/eval_fp_formats.py` 的
  `fake_quant()` 已經實作 bf16/f16/fp8e4m3/fp8e5m2 的 round-trip 模擬，可以
  直接沿用來算逐 node 的 fake-quant 誤差，不需要額外接 onnx-mlir 編譯）
* single-node sensitivity

### 4.2 Edge Feature

第一版使用：

* edge direction：forward／reverse
* `operand_index`
* `producer_output_index`
* tensor rank
* log tensor element count

format-transition edge（producer 與 consumer 格式不同）在低精度版本裡對應
到真正編譯時會插入的 `Cast` 或 `DequantizeLinear -> QuantizeLinear` 邊界，
不只是抽象特徵。

後續可加入：

* residual connection
* tensor statistics

---

## 五、Embedding 與 ML Model

```text
op_type
   ↓
Operation Embedding

Low-Precision format
   ↓
Format Embedding (6 類 one-hot / trainable embedding
                   + bitwidth / is_float_format / requires_calibration)

Numerical Features
   ↓
Log Transform + Normalization
```

將上述 features 串接後輸入：

```text
Node/Edge Features
        ↓
Edge-aware GNN
        ↓
Global Mean + Max Pooling
        ↓
MLP Regression Head
        ↓
Predicted Quantized Accuracy
```

因為 graph 拓樸在所有 configuration 下共用,資料集實作上可以只快取一份
`(edge_index, edge_attr)` per model，每個訓練樣本只需要換 node feature
裡的格式欄位，相較 Posit 版本（每個 configuration 都要重新 lowering 抽
graph）省下大量重複計算。

模型預測量化後準確率：

$$
\widehat{\mathrm{Accuracy}}_{\mathrm{quantized}}
$$

再計算 Accuracy Loss：

$$
\mathrm{AccuracyLoss}
= \mathrm{Accuracy}_{\mathrm{FP32}}- \widehat{\mathrm{Accuracy}}_{\mathrm{quantized}}
$$

主要參考：

* **TpuGraphs**：高階 tensor IR graph feature
* **DNNPerf**：DNN static feature
* **ProGraML**：dependency、edge direction 與 position 概念
* **ONNX-MLIR**：實際 IR feature 抽取
* **MIREncoder**：未來加入預訓練 IR embedding 時參考

---

## 六、Cost Calculator

### 6.1 Model Size

目前定義為 parameter storage size：

$$
\mathrm{ModelSize}
= \sum_{w \in \mathrm{Weights}}
\mathrm{Elements}(w)
\times
\frac{\mathrm{Bitwidth}(w)}{8}
$$

`Bitwidth(w)` 依第四節表格取值（32/16/16/8/8/8）。

注意：

* shared weight 只計算一次
* 不包含 low-precision runtime library（若有）
* 不直接使用 `.so` 檔案大小
* INT8／FP8 的 Gemm/MatMul 要求權重必須是 compile-time constant，若某 node
  的權重不是常數，該 node 就不具備 int8/fp8 的候選資格，Cost Calculator
  算出來的數字也不適用——這點要在 Configuration Generator 產生候選集合時
  一併過濾，而不是等 Cost Calculator 算完才發現不能用

### 6.2 Peak Activation Memory

依照 ONNX graph 的 topological order 進行 tensor liveness analysis。

每個 tensor：

* 在 producer operation 執行後產生
* 存活到最後一個 consumer 執行完成
* 根據該 tensor 的格式（依第四節表格的 bitwidth）計算記憶體

$$
\mathrm{PeakActivationMemory}
= \max_t
\left(
\sum_{v \in \mathrm{Live}(t)}
\mathrm{Elements}(v)
\times
\frac{\mathrm{Bitwidth}(v)}{8}
\right)
$$

第一版計算 logical activation memory，不包含：

* backend workspace
* memory alignment
* runtime allocator overhead

---

## 七、Pareto Selection 與模型更新

使用三個最佳化目標進行 Pareto selection：

```text
Minimize Accuracy Loss
Minimize Model Size
Minimize Peak Activation Memory
```

只選擇少量 Pareto configurations 進入 Real Low-Precision Evaluator，流程
如下（比 Posit 版本多一步校準）：

```text
Selected Configuration
    ↓
為每個 int8/fp8e4m3/fp8e5m2 node 查詢（或計算）校準用的
activation scale / zero-point
    ↓
標註 LOWP_NODE_FORMATS=<node>:<format>[:x_scale:x_zp:y_scale:y_zp],...
    ↓
selective ONNX-to-LowPrecision lowering
(onnx-mlir-opt --convert-onnx-to-lowprecision；
 目前只有 onnx-mlir-opt 有接這個 pass，主 onnx-mlir driver 還沒接)
    ↓
編譯、執行
    ↓
真實 Accuracy / Loss
```

真實評估結果用於：

```text
新增 Dataset
    ↓
Fine-tune Accuracy Predictor
    ↓
提升預測準確度
    ↓
產生下一輪 Configurations
```

因此，整體系統是一個反覆更新的：

> **Active Learning／Surrogate-assisted Multi-objective Optimization 流程**

---

## 八、與 Posit 版本的限制差異

除了 Posit 版本共通的限制（latency 非第一版目標、peak activation memory
為靜態估計、reverse edge 不代表真實執行順序等）之外，這個版本額外繼承
`docs/LowPrecisionFormats.md` 記錄的限制：

* 沒有自動校準：int8/fp8 的 activation scale/zero-point 必須外部提供
* INT8/FP8 的 Gemm/MatMul 要求權重（`B`）必須是 compile-time constant
* FP8 的權重/輸出量化只支援 per-tensor（沒有 per-channel scale）
* 只有 `onnx-mlir-opt` 接了 `--convert-onnx-to-lowprecision`，主
  `onnx-mlir` driver 還沒接
* elementwise op（Relu/Add/Sub/Mul/Div）只支援 bf16/f16，不支援 int8 或
  fp8e4m3/fp8e5m2，所以這類 node 的候選格式集合只有 3 種而非 6 種
