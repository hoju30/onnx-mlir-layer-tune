# Posit 混合精度配置最佳化研究架構

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
              Real Posit Evaluator
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

---

## 二、Configuration Generator

### 2.1 第一階段：Predictor 尚未建立

目前採用：

> **分層隨機抽樣＋少量人工代表配置**

分層條件包含：

* 量化 node 比例，例如 25%、50%、75%、100%
* 不同 Posit format 組合
* 模型前段、中段、後段
* 不同平均 bit-width 區間
* 高敏感度與低敏感度 node

人工代表配置包含：

* 全 FP32
* 全 P8、全 P16
* 單一 node 量化
* 僅前段或後段量化
* 相同格式的連續 region
* sensitivity-guided configuration

這些配置經由 Real Posit Evaluator 取得真實 Accuracy，建立初始訓練資料。

### 2.2 第二階段：Predictor 建立後

改用：

> **Surrogate-assisted NSGA-II**

NSGA-II 使用 Accuracy Predictor 與 Cost Calculator 評估大量 configurations，再選出 Pareto configurations 進行真實 Posit 評估。

---

## 三、Accuracy Predictor

### 3.1 Graph 抽取來源

從完成 shape inference 的：

> **ONNX-MLIR posit dialect**



### 3.2 Graph 定義

* **Node**：每個 posit operation
* **Forward edge**：producer output tensor → consumer operation
* **Reverse edge**：提供 GNN 雙向 message passing
* 第一版只處理 tensor data dependency

Reverse edge 不代表真實執行方向，只用來讓前段 node 接收下游結構資訊。

### 3.3 Order 資訊

保留：

* `topological_index`：operation 執行順序
* `operand_index`：tensor 是 consumer 的第幾個輸入
* `producer_output_index`：tensor 是 producer 的第幾個輸出

---

## 四、Graph Feature

### 4.1 Node Feature

#### Static Feature

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
* activation statistics(activation_mean
activation_std
activation_p99_abs
activation_zero_ratio
activation_dynamic_range)

#### Posit Configuration Feature
* `N`
* `ES`

#### 後續加入
* fake-quant error
* single-node sensitivity

### 4.2 Edge Feature

第一版使用：

* edge direction：forward／reverse
* `operand_index`
* `producer_output_index`
* tensor rank
* log tensor element count

後續可加入：

* residual connection
* precision format transition
* tensor statistics

---

## 五、Embedding 與 ML Model

```text
op_type
   ↓
Operation Embedding

Posit format
   ↓
Format Embedding

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

注意：

* shared weight 只計算一次
* 不包含 Posit runtime library
* 不直接使用 `.so` 檔案大小

### 6.2 Peak Activation Memory

依照 ONNX graph 的 topological order 進行 tensor liveness analysis。

每個 tensor：

* 在 producer operation 執行後產生
* 存活到最後一個 consumer 執行完成
* 根據該 tensor 的格式計算記憶體

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

只選擇少量 Pareto configurations 進入 Real Posit Evaluator。

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