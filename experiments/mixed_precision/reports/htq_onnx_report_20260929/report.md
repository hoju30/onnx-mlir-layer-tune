# HTQ 接入 ONNX-MLIR：搜尋策略與擾動指標的實驗分析

整理日期：2026-09-29。建議分成四部分、約 10 張投影片。

**報告主線：先建立 HTQ 的方法，再說明編譯器接入方式，最後用兩組控制實驗分開檢查「搜尋器」與「perturbation criterion」。**

本文整理現有實驗，沒有重新執行模型評測。實驗數據統一以表格呈現；搜尋對照使用 ORT 模擬，最終 criterion 表使用 onnx-mlir 編譯後的 QDQ 執行結果。

| 主實驗設定 | 內容 |
|---|---|
| 模型 | MobileNetV2（timm_ra） |
| 校準資料 | 10 張 |
| 評測資料 | 1,000 張 held-out |
| FP32 top-1 | 72.1% |
| 預算 f | 0.60、0.65、0.70、0.75、0.80 |

此結果來自上述 held-out 子集，不能直接視為完整 ImageNet validation 準確率。

## 第一部分：HTQ 原本的方法（投影片 1–2）

### 1. 問題與核心想法

混合精度量化要替每一層選擇 weight 與 activation 位元數。降低位元數可以降低理論儲存與運算成本，但量化造成的數值擾動可能降低準確率。HTQ 用敏感度加權的量化擾動估計影響，再搜尋模型大小、BOPs 與擾動之間的折衷。

每層的 connection sensitivity 為：

\[
S_i=\operatorname{mean}\left|W_i\odot\frac{\partial L}{\partial W_i}\right|.
\]

在權重上放置初始為 1 的乘法 mask，mask 的梯度就是上式的乘積，因此一次反向傳播便能收集多層的分數。分數大表示這層的連接對所用 loss 較敏感。

**講稿：**「HTQ 不需要把每一個全模型配置都訓練一遍，而是先估每層的重要性，再估每種精度會產生多少誤差，用兩者相乘作為搜尋時的代價。」

來源：[官方 sensitivity.py](https://github.com/zkkli/HTQ/blob/main/sensitivity.py)。官方程式會先重新初始化 Conv/Linear 權重，使用一個訓練 minibatch 計算敏感度；這與我們在預訓練 ONNX 模型上計算的做法不同。

### 2. Perturbation 與三維搜尋

依目前核對的官方程式，普通 Conv/Linear 層的搜尋分數可寫成：

\[
P_i^{\mathrm{official\ code}}(b_w,b_a)
=S_i\left(\|W_i-Q_{b_w}(W_i)\|_2+
\|W_i-Q_{b_a}(W_i)\|_2\right).
\]

這裡有兩個必須說清楚的實作細節：官方 code 使用未平方的 L2 norm；activation 項也把 weight tensor 用 activation bits 量化，並非測量真實 activation 誤差。這是官方 code 的公式，不將它標成論文 Eq. 5。

候選為 W∈{2,4,8}、A∈{4,8}，普通層有 6 種選項；官方對名稱含 downsample 的層另採 weight-only 分支。每加入一層，就累積：

\[
M=\sum_i |W_i|b_{w,i}/8,\qquad
B=\sum_i\mathrm{MACs}_i b_{w,i}b_{a,i},\qquad
P=\sum_i P_i.
\]

其中 B 式描述普通 W+A 層，downsample 的特殊處理見原碼。搜尋逐層展開，以大小和 BOPs 排序、依擾動剪枝，保留折衷候選。公開程式主要輸出三維 frontier，未提供完整的每層配置重建與部署流程。

來源：[官方 perturbation](https://github.com/zkkli/HTQ/blob/main/utils/pareto_utils.py)、[官方 frontier](https://github.com/zkkli/HTQ/blob/main/pareto_frontier.py)。論文 Eq. 9、QAT 訓練細節此次尚未核對全文，主報告不以這些未核對細節支撐結論。

## 第二部分：如何接入 ONNX-MLIR（投影片 3–4）

### 3. 將選格式與編譯執行串接

```mermaid
flowchart LR
    A[預訓練 ONNX 模型與校準資料] --> B[ORT training 計算敏感度]
    A --> C[校準範圍與實際量化誤差]
    B --> D[每節點每組 W/A 的 criterion]
    C --> D
    D --> E[MILP 或 HTQ-style tree]
    E --> F[configs.json]
    F --> G[MIXEDP_NODE_FORMATS]
    G --> H[ONNX-MLIR mixed precision pass]
    H --> I[編譯與 held-out 評測]
```

| 階段 | 接入方式 | 主要程式 |
|---|---|---|
| 敏感度 | ONNX graph 上建立梯度圖，按節點名稱記錄分數 | `htq_snip_ort.py` |
| 誤差表 | 真實權重、校準 activation；量化運算對齊 pass | `htq_perturbation.py`、`check_shared_fix.py` |
| 搜尋 | 固定 weight bytes 與 BOPs 上限 | `htq_pareto_search.py`、`mp_milp.py` |
| 配置傳遞 | JSON 轉成節點 W/A 格式與 activation scale/zero-point | `eval_random_mixed_precision.py` |
| 編譯與評測 | pass 插入 QDQ，執行並計算 top-1/top-5 | `ONNXToMixedPrecision.cpp` |

這次格式為 W、A 各自從 int4–int8、f16、bf16、fp32 選，共 64 組。搜尋僅調整有常數權重的受支援節點；MobileNetV2 共 53 個，其他節點維持 FP32。Posit 此次排除。

預算 f 的意思是兩個上限同時成立：

\[
M\le fM_{\mathrm{int8}},\qquad B\le fB_{\mathrm{int8\times int8}}.
\]

**講稿：**「我們將選格式放在編譯前的 Python 流程，pass 負責實現選定配置。敏感度與配置都使用 ONNX 節點名稱，因此不用另外維護 PyTorch 層與 ONNX 節點的對應。」

### 4. 必須對齊的量化行為

- 整數 weight：對稱 per-tensor；編譯時將常數權重做 quantize-dequantize。
- 整數 activation：依校準範圍建立 affine scale 與 zero-point。
- 每個選定節點的輸入與輸出都會做 activation QDQ；bias 保持 FP32。
- f16/bf16 使用格式轉換造成的捨入誤差，不套整數 affine 公式。
- 本流程不做 QAT，直接評測配置的 PTQ 效果。

單一有權重節點可概念化為：

\[
\hat y_i=Q_{a_i}\!\left(\operatorname{Op}_i(Q_{a_i}(x_i),Q_{w_i}(W_i),b_i)\right).
\]

此處 Q 表示量化後還原的浮點值。**pass 目前使用 QDQ 模擬，核心運算仍為 FP32。** 所以編譯後準確率是真實執行結果，但 weight bytes 與 BOPs 是格式配置的理論成本／proxy，不能當作已實現的檔案壓縮率或硬體加速倍數。

來源：[pass 實作](../../../../src/Conversion/ONNXToMixedPrecision/ONNXToMixedPrecision.cpp)、[評測器](../../eval_random_mixed_precision.py)。

## 第三部分：為什麼本流程改用 MILP（投影片 5–6）

### 5. 控制實驗：固定 criterion，只換搜尋器

這部分應命名為「HTQ-style 剪枝搜尋在擴大格式空間中的限制」，避免把結論說成所有 Pareto 方法都不能用。

固定模型、校準與評測資料、量化器、`shared_v2` 誤差表，以及相同資源上限。比較現有 HTQ-style tree（每個排序軸 beam cap 2,000）與 MILP。Tree 在保留的可行葉節點中挑 criterion 最小者，沒有使用 Eq. 9 選點，因此差異不能歸因於 Eq. 9。

小選單：W/A 各 {int4,int6,int8}，每節點 9 組；完整選單：每節點 64 組。所有下表準確率皆為 **ORT 模擬、1,000 張 held-out**。

| 格式選項／節點 | f | Tree objective / MILP objective | Tree top-1 | MILP top-1 |
|---|---:|---:|---:|---:|
| 9 | 0.60 | 1.30× | 14.7% | 36.5% |
| 9 | 0.65 | 1.33× | 42.8% | 53.8% |
| 9 | 0.70 | 1.25× | 54.2% | 59.1% |
| 9 | 0.75 | 1.17× | 60.0% | 64.2% |
| 9 | 0.80 | 1.08× | 59.9% | 64.1% |
| 64 | 0.60 | 4.02× | 2.6% | 49.8% |
| 64 | 0.65 | 2.41× | 21.8% | 55.6% |
| 64 | 0.70 | 2.62× | 30.5% | 59.0% |
| 64 | 0.75 | 2.41× | 44.7% | 61.2% |
| 64 | 0.80 | 2.36× | 57.0% | 65.8% |

**講稿：**「即使給 tree 一個已修正、可用的 criterion，它在完整格式選單中仍漏掉較好的配置。表中可看到低預算時，預測誤差與準確率的差距尤其明顯。把 criterion 固定後，這組實驗就能辨識搜尋近似造成的損失。」

來源：[結果 JSON](../../runs/mobilenetv2_timm_ra_tree_vs_milp_v2/results.json)、[實驗腳本](../../check_tree_vs_milp.py)。

### 6. 如何解讀與處理

逐層剪枝會永久移除部分配置；我們另加入的 beam cap 也會減少保留候選。格式選項擴大後，有限候選集合可能無法保留固定預算下的好解。現有實驗顯示這個實作與 beam 設定的搜尋誤差，尚未分別量化原始剪枝與 beam cap 各自的影響。

相同的是資源上限，不是完全相同的資源用量。以下為完整格式選單的資源使用範例，比值皆以全 int8 為基準並四捨五入：

| f | 搜尋器 | Weight size 比值 | BOPs proxy 比值 |
|---|---|---:|---:|
| 0.60 | Tree | 0.594 | 0.534 |
| 0.60 | MILP | 0.598 | 0.600 |

因此使用二元變數 x(i,w,a)，每節點恰選一組格式：

\[
\min_x\sum_{i,w,a}x_{i,w,a}C_i(w,a)
\]

\[
\sum_{w,a}x_{i,w,a}=1,\quad
\sum x_{i,w,a}M_i(w)\le M_{\max},\quad
\sum x_{i,w,a}B_i(w,a)\le B_{\max},\quad x_{i,w,a}\in\{0,1\}.
\]

成功證明最優時，MILP 消除了這個可加總 criterion 的搜尋近似誤差（在求解器容差內）；**它不保證真正的準確率最優**。其用途是把「搜尋沒找好」與「指標預測不好」分開。

補充反例來自較早的 PyTorch 改編流程。其校準數、候選格式與量化位置不同，因此另表呈現；它說明 MILP 的較佳 objective 不等於準確率必勝。

| 流程 | 模型 | Criterion | f | Tree top-1 | MILP top-1 |
|---|---|---|---:|---:|---:|
| 舊 PyTorch 改編流程 | ResNet18 | shared | 0.75 | 68.2% | 67.3% |

來源：[舊流程報告](../../../../../HTQ/runs/search_compare_1000_20260927/report.md)。

## 第四部分：Perturbation 遇到的問題與處理（投影片 7–10）

### 7. 固定 MILP 後，原始移植 criterion 仍然失效

我們最初移植的 shared 是：

\[
C_i^{\mathrm{shared}}=S_i\big(\|Q_w(W_i)-W_i\|_2+\|Q_a(y_i)-y_i\|_2\big).
\]

這已經把官方 code 的 activation 代理換成真實輸出 activation，因此稱 HTQ-style shared，不稱完整 HTQ 重現。

即使用 MILP 求到此分數的最優配置，compiled top-1 仍嚴重下降（完整數據見第 9 張投影片）。表示只更換搜尋器無法解決 criterion 與量化行為不匹配的問題。

兩項診斷：

| 問題 | 原因 | 對選擇配置的影響 |
|---|---|---|
| Weight／activation 誤差尺度不平衡 | 原始 L2 受元素數量、訊號尺度與取樣影響，兩項不能直接視為同等影響 | Weight 在 criterion 中相對便宜，配置偏向較低 weight 精度 |
| 誤差位置不完整 | Pass 量化輸入與輸出，舊 criterion 只量輸出；split 的 activation sensitivity 又計算在輸入 | 預測分數沒有完整對應實際量化位置 |

| 診斷項目 | 條件 | 觀察值 |
|---|---|---:|
| Activation／weight 分數總和比值 | shared，各節點 int4 | 95.6 |
| 選 int4 的權重節點數 | shared，f=0.60 | 31／53 |
| 選 int4 的權重節點數 | shared，f=0.80 | 27／53 |

分數比值是跨層總和的比值，並非每層都具有同樣倍率。低位元配置與 weight 被低估的解釋一致，但尚非單獨的因果證明。

**講稿：**「MILP 忠實地最佳化我們給的分數。如果 weight 在分數裡顯得便宜，它就會大量降低 weight 精度。這時準確率崩掉，不能只怪搜尋器。」

### 8. 修正：相對平方誤差，並涵蓋輸入與輸出

固定敏感度 S_i，將 perturbation 改為：

\[
C_i^{v2}=S_i\left(
\frac{\|\Delta W_i\|_2^2}{\|W_i\|_2^2}+
\frac{\|\Delta x_i\|_2^2}{\|x_i\|_2^2}+
\frac{\|\Delta y_i\|_2^2}{\|y_i\|_2^2}\right).
\]

| 修改 | 處理的問題 |
|---|---|
| 各項改成相對誤差能量 | 減少 tensor 元素數量與訊號尺度造成的失衡 |
| Activation 包含輸入與輸出 | 對齊 pass 實際量化位置 |
| 保留相同 S_i 與 MILP | 比較 perturbation 定義改變後的結果 |

v2 消除單純複製元素造成的 norm 增長，但不代表所有層或 W/A 的任務影響已完全等價。現行程式直接除以 norm，若推廣到零能量 tensor，需另外定義分母為零的處理。

| Criterion | activation／weight 分數總和比值（int4） | f=0.60 top-1 | f=0.70 top-1 | f=0.80 top-1 |
|---|---:|---:|---:|---:|
| shared：raw L2、僅輸出 | 95.6 | 0.0% | 0.1% | 0.2% |
| shared_v2：相對平方誤差、輸入＋輸出 | 2.87 | 49.8% | 59.0% | 65.8% |

此表為同一批實驗的 **ORT 模擬結果**。修正後分數失衡降低，準確率大幅恢復。這個前後比較同時改變正規化、平方及 activation 覆蓋位置，因此不能單憑此表分別量化各項修改的貢獻。

**講稿：**「我們保留原本的重要性分數，修正的是誤差如何定義：用相對誤差處理尺度，並把 pass 會量化的輸入與輸出都納入。表中可以看到，修正後配置的準確率大幅恢復。」

來源：[修正前後結果](../../runs/mobilenetv2_timm_ra_shared_fix/results.json)、[v2 公式實作](../../check_shared_fix.py)。

### 9. SQNR criterion：直接量測傳到最終輸出的誤差

對每個節點、每組 W/A，僅量化該節點，保留其餘配置 FP32，在校準資料上量測：

\[
C_i^{\mathrm{logit}}(w,a)=\sum_{x\in D_{\mathrm{cal}}}
\|z_{i,w,a}(x)-z_{\mathrm{FP32}}(x)\|_2^2.
\]

程式名稱是 SQNR，但儲存的是 logit noise power，沒有直接算 dB；固定參考訊號下，最小化 noise power 對應最大化 SQNR。MILP 使用各節點分數相加。

它能量到同一節點 W+A 同時量化，以及擾動經後續網路傳播的影響。代價是每組格式都要模擬；跨節點同時量化的交互作用，仍未完整包含在可加總分數中。

| f | shared | split_connection | split_direct | shared_v2 | SQNR |
|---|---:|---:|---:|---:|---:|
| 0.60 | 0.1% | 0.0% | 0.3% | 49.4% | **59.3%** |
| 0.65 | 0.1% | 0.2% | 0.7% | 56.9% | **63.4%** |
| 0.70 | 0.1% | 0.1% | 0.8% | 59.1% | **66.9%** |
| 0.75 | 0.2% | 0.4% | 5.7% | 61.5% | **68.6%** |
| 0.80 | 0.3% | 0.9% | 6.5% | 66.4% | **68.8%** |

此表為 **onnx-mlir 編譯後 QDQ 執行**，1,000 張 held-out。v2 與其他 criterion 取自各自已保存的 compiled runs；不能拿它與上一張修正前後比較表的數值差異，直接判定 ORT 與 compiler 的同配置誤差。

來源：[四種 criterion compiled](../../runs/mobilenetv2_timm_ra_criteria_milp/results.json)、[v2 compiled](../../runs/mobilenetv2_timm_ra_shared_v2_compiled/results.json)、[SQNR 實作](../../sqnr_tables.py)。

### 10. 報告應提出的結論

**講稿：**「接入 ONNX-MLIR 時，有兩個獨立問題。第一，擴大格式空間後，目前的 tree 剪枝搜尋會漏掉較好的解，因此用 MILP 控制搜尋誤差。第二，即使搜尋已求到最優，原始移植的 perturbation 仍有尺度與量化位置不一致的問題。改成輸入、輸出的相對誤差後，準確率大幅恢復；直接量測 logit noise 的 criterion 在這組實驗中更好。」

這些結果支持完整 criterion 與配置選擇流程的比較；尚不能單獨宣稱 split_direct 或 split_connection 的敏感度本身較差，因為舊版本仍搭配未修正的 raw L2 誤差。若研究問題是敏感度預測能力，後續需固定誤差定義與量化位置，只替換 sensitivity，再於獨立資料評估 W+A 聯合擾動的排序與配置準確率。

## 表格數據來源

| 報告表格 | 原始數據 |
|---|---|
| 搜尋器比較與資源使用 | [搜尋實驗 JSON](../../runs/mobilenetv2_timm_ra_tree_vs_milp_v2/results.json)、[整理後 CSV](search_comparison.csv) |
| Perturbation 修正前後比較 | [修正實驗 JSON](../../runs/mobilenetv2_timm_ra_shared_fix/results.json) |
| 編譯後準確率比較 | [整理後 CSV](compiled_accuracy.csv)；原始 compiled JSON 見第 9 張投影片 |

以上數值為單組校準／評測資料結果，未提供不同校準種子的重複試驗。這些表格不以此推論所有模型、格式空間或所有 Pareto 搜尋方法。

## 補充實驗：ResNet18 的 shared／shared_v2 比較

使用相同的 10 張校準與 1,000 張 held-out、MILP、非 posit 格式選單。下表是 **ORT 模擬結果**，FP32 top-1 為 70.6%；不能與前述 MobileNetV2 compiled 表混作同一評測後端。編譯後評測亦已完成，另表列於下方。

| 預算 f | shared top-1 | shared_v2 top-1 | 差異（百分點） |
|---|---:|---:|---:|
| 0.60 | 0.6% | 21.3% | +20.7 |
| 0.65 | 1.0% | 56.5% | +55.5 |
| 0.70 | 1.7% | 65.6% | +63.9 |
| 0.75 | 8.8% | 68.1% | +59.3 |
| 0.80 | 13.8% | 69.1% | +55.3 |

本次所有預算均由 shared_v2 取得較高準確率，但最低預算下仍有明顯準確率損失。shared 使用真實 output activation raw L2，並非官方的權重代理方式。

來源：[ResNet18 完整實驗紀錄](../../runs/resnet18_shared_vs_v2_20260929/report.md)。

### ResNet18 編譯後結果

| 預算 f | shared top-1 | shared_v2 top-1 | 差異（百分點） |
|---|---:|---:|---:|
| 0.60 | 0.7% | 20.9% | +20.2 |
| 0.65 | 1.0% | 56.0% | +55.0 |
| 0.70 | 1.8% | 64.8% | +63.0 |
| 0.75 | 8.8% | 67.8% | +59.0 |
| 0.80 | 13.5% | 68.7% | +55.2 |

來源：[編譯結果摘要](../../runs/resnet18_shared_vs_v2_20260929/summary.json)。

## 補充消融：shared_v2 是否需要 input 誤差項

ResNet18，10 張校準、1,000 張 held-out，FP32 70.6%。兩組使用相同敏感度、相對平方誤差、MILP 與格式選單；唯一變因是 criterion 是否計入 input 誤差，實際 input/output QDQ 均維持啟用。

以下為 **ORT 模擬結果**；新增 output-only 配置的編譯評測仍在背景執行。

| f | W＋O top-1 | W＋I＋O top-1 | 加入 input 後差異（百分點） |
|---|---:|---:|---:|
| 0.60 | 19.0% | 21.3% | +2.3 |
| 0.65 | 55.7% | 56.5% | +0.8 |
| 0.70 | 65.0% | 65.6% | +0.6 |
| 0.75 | 65.9% | 68.1% | +2.2 |
| 0.80 | 69.1% | 69.1% | +0.0 |

加入 input 在四個預算下較高、一個持平；提升幅度有限，不能推論所有預算或模型都一定有利。

[完整消融紀錄與編譯進度](../../runs/resnet18_input_error_ablation_20260929/report.md)。
