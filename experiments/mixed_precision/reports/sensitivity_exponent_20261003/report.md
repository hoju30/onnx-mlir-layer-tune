# Sensitivity 次方與 W／input／output 比例對照（2026-10-03）

固定平方量化誤差後，比較 mean(g²) 與 mean(|g|)，再配平 W／input／output 總量，最後追加 weight 使用 [mean(abs(g))]² 的同量綱對照。共檢查 51 個配置；GPT-2 寬預算找到改善候選，但目前沒有跨模型、跨預算的一致勝出方法。

## 設定與公式

| 項目 | GPT-2 small | MobileNetV2 |
|---|---|---|
| 資料 | WikiText-2 raw | ImageNet-1K |
| 校準／評測 | 10／32 個長度 128 視窗；4064 個預測 token | 10／1000 張影像 |
| 搜尋節點 | 48 個投影節點 | 53 個 Conv／Linear 節點 |
| 格式 | int4–8、f16、bf16、fp32 | 同左 |
| 搜尋／後端 | MILP／ORT QDQ simulation | 同左 |

所有方法實際同時量化 W、input、output，input/output 共用同層 activation 格式。兩個預算皆以搜尋節點全 W8A8 為基準：weight bytes ≤ f×基準、BOPs ≤ f×基準。

令 t 為 W、X 或 Y，g=∂L/∂t，Δt=Q(t)−t；省略共同校準樣本累加因子。

| 方法 | 每個 tensor 的分數 |
|---|---|
| Fisher scalar (`2nd_tensor`) | mean(g²) × Σ(Δt)² |
| 絕對梯度、固定平方誤差 (`abs_g_sq_error`) | mean(abs(g)) × Σ(Δt)² |
| 絕對梯度、配平三項 (`abs_matched_totals`) | α_t × mean(abs(g)) × Σ(Δt)² |

配平係數 α_t = [Σ層 Fisher_t(int8)] / [Σ層 abs_t(int8)]。W、X、Y 分別配平，僅使用校準資料；沒有用 test 指標選係數。這只配平全 int8 下的跨層總分，不會讓各層分數相同，也不保證混合格式時比例相同。

`mean(|g|) × 平方誤差` 是 sensitivity 次方消融，不是原始 HTQ，也不是一階 Taylor 或 Fisher 的同尺度估計。

## 完整預算結果

### GPT-2：PPL ↓

| f | Fisher scalar | 絕對梯度＋平方誤差 | 再配平 W／X／Y |
|---:|---:|---:|---:|
| 0.80 | 151.22 | 186.51 | 300.86 |
| 1.00 | 75.91 | 81.53 | 88.97 |
| 1.25 | 68.05 | 69.88 | 70.25 |
| 1.50 | 66.83 | 66.84 | 66.55 |

[固定誤差原始結果](../../runs/gpt2_sensitivity_exponent_20261003/results.json) · [配平原始結果](../../runs/gpt2_sensitivity_balance_20261003/results.json)

### MobileNetV2：Top-1（%）↑

| f | Fisher scalar | 絕對梯度＋平方誤差 | 再配平 W／X／Y |
|---:|---:|---:|---:|
| 0.60 | 59.00 | 0.20 | 48.90 |
| 0.65 | 62.90 | 0.60 | 57.90 |
| 0.70 | 66.80 | 0.40 | 61.30 |
| 0.75 | 69.10 | 1.70 | 65.20 |
| 0.80 | 68.60 | 2.60 | 67.80 |

[固定誤差原始結果](../../runs/mbv2_sensitivity_exponent_20261003/results.json) · [配平原始結果](../../runs/mbv2_sensitivity_balance_20261003/results.json)

MobileNetV2 的配平版本在 f=0.80 的 Top-1 為 67.8%，低於 Fisher 的 68.6%；但 CE loss 為 1.4515，優於 Fisher 的 1.4944。不可把 Top-1 較低解讀成所有指標都較差。

GPT-2 在 f=1.50 幾乎持平：配平版 PPL 66.55，略低於 Fisher 66.83；視窗 bootstrap 的 ΔNLL 區間包含 0，不能視為確立改善。

## 三項分數的比例

下表是全 int8 校準 criterion 總分占比，**不是硬體預算占比，也不是真實 loss 貢獻占比**。

| 模型 | sensitivity（誤差皆平方） | W | input | output |
|---|---|---:|---:|---:|
| gpt2 | mean(g²) | 38.93% | 17.42% | 43.65% |
| gpt2 | mean(abs(g)) | 21.28% | 7.04% | 71.69% |
| mbv2 | mean(g²) | 66.25% | 0.77% | 32.98% |
| mbv2 | mean(abs(g)) | 3.51% | 8.63% | 87.86% |

MobileNetV2 的 weight 項由 66.25% 降至 3.51%。配平後 Top-1 大幅恢復，支持「三項尺度失衡」是這個絕對梯度版本失效的重要原因；並不證明 Fisher 的比例是唯一正確比例。

## 單項替換：保留各項 int8 總量

其餘兩項保持 Fisher，只替換指定項為配平後的絕對梯度分數。只測各模型一個緊預算，不能推廣到所有預算。

| 設定 | GPT-2 f=0.80 PPL ↓ | MobileNetV2 f=0.60 Top-1 ↑ |
|---|---:|---:|
| 全部 Fisher | 151.22 | 59.0% |
| 只換 weight | 205.83 | 54.8% |
| 只換 input | 150.10 | 58.8% |
| 只換 output | 153.17 | 59.6% |
| 三項皆替換 | 300.86 | 48.9% |

[GPT-2 單項結果](../../runs/gpt2_sensitivity_term_20261003/results.json) · [MobileNetV2 單項結果](../../runs/mbv2_sensitivity_term_20261003/results.json)

在這兩個測點，weight sensitivity 的替換造成最大的單項退步。input/output 的小幅變化不能直接認定更好；三項一起替換也不能由三個單項結果相加預測。每次均重新做 MILP，因此這是 criterion 的配置效果，不是固定配置下的真實量化損傷分解。

## 可以支持的解釋

1. 固定平方誤差，直接以 mean(|g|) 取代 mean(g²) 仍會退步，因此前一輪失效不只是 L1 誤差造成。
2. MobileNetV2 的嚴重退化有很大一部分能用 W／X／Y 的分數比例修正；不宜全部歸因於層重要性排序。
3. 配平後，緊預算仍有差距。單項替換支持 weight 項的層間相對評分是下一個優先研究方向；尚未定位到個別層的因果影響。
4. 保留目前 Fisher scalar 作基準。mean(g²) 不是精確 Hessian；目前 GPT-2 取每個視窗平均 CE 的梯度再平方，也不是逐 token empirical Fisher。

## 延伸：同量綱的 weight 梯度統計對照

前面的單項測試顯示 weight 的影響最大，因此這一輪固定 input/output 為 Fisher，只改 weight：

| 方法 | weight 分數（省略共同校準數因子） |
|---|---|
| Fisher scalar | mean(g²) × Σ(ΔW)² |
| 先平均再平方 | [mean(abs(g))]² × Σ(ΔW)² |
| 先平均再平方，配平 W 總量 | β × [mean(abs(g))]² × Σ(ΔW)² |

β 僅用校準分數，令新 weight 項的全 int8 總分等於原 Fisher 的 weight 項；input/output 完全保持原分數。β 為 GPT-2 2.95449、MobileNetV2 8.72535。

這裡兩種 sensitivity 的梯度次方相同，差別為：

$$\mathrm{mean}(g^2)=[\mathrm{mean}(|g|)]^2+\mathrm{Var}(|g|).$$

平均與變異數均涵蓋權重元素和校準樣本。Fisher 多出的部分會提高梯度大小不均勻的層的相對分數；這是統計量的恆等式，不表示變異數本身就等於真實量化損傷。

### GPT-2：PPL ↓

| f | Fisher scalar | W 先平均再平方 | W 先平均再平方＋配平 |
|---:|---:|---:|---:|
| 0.80 | 151.22 | 185.18 | 152.13 |
| 1.00 | 75.91 | 85.99 | 77.55 |
| 1.25 | 68.05 | 71.05 | 68.39 |
| 1.50 | 66.83 | 67.01 | 64.93 |

[完整結果](../../runs/gpt2_weight_gradient_moment_20261003/results.json) · [每層梯度統計](../../runs/gpt2_weight_gradient_moment_20261003/weight_gradient_moments.json)

### MobileNetV2：Top-1（%）↑

| f | Fisher scalar | W 先平均再平方 | W 先平均再平方＋配平 |
|---:|---:|---:|---:|
| 0.60 | 59.00 | 47.40 | 58.30 |
| 0.65 | 62.90 | 58.70 | 63.50 |
| 0.70 | 66.80 | 65.40 | 66.80 |
| 0.75 | 69.10 | 67.30 | 68.10 |
| 0.80 | 68.60 | 68.40 | 68.00 |

[完整結果](../../runs/mbv2_weight_gradient_moment_20261003/results.json) · [每層梯度統計](../../runs/mbv2_weight_gradient_moment_20261003/weight_gradient_moments.json)

### 這個延伸能說明什麼

在 GPT-2，配平後的先平均再平方版本比直接 mean(abs(g)) 更接近 Fisher；f=1.50 PPL 從 66.83 降至 64.93。此測點的配對視窗 bootstrap ΔNLL 約 −0.02894，95% 區間約 [−0.040, −0.017]，在目前子集上有改善訊號；其他三個 GPT-2 預算的 ΔNLL 區間均包含 0。

MobileNetV2 配平版在 f=0.65 Top-1 63.5%（Fisher 62.9%），f=0.70 同為 66.8%，其他預算稍低。沒有跨預算的一致優勢。未配平版 f=0.80 Top-1 68.4% 略低於 Fisher 68.6%，但 CE loss 1.3853 低於 Fisher 1.4944；Top-1 與 loss 必須分開評估。

因此，現有結果不支持「一定要保留 mean(g²) 中的所有梯度大小變異才有效」。gradient sensitivity 的量綱、W／A 相對尺度和層間分配都會影響結果。先平均再平方＋配平是可保留驗證的候選，尚不足以全面取代 Fisher。

這一輪新增 18 個配置，加上前述 33 個共 51 個，皆通過重新計算預算的檢查，並回報 MILP 最佳解。梯度矩由既有校準表和真實權重量化誤差還原，通過 int4／int8 一致性與 Jensen 不等式檢查；未重新使用 test label 計算梯度。

重跑時使用 `run_sensitivity_balance.py --variants mean_abs_squared_w,mean_abs_squared_w_matched`，其餘模型、source-run 和 out-dir 參數同前。

## 驗證與限制

重新執行的 9 個 Fisher 參考配置與舊實驗的格式和指標完全一致。33 個配置皆回報 MILP 最佳解，並依模型形狀重新計算 size／BOPs，全部符合預算。MobileNetV2 非有限輸出數全為 0；GPT-2 執行亦通過有限值檢查。

配對 bootstrap 使用固定 seed 20261003、5000 次重抽樣：GPT-2 以整個視窗為單位，MobileNetV2 以影像為單位。完整 ΔNLL、ΔTop-1、Δloss 區間與分數比例見 [analysis.json](analysis.json)。同一文本鄰近視窗仍可能相關，且評測集已反覆用於方法探索；這些區間不涵蓋校準集變化或方法選擇偏差。

目前全部為 ORT QDQ 模擬；未新增 ONNX-MLIR 原生推論驗證。沒有重新訓練或使用評測 label 計算 criterion。

## 重跑方式

從 repository 根目錄執行；輸出目錄必須尚未存在。兩個原 runner 的預設一階比較模式保持不變，新增 --variants 可選固定平方誤差對照。

```bash
venv/bin/python3 experiments/mixed_precision/run_gpt2_first_order.py --variants 2nd_tensor,abs_g_sq_error --out-dir experiments/mixed_precision/runs/gpt2_sensitivity_exponent_NEW
venv/bin/python3 experiments/mixed_precision/run_mbv2_first_order.py --variants 2nd_tensor,abs_g_sq_error --out-dir experiments/mixed_precision/runs/mbv2_sensitivity_exponent_NEW
venv/bin/python3 experiments/mixed_precision/run_sensitivity_balance.py --model gpt2 --source-run experiments/mixed_precision/runs/gpt2_sensitivity_exponent_NEW --out-dir experiments/mixed_precision/runs/gpt2_sensitivity_balance_NEW
venv/bin/python3 experiments/mixed_precision/run_sensitivity_balance.py --model mbv2 --source-run experiments/mixed_precision/runs/mbv2_sensitivity_exponent_NEW --out-dir experiments/mixed_precision/runs/mbv2_sensitivity_balance_NEW
```

單項替換使用相同 balance runner，加 --variants abs_w_matched,abs_x_matched,abs_y_matched，並設定 --budgets 0.8（GPT-2）或 --budgets 0.6（MobileNetV2）。
