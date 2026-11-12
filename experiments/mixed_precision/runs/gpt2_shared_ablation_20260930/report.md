# GPT-2 small：混合精度 criterion 初步測試

**評測後端：ONNX Runtime QDQ 模擬；尚未宣稱 ONNX-MLIR 編譯後驗證。**

| 設定 | 內容 |
|---|---|
| 模型 | openai-community/gpt2，預訓練 small 模型 |
| 資料 | WikiText-2 raw，validation 校準／test 評測 |
| 校準／評測 | 10／32 個獨立視窗 |
| 視窗長度／預測 token 數 | 128／4064 |
| 格式 | int4–8、f16、bf16、fp32 |
| 搜尋 | MILP；只包含 48 個 transformer 投影節點 |
| 固定 FP32 | Embedding、共享權重的 LM head、LayerNorm、attention 動態 MatMul 與其他節點 |
| f 的基準 | 48 個投影層全 W8A8 的理論 weight bytes 與 BOPs；不是全模型成本 |
| Sensitivity | 相同預訓練權重，以 causal next-token CE 計算 mean(abs(W*dL/dW))，再平均校準視窗 |

shared 使用 raw weight/output L2；W＋O 使用相對平方 weight/output 誤差；W＋I＋O 再加入相對平方 input 誤差。三者均使用相同 sensitivity。實際量化在各選定節點同時作用於 weight、input 和 output。

| 方法 | f | NLL ↓ | Perplexity ↓ | Next-token top-1 ↑ |
|---|---:|---:|---:|---:|
| FP32 | — | 4.1383 | 62.69 | 30.12% |
| W8A8 (projections) | — | 4.6708 | 106.79 | 25.30% |
| shared | 0.6 | 8.2021 | 3648.64 | 1.77% |
| relative_wo | 0.6 | 8.4403 | 4630.12 | 1.06% |
| shared_v2 | 0.6 | 8.6917 | 5953.47 | 1.40% |
| shared | 0.65 | 8.0472 | 3124.94 | 3.08% |
| relative_wo | 0.65 | 7.9791 | 2919.20 | 5.02% |
| shared_v2 | 0.65 | 7.9065 | 2714.88 | 5.29% |
| shared | 0.7 | 8.1511 | 3467.08 | 3.89% |
| relative_wo | 0.7 | 7.5338 | 1870.16 | 6.64% |
| shared_v2 | 0.7 | 7.3209 | 1511.53 | 7.38% |
| shared | 0.75 | 8.4260 | 4564.17 | 2.29% |
| relative_wo | 0.75 | 6.9699 | 1064.10 | 8.81% |
| shared_v2 | 0.75 | 6.8140 | 910.53 | 10.29% |
| shared | 0.8 | 5.7958 | 328.93 | 14.42% |
| relative_wo | 0.8 | 6.3811 | 590.57 | 11.76% |
| shared_v2 | 0.8 | 5.9238 | 373.82 | 15.03% |

本結果使用固定短視窗的 test 子集，視窗之間不傳遞 KV cache；不能與完整 WikiText-2 或長滑動視窗的公開 PPL 直接比較。這是同設定下的 criterion 對照。

[原始結果](results.json) · [設定與資料來源](experiment.json) · [配置](configs.json)

ONNX-MLIR 相容性檢查：mixed_pass_passed。已驗證匯入與 shared_v2、f=0.80 的量化 pass；這項檢查不代表已完成原生執行。
[編譯相容性檢查紀錄](compiler_smoke/status.json)

本輪各預算最低 PPL：f=0.60：shared；f=0.65：shared_v2；f=0.70：shared_v2；f=0.75：shared_v2；f=0.80：shared。這是小型子集結果，不能據此宣稱某個 criterion 普遍較好。
即使投影層全部 W8A8，PPL 也高於 FP32。後續若要找出退化原因，應固定配置拆分 weight-only、activation-only 與 W+A，並增加校準文字；目前結果本身無法判定主要成因。
這裡的 shared 是目前 ONNX 流程的 raw weight/output L2 criterion；不是官方 HTQ 以權重誤差代理 activation 的完整重現。

重跑方式（輸出目錄須尚未存在）：
```bash
venv/bin/python3 experiments/mixed_precision/run_gpt2_criteria.py --out-dir experiments/mixed_precision/runs/gpt2_new_run
```
