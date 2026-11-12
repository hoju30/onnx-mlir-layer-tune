# Qwen3-0.6B：L2／L27 Down activation 的 FP16 保護是否必要

日期：2026-10-24。腳本：`run_qwen_down_fp16_protection.py`（sha256 `3c62a959bcb68430…`）。計畫：`plan.json`／`plan.md`（00:55:37 寫入）。

## 性質

這是固定配置的局部介入：不改 criterion、不重新搜尋、不重選 knee、不 fine-tune、不換 calibration。

- **主要結果**：GPU PyTorch QDQ（`run_lp_pipeline_eval.LMReplay`）在 512 篇新 WikiText-103 文章上的評估。
- **native**：onnx-mlir 只在原 10 個 calibration 窗口上交叉驗證。
- **結論範圍**：依預先登記的後端規則，所有結論都限定為 PyTorch QDQ 診斷（理由見第 2 節）。

## 先回答

**1. 原 FP16 保護是否有必要？**

| 位置 | 判斷 | 依據 |
|---|---|---|
| L2 Down | 必要 | 改成 p8e1（B）每 token 增加 0.305 nat，校正 CI [0.298, 0.312]，是容許界限 δ（0.001）的約 300 倍；512 篇文章全部變差 |
| L27 Down | 在 PyTorch QDQ 下不必要 | 改成 p8e1（C）反而降低 0.0085 nat/token，校正 CI [−0.0096, −0.0075]；512 篇中 403 篇變好 |

- L2 的結果與 native 在 calibration 窗口上的方向、大小一致：native +0.303，PyTorch +0.306。
- L27 的結果在 native 上沒有得到同等支持：calibration 窗口上 native −0.0018 ± 0.0045，PyTorch −0.0045；兩後端的 paired 差 0.0027 ± 0.0015，已達 δ 的量級。

**2. 哪些替換在 δ 內、哪些超過、哪些未定？**

| 配置 | 判讀 |
|---|---|
| B（只改 L2） | 超過容許（校正下界 > δ） |
| C（只改 L27） | 支持品質改善（校正上界 < 0），因此也在 δ 內 |
| D（兩者都改） | 超過容許 |

沒有未定的配置。

**3. 實際省下的全模型 BOPs proxy 有多大？**

- 每個被修改的投影：BOPs 減半，各減少 2.577e10。
- 全模型（整個搜尋範圍 196 個投影）：
  - B 或 C：A 的總 BOPs 減少 0.814%；正規化 BOPs 0.877976 → 0.870833。
  - D：減少 1.627%，正規化 BOPs → 0.863690。
- weight bytes 完全相同（size 0.759821）。A 的 R 由 BOPs 主導，所以 R 與 BOPs 降幅相同：
  - B 或 C：R 0.877976 → 0.870833（−0.0071）；
  - D：R → 0.863690（−0.0143）。
- BOPs 只是 proxy，不代表 GPU QDQ 或低位元 kernel 會等比例變快。model size 沒有下降。

**4. 兩層一起替換與單獨替換是否不同？**

- 是。逐文章 I = NLL(D) − NLL(B) − NLL(C) + NLL(A) = +0.0136，95% CI [+0.0122, +0.0150]。
- 也就是說，聯合替換的額外損傷高於兩個單獨替換之和：C 單獨時的改善（−0.0085），在與 B 一起時不但沒有出現，還多了約 0.005。
- 這是完整配置中兩個局部替換的共同作用，本輪不歸因於任何特定機制。

**5. 結論是 PyTorch QDQ，還是也得到 native 支持？**

- 只有 PyTorch QDQ 診斷。
- native 只跑了 10 個 calibration 窗口。B−A、C−A、D−A 三個比較的「native − PyTorch」paired 差都沒有低於 δ 的規則門檻，所以依規則，全部結論限定為 PyTorch QDQ。
- 描述性補充：B、D 的 native 方向與大小一致（10/10 窗口同號）；C 的 native 改善較小，也不顯著。
- 不能把任何結果稱為 native pipeline 已確認的收益。

## 1. 配置與控制

### 配置

- **A**：即 `runs/lp_pipeline_regression_20261016/C/qwen3/configs/ref.json`。
  - 已核對：與該輪 freeze 的檔案 sha256 相同、與 MILP 原規則 knee（`search_milp_orig.json`）的 config hash 相同（`36205a06…`）。
  - L2、L27 Down 都是 w=int8、a=f16，與需求所述相符。
  - `configs/A.json` 與原檔位元組相同。
- **B**：只把 L2 Down 的 a 從 f16 改成 p8e1。
- **C**：只把 L27 Down 的 a 從 f16 改成 p8e1。
- **D**：兩者都改。

所有配置的 196 個 w 都相同，其餘投影的 a 都沿用 A。

### a 的語意

- native pass：`MIXEDP_NODE_FORMATS a=` 套用在 MatMul 的 activation input 與 result。
- LMReplay：forward pre-hook 用 x range 量化 input，forward hook 用 y range 量化 output。
- 兩者一致。因此本輪替換的是完整的 a（input 與 output），不是上一輪只量化 Down input 的設定。

### 控制（全部通過；`controls.json`，含 `controls_gpu.json` 與 `native_crosscheck.json`）

| 控制 | 結果 |
|---|---|
| config diff | B、C 各改 1 個 a，D 改 2 個 a，都只改 a，沒有改任何 w |
| A–D 的量化後權重 hash | 全部相同（`46c623ad…`）；FP32 的權重 hash 不同 |
| 切換配置後的 activation map | 每次都等於配置 |
| INT QDQ | scale／zp 放在 tensor 所在裝置（LMReplay.qact） |
| GPU 設定 | TF32 關閉（matmul 與 cuDNN）、deterministic、matmul precision highest、FP32 |
| p8e1 LUT vs Universal | 1,501,279 點，0 不一致 |
| L2、L27 Down 的 input／output 實際格式（FP32、A、B、C、D） | 全部符合配置：f16 位置不是 posit-only；p8e1 位置是 LUT 不動點 |
| hook 數 | 196 個投影各 1 個 pre-hook 與 1 個 forward hook，前後不變；診斷 hook 已移除 |
| FP32 identity vs 無 hook 模型（10 個 calibration 窗口） | 逐位元相同 |
| A 重複 4 次、其他配置各 2 次，兩種交錯順序 | 逐位元相同 |
| 模型設定 | RMSNorm epsilon 1e-6；calibration tokens = prepare 的 validation 窗口；ranges = native_eps_fix |
| tokenizer | 重現 prepare 的 validation／test 窗口 |
| native A 重跑 | summed CE 差 0 |
| 資源模型重現 A 的 knee | size、BOPs、R、BC 總分都完全相同（0.759821／0.877976／0.877976／330.4633） |

## 2. native 交叉驗證（10 個 calibration 窗口；`native_crosscheck.json`、`native/`）

編譯方式沿用 `run_lp_pipeline_eval.stage_c_native`：Krnl -O3、clang -O0、constants to file、per-channel 權重。已核對編譯環境中 196 個條目的 w／a 都與配置相同。

### 絕對 NLL

| 配置 | native NLL | PyTorch NLL | 每窗口最大差 |
|---|---|---|---|
| FP32 | 3.499090 | 3.499089 | 2.7e-6 |
| A | 3.560071 | 3.554400 | 0.042 |
| B | 3.863064 | 3.860415 | 0.058 |
| C | 3.558227 | 3.549870 | 0.043 |
| D | 3.867337 | 3.869888 | 0.057 |

### paired 效應（相對 A）

| 比較 | native | PyTorch | native − PyTorch（平均 ± SE） | 同號窗口 | 低於 δ 規則 |
|---|---|---|---|---|---|
| B−A | +0.3030 ± 0.0233 | +0.3060 ± 0.0235 | −0.0030 ± 0.0108 | 10/10 | 否 |
| C−A | −0.0018 ± 0.0045 | −0.0045 ± 0.0045 | +0.0027 ± 0.0015 | 10/10 | 否 |
| D−A | +0.3073 ± 0.0274 | +0.3155 ± 0.0264 | −0.0082 ± 0.0112 | 10/10 | 否 |

- **規則判讀**：預先登記的條件是 |平均| + 2 SE < δ，三個比較都不成立。所以後端差異可能改變 δ 等級的判讀，結論限定為 PyTorch QDQ 診斷。
- **描述**：
  - B、D 的損傷比後端差大兩個數量級。
  - C 的後端差（0.0027）與 δ 同量級，也與 C 在新文章上的改善（0.0085）同量級。
- 計畫前的冒煙測試已在相同的舊窗口上看到 PyTorch 的這些數字（見第 9 節）。

## 3. 資源（`resources.json`；原 pipeline 成本函式）

成本函式：`run_solver_regression.dense_problem('qwen3')` 加上 `mp_auto` 的 record 公式。

| 配置 | weight bytes | size | BOPs | BOPs／int8 | R | 主導 | 原 BC 總分 | L2／L27 Down 的 C(a) |
|---|---|---|---|---|---|---|---|---|
| A | 334,626,816 | 0.759821 | 3.1675e12 | 0.877976 | 0.877976 | BOPs | 330.46 | 0.172／0.000326 |
| B | 334,626,816 | 0.759821 | 3.1418e12 | 0.870833 | 0.870833 | BOPs | 873,346.94 | 873,016.6／0.000326 |
| C | 334,626,816 | 0.759821 | 3.1418e12 | 0.870833 | 0.870833 | BOPs | 490.43 | 0.172／159.97 |
| D | 334,626,816 | 0.759821 | 3.1160e12 | 0.863690 | 0.863690 | BOPs | 873,506.90 | 873,016.6／159.97 |

- 每個被修改的投影：BOPs 5.154e10 → 2.577e10（減半）。w 都是 int8，W 項（B_w）不變：L2 6.67、L27 0.915。
- 全模型（搜尋範圍 196 投影）：
  - B、C 各減少 0.814% 的 A 總 BOPs；正規化 BOPs 與 R 各 −0.00714。
  - D 減少 1.627%；R −0.01429。
- 實際驗證 weight bytes 完全相同。
- 原 BC 認為兩處都應保護：L2 改 p8e1 的預測成本增加約 87 萬，L27 增加約 160。

## 4. 資料（`segments.json`、`exclusions.json`、`data_freeze.json`、`audit.json`）

- 512 篇 WikiText-103 train 文章 × 3 個連續、不重疊的 128-token 窗口，共 1536 個窗口。使用 Qwen3 tokenizer，所有配置用同一批文章。
- **verified 掃描**排除 7849 篇歷史文章，其中包含 qwen3_down_spike_diag 的 128 篇（已逐一核對都在排除集合中）。
- **歷史 Qwen token audit**（凍結後重跑）：
  - 50 個檔：43 個 WikiText-2；6 個合成 id；1 個是 spike 診斷的 tokens，其 384 個窗口都落在該輪 manifest 所列的 WT103 文章中。
  - 無法解析：0。
  - 全部 2468 個窗口都加入重疊排除。
- **抽樣**：合格池 19546 篇，seed 順序 28101、起點 28102，訪問 609 篇。跳過的原因：
  - 與歷史 WT103 近重複：54
  - 與歷史 Qwen 窗口重疊：24
  - 與 WT2 近重複：15
  - 太短：3
  - 與已接受文章近重複：1
- 接受文章的最大 shingle share 為 0.096，低於 10% 門檻。
- 抽樣在凍結之後，看到任何新 loss 之前；沒有增減文章。

## 5. 主要結果（PyTorch QDQ，512 篇，`stats.json`）

| 配置 | NLL | PPL | 相對 FP32 的 ΔNLL [95% CI] |
|---|---|---|---|
| FP32 | 3.968967 | 52.93 | — |
| A | 4.032680 | 56.41 | +0.0637 [+0.0614, +0.0660] |
| B | 4.337761 | 76.54 | +0.3688 [+0.3624, +0.3751] |
| C | 4.024173 | 55.93 | +0.0552 [+0.0529, +0.0575] |
| D | 4.342867 | 76.93 | +0.3739 [+0.3675, +0.3802] |

| 比較 | 平均 | 95% CI | Bonferroni 98.333% CI | 變好／變差篇數 | 判讀 |
|---|---|---|---|---|---|
| d_B = B − A | +0.30508 | [+0.29921, +0.31085] | [+0.29795, +0.31213] | 0／512 | 新增損傷超過容許（校正下界 > δ） |
| d_C = C − A | −0.00851 | [−0.00936, −0.00765] | [−0.00957, −0.00747] | 403／109 | 支持品質改善（校正上界 < 0） |
| d_D = D − A | +0.31019 | [+0.30442, +0.31585] | [+0.30316, +0.31713] | 0／512 | 新增損傷超過容許 |

- bootstrap：文章層級 paired，20000 次，seed 28103，所有配置使用同一組重抽。
- 文章間 SD：d_B 0.067、d_C 0.010、d_D 0.066。
- C 的改善約等於 A 總量化損傷（0.0637）的 13%。

## 6. 兩層共同作用（次要）

- I = NLL(D) − NLL(B) − NLL(C) + NLL(A) = +0.01361，95% CI [+0.01220, +0.01500]；413 篇為正。
- 判讀：I > 0 已確認，聯合的額外損傷高於兩個單獨損傷之和。
- 這只描述完整配置中兩個局部替換的共同作用，本輪不歸因於 SwiGLU Gate/Up 交互作用、position 0 或任何二階近似問題。

## 7. 事後機制解釋（只作說明，不據此改設定）

**L2：範圍超出 Posit8 的上限**
- L2 Down 的 output（y）calibration range 上限是 6861，超過 p8e1（es=1）的 maxpos 4096；在 2^11–2^12 之間，p8e1 也幾乎沒有 fraction bit。
- 原 BC 的 L2 p8e1 activation 成本約 76% 來自 y 槽位（x 佔 24%；取自上一輪的 x+y 重建）。
- 上一輪只量化 Down input 時，L2 p8e1 的單層 ΔNLL 只有 +0.0001。本輪把 a 完整換掉（含 output）後，損傷達 +0.305。這說明只看 input 的單點結果，不足以推翻原配置的 FP16 保護。

**L27：BC 的方向與實測相反**
- L27 Down 的 y range 下限是 −5034，也超過 maxpos。但 C 在 PyTorch QDQ 下反而降低 loss。
- 原 BC 預測 p8e1 比 f16 差（+160），實測方向相反（PyTorch）。上一輪 input-only 的 L27 p8e1 ΔNLL 也是負值（−0.0025）。
- 本輪沒有檢驗原因，也不把它解讀為已找到 BC 誤排的完整原因。

## 8. 耗時（`timing.json`；不是 pipeline 搜尋 latency）

| 項目 | 時間 |
|---|---|
| 配置準備與資源計算 | 3.0 s（plan 階段共 29 s，多數是輸入 hash 與語料掃描） |
| LMReplay 初始化（含 p4e1／p6e1／p8e1 LUT 建立與驗證） | 6.5 s |
| 量化權重（4 配置 + FP32） | 19.6 s |
| GPU 控制（含重複與交錯順序 12 s） | 共 49 s |
| native：posit runtime | 17 s |
| native：編譯 | FP32 21 s，A–D 各 38–39 s |
| native：10 窗口執行 | FP32 102 s、A 172 s（兩次）、B／C／D 81–87 s |
| native 階段總計 | 730 s |
| audit | 33 s |
| 抽文章 | 232 s |
| 正式評估（GPU，1536 窗口） | FP32 44 s，A–D 各 96–103 s，共 447 s；GPU 峰值 4.3 GB |
| 分析 | 6 s |

## 9. 決策

**C（L27 改 p8e1）**
- 在容許內，品質還有改善，並降低資源：R −0.0071、全模型 BOPs −0.81%。
- 依規則只能說：L27 這個已選定位置，在 PyTorch QDQ 診斷下存在降低保護的機會。
- native 對此沒有同等支持。下一步若要研究，應是「能否由 calibration 自動識別這類位置」，另行登記。
- 不把 L27 硬編碼進正式 pipeline。

**B、D（含 L2 改 p8e1）**
- 損傷遠超容許，L2 保留 FP16。
- 只量化 input 的單點 INT／Posit 結果，不足以推翻原配置。

**整體**
- 不改正式預設、不啟動新搜尋。
- 不把 C 的結果稱為 native pipeline 收益，也不稱為已找到 BC 誤排的完整原因。

## 10. 冒煙測試、重現與 hash

**冒煙測試（計畫前，在 scratchpad，不在 runs/，只用舊資料）**
- GPU 控制在 10 個 calibration 窗口上跑過，看到了 FP32／A／B／C／D 的 PyTorch NLL：B、D 約比 A 高 0.3，C 約低 0.005。
- native 只建了 FP32／A，跑 2 個窗口。
- 資料階段用其他 seed（999101／999102）抽了 6 篇，沒有評估。
- eval／analyze 只用 9 個 calibration 窗口。
- 修正了 1 個程式錯誤（缺少比較時的 native scope 文字）。
- 沒有評估任何新文章；配置、δ、資料量與規則都來自需求，未因此更改。這件事已記錄在 `plan.json["pre_plan_smoke_test"]`。

**修改紀錄**：計畫寫入後腳本未再變更（每個階段都檢查程式與輸入 hash），沒有 amendment。

**環境**：nvidia-smi 回報 NVML 版本不符，但 CUDA 正常運作，沒有動驅動。

```
O=runs/qwen3_down_fp16_protection_20261024
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$CPU run_qwen_down_fp16_protection.py --stage plan     --out-dir $O    # 只能寫一次；新實驗請換新目錄
$GPU run_qwen_down_fp16_protection.py --stage controls --out-dir $O
$CPU run_qwen_down_fp16_protection.py --stage native   --out-dir $O
$CPU run_qwen_down_fp16_protection.py --stage freeze   --out-dir $O
$CPU run_qwen_down_fp16_protection.py --stage audit    --out-dir $O
$CPU run_qwen_down_fp16_protection.py --stage data     --out-dir $O
$GPU run_qwen_down_fp16_protection.py --stage eval     --out-dir $O
$CPU run_qwen_down_fp16_protection.py --stage analyze  --out-dir $O
```

**hash（sha256；腳本、計畫、配置 A 為完整碼，其餘為前 16 碼）**

| 檔案 | sha256 |
|---|---|
| 腳本 | `3c62a959bcb6843071cf6c395db72bb9363975c4ad09ba58b9340bf30f5b8156` |
| plan.json | `55445bb8736bf45d7cafafa7009dabf908f0644674ef1f04da47edcaf7f3a682` |
| plan.md | `d2eb25f573119aced5d5ea4f5493641ab70202f9876e94fbfc81e21f548261d9` |
| configs/A.json（= 原 ref.json） | `ac10afa96c7833cacc6f7886857a97c9ba9b4e2cc151712194373f0e9490c06b` |
| configs/B.json | `d77c7bdf2a79add3` |
| configs/C.json | `c1c8a0834104a1fb` |
| configs/D.json | `1ac9c7ed99939fb7` |
| freeze.json | `067d115051741766` |
| controls.json | `6b6c70244396c35a` |
| controls_gpu.json | `41418c13bfaeda66` |
| native_crosscheck.json | `265c1d7a31a7b2ce` |
| resources.json | `1999a8442385842e` |
| audit.json | `95d7ef7c78b47378` |
| segments.json | `80f68d4a0238802a` |
| exclusions.json | `20b38e8dc7bc831c` |
| data_freeze.json | `1553699f3b51641b` |
| tokens.npy | `71c04df2c0934a71` |
| eval_ce.npz | `9c7c49f209b87db3` |
| per_window_losses.csv | `40331e64df78a1bb` |
| per_article_losses.csv | `726203804c59c49e` |
| stats.json | `18fae3e99a79ef90` |

模型、range、原表格等輸入的 hash 記錄在 `plan.json["inputs"]`；配置的 config hash 記錄在 `plan.json["configs"]`。
