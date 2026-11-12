# Qwen3-0.6B：L27 Down activation f16 → p8e1 的 native 獨立確認

日期：2026-10-24。腳本：`run_qwen_l27_native_confirm.py`（sha256 `1fd15c0967fe0ec6…`）。計畫：`plan.json`／`plan.md`（02:11:20 寫入）。

## 性質

這是固定兩個完整配置的獨立確認實驗：不改 criterion、不重新搜尋、不重選 knee、不 fine-tune、不換 calibration，也不改 pipeline。

- **主要證據**：onnx-mlir 編譯模型（native）在 128 篇新文章上的結果。
- **次要診斷**：同一批窗口上的 GPU PyTorch QDQ。
- **兩個配置**：A 是原 BC knee；C 與 A 只差 L27 Down 的 a（f16 → p8e1，input 與 output 都量化）。

## 先回答

**1. L27 的改善是否在 native 新文章上成立？**

成立。128 篇新 WT103 文章、每篇 1 個窗口：

| 量 | 值 |
|---|---|
| native NLL(C) − NLL(A) | −0.00868 nat/token |
| 95% CI（文章 paired bootstrap 20000 次） | [−0.01179, −0.00555] |
| 變好／變差文章數 | 85／43 |

CI 上界 < 0，依預先規則判讀為「支持 native 品質改善」。這是單一已選定案例的獨立確認，不代表其他層或其他模型也成立。

**2. 是否在 δ 內？省下多少資源？**

- 是。改善本身就在 δ（0.001）之內，而且 CI 上界低於 0。
- 資源以原成本函式重算，並與前輪逐值核對：
  - weight bytes 不變（size 0.759821）。
  - 搜尋範圍（196 個投影）的總 BOPs proxy 下降 0.814%；L27 Down 這個投影本身的 BOPs 減半。
  - 正規化 BOPs 與 R 由 0.877976 降到 0.870833（−0.00714），R 由 BOPs 主導。
- BOPs 是 proxy，不是量到的 kernel 加速。

**3. 同批資料上，兩後端的效應是否一致？**

一致。同一批 128 個窗口上：

| 量 | 值 |
|---|---|
| PyTorch QDQ 的 C − A | −0.00834，95% CI [−0.01146, −0.00523] |
| 逐文章 b = d_native − d_pytorch | −0.00035，95% CI [−0.00242, +0.00172] |
| 兩後端效應的相關 | 0.78 |
| 逐文章符號一致 | 102／128 |

- 前輪在 10 個 calibration 窗口上看到的後端差 +0.0027 ± 0.0015，這次在新文章上沒有重現：本輪 b 的 CI 上界是 +0.0017。
- 這只說明差異沒有重現，不據此判定任何一個後端有錯。

**4. 原 BC 是否錯失一個 native 上較好的配置？**

- 就這個案例而言，是。原 BC 給 L27 Down p8e1 的成本比 f16 高約 160，所以搜尋選了 f16。
- 但在 native、獨立新文章上，C 的品質較好、BOPs proxy 也較低。也就是說，原搜尋錯失了一個在這兩個指標上都不差於 A 的配置。
- 這不等於找到 BC 誤排的完整原因，也不代表 knee 整體應該改變。

**5. 證據不足之處**

- 只有單一模型、單一層，而且這個位置是看過前幾輪診斷後才挑出來的。新資料與 native 排除了「同資料選擇又評估」的偏差，但不能推廣到其他層或模型。
- 每篇只用 1 個 128-token 窗口、共 128 篇，都來自 Wikipedia（WT103）。沒有測其他文本或下游任務。
- 依計畫，不與前輪 512 × 3 窗口合併做主要推論。
- 沒有量測實際延遲；BOPs 只是 proxy。
- 改善的機制本輪沒有檢驗。
- calibration 能否自動識別這類位置、在 knee 重新搜尋時是否仍成立，本輪都沒有回答。

## 1. 配置與編譯

### 配置

- A、C 直接取自 `runs/qwen3_down_fp16_protection_20261024/configs/`，檔案 sha256 與前輪相同。
  - A：`ac10afa9…`，config hash `36205a06…`
  - C：`c1c8a083…`
- 核對結果：
  - 196 個投影中，唯一差異是 `/model/model/layers.27/mlp/down_proj/MatMul` 的 a（f16 → p8e1），w 都是 int8。
  - 所有 w 相同；L2 Down 在兩者都是 int8／f16。

### 編譯（`builds.json`、`native/A|C/compile_env.json`）

- 參數：Krnl O3、clang O0、constants to file、MLIR bytecode、`MIXEDP_WEIGHT_PER_CHANNEL=1`。
- 基底與 range：`runs/native_eps_fix_20261014/qwen3/model.mlirbc` 與其 calibration ranges。
- `MIXEDP_NODE_FORMATS` 由配置重新產生：
  - 與前輪的 compile_env.json 逐字相同（A、C 都是）；
  - 196 個條目的 w／a 都與配置一致；
  - 條目內容：L27 在 A 是 `w=int8:a=f16`，在 C 是 `w=int8:a=p8e1`；L2 在兩者都是 `w=int8:a=f16`。
- 建置時間：A 40.4 s，C 40.7 s，posit runtime 16.9 s。
- 產物 hash：

| 產物 | sha256 |
|---|---|
| A.so | `60b93866ce24b4c0…` |
| C.so | `175c73cc7b1d9e47…` |
| model.constants.bin（A、C 相同，權重本來就相同） | `76a491c14bd1b9d8…` |

- 工具鏈 hash 記錄在 `builds.json["tools_sha256"]`：onnx-mlir、onnx-mlir-opt、mlir-translate、clang++、libcruntime.a、posit runtime 原始碼、PyRuntime。
- .so、constants、compile_env 都保留在 `native/A|C/`，共 5.7 GB。

## 2. 控制（`controls.json`；全部通過，沒有放寬容差）

| 控制 | 結果 |
|---|---|
| config diff | 只有 L27 Down a 一項 |
| 196 個投影的編譯設定 | 與配置一致，且與前輪 compile_env 逐字相同 |
| calibration 窗口 0–1，native A 的 summed CE vs 前輪 native/A.json | 最大絕對差 0.0（容差 1e-8） |
| calibration 窗口 0–1，native C vs 前輪 native/C.json | 最大絕對差 0.0 |
| A 的窗口 0 在另一個新 process 重跑 | 差 0.0 |

控制窗口只用來檢查，沒有計入正式評估。

## 3. 資料（`segments.json`、`exclusions.json`、`data_freeze.json`、`audit.json`）

**抽樣**
- 128 篇 WikiText-103 train 文章，每篇 1 個 128-token 窗口，起點隨機。使用 Qwen3 tokenizer，A／C 用完全相同的窗口。
- seed：順序 29101、起點 29102。
- verified 掃描排除 8361 篇歷史文章，其中包含 spike 診斷的 128 篇與前輪的 512 篇（都已核對在排除集合中）。
- 合格池 19034 篇，訪問 150 篇，跳過 22 篇：

| 跳過原因 | 篇數 | 說明 |
|---|---|---|
| 與歷史 WT103 文章近重複 | 10 | |
| 與歷史 Qwen 窗口重疊 | 8 | 例如 Trey Burke，窗口 share 0.76 |
| 與 WT2 近重複 | 4 | |

**歷史 Qwen token audit**（凍結後重跑）
- 52 個檔、4014 個窗口，無法解析的來源 0 個：
  - 44 個 WikiText-2；
  - 6 個合成 id；
  - 1 個是 spike 診斷的 tokens，全部落在該輪 manifest 所列的 WT103 文章；
  - 1 個是前輪 fp16_protection 的 tokens，1535 個窗口落在其 manifest 所列的 WT103 文章，另 1 個窗口判為 WikiText-2。
- 前輪那個窗口的註記（事後描述）：
  - 位置：第 1459 個窗口，文章「Never Too Far」。
  - 重疊：該窗口 62% 的 5-gram 出現在 WT2 文字中，但文章層級的 WT2 share 只有 6.6%，低於 10% 的排除門檻。
  - 所以那篇文章依前輪規則屬於合格資料。這是窗口層級的部分重疊，前輪規則只在文章層級檢查。
- 本輪 128 個窗口的同類描述性檢查（`window_wt2_overlap.json`，計畫後才做，不是排除規則）：最高 11%，中位數 1%，沒有 ≥ 50% 的窗口。

## 4. 主要結果：native（`stats.json`、`native_ce.npz`、`native_results/fresh/*/complete.json`）

| 後端 | NLL(A) | NLL(C) | PPL(A) | PPL(C) |
|---|---|---|---|---|
| native | 4.003444 | 3.994760 | 54.79 | 54.31 |
| PyTorch QDQ（同窗口） | 4.002506 | 3.994168 | 54.74 | 54.28 |

| d = NLL(C) − NLL(A) | 平均 | 95% CI | 中位數 | SD | 變好／變差 |
|---|---|---|---|---|---|
| native（主要） | −0.00868 | [−0.01179, −0.00555] | −0.00665 | 0.0181 | 85／43 |
| PyTorch（次要） | −0.00834 | [−0.01146, −0.00523] | −0.00693 | 0.0180 | 91／37 |

**判讀**：native 品質改善獲得支持（CI 上界 < 0）。這只是一個主要比較，沒有做多重比較校正。

**逐文章分布（native）**
- 分位數 5／25／50／75／95%：−0.0377／−0.0214／−0.0066／+0.0036／+0.0187。
- |d| > 0.01 的文章有 72 篇，|d| > 0.05 的有 0 篇。沒有單篇主導結果。

**極端案例**

| 方向 | 文章 | d_native | d_pytorch |
|---|---|---|---|
| 改善最多 | Ozzie Smith | −0.0473 | −0.0452 |
| 改善最多 | Churches of Christ | −0.0420 | −0.0406 |
| 改善最多 | Ivan the Russian | −0.0400 | −0.0307 |
| 改善最多 | History of the Great Wall of China | −0.0398 | −0.0597 |
| 改善最多 | Titchwell Marsh | −0.0396 | −0.0359 |
| 變差最多 | 2008 Turkish Grand Prix | +0.0460 | +0.0171 |
| 變差最多 | 2000–01 Arsenal F.C. season | +0.0330 | +0.0203 |
| 變差最多 | HMS Royal Oak (08) | +0.0301 | +0.0250 |
| 變差最多 | Hawker Siddeley P.1154 | +0.0262 | +0.0198 |
| 變差最多 | Charles R. Forbes | +0.0223 | +0.0199 |

變差最多的案例裡，兩後端的方向一致，但 native 的幅度普遍較大。這些極端案例是依 native 排序挑出來的，本來就會偏向 native 較大，不能據此推論後端差異。

## 5. 次要：同資料的後端差異

**效應差 b = d_native − d_pytorch**
- 平均 −0.00035，95% CI [−0.00242, +0.00172]，SD 0.0120。
- 兩後端效應的相關 0.78；符號一致 102／128。

**絕對 NLL 差（native − PyTorch）**
- A：+0.0009，CI [−0.0036, +0.0054]，逐文章 SD 0.026。
- C：+0.0006，CI [−0.0040, +0.0051]。

**解讀**
- 兩後端的單篇絕對 NLL 差異不小（SD 約 0.026），但 C−A 這個 paired 效應在兩後端之間的平均差很小，CI 包含 0。
- 本輪不據此判定任何一個後端有錯，也不用它改變主要的 native 判讀。

## 6. 資源（`resources.json`；重算後與前輪逐值相同）

| 配置 | weight bytes | size | BOPs | BOPs／int8 | R | 原 BC 總分 |
|---|---|---|---|---|---|---|
| A | 334,626,816 | 0.759821 | 3.1675e12 | 0.877976 | 0.877976 | 330.46 |
| C | 334,626,816 | 0.759821 | 3.1418e12 | 0.870833 | 0.870833 | 490.43 |

- C − A：BOPs −2.577e10，佔搜尋範圍總 BOPs −0.814%；R −0.00714；原 BC 總分 +159.97。
- 搜尋範圍是 196 個 transformer 投影；embedding、LM head、norm 不在其中。
- BOPs 是 proxy，不代表實際 kernel 加速。

## 7. 耗時（`timing.json`；診斷實驗耗時，不是搜尋 latency）

| 項目 | 時間 |
|---|---|
| plan（含資源重算 2.9 s、輸入 hash、語料掃描） | 30 s |
| native 編譯（posit runtime 17 s；A 40.4 s；C 40.7 s） | 100 s |
| 舊資料控制（A／C 各 2 窗口、A 重跑 1 窗口） | 62 s |
| audit | 37 s |
| 抽樣 | 102 s |
| native 評估：A（4 個 worker 並行，每 worker 1 thread） | 1014 s |
| native 評估：C | 1148 s |
| PyTorch 評估（GPU；初始化 6.5 s，A 17 s，C 8 s） | 33 s |
| 分析 | 0.2 s |

- native 每個 worker 處理 32 個窗口，峰值 RSS：A 3.11 GB、C 3.36 GB。
- 並行執行的總時間不是單次推論 latency；每窗口約 31–36 s 也是在 4 個 process 並行下量到的。

## 8. 決策

- 記為「已選定案例（L27 Down）的 native 獨立確認」：在 128 篇新文章上，C 的 native 品質較好，BOPs proxy 也較低。
- **不修改正式配置、criterion 或 pipeline**，也不把 L27 硬編碼進 pipeline。
- 後續若要研究，應另行登記，問題依序是：
  1. calibration 統計能否自動識別這類位置；
  2. 是否跨層、跨模型成立；
  3. 在 knee 重新搜尋時是否仍有利。

## 9. 修改紀錄、冒煙測試與重現

**草稿審查後的修改（都在計畫寫入前）**
- 續跑：每個 part 檔加上含 tokens／.so／constants／程式／配置 hash 與窗口範圍的標頭，續跑時必須相符；截斷的最後一行丟棄後重算；加上「同一 checkpoint 只允許一個寫入者」的鎖。
- 改用 `run_native_bc_pipeline.cross_entropy`，與既有計算一致。
- 由配置重新產生 compile env，並與前輪逐字核對。
- 用原成本函式重算資源並與前輪核對。
- 記錄 worker 的 thread 設定與 VmHWM。
- 統計補上「小幅變差但在容許內」的分支、後端相關與符號一致率、極端案例。
- 無法解析的歷史來源會讓 audit 停止。
- plan.md 補齊所有計畫內容。

**冒煙測試（計畫前，在 scratchpad，不在 runs/）**
- 建了 A／C，跑過控制、續跑與鎖定測試。
- native／PyTorch 只在 calibration 窗口 0–9 上評估，這些窗口前輪都已量過。
- 以其他 seed（999201／999202）測試 audit／data，抽了 4 篇，沒有評估。
- 沒有評估任何新文章。

**沒有 amendment**：計畫寫入後腳本未變更，每個階段都檢查了程式、配置與輸入 hash。

```
O=runs/qwen3_l27_native_confirm_20261024
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$CPU run_qwen_l27_native_confirm.py --stage plan         --out-dir $O   # 只能寫一次；新實驗請換新目錄
$CPU run_qwen_l27_native_confirm.py --stage build        --out-dir $O
$CPU run_qwen_l27_native_confirm.py --stage controls     --out-dir $O   # 控制通過才寫 freeze.json
$CPU run_qwen_l27_native_confirm.py --stage audit        --out-dir $O
$CPU run_qwen_l27_native_confirm.py --stage data         --out-dir $O
$GPU run_qwen_l27_native_confirm.py --stage pytorch_eval --out-dir $O
$CPU run_qwen_l27_native_confirm.py --stage native_eval  --out-dir $O   # 可續跑：已完成的窗口依標頭核對後沿用
$CPU run_qwen_l27_native_confirm.py --stage analyze      --out-dir $O
```

**hash（sha256；腳本、計畫為完整碼，其餘為前 16 碼）**

| 檔案 | sha256 |
|---|---|
| 腳本 | `1fd15c0967fe0ec699860402423362b4c1a479404df48dae9e4b7e80761bb3ab` |
| plan.json | `3dc834f820cccb1250e06a38424ce3d636f6e7c0e885ae78b050e1ef33e6ecaf` |
| plan.md | `f6464ae265ba28f67c17100578b5f212001cb9731ef533d776ead0283c41cbf4` |
| configs/A.json | `ac10afa96c7833ca` |
| configs/C.json | `c1c8a0834104a1fb` |
| freeze.json | `9de72ab2147f8ea6` |
| builds.json | `e0e91a9842ddd717` |
| controls.json | `3b6a370829284db6` |
| native/A/compile_env.json | `6b6b6890f58d8794` |
| native/C/compile_env.json | `fe34ed9648bd8add` |
| resources.json | `b06cf97f80a56b5f` |
| audit.json | `415561dfed577e42` |
| segments.json | `7bc1b0febb06bd1e` |
| exclusions.json | `a2c41f1a31d197fa` |
| data_freeze.json | `8eaa333fd7420539` |
| tokens.npy | `ea2dc312b1fc82dd` |
| native_ce.npz | `ef96a094635f1b0c` |
| native_results/fresh/A/complete.json | `7d36119a12d99009` |
| native_results/fresh/C/complete.json | `57ab1c0f5041ee4b` |
| pytorch_ce.npz | `82167f3076428568` |
| per_article_losses.csv | `22a1c659be69f797` |
| stats.json | `56e09ea66d759ba7` |
| timing.json | `99c5d2001edabcc5` |

模型、範圍、工具鏈與歷史資料的 hash 記錄在 `plan.json["inputs"]` 與 `builds.json["tools_sha256"]`。
