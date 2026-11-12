# OLMoE：完整更新一次 BC proxy 是否有收益

日期：2026-10-22
腳本：`run_olmoe_proxy_refresh.py`（plan → static → search_static → refresh → search_refresh → freeze → data → eval → analyze）
輸出：本目錄

**本輪的範圍**
- 只回答「在量化工作狀態下完整更新一次 BC 統計，是否比固定的 FP32 proxy 選出更好的配置」。
- 沒有做 sensitivity-only／error-only 消融；沒有修改 BC 公式；沒有 router 微調、量化粒度變更或一階項；也沒有整合為預設。
- 這不是 GEMQ 的重現：GEMQ 使用 MoE 輸出層級的 proxy，並搭配 router fine-tuning。
- 全模型是 PyTorch GPU 上的 QDQ，不代表 native kernel 的結果。

**流程**
- `plan.json`／`plan.md` 在任何新 proxy、搜尋與評估資料之前寫定，沒有修訂。
- 每個階段用的都是同一份腳本（sha `e8edbbd2…`）。
- 輸入檔的 hash 在實驗前後一致（`checks.json`）。

## 結論

**完整更新一次後的配置 q1，點估計比 static 的 q0 好約 0.002 nat/token，但 95% CI 的上界略大於 0，依預先規則判定為「尚無法確認改善」；也不能判定等價。這次更新大約讓 offline 時間增加一倍。**

| | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| FP32 dynamic | 2.93670 | 18.854 | — |
| q0（static BC） | 2.97230 | 19.537（+3.62%） | +0.03560 [+0.03207, +0.03928] |
| q1（refresh 一次） | 2.97035 | 19.499（+3.42%） | +0.03364 [+0.03002, +0.03758] |

**主要終點**（128 篇新的 WikiText-103 文章 × 4 個窗口，以文章配對）
- Δ = NLL(q1) − NLL(q0) = **−0.00196 nat/token，95% CI [−0.00415, +0.00017]**。
- 判定：上界 > 0，所以**尚無法確認改善**。
- CI 下界 −0.00415 超出 −0.002，所以也**不能判定等價**。
- 128 篇中有 54 篇的 Δ 為正（q1 較差）；文章級 SD 為 0.0125。

**四個問題**
1. **完整更新是否改善相同資源下的 dynamic 品質？** 尚未確認。
   - 點估計是改善 0.0020 nat/token，大約是 q0 損傷的 5.5%，PPL 從 +3.62% 降到 +3.42%。
   - 但 95% CI 包含很小的變差（上界 +0.0002）。
   - 本輪沒有足夠證據宣稱改善，也不能宣稱兩者等價。
2. **新增多少 offline 時間？** 約 108 秒的牆鐘時間，相當於 static 基準流程（約 104 秒）的 104%，也就是讓 offline 時間大約翻倍。
   - 計時器內的計算只有 56 秒，主要是 backward 28 秒和 activation 誤差 18 秒。
   - 其餘約 45 秒是 checkpoint 逐層載入、權重誤差快取的讀取與存檔等，沒有分項計時。
3. **變化主要出現在 weight 還是 activation 格式？** 兩者都有，activation 稍多。
   - q0 和 q1 有 292 個投影不同：weight 格式改變 125 個，activation 格式改變 182 個（部分投影兩者都變）。
   - weight 大多是 int4／int5／int6 之間的重新分配。
   - activation 大多是 p8e1↔p6e1、int7↔p8e1／p6e1 互換。
   - 改變分散在 16 層和三種投影。本輪無法判斷 loss 差來自哪一種變化。
4. **是否值得再做 sensitivity-only／error-only 的省時消融？** 目前還不值得優先做。
   - 先決條件是確認完整更新確實有收益。現在 CI 跨 0，若真實收益是 0，消融省下的時間就沒有意義。
   - 建議先預先登記一次確認實驗：以文章級 SD 0.0125 估算，CI 半寬要到 0.001 約需 600 篇；或換一個資源點，例如 knee。
   - 若確認有收益，額外成本主要在 backward（28 秒）和 activation 誤差重算（18 秒），正好分別對應 error-only 和 sensitivity-only 這兩種消融能省下的部分。

**不能由本輪宣稱**
- 已驗證其他資源點、其他模型或 native kernel。
- BC proxy 的數值精確近似真實 loss。
- 已完整重現 GEMQ。
- 所有 progressive proxy 方法都無效（本輪本來就沒有否定的結果）。

## 方法（`plan.json`）

**固定的條件**（兩組相同）
- 資源上限：aggressive 配置的精確資源。
  - weight bytes ≤ 3,804,495,872；BOPs ≤ 39,226,380,910,592（各自除以 int8 參考後 M_rel 0.5946001、B_rel 0.5946031）。
  - BOPs 用 FP32 calibration 的 routed rows，兩次搜尋都相同。
- 3051 個搜尋投影（每個 expert × gate／up／down）；GRID11 × GRID11。
- stage 1 的 calibration：10 個窗口 × 128 tokens。
- 固定的 ranges（`runs/olmoe_multipoint_20261015/ranges.json`）。
- 修正後的共用量化器：DevQuant 的權重量化（per-output-channel int、Universal posit LUT、f16／bf16）加上修正後的 int activation QDQ。
- router、attention、norm、embedding 和 LM head 都是 FP32。
- 21 個未搜尋的投影維持 bf16 權重和 FP32 activation。

**BC 的形式**（和 `run_olmoe_multipoint.stage_table` 相同）
- **Weight 項**：B[f] = Σ_c (G_c／n_in)·Σ_in ΔW²_c。
  - G_c = Σ_window Σ_in (GYᵀX)²_c，channel 軸是輸出 channel。
  - ΔW = Q_f(W) − W，W 是原始 checkpoint 權重。
- **Activation 項**：C[f] = Σ_{slot∈x,y} Σ_ch [G2／n·(Σ err² − n·μ²) + GS·μ²]。
  - 前半是 variance 項，後半是平均偏差（bias）項，兩者都重算。
  - G2 = Σ_rows gt²；GS = Σ_window (Σ_rows∈window gt)²；μ 是逐 channel 的平均誤差；n 是該 expert 的 routed rows。
- fp32 選項的成本為 0。總成本 = B[w] + C[a]。

**A：static**
- 在 FP32 reference 上：calibration 的 FP32 dynamic forward（記錄路由），把路由視為常數，對 summed CE 做 backward，再依上述形式建表。
- 接著 LP 優先搜尋，得到 q0。

**B：refresh**（只做一次，不依 loss 回退）
- **Forward**：q0 配置、calibration 窗口、正常 dynamic routing（記錄自己的 ID 和權重），套用 q0 的 weight 與 activation QDQ。
- **Backward**：用記錄下來的 q0 路由（ID 與 routing weight 皆為常數）逐層重算後反傳。
  - 沒有 router 權重路徑的梯度，也沒有對 top-k 切換求導。本輪更新了路由所在的工作狀態，但沒有對路由選擇求導。
  - activation QDQ 經過 **identity STE**：用自訂的 autograd Function，forward 在無梯度下直接回傳既有 QDQ 的值（逐位元相同，不使用 `x + (q − x).detach()` 這種可能改變浮點位元的寫法）；backward 把梯度原樣傳回，被 clamp 的值也一樣。這不是離散量化器的真實導數。
- **Weight 項**：GY 是對 **y-QDQ 前** linear 輸出的梯度（經 STE 等於對 QDQ 後輸出的梯度）；X 是 linear **實際吃到的、x-QDQ 後**的輸入。ΔW 用原始 W 生成，直接重用 static 建表時的結果，不對 q0 的量化權重再量化。
- **Activation 項**：
  - x 槽：gate 和 up 用共享的 **QDQ 前** expert 輸入，配上它的 total gradient（兩個 consumer 經 STE 的梯度和），維持原本 Gate／Up 共享輸入的語意；down 用 **QDQ 前**的 SiLU(g)·u。
  - y 槽：**QDQ 前**的 linear 輸出和它的梯度。
  - err = QDQ_f(t) − t，t 是 q0 工作狀態下的 QDQ 前張量，ranges 固定。
  - 所有誤差都相對該工作狀態的 QDQ 前張量計算。
- **覆蓋處理**：搜尋節點集合固定；q0 calibration 中沒有 token 的原搜尋 expert 會沿用 static 表的值（本輪為 0 個）；原本不搜尋的 expert 即使有了 token，也不加入搜尋；n 用 q0 的 routed rows。
- 這是沿用 BC 形式的「工作狀態刷新」，不宣稱是精確的 loss 增量。

**搜尋**（兩組相同）
- `run_olmoe_search_final.make_solver("lpfirst")`：LP → 取整 → 可行性修復。
- 和 LP 下界的差距 ≤ max(1e-3·目標值, 1e-6·scale) 才接受；否則退回修正後 MILP（1 ppm 重試、修復），時限 300 秒。
- 兩張表各自做精確支配剪枝，並做精確資源檢查。

## 結果細節

**搜尋**（`search_static.json`、`search_refresh.json`）

| | static → q0 | refresh → q1 |
|---|---|---|
| 路徑 | LP 被接受（不需要 MILP） | LP 被接受 |
| 目標值（各自的表）／LP 下界 | 42.2003／42.1827 | 43.9457／43.9402 |
| 剪枝後選項數 | 70,153 | 70,115 |
| 求解時間 | 0.9 秒（剪枝另 4.1 秒） | 0.8 秒（剪枝另 4.1 秒） |
| size 用量 | 100% | 100% |
| BOPs 用量 | 99.988% | 99.996% |
| weight 格式 | int5 1549、int4 1161、int6 310、int7 16、int8 11、bf16 4 | int5 1576、int4 1147、int6 293、int7 17、int8 15、bf16 3 |

**交叉目標值**（描述）

| 用哪張表計算 | q0 | q1 |
|---|---:|---:|
| static 表 | 42.20 | 42.95（+1.8%） |
| refresh 表 | 50.70 | 43.95 |

refresh 表對整體的評價比較嚴，而且認為 q0 比 q1 差 15%。這只是 proxy 數值，不是 loss 的估計。

**和原 aggressive 的關係**：q0 和原 aggressive 只有 6 個 activation 格式不同。所以本輪重建 static 表加上 LP 優先搜尋，等於重現了原本的配置。

**q0 → q1 的格式變化**（`freeze.json`）
- 292 個投影不同。
- 依投影類型：gate 114、up 90、down 88。
- 依層：每層 11–25 個。
- weight 格式變化（125 個）：
  - int4→int5 44、int5→int4 33、int6→int5 25、int5→int6 8；
  - 少數升到 int7／int8。
- activation 格式變化（182 個）：
  - p8e1→p6e1 38、p6e1→p8e1 34；
  - int7→p8e1 20、int7→p6e1 19、p8e1→int7 16、p6e1→int7 11；
  - 其餘較少。

**q0 的工作狀態**（`refresh_checks.json`）
- calibration CE 為 2.5501 nat/token。
- 搜尋的 1017 個 expert 全部有被路由到，沒有需要沿用 static 值的節點。
- routed rows 總數：FP32 163,840，q0 163,838。

## 控制（`checks.json`；全部通過）

**static 建表**
- FP32 calibration 路由和 stage 1 的快取逐位元相同；CE 差 0。
- 逐層重算（含梯度）和 forward 的差是 0。
- routed rows、ranges 和歷史表一致；3051 個節點全部有被路由到。
- 和歷史表比較：
  - B 的最大相對差 4.4e-16；
  - C 在 posit、f16、bf16 上相同；在 int4–int8 上差 1.1e-6 到 2.2e-5。原因是歷史表用的是修正前的 GPU int activation QDQ（用 CPU 上的 0 維 scale 做除法，每個窗口有 1–5 個 code 翻轉）。

**q0 refresh**
- 逐層重算（含梯度與 STE）和 no-grad forward 的差，16 層全部是 0。
- 我寫的 dynamic forward 和已驗證的 `run_olmoe_dynamic_route.forward` 相比：CE 差 0，路由逐位元相同。
- 把記錄的 calibration 路由重播一次，CE 差 0。
- STE 單元測試（8 種格式）：forward 和 QDQ 逐位元相同；backward 等於上游梯度；梯度有限且不全為 0。
- 梯度、proxy、loss 全部是有限值。

**搜尋**：q0 和 q1 都在預算內，所有格式都合法。

**評估**
- q0 和 q1 記錄的路由重播：512／512 個窗口逐位元相同。
- q1 第一批窗口重跑：逐位元相同。
- 程式控制：aggressive 在舊窗口上的 dynamic CE 和先前的結果差 0。
- router 未被替換；量化模型自己選路由：q0 有 30.5%、q1 有 30.0% 的 (token, 層) 選擇和 FP32 不同。

## 評估資料（`segments.json`、`data_freeze.json`）

- 去重後的 WikiText-103 train，使用嚴格文章邊界。
- 排除：WikiText-2 的全部文章、WikiText-103 內部重複、近重複（≥10% 的 5-gram shingle）。
- 也排除先前所有 OLMoE 用過的文章：route_mass_restore 103 篇、route_dynamic_prop 242 篇、precision_protection 的 selection 32 篇與 final 169 篇。
- 排除後可用 26,849 篇。
- 依 seed 22101 的順序接受 128 篇，另有 5 篇因近重複被跳過。窗口起點用 seed 22102。
- 每篇的內容 hash、窗口位置和 tokens hash 都已記錄。
- 抽樣在配置凍結之後才進行（`freeze.json` → `data_freeze.json`）。

## Offline 時間（`timing.json`、`timing.jsonl`）

- 每項只量了一次。
- static 先跑，因此 refresh 讀 checkpoint 時可能受益於作業系統的頁面快取，並不完全是 cold 的計時。
- 沒有使用任何歷史快取。

**1. 推導 q0 的基準成本：約 104 秒（牆鐘）**

| 項目 | 秒 |
|---|---:|
| 量化器初始化 | 2.5 |
| FP32 calibration forward／路由 | 0.8 |
| backward／敏感度統計 | 27.6 |
| 權重誤差 ΔW（10 種格式 × 3051 個投影） | 35.4 |
| activation 誤差統計 | 18.3 |
| 組表 | 0.8 |
| 未分項的 checkpoint 載入、歷史表核對、存檔 | 約 11.6 |
| **static stage 合計** | **97.0** |
| 初次搜尋（剪枝 4.1、LP＋取整 0.9） | 6.9 |

**2. 從 q0 得到 q1 的額外成本：約 108 秒（牆鐘，不含控制）**

| 項目 | 秒 |
|---|---:|
| q0 權重與 activation 量化準備（forward 3.6、backward 重算 3.5） | 7.1 |
| q0 calibration forward／路由 | 1.6 |
| backward／敏感度統計 | 28.4 |
| 候選 activation 誤差統計 | 18.3 |
| 權重誤差 | 0（精確重用 static 的結果） |
| 組表 | 0.8 |
| 未分項的兩次逐層 checkpoint 載入、權重誤差快取（約 250 MB）讀取、存檔 | 約 45 |
| **refresh stage 合計（扣除 16.3 秒的控制）** | **約 101.6** |
| 第二次搜尋（剪枝 4.1、LP 0.8） | 6.7 |

額外成本中，backward 不是唯一的主要成本：activation 誤差重算和 I/O 也各占相當比例。

**3. 正確性控制與最終評估：約 6 分鐘**

| 項目 | 秒 |
|---|---:|
| refresh 內的控制（`dr.forward` 8.2、重播 8.2） | 16.3 |
| 評估：FP32 | 29.0 |
| 評估：q0、q1 | 各 62.3 |
| 評估控制：重播 2 次 | 124.6 |
| 評估控制：q1 重跑 | 20.7 |
| 評估控制：程式控制 | 20.6 |
| 抽評估資料 | 33.2 |
| plan | 22.6 |
| 分析 | 0.3 |

**峰值記憶體**
- GPU：static 3.8 GB；refresh 4.2 GB；評估 5.9 GB。
- CPU RSS：約 14.8 GB，包含以 mmap 讀入的 checkpoint 頁面。

這些都是研究流程的成本，不代表部署時的 latency 或端到端的加速。

## 下一步建議（未執行）

1. **先確認收益**：在同一資源點，用約 600 篇新文章預先登記一次確認實驗，CI 半寬約 0.001；或另選 knee 點。只評估已凍結的 q0 和 q1，不重新搜尋。
2. **若確認有收益**，再做省時消融：
   - error-only：只重算 activation 誤差，可省下 backward 的 28 秒；
   - sensitivity-only：只重算梯度，可省下 18 秒的 activation 誤差重算。
   - 同時把 I/O（約 45 秒）優化掉，例如在 forward 和 backward 之間保留量化後的權重。這需要更多記憶體：每層量化權重 1.6 GB，全部共 25.8 GB。
3. **若確認沒有收益**：維持 static proxy；不推廣到其他 progressive proxy 方法。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_proxy_refresh_20261022      # 新實驗請改用新目錄
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_proxy_refresh.py
$CPU $S --stage plan --out-dir $O
$GPU $S --stage static --out-dir $O            # 約 1.5 分鐘
$CPU $S --stage search_static --out-dir $O
$GPU $S --stage refresh --out-dir $O           # 約 2 分鐘
$CPU $S --stage search_refresh --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage eval --out-dir $O              # 約 5.5 分鐘
$CPU $S --stage analyze --out-dir $O
```

**環境**
- GPU：torch 2.14.0+cu130，RTX 3060，TF32 關閉，matmul precision highest。
- 開始前確認過 `torch.cuda.is_available()` 為 True；nvidia-smi 的 NVML 版本不一致沒有影響 CUDA，系統驅動未改動。
- CPU：torch 2.14.0+cpu、scipy HiGHS。

## 檔案

- **預先登記**：`plan.json`、`plan.md`、`input_hashes_before.json`。
- **表**：`table_static.json`、`table_refresh.json`、`weight_error_d2.npz`；`static_calibration_routes.npz`、`q0_calibration_routes.npz`。
- **搜尋與凍結**：`search_static.json`、`search_refresh.json`、`configs/q0.json`、`configs/q1.json`、`freeze.json`。
- **控制**：`checks.json`（彙整）、`static_checks.json`、`refresh_checks.json`、`eval.json` → `controls`。
- **評估**：`segments.json`、`tokens.npy`、`data_freeze.json`、`eval.json`、`per_article.csv`、`analysis.json`。
- **時間**：`timing.json`、`timing.jsonl`；`logs/` 含 HiGHS 的 log。
