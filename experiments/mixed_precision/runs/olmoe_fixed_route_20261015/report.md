# OLMoE-1B-7B：固定路由下原版 BC 對 expert 量化損傷的預測（第一階段）

日期：2026-10-15　腳本：`experiments/mixed_precision/run_olmoe_fixed_route.py`、`olmoe_common.py`　輸出：本目錄

本輪**沒有修改** BC 公式，沒有做 dynamic-route 量化，沒有啟動全模型所有 experts 的搜尋，也沒有覆寫既有實驗或預設。


## 更正與後續（2026-10-16）

以下是報告完成後，由後續實驗發現的更正。原文保留不改，閱讀時請以本節為準。

- **GPU 量化器（2026-10-16 修正）**：本報告的 GPU 重播所用的 int 活化量化器（`qwen3_swiglu_common.int_act_qdq`），在 GPU 上是「乘以 scale 的倒數」，和 CPU 及 QuantizeLinear 的真除法相差最後一個位元，每個窗口約有 1–5 個量化值不同。量化流程會把這種差異放大成單窗口約 0.01 nat/token 的變動；以 64 個窗口的配對平均來看，偏移約 1e-3（`runs/olmoe_search_final_20261016` 實驗 A）。共用函式已修正並驗證（`runs/fix_integration_20261016`），本報告的數字沒有重算。本報告的單一 expert 損傷約 1e-4，這種放大對它影響多大沒有量過 **[未驗證]**；「單一 expert 層級的 BC 排序接近雜訊」這個結論，可能有一部分反映了這種實作變異。
- **CPU 交叉檢查**：系統有負載時，CPU 重播會因 OpenMP／MKL 動態調整執行緒數而無法重現；設 `OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE` 才能逐位元重現（`runs/olmoe_search_final_20261016`）。本報告的 CPU 子集結果不應視為固定值。

## 結論（回答七個問題）

標記說明：**[量測]** 實際量到；**[推論]** 由量測推得；**[未驗證]** 本輪沒有資料；**[資源限制]** 受本機資源所限。

1. **固定路由的 forward／backward 正確**。[量測]
   - tiny 模型：transformers 整個模型、逐層組合、本實作三者 logits 完全相同；fixed 與 dynamic 也完全相同；FP64 下分層反向與整圖 autograd 逐位元相同；FP32 對 FP64 的中位數差 8e-7。
   - 真實模型：
     - forward：與 transformers 原始層相比，logits 相對差 6.7e-7，summed CE 到小數第 4 位相同；fixed 與 dynamic 重播完全相同（42 個窗口）；
     - backward：layer 13–15 上，分層反向與整圖 autograd 完全相同；
     - 重跑：GPU 重播結果逐位元一致。
   - routing weights 是常數，不帶梯度；換用別的窗口的路由，logits 會改變 72%，表示快取沒有混用。
2. **哪些是 native、哪些是 PyTorch**。
   - **native**（onnx-mlir）：tiny OLMoE 的 fixed-route forward，以及 expert 投影的梯度統計（per-channel Σg²、Σg、ΣdW²），兩個窗口的中位數差 7e-7、最大 3.5e-6。[量測]
   - **PyTorch**：真實模型的 BC 表、range 與誤差統計、所有 truth。
   - 真實模型的 native 沒有嘗試。[資源限制] FP32 權重 27.7 GB，大於本機可用 RAM（約 28 GB 且還要扣其他程式）；每個窗口的路由不同，selector 形式需要每窗口一張圖（40 個窗口就是 40 次編譯）；而 Qwen3.5-0.8B 在 512 tokens 時統計編譯就已用到 33 GB。
   - MoE 激活的 range／誤差統計沒有 native 核對，只做了 PyTorch 內的「gathered vs dense 遮罩」一致性檢查（通過）。
3. **expert 校準覆蓋只部分足夠**。[量測]
   - 10 個 calibration 窗口下，每層每個 expert 的路由 token 數中位數 126–154，最多 856–960；6/16 層各有 1–2 個 expert 完全沒被選到，這些 expert 不給分數。
   - 選出的 9 個 expert 中，2 個依規則標為低覆蓋：L8.E3 9 個 token、L15.E59 3 個 token。
   - 「覆蓋足以計算統計」不等於「足以支持排序」：真實損傷的排序在多數 expert 上不能跨窗口重現（第 4 點）。
4. **BC 在充分覆蓋的 expert 上排序不合理**。[量測]
   - 全部 461 個決定性同資源配對，BC 對 56.8%；weight-only 53.4%、activation-only 62.1%、wa 59.4%；各 expert 內的 Spearman 介於 −0.59 到 +0.74。
   - **高覆蓋 expert**：BC 只對 44.2%（208 個配對）；而事後算的參考「直接用 calibration 實測的損傷來預測」對 78.8%。真實訊號存在且可重現，但 BC 沒抓到。
   - **中覆蓋 expert**：連 calibration 實測都只對 41.2%，排序本身被窗口雜訊主導。
   - **損傷量級**：單一 expert 量化的損傷極小，每 token ΔNLL 多在 1e-5 到 3e-4，最大約 1e-3；int4 的損傷「幅度」比 int8 大，但「方向」不固定，只有約一半的投影 int4 > int8。
5. **瓶頸是單支評分，不是聯合的非加成性**。[量測]
   - Gate+Up 聯合幾乎可加：交互項／聯合損傷的中位數 1–10%；用 diagnostic 實測的單支和排序聯合配對，對 92–100%（同窗口 oracle，不可部署）。
   - BC 的聯合正確率只有 31–87%。
   - 用 calibration 實測的單支和（可部署的估計）來排序 diagnostic 的聯合配對：29–96%，中位數約 0.6。
6. **還不值得進入 fixed-route 自動搜尋**。[推論]
   - BC 在單一 expert 層級排不出候選的順序。
   - 但單一 expert 的損傷只有約 1e-4 nat／token，自動搜尋真正在乎的是許多 expert 一起量化後的總損傷，而那個層級本輪沒有驗證。[未驗證]
   - 建議先做「整層或多個 expert 同時量化」的 fixed-route 驗證（§11），再決定要不要搜尋。
7. **dynamic-route 下一階段需要另外驗證的因素**（§11）：
   - 量化後 top-8 選擇翻轉的比例與位置；
   - routing weight 的變化量；
   - 梯度要包含 router 路徑（梯度定義會改變）；
   - top-k 的不連續性使二階近似失效；
   - expert 負載與覆蓋隨量化改變；
   - router 本身是否量化；
   - 部署時沒有 FP32 參考路由。

**不要因為這批結果立刻修改 criterion**。[推論]
- 這輪的真實損傷極小，而且多數 expert 的排序不能跨窗口重現，所以「BC 判錯」大多不是可修正的訊號。
- 只有高覆蓋 expert 有 BC 沒捕捉到、但可重現的訊號，那裡的候選來源是一階（方向）項；這是假說，要另外預先登記、用新窗口驗證。

## 1. 模型與資源預檢（`environment.json`）

**模型**
- `allenai/OLMoE-1B-7B-0924`，revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，下載 12.9 GB（3 個 BF16 safetensors，130 s）。
- 實際 config 與官方一致：16 層、64 experts、top-8、`norm_topk_prob=false`；hidden 2048、expert FFN 1024、16 heads（MHA）、vocab 50304、embedding 不共用、`rms_norm_eps` 1e-5（預設值）、QK-norm 作用在整個 2048 維。
- 參數 6,919,161,856，全部 BF16：experts 6.44B、attention 2.69 億、embedding 1.03 億、LM head 1.03 億、router 210 萬、norms 6.8 萬。
- transformers 5.19.0（隔離的 `qwen35_assets/python_deps`，沒有改動任何 venv）。原始碼要點：
  - expert 權重載入時拼成 `gate_up_proj` [64, 2048, 2048] 與 `down_proj`；
  - router 用 float32 softmax、top-8、不重新正規化；
  - expert 輸出先過 down，再乘 routing weight，最後 index_add；
  - 預設 experts 實作是 `grouped_mm`、attention 是 sdpa，本研究的參考一律指定 eager。

**資源**
- RAM 39 GB（可用約 28 GB）、swap 7 GB；GPU RTX 3060 12 GB（可用 11.2 GB）；磁碟可用 171 GB；12 核 CPU。
- FP32 全模型 27.7 GB 放不進 GPU，RAM 也很勉強。所以採**分層串流**：每層只把 BF16 權重轉成 FP32（精確轉換）放上 GPU。
- 反向時從存好的每層輸入重算該層（checkpointing），等價性已驗證（§3）。
- 沒有把 FP32 baseline 換成 BF16，也沒有 detach 中間 hidden state。

## 2. fixed-route 的定義與實作

**快取**：每個窗口先跑一次未量化的 FP32 dynamic forward，每層、每個 token 存下：
- top-8 expert ids（int16）；
- 對應的 routing weights（float32，未重新正規化）；
- 窗口 tokens。

快取附 hash、shape、dtype 與 revision（`routes_manifest.json`），calibration、diagnostic、check 三組分開產生。

**重播**
- 每層都用該窗口自己的快取：y = Σ_e r_FP32 · Expert_e(h)。r 是常數（不帶梯度）；h、expert 輸入與輸出都重新計算；不重選 top-k，也不重新正規化。
- 實作兩份：
  - transformers 的 `OlmoeDecoderLayer`（FP32，eager），作為 dynamic 參考；
  - 本實作 `layer_fixed`：每個 op 都用輸入的 dtype。transformers 的 RMSNorm、router softmax、rotary 和 attention softmax 會強制用 float32，所以呼叫 `.double()` 不等於 FP64；本實作的 FP64 才是全程 FP64。
  - Gate、Up、Down 分開 linear，expert 與 token 的處理順序和 transformers 相同。
- **梯度**：fixed-route 模型的梯度不經過 router，和 dynamic 模型不同。BC 統計和 truth 都用 fixed route。loss 是 summed next-token CE，不含 router auxiliary loss。

## 3. 正確性（`correctness.json`、`determinism_check.json`）

**A. Forward**

| 檢查 | 結果 |
|---|---|
| tiny：transformers 整個模型 vs transformers 逐層組合 vs 本實作 | 0／0（完全相同） |
| tiny：fixed（快取路由）vs dynamic | 0 |
| tiny：換用另一窗口的路由 | logits 相對差 0.72（快取沒有混用） |
| 真實模型（check 窗口）：transformers 層 vs 本實作 | logits 6.7e-7；MoE 輸出最大 5.7e-7；層輸出最大 2.7e-7；summed CE 336.379／358.0885 兩邊相同 |
| 真實模型：fixed 重播 vs dynamic（check 2、calibration 10、diagnostic 30 個窗口） | logits 0，CE 0 |

**B. Backward**

| 檢查 | 結果 |
|---|---|
| tiny：FP64 分層反向 vs FP64 整圖 autograd（tap 梯度與權重梯度） | 0（逐位元相同） |
| tiny：FP32 分層 vs FP64 整圖 | 中位數 7–8e-7，最大 2.6e-6 |
| 真實模型：layer 13–15，分層 vs 整圖（10 個 tap 梯度、6 個 dW、h13 梯度） | 0 |
| routing weights 帶梯度？ | 否 |
| 重跑一致性（GPU，baseline 與一個候選各跑兩次） | 0（與存下的 truth 也是 0） |

整圖檢查的 dW 用 tap 組出（dW = gYᵀX），和建表的計算相同；第一次直接對整個 expert 堆疊權重取梯度時 GPU 記憶體不足，改成這個方式。

**C. Native（tiny OLMoE：3 層、8 experts、top-2）**
- **匯出方式**：每層每個被路由的 expert，用常數 one-hot selector S_e [n_e, T] 做 MatMul 取出它的 token 列（精確），算完乘上常數 routing weight，再用 S_eᵀ 寫回；expert 由小到大，和 transformers 的累加順序相同。
- **沒有使用 Gather／Scatter 派送**：整張圖只有 1 個 Gather（embedding）。因為 selector 是常數，每個窗口要一張圖。
- **結果**（窗口 0／1）：
  - 匯出模型 vs PyTorch fixed-route：0／0；
  - native logits vs FP64：6.2e-7／4.9e-7；
  - 對應到的 expert 投影 66／63 個，等於實際被路由的數量；
  - 梯度統計 132／126 個量化點，與 FP64 比較的中位數 7.1e-7／6.8e-7，最大 3.5e-6／2.4e-6。

**沒有做的**：真實模型的 native forward／backward／統計（§1、結論第 2 點）；MoE 激活誤差統計的 native 核對。

## 4. 資料與預先登記（`plan.json`、`freeze.json`）

- **資料**：WikiText-2 raw（與先前研究同一份 parquet，空行連接），OLMoE 自己的 tokenizer，不加特殊 token；記錄了原文字元位置。

  | 集合 | split | token | 原文字元 |
  |---|---|---|---|
  | calibration（10 × 128） | validation | 0–1279 | 0–5451 |
  | diagnostic（30 × 128） | validation | 1280–5119 | 5451–22584 |
  | check（2 × 128，只用於正確性） | test | 0–255 | 0–1157 |

  - calibration 與 diagnostic 的原文字元不重疊。
  - 先前 Qwen／GPT-2 研究用其他 tokenizer 看過這兩個 split 的開頭；token 偏移不能證明跨模型的文字不重疊，但那些研究沒有對 OLMoE 做任何決定。
  - 本輪沒有用 final test。
- **expert 挑選規則**（在看到覆蓋之前寫入 plan）：
  - 層：L0、L8、L15；
  - 每層取至少有 1 個路由 token 的 expert，依 (token 數降序, id 升序) 排序，取 round-half-up(q·(n−1)) 位置，q = 0.05／0.5／0.95；
  - 路由 token 少於 32 或出現在少於 3 個窗口者，標為低覆蓋。
- **順序**：plan → 路由快取 → 覆蓋與挑選 → BC 表 → 凍結候選（第 1 輪 w；第 2 輪 a、wa）→ truth。diagnostic 的路由與損傷只用於評估。

## 5. expert 覆蓋（`expert_coverage.csv`、`coverage.json`）

- 每層 calibration 有 1280 × 8 = 10240 個路由位置。每個 expert 的 token 數：L0 是 41–856、L8 是 0–960、L15 是 0–857，中位數 126–154。
- 完全沒被選到的 expert：L5 2 個；L6、L8、L10、L11、L15 各 1 個；其他層 0。這些 expert 沒有統計，不給分數，也不能當成免費的低精度選項。
- 路由權重很小：每個 expert 的 routing weight 中位數約 0.04–0.06（top-8 且不正規化），所以單一 expert 的擾動在輸出端會被縮小約 20 倍。

**選出的 9 個 expert**

| 層 | high | mid | low |
|---|---|---|---|
| L0 | e0：444 tokens、10 個窗口 | e7：126、10 | e57：62、10 |
| L8 | e33：347、10 | e23：150、10 | **e3：9 tokens、4 個窗口（低覆蓋）** |
| L15 | e17：413、10 | e23：159、10 | **e59：3 tokens、3 個窗口（低覆蓋）** |

這只是 9 個 expert、27 個投影，不是 64 個 experts 的完整量化搜尋。

## 6. BC 與量化語意（`table.json`、`ranges.json`、`tables.json`）

- **公式不變**：C(w,a) = W(w) + A(a)。
  - W = Σ_c (G_c / n_in) Σ_in ΔW²，其中 G_c = Σ_window（該窗口每個輸出 channel 的 Σ(∂L/∂W)²）；
  - A 是輸入、輸出 tap 的原版變異項加平均偏差項。
- **MoE 計數**：
  - n 是該 expert 在 calibration 的路由列數（沒有 padding）；
  - G2 是路由列上的 Σg²；GS = Σ_window（該窗口路由列的 Σg）²；
  - 沒有除以啟用次數，也沒有另外乘 routing 機率。routing weight 的作用已經包含在梯度裡。
- **共享輸入**：Gate 與 Up 讀同一組 expert 輸入列，tap 用兩者的總梯度；量化時各自量化自己的副本。
- **QDQ 順序**：x → QDQ（每個使用者各自）→ W_q → QDQ(y)；Down 是 QDQ(silu(g)·u) → W_q → QDQ(y)；接著乘固定的 routing weight，再 index_add。
- **實作**：PyTorch，fixed-route，FP32，GPU。統計語意在 tiny 模型上和 native 核對過（§3C）。
- **激活統計核對**：expert 輸入的 range 與列數，用 gathered 路徑和 dense 遮罩重算的結果完全相同（9 個 expert 全部通過）。依預先登記，這是做 activation 模式的條件。

## 7. 排序結果（diagnostic 30 個窗口；`analysis.json`、`pairs.csv`、`candidates.csv`）

**候選**：每個 expert、每種模式 40 個（Gate／Up／Down 單獨 × 5 種格式，加上 Gate+Up 的 25 種格式組合）；w 360 個，a、wa 各 360 個，共 1080 個。

**同資源配對**（同 expert、同模式，M 與 BOPs 都相同）：
- 共 2583 對；
- 決定性：2000 次配對窗口 bootstrap 中，至少 95% 同向；所有候選共用同一組重抽樣索引。

| 範圍 | 配對 | 決定性 | BC 對 | 正確率 | 判錯配對的 \|差\| 中位數 |
|---|---:|---:|---:|---:|---:|
| 全部 | 2583 | 461 | 262 | 0.568 | 2.6e-4 |
| w | 909 | 238 | 127 | 0.534 | 1.5e-4 |
| a | 909 | 95 | 59 | 0.621 | 3.4e-4 |
| wa | 765 | 128 | 76 | 0.594 | 2.3e-4 |
| 高覆蓋 | 861 | 208 | 92 | 0.442 | — |
| 中覆蓋 | 861 | 182 | 114 | 0.626 | — |
| 低覆蓋（含 2 個低覆蓋 expert） | 861 | 71 | 56 | 0.789 | — |
| Gate vs Up 同格式 | 135 | 27 | 18 | 0.667 | 1.9e-4 |
| 同分支 int vs posit | 162 | 24 | 14 | 0.583 | 2.2e-4 |
| Gate／Up 互換 | 270 | 57 | 40 | 0.702 | 2.6e-4 |
| L0／L8／L15 | 861 每層 | 126／127／208 | 84／82／96 | 0.667／0.646／0.462 | — |

**依模式與覆蓋**（決定性正確率）

| | high | mid | low |
|---|---:|---:|---:|
| w | 0.440 | 0.575 | 0.688 |
| a | 0.353 | 0.694 | 0.880 |
| wa | 0.486 | 0.700 | 0.857 |

**損傷量級與組內排序**（每 token ΔNLL）
- 各 (expert, 模式) 的範圍大約在 [−1e-3, +1.1e-3]，多數 |ΔNLL| < 3e-4。顯著為負的候選，在 L15.E23|w 有 12 個、L15.E17|w 有 9 個。
- 組內 Spearman（BC vs diagnostic）：w 介於 −0.31 到 +0.48，a 介於 −0.59 到 +0.74，wa 介於 −0.24 到 +0.70。

**事後追加的分析**（不在預先登記內，用來區分實作、雜訊與 BC 的問題）
- **int4 是否比 int8 損傷大**：w 模式只有 11/27 個投影成立，a 15/27，wa 17/27。int4 的 |ΔNLL| 確實大一到兩個數量級，但符號不固定。
- **真實損傷的排序能不能跨窗口重現**：calibration 與 diagnostic 實測損傷的組內 Spearman 介於 −0.60 到 +0.92。
  - 可重現的：L8.E33|w 0.85、L15.E17|w 0.85、L15.E59|w 0.92、L8.E33|wa 0.87；
  - 多數介於 ±0.4。
- **參考預測器**：用 calibration 實測的 ΔNLL 預測 diagnostic 的決定性配對（`reference_predictor.json`）。

  | | 決定性 | BC | calibration 實測 |
  |---|---:|---:|---:|
  | 全部 | 461 | 0.568 | 0.627 |
  | w | 238 | 0.534 | 0.664 |
  | a | 95 | 0.621 | 0.568 |
  | wa | 128 | 0.594 | 0.602 |
  | 高覆蓋 | 208 | **0.442** | **0.788** |
  | 中覆蓋 | 182 | 0.626 | 0.412 |
  | 低覆蓋 | 71 | 0.789 | 0.704 |

- **量級對照**（L15.E17 的 Gate int4）：BC 換算約 2e-4 nat／token（B = 0.236，對 10 個窗口 × 127 個 token 取平均），實測 +1.3e-4。二階項的量級和實測同級，但方向由其他成分決定。

## 8. 單支 vs 聯合（`analysis.json` 的 `joint`）

| | 範圍（27 個 expert × 模式組合） |
|---|---|
| 交互項／聯合損傷，中位數 | 0.009–0.095 |
| BC 聯合排序（聯合同資源配對） | 0.31–0.87 |
| diagnostic 實測的單支和 → 同一批 diagnostic 的聯合（同窗口 oracle，不可部署） | 0.92–1.00 |
| calibration 實測的單支和 → diagnostic 聯合（可部署的估計） | 0.29–0.96 |

BC 的聯合分數完全等於兩支相加（`bc_additive_exact` 全部為真）。實際損傷也幾乎可加，所以聯合的錯誤來自單支的估計；不過單支的「真實值」本身也不容易從 calibration 預測（最後一列）。

## 9. 資源與成本（`timing.json`）

**儲存**
- 每個 expert 投影 2048 × 1024 = 2.1M 個元素；每個 expert 6.3M，FP32 為 25.2 MB。
- 全部 experts 6.44B 參數：FP32 25.8 GB、BF16 12.9 GB。
- 候選的 M（`candidates.json` 的 `M_bytes`）只算該 expert 的三個投影。
- 搜尋時所有儲存的 expert 都要計入，不能只算每個 token 用到的 top-8。
- 固定 FP32 的模組：attention 2.69 億、router 210 萬、embedding 1.03 億、LM head 1.03 億、norms 6.8 萬參數。

**BOPs**
- = 該 expert 在 calibration 的路由列數 × 元素數 × 權重位元 × 激活位元。routing weight 的大小不代表執行次數，不能拿來折算。
- 本實作只計算被路由的列（gather），所以實際執行量和語意上的稀疏 BOPs 相同。tiny native 的 selector 形式多了 selector MatMul，那是驗證用的匯出，不是部署方式。

**時間**（wall time）

| 階段 | 時間 | 備註 |
|---|---:|---|
| 下載 | 130 s | 12.9 GB |
| 預檢（含 hash） | 10 s | |
| tiny 正確性（含 native 2 次編譯） | 27 s | 兩次失敗：tiny 權重提供者回傳共用張量的 bug，已修正 |
| prepare | 3 s | 一次失敗：缺 pandas，改用 pyarrow |
| FP32 路由擷取 | 29 s | check 6 s、calibration 4 s、diagnostic 5 s；含 transformers 對照 |
| 真實模型反向檢查 | 5 s | 一次 GPU OOM，改用 tap 組 dW |
| BC 表（9 個 expert、27 個投影） | 26 s | forward 4 s、backward 5 s |
| truth：w | 520 s | diagnostic 354 s、calibration 163 s |
| truth：a、wa | 984 s | diagnostic 685 s、calibration 299 s |
| 分析 | 2 s | |

- **峰值記憶體**：host 14.6 GB；GPU 最高 10.4 GB（a、wa 的 diagnostic truth，120 個候選的 hidden state 同時在 GPU 上）。
- **FP32 參考成本**：固定路由需要先跑一次 FP32 forward 取得路由（每 40 個窗口約 10 s），這是本診斷流程的固有成本。本輪**不是**無參考的部署流程，也不宣稱任何硬體低位元加速。

## 10. 判讀

**[量測]**
- 固定路由的 forward 與 backward，在 tiny（含 native）和真實模型上都正確，而且可重現。
- 單一 expert 的量化損傷極小（約 1e-4 nat／token），方向不固定。
- BC 在決定性配對上的正確率約 57%，高覆蓋 expert 只有 44%。
- 同樣的配對，用 calibration 實測損傷預測，高覆蓋 expert 可達 79%，中覆蓋只有 41%。
- 聯合幾乎可加。

**[推論]**
- 單一 expert 的輸出被很小的 routing weight（約 0.05）縮放，量化擾動很小，所以 loss 變化中一階（方向）成分相對變大，而 BC 只估非負的二階項。
- 高覆蓋 expert 的一階成分在同一個語料上可以重現，所以 calibration 實測能預測它，BC 不能。中覆蓋 expert 的成分則被窗口雜訊主導。
- 這和 Qwen 研究中「小擾動、負 ΔNLL 時 BC 失準」是同一類現象，只是在 MoE 的單一 expert 上更極端。

**[未驗證]**
- 多個 expert 或整層一起量化時，BC 是否能預測總損傷。一階項在 expert 之間可能部分抵消，二階項會同向累加。
- 一階項是否能從 calibration 轉移到新窗口：只在本輪的事後分析看到跡象。
- dynamic route 下的任何結論。
- 真實模型的 native 統計。

**[資源限制]**
- 真實模型的 native 在本機無法執行（§1、§3C）。
- GPU 12 GB 限制了一次能同時重播的候選數（目前 120 個，峰值 10.4 GB）。

## 11. 下一步

1. **不修改 criterion**。這輪的失準大多發生在不可重現的小訊號上。
2. **建議下一個實驗**（另外預先登記、用新窗口）：fixed-route 的**多 expert／整層配置**。
   - 例如 L8 的 64 個 experts 全部量化成同一格式，或混合格式；
   - 比較 BC 總和與實測總損傷的趨勢，和 Qwen3.5 多資源點驗證的做法相同。
   - 這直接回答「能不能進入 fixed-route 搜尋」。
   - 成本估計：PyTorch BC 表擴到每層 64 個 experts 約 1–2 分鐘／層；每個配置的 truth 約數秒（GPU）。
3. **若第 2 步仍失準**：再針對高覆蓋 expert，預先登記一個「一階項從 calibration 轉移到新窗口」的檢驗（本輪事後分析看到可重現的跡象）。
4. **dynamic route**（另一輪，預先登記，不在本輪中途切換定義）：
   - 用量化後的 hidden state 重新計算 router；
   - 量測量化造成的 top-8 翻轉比例、翻轉的層與位置、routing weight 的變化；
   - 梯度要包含 router 路徑，BC 統計需重建；
   - top-k 的不連續性會讓二階近似在翻轉處失效；
   - 決定 router 是否量化；
   - 部署時沒有 FP32 參考路由。

## 12. 重跑與檔案

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/olmoe_fixed_route_20261015     # 新實驗請換新目錄
GPU="env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$PY  run_olmoe_fixed_route.py --stage precheck --out-dir $O
$PY  run_olmoe_fixed_route.py --stage tiny     --out-dir $O        # 含 native（--skip-native 只跑 PyTorch）
$PY  run_olmoe_fixed_route.py --stage prepare  --out-dir $O
$GPU run_olmoe_fixed_route.py --stage capture  --out-dir $O --device cuda
$PY  run_olmoe_fixed_route.py --stage coverage --out-dir $O
$GPU run_olmoe_fixed_route.py --stage bcheck   --out-dir $O --device cuda
$GPU run_olmoe_fixed_route.py --stage tables   --out-dir $O --device cuda
$PY  run_olmoe_fixed_route.py --stage freeze   --out-dir $O --modes w
$GPU run_olmoe_fixed_route.py --stage truth    --out-dir $O --device cuda --sets diagnostic,calibration
$PY  run_olmoe_fixed_route.py --stage freeze   --out-dir $O --modes a,wa
$GPU run_olmoe_fixed_route.py --stage truth    --out-dir $O --device cuda --sets diagnostic,calibration --modes a,wa
$PY  run_olmoe_fixed_route.py --stage analyze  --out-dir $O
```

truth 會依候選續跑：已完成的會略過。模型檔在 `olmoe_assets/model`（12.9 GB），revision 固定。

| 檔案 | 內容 |
|---|---|
| `plan.json`、`environment.json`、`freeze.json` | 預先登記、版本與 hash、兩輪凍結 |
| `routes/`、`routes_manifest.json` | 每個窗口的路由快取（ids、weights、tokens）與 hash |
| `tokens_*.npy` | calibration、diagnostic、check 窗口 |
| `correctness.json`、`tiny_native/`、`determinism_check.json` | tiny（含 native）與真實模型的正確性、重跑一致性 |
| `expert_coverage.csv`、`coverage.json` | 每層每個 expert 的覆蓋與挑選結果 |
| `table.json`、`ranges.json`、`tables.json` | BC（B = W、C = A）、range、激活統計核對與耗時 |
| `candidates.json`、`candidates.csv` | 1080 個候選（BC、M、BOPs、實測 ΔNLL 與 CI） |
| `truth_diagnostic.json`、`truth_calibration.json` | 逐窗口 summed ΔCE |
| `analysis.json`、`pairs.csv`、`reference_predictor.json` | 排序、配對、聯合、覆蓋分組；事後的參考預測器 |
| `timing.json`、`*.log` | 每次執行的時間、狀態、峰值記憶體（含失敗紀錄） |
