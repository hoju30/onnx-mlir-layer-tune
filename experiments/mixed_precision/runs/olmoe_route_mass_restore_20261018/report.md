# OLMoE aggressive 配置：固定 expert 選擇、恢復 routing weight 總和

日期：2026-10-18
腳本：`run_olmoe_route_mass_restore.py`（plan → pilot → data → run → analyze）
輸出：本目錄

**本輪的性質**
- 這是用了 FP32 資訊的**診斷**，不是可部署的修正。
- 「weight」一律指 routing weight，不是 expert 參數 W。
- 本輪測的是 GPU 上的 QDQ 行為，不代表 native kernel 的結果。

**本輪沒有改動的東西**
- BC、搜尋、配置（aggressive，sha256 `e51dd107…`）、量化器、range、router。
- 沒有加入一階項。

**標記**
- [預先登記]：寫在 `plan.json` 的分析。
- [描述]：不做推論檢定的描述統計。
- [推論]：由資料推得、尚未直接驗證的判斷。

## 結論

**把所選權重的總和恢復成 FP32 的值，沒有帶來可確認的改善；權重路徑的損傷幾乎全部還在。損傷不是來自「總和下降」，比較可能來自 expert 之間的相對比例改變（這一點是推論）。**

| 指標 | 定義 | 平均 | 95% CI | Bonferroni ×3（96.67%） | 判定 |
|---|---|---:|---|---|---|
| **Δ_restore（主要）** | Lmass − L01 | **+0.00094** | **[−0.00034, +0.00222]** | — | 無法和 0 區分 |
| Residual | Lmass − L00 | +0.00475 | [+0.00327, +0.00626] | [+0.00295, +0.00659] | 明確為正，不在 ±0.002 內 |
| E_weight（重現） | L01 − L00 | +0.00381 | [+0.00205, +0.00561] | [+0.00166, +0.00601] | 重現 |
| E_total（重現） | L11 − L00 | +0.00720 | [+0.00485, +0.00954] | [+0.00433, +0.01007] | 重現，而且超過 ε |
| L11 − Lmass（描述） | | +0.00245 | [+0.00030, +0.00461] | — | — |

- 單位 nat/token。
- 資料是 103 篇 WikiText-103 train 文章（412 個窗口），已排除所有和 WikiText-2 重疊的文章。

**五個問題**

1. **權重干預的損傷是否重現？** 重現，而且是在新的資料來源上。
   - E_weight = +0.0038，校正後下界 0.0017；上一輪在 WikiText-2 上是 +0.0037。
   - E_total = +0.0072，下界 0.0043 > ε；上一輪是 +0.0068。
2. **恢復總和是否改善？** 沒有確認的改善。
   - Δ_restore 的點估計是 +0.0009，方向反而是變差，CI 含 0。
   - 95% CI 可以排除「改善超過 0.00034 nat/token」，這大約只是權重路徑損傷的 9%。
   - 依預先判讀，不宣稱有效，也不宣稱總和完全無關。
3. **恢復總和後還剩多少損傷？** 幾乎全部。
   - Residual = +0.0047，95% CI [+0.0033, +0.0063]，校正後明確為正，也不在 ±0.002 內。
   - 「恢復總和」不足以消除權重路徑的損傷。
4. **有沒有數值或新資料來源干擾？** 沒有找到。
   - 所有控制都逐位元通過：關閉量化時，Lmass 的縮放比例恰好是 1，結果和 FP32 逐位元相同。
   - 恢復後的總和和 m0 的最大相對差是 2.9e-7。
   - 新資料上 D_L00、E_weight、E_total 都和 WikiText-2 的數值接近，沒有看到來源偏移；但結果只描述「不在 WikiText-2 中的 WikiText-103 train 文章」。
   - 有一次控制門檻的修正，發生在抽新資料之前（見「修訂」）。
5. **下一步是否值得研究可部署的替代方案？** 以「恢復總和」為目標的可部署方案，目前**不值得**投入。
   - 即使用 FP32 的 m0 直接恢復，也沒有改善。
   - 下一步應檢驗另一面：保持目前的總和，只把 expert 之間的比例恢復成 FP32。見最後一節。

## 條件（`plan.json`；每個條件各自完整 forward）

**基準**
- 完整 FP32 dynamic。
- 保存每層、每個 token 的 S0（top-8 ID）和 p0（全部 64 個 expert 的 float32 softmax 機率）。
- w0 = p0[S0]，m0 = sum(w0)。m0 是每個 token、每層各自的值，不是 1；平均約 0.32–0.54，依層而異。

| 條件 | expert ID | routing weight |
|---|---|---|
| L00 | S0 | w0 |
| L01 | S0 | w = p[S0]，p 來自該條件自己的 hidden state |
| **Lmass** | S0 | w × (m0／m)，m = sum(w) |
| L11 | top-8(p) | p[ID] |

**Lmass 的數值實作**（預先固定）
- w 是 float32。
- m 和 m0 都用 `torch.sum`，以 float32 依 S0 的順序（基準 top-k 的順序）加總。
- scale = m0／m 用 float32 計算；權重 = w × scale。
- 不 normalize 到 1，不改 logits，不 clip，不加 epsilon。
- 若 m 為零或任何值非有限，直接停止。本輪沒有發生。

**語意**
- 恢復總和會改變後續的 hidden state，所以之後各層的 p 和 L01 不同。
- 這是「逐層恢復總量」的端到端干預，不能拆成「總量占 X%、比例占 Y%」。

## 數值與語意控制（`results.json` → `controls`）

全部通過。

| 控制 | 結果 |
|---|---|
| 程式控制（pilot，上一輪 route-decomposition 的窗口 0–127，舊資料） | 基準的 S0、w0、p0 和 CE，以及 L00、L01、L11 的 CE，都和上一輪逐位元相同 |
| Lmass 的 ID | 全部等於 S0 |
| Lmass 的總和 | 和 m0 的最大差 1.8e-7，最大相對差 2.9e-7；總和 = 1 的比例是 0（基準也是 0） |
| Lmass 的比例 | 權重 = float32(w × scale)，逐位元相同；權重 ÷ w 在每個 token 內的相對離散 ≤ 1.2e-7 |
| m0 對應到正確的 token 和層 | 記錄的 m0 和用 numpy 對 p0[S0] 加總的結果，最大差 1.2e-7（約 2 ulp） |
| 關閉量化（控制子集 0–127） | L00、L01、Lmass、L11 都和 FP32 逐位元相同；Lmass 的 scale 恰好是 1 |
| 重跑（控制子集） | 四個條件都 128／128 逐位元相同，ID 和權重也相同 |
| 重播：各條件實際用的 ID 和權重，交給已驗證的 `run_olmoe_dynamic_route.forward` | 四個條件都 412／412 逐位元相同 |
| 配置與數值 | 6102 個 hook，只掛在 expert 投影上；router 未被替換；所有值有限；scale 沒有非有限值 |

**數值結論**：本輪的縮放沒有引入可辨識的數值差異。不量化時，Lmass 和 FP32 逐位元相同，所以 Δ_restore 不是浮點雜訊造成的。

## 資料（新來源，`data_freeze.json`、`segments.json`）

**來源**
- Salesforce/wikitext `wikitext-103-raw-v1` 的 train，兩個 parquet 檔，sha256 和 hub 的 LFS oid 一致。
- 下載到獨立目錄 `gpt2_assets/data_wt103/`。
- WikiText-103 的 validation／test 和 WikiText-2 是同一個檔案（sha 相同），所以只用 train。

**去重**（在 plan 中固定，早於任何評估）
- 嚴格文章邊界規則：共 28,472 篇文章。
- 排除標準化標題或內容 hash 和 WikiText-2（全部 720 篇）相同的 613 篇。WikiText-2 的所有文章都排除，因此先前 OLMoE 用過的文本也全部排除。
- 排除 WikiText-103 內部重複的 464 篇（標題或內容重複的整組）。
- 剩下 27,395 篇。

**抽樣**
- 依隨機順序（seed 18111）逐篇處理，看了 105 篇、接受 103 篇：
  - 1 篇不足 512 token，跳過；
  - 1 篇近重複，跳過：「Entombed (video game)」有 30% 的 5-gram 和某篇 WikiText-2 文章相同。
- 已接受文章和 WikiText-2 的最大 shingle 重疊是 8%，彼此之間的最大重疊是 0.6%。
- 每篇單獨 tokenize，取 4 個連續窗口，起點 seed 18112。
- 文章標題、行範圍、hash、token 範圍都已記錄。

**樣本數**
- pilot（32 篇舊文章）中，Δ_restore 的文章級 SD 為 0.00775。
- n = ⌈(1.96·s／0.0015)²⌉ = 103 篇。
- 實際 95% CI 半寬 0.00128，達到目標。

**外推限制**
- 結果只描述「不在 WikiText-2 中的 WikiText-103 train 文章」。
- 不和先前的 WikiText-2 確認集合併。

## 品質（`analysis.json` → `ce_ppl`）

| | FP32 | L00 | L01 | Lmass | L11 |
|---|---:|---:|---:|---:|---:|
| CE | 2.96300 | 2.99171 | 2.99552 | 2.99646 | 2.99890 |
| PPL | 19.356 | 19.920 | 19.996 | 20.014 | 20.064 |
| 相對 FP32 損傷（nat/token） | — | +0.0287 | +0.0325 | +0.0335 | +0.0359 |

**逐篇** [描述]

| 指標 | 為正的篇數（共 103） | 文章級 SD |
|---|---:|---:|
| Δ_restore | 57 | 0.0067 |
| Residual | 75 | 0.0078 |
| E_weight | 65 | 0.0092 |

## 總和、比例與 scale（`per_layer.csv`）[描述]

| 層 | m0 平均 | L01 總和 − m0 | L11 總和 − m0 | 比例偏移 L1（L01／Lmass） | scale 平均 [1%, 99%]，最大值 | MoE 輸出 norm：FP32／L00／L01／Lmass／L11 |
|---|---:|---:|---:|---|---|---|
| 0 | 0.379 | 0 | 0 | 0／0 | 1.000 | 0.138／0.137／0.137／0.137／0.137 |
| 1 | 0.316 | −0.0002 | −0.0000 | 0.025／0.025 | 1.001 [0.97, 1.03]，1.21 | 0.220／0.215／0.215／0.216／0.215 |
| 4 | 0.340 | −0.0006 | −0.0002 | 0.035／0.035 | 1.002 [0.95, 1.06]，1.39 | 0.287／0.287／0.288／0.288／0.288 |
| 8 | 0.385 | −0.0021 | −0.0016 | 0.036／0.036 | 1.006 [0.96, 1.07]，2.24 | 0.483／0.481／0.480／0.482／0.480 |
| 12 | 0.529 | −0.0020 | −0.0014 | 0.039／0.038 | 1.004 [0.96, 1.06]，2.62 | 1.887／1.880／1.877／1.882／1.877 |
| 15 | 0.542 | −0.0038 | −0.0030 | 0.044／0.043 | 1.008 [0.95, 1.09]，2.34 | 6.417／6.418／6.389／6.422／6.396 |

**scale（m0／m）**
- 整體平均 1.0046，55% 的值 > 1。
- 0.1% 和 99.9% 分位數為 0.92 和 1.18。
- 極端值 0.49 和 3.66，沒有非有限值。

**總和的缺口很小，比例的偏移較大**
- L01 的總和缺口只占 m0 的約 0.1–0.7%。
- 但所選 expert 之間的相對比例偏移（正規化後的 L1）是 0.025–0.044。
- Lmass 保留了 L01 的比例偏移：兩者幾乎相同。

**MoE 輸出 norm**
- 第 15 層 L01 的輸出 norm 比 L00 小（6.389 對 6.418），Lmass 則恢復到 6.422。
- 但 loss 沒有改善。
- 也就是說，輸出 norm 的下降不是權重路徑損傷的主要來源。這是描述，不是因果證明；不同 expert 的輸出也可能互相抵銷。

[推論]
- 權重路徑的損傷，比較可能來自 router 對所選 expert 的相對偏好改變，而不是總量下降。
- 本輪沒有直接檢驗這一點。

## 判讀（依預先固定的規則）

| 規則 | 本輪 |
|---|---|
| Δ_restore 的 CI 完全低於 0：恢復有益 | 不成立 |
| Δ_restore 無法和 0 區分 | **成立**：不宣稱有效，也不宣稱總和完全無關；CI 排除了大於 0.00034 nat/token 的改善 |
| Δ_restore 明顯為正：恢復有害 | 不成立：點估計為正，但 CI 含 0 |
| 改善後 Residual 仍明顯為正：恢復總量不足以消除權重路徑損傷 | **成立**：Residual +0.0047，Bonferroni 下界 0.0030 |
| E_weight 和 E_total 是否在新來源重現 | **都重現** |
| 範圍限制 | 本輪固定 FP32 expert ID 並用 FP32 的 m0，即使有效也不能直接套用到完整 dynamic |

## 下一步建議（需另行預先登記）

**1. 互補的比例干預 Lprop**
- 設定：ID 固定為 S0；權重 = m × (w0／m0)，也就是保持該條件自己目前的總和，只把 expert 之間的比例恢復成 FP32。
- 主要對比：Lprop − L01。
- 若明確為負：支持「比例偏移」才是權重路徑損傷的主要來源。接著再研究可部署的方向，例如讓 router 的輸入更接近 FP32：提高餵給 router 的上游 hidden state 精度，或在 BC 中納入 router-logit 敏感度（維持梯度式 criterion 的原則）。
- 若也沒有改善：表示損傷來自後續各層的連鎖反應，而不是單層的權重數值；應改做逐層開關的干預。

**2. 資料**
- WikiText-103 train（排除 WikiText-2 後）還有約 27,290 篇可用，足夠。
- 若延用同一個來源，要另行登記，並排除本輪的 103 篇。

**目前不建議**：任何「恢復權重總和」的可部署修正（例如 router 輸出的總和補償）；修改 BC、搜尋或配置。

## 修訂（`amendments.json`）

- pilot 第一次執行時，「記錄的 m0 等於 numpy 的 float32 加總」這項檢查用了 1e-7 的容許值。GPU `torch.sum` 和 numpy float32 的加總順序不同，會差 1–2 個 ulp（6e-8 到 1.2e-7），所以這項檢查沒有通過；其他控制全部通過。
- 改為記錄實際最大差，容許值 1e-6，和 m 的檢查相同。若真的對錯 token 或層，差異會在 0.1 的量級，仍然能檢出。
- 這項修正發生在**抽任何新資料之前**。
- 方法、資料規則、樣本數規則和終點都沒有改變。
- 失敗的紀錄保留在 `pilot_failed_m0_tolerance.json` 和 `logs/pilot_failed_m0_tolerance.log`。
- 腳本 sha：plan 時是 `1eed0917…`；修正後是 `a2395f10…`，用於 pilot 重跑、data、run 和 analyze。

## 時間成本

所有條件都在同一套 GPU 程式中執行，以下只是本次實驗的成本，不代表部署時的 latency。

| 項目 | 時間 |
|---|---|
| FP32 基準（含全部 64 個機率，412 個窗口） | 24.5 秒 |
| L00／L01／Lmass／L11 | 52.4／53.2／52.7／53.5 秒 |
| 控制：重播 4 次 | 210.0 秒 |
| 控制：關閉量化與重跑（子集）、存檔 | 約 133 秒 |
| run stage 合計 | 579.7 秒 |
| pilot（舊資料，含第一次失敗的那次） | 103.6 秒 + 100.2 秒 |
| plan（含 WikiText-103 清點）／抽資料（含逐篇 tokenize 與近重複檢查） | 23.7／30.6 秒（CPU） |
| 統計分析 | 0.8 秒 |

## 已確立與未確定

**已確立**
- 在 WikiText-103 的新文章上，aggressive 配置的權重路徑損傷（E_weight +0.0038）和完整 dynamic 損傷（E_total +0.0072，超過 ε）都重現。
- 用 FP32 的 m0 恢復所選權重總和，沒有可確認的改善：改善最多 0.00034 nat/token（95% CI 下界）。
- 恢復總和後，損傷仍明確大於 0 且超出 ±ε（Residual +0.0047）。
- 所有數值控制都逐位元通過，縮放本身不是干擾因素。

**未確定**
- 比例偏移是否就是權重路徑損傷的主要來源。
- 這些結論在完整 dynamic（ID 也放開）下是否成立。
- 其他資源點、其他模型、native kernel。
- 如何把診斷結果轉成可部署的修正。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_route_mass_restore_20261018      # 新實驗請改用新目錄
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
# 資料：gpt2_assets/data_wt103/train-0000{0,1}-of-00002.parquet
#   來源 https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/
#   sha256 74da360f… / ba090ac3…
../../venv/bin/python run_olmoe_route_mass_restore.py --stage plan --out-dir $O
$GPU run_olmoe_route_mass_restore.py --stage pilot --out-dir $O      # 約 2 分鐘
../../venv/bin/python run_olmoe_route_mass_restore.py --stage data --out-dir $O
$GPU run_olmoe_route_mass_restore.py --stage run --out-dir $O        # 約 10 分鐘
../../venv/bin/python run_olmoe_route_mass_restore.py --stage analyze --out-dir $O
```

**環境**
- GPU：torch 2.14.0+cu130，RTX 3060，TF32 關閉，matmul precision highest。
- CPU：torch 2.14.0+cpu。

## 檔案

- **預先登記與凍結**：`plan.json`、`plan.md`、`pilot.json`、`pilot_failed_m0_tolerance.json`、`amendments.json`、`segments.json`、`data_freeze.json`、`tokens.npy`。
- **路由**：`baseline_routes.npz`（S0、w0、全部 64 個 p0、MoE 輸出 norm）、`routes/{L00,L01,Lmass,L11}.npz`（ID、權重、MoE 輸出 norm；Lmass 另有 scale、m、m0、w_raw）。
- **結果與分析**：`results.json`、`analysis.json`、`per_article.csv`、`per_layer.csv`。
- **紀錄**：`timing.jsonl`、`logs/`。
