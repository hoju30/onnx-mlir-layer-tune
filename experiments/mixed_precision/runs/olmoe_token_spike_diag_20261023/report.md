# OLMoE token-level activation spikes 與 BC 排序（診斷）

> **更正註記（2026-10-23，資料隔離）**
> 本輪確認文章 #39「Brushstrokes series」與 act_outlier 輪重複（content_sha256 相同，token 窗口不同），違反文章級隔離。
> 原因：`used_wt103_all()` 只依 source／split 欄位判定來源，act_outlier 輪的 segments.json 缺這兩個欄位，因此其 64 篇 WT103 文章沒有被排除。
> 修正：`wt_manifest_scan.py`（內容 hash 判定）。排除該文章後以 95 篇重算：
> - p4e1 強弱配對差 +0.001344，97.5% CI [+0.000615, +0.002072]；int4 -0.000474，97.5% CI [-0.001162, +0.000183]。
> - L1.E9 上一輪誤排的重現：int4 vs p6e1 +0.00355 [+0.00194, +0.00516]（決定性、仍誤排）；int6 vs p6e1 +0.00126 [-0.00001, +0.00251]（CI 含 0）；int8 vs p6e1 +0.00040 [-0.00045, +0.00125]（CI 含 0）；int8 vs p8e1 +0.00024 [-0.00060, +0.00110]（CI 含 0）。
> - 新 tap 排序：決定性 63，BC 正確 54（原 61／53）；L1.E9 決定性 9，BC 正確 8（原 10／8）。誤排：L1.E37 int8 vs p6e1、L6.E59 int4 vs int6、L6.E59 int4 vs int8、L6.E59 int4 vs p6e1、L6.E59 int4 vs p8e1、L2.E14 int4 vs int6、L2.E14 int4 vs int8、L2.E14 int4 vs p6e1、L2.E14 int4 vs p8e1、L1.E9 int4 vs p6e1。
> - 詳見 `correction_20261023/correction.json` 與 `correction.md`；原始輸出與其 hash 保留不變（本註記改變了本檔 hash，原 hash 見 correction.json）。


- 日期：2026-10-23
- 程式：`run_olmoe_token_spike_diag.py`（sha256 `8a81bc07…4dca`），所有階段使用同一版本，沒有修正紀錄。
- 預先登記：`plan.json`、`plan.md`（02:58:51），在 discovery 與任何確認資料之前。
- 凍結：
  - `freeze.json`（03:00:40）：tap、spike 門檻、outlier channel、BC 預測、候選格式；
  - `data_freeze.json`（03:01:49）：新文章。
- 參考：Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs（arXiv 2405.14428）。只借用它的 token-wise 尺度量測，沒有實作 QFeM／QFeP。
- 本輪只做診斷：criterion、量化器、搜尋、knee、pipeline 都沒有修改，沒有 fine-tune，也沒有任何干預。

## 結論摘要

1. **OLMoE 有 token-level spike，但只部分重現論文的描述。**
   - **在 Down 輸入 h = SiLU(gate)·up 上確實有 token spike。**
     - 單一 expert 內的 R_token 最大為 255，中位數 8.2。
     - 1013 個 expert 中，有 67 個出現 ≥ 20× 中位數的強 spike，4 個出現 ≥ 100× 的極端 spike。
   - **觀察到兩類 spike。**
     - **位置 0 spike**：每個窗口的第一個 token 都會在固定的 expert 與 channel 上出現強 spike，202 / 202 個窗口都是如此：L1.E11（c 808）、L1.E18（c 627）、L2.E30（c 742）、L3.E39（c 438）。OLMoE 沒有 BOS，窗口也是從文章中間切出，所以這與 token 身分無關。
     - **特定 token spike**：L1.E9 的 401 個強 spike 全部是 ' the'（350）或 ' a'（51），都在 channel 915。
   - **論文提到的分隔符號沒有在 OLMoE 重現。** 強 spike 中，換行與標點的比例都低於 base rate（lift 約 0.35）。
2. **層分布：集中在前段，後段沒有重現。**
   - 前段：L0–L1 的 expert R_token 中位數 16.0，128 個 expert 中有 43 個有強 spike。L1 最嚴重（中位數 23.6，35 / 64 個 expert 有強 spike）。
   - 中段：L2、L3、L4、L6 也有少數特定 expert。
   - 後段：L14–L15 的中位數 5.3，126 個 expert 中沒有任何強 spike。
   - FP32 與 q0 的結果幾乎相同：R_token 的 Spearman 為 0.95。
3. **channel outlier 與 token spike 的關係**
   - 兩者分子相同，在 expert 之間高度相關（Spearman 0.79）。
   - 但有兩種不一致：
     - **持續性大 channel**：例如 L15.E33，R_channel 15.7，R_token 只有 2.7，確認資料上沒有任何 spike row。
     - **位置 0 spike**：固定落在單一 channel，因此兩個指標同時變大，例如 L2.E30 的 R_channel 307、R_token 73。
4. **spike 與損傷（4 組匹配配對，只對這些 tap 成立）**
   - **p4e1**：token 強的 tap 損傷較大。配對差 +1.37e-3，97.5% CI [+6.5e-4, +2.1e-3]，在 Bonferroni 下確認。
   - **int4**：沒有一致差異。配對差 −4.9e-4，97.5% CI [−1.17e-3, +1.75e-4]；四組配對方向不一（+1.5e-3、−1.2e-3、0、−2.3e-3）。
5. **BC 排序：新 tap 上，spike 沒有讓 BC 更容易排錯。**
   - 新 tap 有 61 / 180 個決定性配對，BC 對了 53 個（86.9%）。
     - Posit–Posit 15 / 15，同位元 INT／Posit 8 / 8，跨家族 29 / 33，INT–INT 9 / 13。
   - 8 個誤排全部集中在兩個 tap：
     - L6.E59（token 強）；
     - L2.E14（channel 主導，確認資料上 6489 個 rows 中只有 1 個 spike row）。
   - 這 8 個都是「int4 讓 NLL 變好」的情況：int4 ΔNLL 分別為 −2.2e-3 與 −1.9e-3，CI 都排除 0。BC 的成本恆為正，無法預測量化反而改善，所以這類誤排與 spike 沒有對應關係。
   - token 強減弱的誤排率差 +0.14，tap bootstrap CI [0, 0.43]，**不成立**。
6. **L1.E9：誤排可重現，但目前只在 L1.E9 看到。**
   - 上一輪的 4 個決定性誤排中，有 2 個在新資料上再次決定性誤排：
     - int4 vs p6e1：+3.5e-3，99% 也成立；
     - int6 vs p6e1：+1.2e-3。
   - 另外 2 個方向相同，但 CI 含 0。
   - 機制線索：在 calibration 上，L1.E9 的 22 個 spike row（全是 ' the'／' a'）佔了 BC p6e1 變異項的 98%，但實測 p6e1 損傷很小。也就是 BC 在這些 spike token 上高估了 Posit 成本。
   - 新 tap 沒有出現這種誤排。但同等極端的位置 0 expert（L1.E11、L1.E18、L2.E30）不在隨機抽中的 tap 裡，所以**無法判定**這個現象是否只屬於 L1.E9。
7. **要建立因果，需要下一輪的干預。** 本輪只看到關聯，不修改 criterion。第 7 節提出預先登記用的設計。

## 1. 設定

### 1.1 資料

| 用途 | 資料 |
|---|---|
| discovery | BC calibration（10 × 128，WikiText-2 validation，連續文本）＋上一輪的確認資料（64 篇 WikiText-103 × 3 窗口，結果已看過）＝ 202 窗口 |
| BC token 分解 | 只用 calibration 窗口 |
| 確認 | 凍結後才抽的 96 篇新 WikiText-103 train 文章 × 3 窗口（128 token） |

確認資料的抽取：
- 排除 6074 篇歷史文章。
- 造訪 109 篇，其中 13 篇因近似重複跳過。

### 1.2 指標

| 指標 | 定義 |
|---|---|
| s_t | max_c \|h[t,c]\| |
| R_token | max_t s_t ÷ median_t s_t（每個 expert） |
| R_channel | max_c(max_t \|h[t,c]\|) ÷ median_c(max_t \|h[t,c]\|) |
| R_channel_cal | 同 R_channel，只用 calibration（＝上一輪的 R_cross） |

- 兩者分子相同；log R_token − log R_channel 等於兩個分母的差。
- spike 門檻（主要）：s_t ≥ 6 × median_t s_t。強 spike ≥ 20×，極端 spike ≥ 100×。

### 1.3 量化事實

- INT activation 是 per-tensor 的 scale／zero-point，來自 FP32 calibration 的 range。
  - calibration 中的 spike 會撐大所有 token 的步長；
  - 新文本中超出 range 的 spike 會被 clipping。
- Posit 沒有 range，INT 的 range 機制不適用。

### 1.4 狀態

- FP32 與 q0（`q0_knee.json`）都用 dynamic route。
- q0 knee 已經把幾個極端 spike expert 的 Down activation 設為 f16：L1.E9、L1.E11、L1.E18、L2.E30、L6.E59。

## 2. token spike 是否存在、在哪裡（discovery）

### 2.1 整體（expert rows ≥ 512：FP32 1013 個，q0 1013 個）

| 指標 | FP32 p50 / p90 / p99 / max | q0 p50 / p90 / p99 / max |
|---|---|---|
| R_token | 8.2 / 16.4 / 43.7 / 255.5 | 8.0 / 16.0 / 43.7 / 248.5 |
| R_token@256（固定樣本量） | 5.7 / 10.4 / 37.1 / 228.9 | 5.6 / 10.3 / 36.5 / 223.8 |
| R_channel | 7.5 / 15.1 / 36.6 / 306.7 | 7.3 / 14.4 / 40.1 / 309.7 |
| R_channel_cal | 12.5 / 26.7 / 95.2 / 636.9 | 12.3 / 26.7 / 99.1 / 637.5 |
| 窗口內 R_token 中位數（rows ≥ 8） | 2.6 / 3.8 / 9.3 / 153 | 2.5 / 3.8 / 9.5 / 151 |
| 窗口內 R_token 最大值 | 7.8 / 18.1 / 56.6 / 348 | 7.7 / 18.1 / 54.3 / 303 |
| spike 數／1000 rows | 1.6 / 14.0 / 87.0 / 146 | 1.4 / 14.9 / 85.9 / 150 |
| spike rows 的能量占比 | 0.9% / 8.6% / 57% / 99.5% | 0.8% / 8.4% / 57% / 99.5% |

**spike 事件數**（FP32 總 rows 3,309,568）

| 門檻 | 事件數 | 涉及的 expert 數 |
|---|---|---|
| ≥ 6× | 20,147 | 805 |
| ≥ 20× | 2,045 | 67 |
| ≥ 100× | 428 | 4 |

q0 分別為 19,657 / 1,954 / 401 個事件。

**FP32 vs q0 的 Spearman**：R_token 0.95，R_channel 0.94。

**覆蓋量**
- 每個 expert 的 rows 中位數為 3,056。
- rows 與各指標的 Spearman：
  - R_token：0.08；
  - R_token@256：−0.04；
  - R_channel：−0.19；
  - spike 率：−0.03。
- 在 rows ≥ 512 的範圍內，max／median 受樣本量的影響很小（見 `figures/coverage.png`）。
- 選點時仍匹配了 log rows。

### 2.2 層分布（FP32）

所有 expert 的分布見 `figures/layer_distributions.png`，層級 MoE 類比見 `figures/layer_level.png`。

| 層 | R_token 中位數 / p90 / max | R_channel 中位數 / max | 強 spike 的 expert 數 | 極端 spike 的 expert 數 | 層級 R_token^layer（max 所在） |
|---|---|---|---|---|---|
| 0 | 13.2 / 20.8 / 33.2 | 11.8 / 34.3 | 8 | 0 | 10 |
| **1** | **23.6 / 51.7 / 255.5** | 15.0 / 228 | **35** | **3** | **98（E9）** |
| 2 | 11.2 / 15.6 / 72.5 | 9.0 / 307 | 1 | 0 | 59（E30，位置 0） |
| 3 | 9.9 / 14.7 / 27.1 | 7.9 / 40.7 | 1 | 0 | 10 |
| 4 | 8.7 / 17.8 / 144 | 7.6 / 223 | 5 | 1 | 47（E27） |
| 5 | 8.8 / 22.6 / 31.3 | 8.3 / 35.7 | 8 | 0 | 14 |
| 6 | 7.7 / 13.4 / 65.7 | 6.9 / 98.3 | 5 | 0 | 30（E59） |
| 7 | 7.2 / 10.9 / 31.2 | 6.6 / 25.9 | 1 | 0 | 9 |
| 8 | 7.0 / 10.2 / 20.4 | 6.2 / 22.3 | 1 | 0 | 7 |
| 9 | 8.0 / 11.9 / 25.2 | 7.3 / 46.0 | 1 | 0 | 7 |
| 10 | 8.1 / 13.4 / 17.5 | 7.5 / 24.2 | 0 | 0 | 6 |
| 11 | 7.4 / 11.7 / 13.2 | 7.3 / 16.8 | 0 | 0 | 5 |
| 12 | 6.9 / 10.9 / 15.4 | 6.5 / 13.8 | 0 | 0 | 7 |
| 13 | 7.3 / 10.7 / 22.9 | 6.7 / 16.1 | 1 | 0 | 7 |
| 14 | 5.9 / 7.9 / 11.3 | 5.7 / 15.2 | 0 | 0 | 5 |
| 15 | 4.2 / 8.8 / 14.9 | 6.8 / 24.5 | 0 | 0 | 8 |

**前段／中段／後段（FP32）**

| 分段 | R_token 中位數 | 強 spike 的 expert |
|---|---|---|
| 前段 L0–L1 | 16.0 | 43 / 128 |
| 中段 L2–L13 | 8.0 | 24 / 759 |
| 後段 L14–L15 | 5.3 | 0 / 126 |

- 前段集中成立，主要來自 L1。
- 後段集中**沒有**重現，L14–L15 反而最低。

### 2.3 兩類強 spike（FP32，≥ 20×，共 2,045 個）

**位置 0 spike**
- 共 867 個，占 42%。所有 rows 中，位置 0 只占 0.78%。
- 集中在少數 expert 與固定 channel：

  | expert | 出現的窗口數 | channel |
  |---|---|---|
  | L1.E11 | 201 / 202 | 808 |
  | L1.E18 | 195 / 202 | 627 |
  | L2.E30 | 202 / 202 | 742 |
  | L3.E39 | 202 / 202 | 438 |
  | L1.E45 | 45 | — |

- 202 個窗口全部都有位置 0 的強 spike。
- 窗口的第一個 token 是文章中間的任意 token，OLMoE 也沒有 BOS，所以這是「序列第一個位置」的效應，不是 BOS 或特定 token 的效應。

**非位置 0 spike**
- 主要來自：
  - L1.E9：401 個；
  - L6.E9：99 個，c 110；
  - L1.E38：65 個；
  - L6.E49：65 個，c 1017；
  - L1.E14：64 個；
  - L1.E3：58 個；
  - L6.E59：31 個，c 750。

**L1.E9 的細節**
- 401 個強 spike 全在 channel 915，token 只有 ' the'（350）與 ' a'（51），都不在位置 0。
- discovery 中所有 1,350 個 ' the' 在 L1 都路由到 E9，但只有 26% 成為強 spike。
- 強 spike 的比例與 token 在窗口中的位置有關：

  | 條件（' the'／' a'） | rows | 強 spike 率 |
  |---|---|---|
  | 窗口內第一個 ' the'／' a' | 200 | 83% |
  | 之後的 ' the'／' a' | 1,537 | 15% |
  | 位置 0–15 | 211 | 66% |
  | 位置 64–127 | 872 | 13% |

- 「首次出現」與「位置靠前」高度重疊，本輪無法分開。這只是共現描述，不是因果。

**強 spike 的 gate／up 來源**（只是來源描述，不是誤差放大的證據）
- 相對於該 expert 的典型 token，100% 的強 spike 都是 SiLU(gate) 與 up 同時變大。
- gate 增幅較大：Δlog 中位數 gate 2.16、up 1.55；87.6% 的 gate 增幅大於 up。
- 極端 spike 中，gate 增幅大於 up 的比例為 99.5%。

### 2.4 spike 身分與 base rate（FP32）

| 項目 | ≥ 6×（20,147） | ≥ 20×（2,045） | 所有 rows |
|---|---|---|---|
| 位置 0 | 10.1% | 42.4% | 0.78% |
| 位置 1–3 | 4.9% | 2.4% | 2.3% |
| 位置 64–127 | 38.9% | 21.8% | 50% |
| 窗口內首次出現 | 77.4% | 82.4% | 66.3% |
| 換行（lift） | 1.6%（1.52） | 0.4%（0.36） | 1.1% |
| 標點（lift） | 8.4%（0.59） | 5.0%（0.35） | 14.3% |
| 空白（lift） | 2.6%（2.41） | 3.5%（3.22） | 1.1% |
| 一般詞（lift） | 85.5%（1.06） | 90.2%（1.12） | 80.8% |
| 不同 token ID 數 | 4,041 | 542 | — |

- 次數最多的 token 是 ' the'：≥ 6× 有 1,324 次，≥ 20× 有 406 次。
- 但這是因為 ' the' 本身常見：以每次出現計，spike 率只有 0.8%（≥ 6×）與 0.2%（≥ 20×）。
- 沒有任何單一 token ID 普遍造成 spike。L1.E9 的 ' the'／' a' 是限於單一 expert 的現象。
- 完整清單見 `spike_tokens_fp32.csv`（≥ 6×），強度最高的 40 個見 `analysis.json` 的 `top_strong_spikes`。

### 2.5 channel outlier vs token spike

| 項目 | 結果 |
|---|---|
| expert 之間的 Spearman(R_token, R_channel) | 0.79；R_token@256 對 R_channel 為 0.76 |
| 強 spike 第二大 \|h\| ÷ s_t 的中位數 | 0.10（spike 幾乎只在一個 channel） |

**不一致的例子**

| 類型 | expert | R_token | R_channel |
|---|---|---|---|
| channel 主導 | L15.E33 | 2.7 | 15.7 |
| channel 主導 | L2.E14 | 6.4 | 20.4 |
| 位置 0 spike | L2.E30 | 73 | 307 |
| 位置 0 spike | L1.E18 | 204 | 228 |

- channel 主導的 expert，是某個 channel 對很多 token 都大，所以 token 的 max 中位數本身就大。
- 位置 0 spike 固定落在單一 channel，所以兩個指標同時變大。

## 3. 量化誤差分布（discovery，FP32，固定 range）

### 3.1 spike rows（≥ 6×，占所有 rows 0.61%）的誤差

| 格式 | spike rows 的誤差能量占比 p50 / p90 | 每元素 MSE 比（spike ÷ 一般）p50 / p90 | spike × outlier ch | 一般 × outlier ch |
|---|---|---|---|---|
| int4 | 0.9% / 5.4% | 2.7 / 7.7 | 0.2% | 0.4% |
| int6 | 2.5% / 14.4% | 8.2 / 49.6 | 1.2% | 1.0% |
| int8 | **15.5% / 53.2%** | **79 / 295** | 8.5% | 5.4% |
| p4e1 | 0.8% / 5.6% | 2.5 / 13.2 | 3.4% | 4.9% |
| p6e1 | 1.5% / 11.9% | 4.9 / 36.2 | 6.3% | 9.2% |
| p8e1 | 1.6% / 14.8% | 5.2 / 39.2 | 7.7% | 10.1% |

- 占比是對 801 個有 spike 的 expert 取分位數。
- 最後兩欄是彙整的 2×2 誤差能量占比。

### 3.2 INT clipping，依來源分開

- **calibration 窗口**：沒有任何 clipping，因為 range 就來自這批資料。
- **新文本**（上一輪的 192 個窗口）：spike rows 超出 calibration range，clipped 誤差占 spike rows 誤差能量的：
  - int4 58%；
  - int6 86%；
  - int8 98%。
- 一般 rows 的對應比例是 0.9% / 4.8% / 38%。
- 所以 int8 在 spike row 上的誤差幾乎全來自 clipping，int4 則仍以捨入為主。

### 3.3 BC token 分解（calibration rows，spike rows 占 0.65%）

| 格式 | spike rows 的變異項占比 p50 / p90 | spike rows 的偏差分攤占比 p50 / p90 | 偏差項占 T 的比例 p50 / p90 |
|---|---|---|---|
| int4 | 1.6% / 8.1% | 1.1% / 4.9% | 7.9% / 15.5% |
| int6 | 1.2% / 5.6% | 1.0% / 4.4% | 1.4% / 3.6% |
| int8 | 1.2% / 5.1% | 1.2% / 5.9% | 1.0% / 1.8% |
| p4e1 | 2.4% / 16.1% | 1.3% / 13.1% | 2.0% / 4.4% |
| p6e1 | 4.0% / **35.2%** | 2.5% / 31.2% | 1.0% / 1.8% |
| p8e1 | 4.7% / **41.8%** | 3.7% / 28.8% | 0.9% / 1.6% |

- 統計範圍：274 個在 calibration 中有 spike row 的 expert。
- **可加性**：
  - 變異項可以精確拆到每個 row：v_t = Σ_c G2_c/n·(e_tc − μ_c)²，μ 由所有 row 決定。
  - 偏差項 Σ_c GS_c μ_c² 有跨 row 項，只能分攤成 b_t = Σ_c GS_c μ_c e_tc / n。分攤後加總精確（v + b 與 T 的相對差 ≤ 7e-16），但單一 row 的值可為負。
  - **兩者都不是單一 token 的真實 BC 成本**，也沒有對 G2／GS 做 token 分解（cache 只有 channel 彙整）。
- **選中 tap 的 spike rows 占 Posit 變異項的比例**：
  - L1.E9：22 / 175 rows 是 spike，占 p4e1 90%、p6e1 98%、p8e1 99.8%（int4 68%）。
  - L6.E59：6 / 184 rows 是 spike，占 p4e1 84%、p6e1 93%、p8e1 97%（int4 8%）。
  - 其他新 tap 的 spike rows 占比都 ≤ 23%。
- **與 static cache 的比對**（1017 個 expert）：
  - 重算的 calibration rows 數與 cache 完全相同。
  - 重算的 T 與 cache 的相對差中位數約 1e-7，最大 5.4e-4。
  - 逐 channel 的 S1 不是逐位元相同：signed 加總會互相抵消，最大差可到該 slot 最大 |S1| 的 53%；S2 最大差 1.6%。
  - 這是 batch 組成不同造成 h 的極小差異，進而改變部分捨入結果；不影響確認實驗，因為確認實驗的 BC 預測直接使用 cache。

## 4. 確認：選點（凍結於確認資料之前）

**合格條件**
- searched expert 的 Down 輸入；
- calibration rows ≥ 64，discovery rows ≥ 512；
- 排除上一輪用過的 24 個 expert。

共 801 個合格；R_token 的 p90 為 17.2、p50 為 8.4，R_channel 的 p90 為 15.1。

| id | 角色 | tap | R_token | R_channel | R_channel_cal | rows（全部／cal） | q0 act | 配對品質 |
|---|---|---|---|---|---|---|---|---|
| 0 | 配對 0 強 | L0.E46 | 33.2 | 24.2 | 56.5 | 4747 / 193 | p8e1 | – |
| 1 | 配對 0 弱 | L0.E12 | 7.0 | 5.1 | 11.6 | 4615 / 65 | int7 | 同層、同位元類別 |
| 2 | 配對 1 強 | L1.E49 | 27.4 | 12.5 | 24.6 | 3099 / 108 | p8e1 | – |
| 3 | 配對 1 弱 | L1.E37 | 8.0 | 6.3 | 15.1 | 5117 / 251 | p8e1 | 同層、同類別，\|Δlog rows\| 0.50 |
| 4 | 配對 2 強 | L1.E23 | 28.9 | 19.4 | 34.9 | 3885 / 218 | p8e1 | – |
| 5 | 配對 2 弱 | L0.E9 | 7.4 | 5.5 | 11.4 | 3901 / 568 | p8e1 | **跨一層**（L1 沒有可用的弱 tap） |
| 6 | 配對 3 強 | L6.E59 | 65.7 | 98.3 | 181 | 3848 / 184 | **f16** | – |
| 7 | 配對 3 弱 | L6.E44 | 7.5 | 6.4 | 9.4 | 3868 / 200 | p8e1 | 同層，**位元類別不同** |
| 8 | token 主導 | L0.E30 | 17.4 | 9.2 | 16.4 | 3271 / 135 | p8e1 | R_channel 百分位 0.69 |
| 9 | token 主導 | L0.E39 | 17.3 | 10.7 | 6.2 | 2522 / 67 | int6 | R_channel 百分位 0.77 |
| 10 | channel 主導 | L15.E33 | 2.7 | 15.7 | 25.0 | 687 / 72 | p8e1 | R_token 百分位 0.005 |
| 11 | channel 主導 | L2.E14 | 6.4 | 20.4 | 23.4 | 4709 / 252 | p8e1 | R_token 百分位 0.23 |
| 12 | 已知案例 | L1.E9 | 255.5 | 215 | 637 | 3757 / 175 | f16 | 單獨報告 |

**選點的限制**
- token 主導案例的分歧不大：R_channel 百分位仍有 0.69 與 0.77，因為兩個指標高度相關。
- 位置 0 的極端 expert 都在強 pool 中，但這次的隨機抽樣沒抽到：L1.E11、L1.E18、L2.E30；L3.E39 則因 calibration rows 只有 56 而不合格。

## 5. 確認：損傷與排序（96 篇 × 3 窗口，FP32 背景，dynamic route）

### 5.1 各 tap 的實測 ΔNLL（nat／token，相對 FP32）

| tap | 角色 | int4 | int6 | int8 | p4e1 | p6e1 | p8e1 | 決定性 | 誤排 | 確認資料 R_token | 確認資料 spike rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0.E46 | 配對強 | +9.4e-4 | +4.5e-4 | −2.9e-4 | **+4.1e-3** | −9.4e-5 | −1.9e-4 | 7 | 0 | 32.4 | 114 / 6978 |
| L0.E12 | 配對弱 | −6.0e-4 | −1.1e-4 | −1.4e-4 | +1.4e-3 | −3.2e-4 | −5.7e-5 | 5 | 0 | 8.1 | 9 / 7064 |
| L1.E49 | 配對強 | +1.3e-4 | +2.0e-4 | +8.6e-5 | +1.4e-3 | −2.1e-5 | −9.9e-5 | 5 | 0 | 25.7 | 314 / 4808 |
| L1.E37 | 配對弱 | +1.3e-3 | −3.9e-4 | +1.1e-4 | +1.5e-4 | −2.9e-4 | −2.8e-5 | 4 | 0 | 9.8 | 24 / 6622 |
| L1.E23 | 配對強 | +4.7e-4 | +7.3e-4 | −2.2e-4 | +1.8e-3 | +1.3e-4 | −2.6e-4 | 7 | 0 | 28.5 | 304 / 5545 |
| L0.E9 | 配對弱 | +4.9e-4 | +9.0e-5 | −5.2e-6 | +4.4e-4 | +7.0e-5 | −8.4e-5 | 0 | – | 7.0 | 4 / 4907 |
| L6.E59 | 配對強 | **−2.2e-3** | +1.9e-4 | +2.7e-4 | +5.3e-4 | −4.0e-5 | −6.1e-5 | 7 | **4** | 66.2 | 46 / 5521 |
| L6.E44 | 配對弱 | +8.0e-5 | +1.9e-4 | −2.5e-5 | +4.3e-4 | +1.0e-5 | −1.3e-5 | 4 | 0 | 7.0 | 2 / 5596 |
| L0.E30 | token 主導 | +1.1e-3 | +7.3e-5 | −1.4e-4 | +2.3e-3 | +3.6e-4 | +1.7e-4 | 8 | 0 | 15.9 | 119 / 4393 |
| L0.E39 | token 主導 | +1.5e-4 | −1.2e-4 | +1.7e-5 | +7.1e-4 | +1.2e-4 | −2.7e-5 | 0 | – | 25.2 | 118 / 3562 |
| L15.E33 | channel 主導 | +1.2e-4 | +1.3e-5 | +1.6e-6 | −7.0e-6 | +5.3e-6 | +1.0e-6 | 5 | 0 | 2.6 | 0 / 741 |
| L2.E14 | channel 主導 | **−1.9e-3** | +1.7e-4 | −1.6e-4 | **+3.5e-3** | +1.3e-5 | −2.8e-4 | 9 | **4** | 6.6 | 1 / 6489 |
| L1.E9 | 已知案例 | +3.3e-3 | +1.0e-3 | +1.4e-4 | **+3.0e-2** | −2.2e-4 | −1.1e-4 | 10 | 2 | 254.9 | 644 / 5053 |

- 確認資料的 R_token 與 discovery 高度一致：13 個 tap 的 Spearman 為 0.97。spike 程度在新文本上可以重現。
- 13 個 tap 中只有 2 個，6 個 ΔNLL 的 CI 全部含 0。
- 下游路由變化：int4 在 2–11%，p4e1 在 3–18%。L15 沒有下游路由。

### 5.2 排序（P1–P3；新 tap 共 12 個，L1.E9 另列）

**依配對類型**

| 組別 | 配對 | 決定性 | 不可判定 | BC 平手 | BC 正確 / 決定性 |
|---|---|---|---|---|---|
| 新 tap 全部 | 180 | 61 | 119 | 0 | **53 / 61（86.9%）** |
| 新 tap 全部（99% CI） | 180 | 45 | 135 | 0 | 38 / 45（84.4%） |
| INT–INT | 36 | 13 | 23 | 0 | 9 / 13 |
| Posit–Posit | 36 | 15 | 21 | 0 | **15 / 15** |
| 跨家族 | 108 | 33 | 75 | 0 | 29 / 33 |
| 同位元 INT／Posit | 36 | 8 | 28 | 0 | **8 / 8** |

- 同位元的細項：int4–p4e1 7 / 7；int6–p6e1 1 / 1；int8–p8e1 沒有決定性配對。
- 實測差 < 1e-6 的配對只有 1 個。

**依 tap 角色**

| 角色 | 決定性 | BC 正確 |
|---|---|---|
| 配對強 | 26 | 22 |
| 配對弱 | 13 | 13 |
| token 主導 | 8 | 8 |
| channel 主導 | 14 | 10 |
| 已知案例 L1.E9 | 10 | 8 |

L1.E9 的細項：同位元 1 / 2，跨家族 4 / 6。

### 5.3 新 tap 的 8 個誤排

| tap | 配對 | 實測差（前 − 後） | 95% CI | 99% 也決定性 |
|---|---|---|---|---|
| L6.E59 | int4 vs int6 | −2.4e-3 | [−3.3e-3, −1.5e-3] | 是 |
| L6.E59 | int4 vs int8 | −2.5e-3 | [−3.7e-3, −1.3e-3] | 是 |
| L6.E59 | int4 vs p6e1 | −2.2e-3 | [−3.4e-3, −9.6e-4] | 是 |
| L6.E59 | int4 vs p8e1 | −2.2e-3 | [−3.4e-3, −9.4e-4] | 是 |
| L2.E14 | int4 vs int6 | −2.1e-3 | [−3.2e-3, −9.6e-4] | 是 |
| L2.E14 | int4 vs int8 | −1.8e-3 | [−2.8e-3, −6.6e-4] | 是 |
| L2.E14 | int4 vs p6e1 | −2.0e-3 | [−3.1e-3, −7.5e-4] | 是 |
| L2.E14 | int4 vs p8e1 | −1.7e-3 | [−2.8e-3, −4.5e-4] | 否 |

**共同特徵**
- 8 個誤排都是「int4 比 FP32 和其他格式都好」。BC 的成本恆為正，並預測 int4 是最差的 INT 格式。
- 兩個 tap 的 R_channel 都偏高：L6.E59 為 98（cal 181），L2.E14 為 20（cal 23）。
- 但只有 L6.E59 有 token spike；L2.E14 在確認資料上 6489 個 rows 中只有 1 個 spike row。
- 所以這類誤排不是 spike 特有的。它與先前 OLMoE 觀察到的「誤差方向剛好有利」一致，但本輪沒有檢驗這個機制。

### 5.4 token 強 vs 弱（P4，4 組匹配配對；結論只限這些 tap）

| 格式 | 平均配對差 ΔNLL(強) − ΔNLL(弱) | 97.5% CI（Bonferroni 2 個檢定） | 各配對 |
|---|---|---|---|
| **int4（主要）** | −4.9e-4 | [−1.17e-3, +1.75e-4] **不成立** | +1.5e-3*、−1.2e-3*、−1.9e-5、−2.3e-3* |
| **p4e1（主要）** | **+1.37e-3** | **[+6.5e-4, +2.1e-3] 確認** | +2.7e-3*、+1.2e-3、+1.4e-3、+1.0e-4 |
| int6（次要） | +4.5e-4 | [−7.9e-5, +1.0e-3] | – |
| p6e1（次要） | +1.3e-4 | [−1.1e-4, +3.7e-4] | – |
| int8（次要） | −2.3e-5 | [−3.6e-4, +3.2e-4] | – |
| p8e1（次要） | −1.1e-4 | [−2.3e-4, +8.9e-6] | – |

- 「*」表示該配對自己的 95% CI 排除 0。
- **誤排率（P4b）**：強 tap 0.14（4 個誤排都在 L6.E59），弱 tap 0（L0.E9 沒有決定性配對，不計）。差 +0.14，tap bootstrap CI [0, 0.43]，**不成立**；強、弱各只有 3–4 個 tap。
- **p4e1 的差異不能直接歸因於 spike rows 的誤差**：在確認資料上，4 個 token 強 tap 中有 3 個，spike rows 只占 p4e1 誤差能量的 1.6–6%（L6.E59 為 49%）。
- **int4 的配對差異方向不一**：其中 L6.E59 的 int4 是「有利」方向，且它的配對位元類別不同。

### 5.5 R_token vs R_channel（P5，12 個新 tap，描述性）

| Spearman | int4 ΔNLL | p4e1 ΔNLL | 誤排率（n = 10） |
|---|---|---|---|
| R_token | 0.22 | 0.39 | 0.16 |
| R_token@256 | 0.13 | 0.43 | 0.16 |
| R_channel | −0.29 | 0.41 | 0.62 |

- **誤排**：R_token 沒有在 R_channel 之外提供額外解釋。誤排發生在兩個 R_channel 高的 tap，其中一個沒有 token spike。
- **p4e1 損傷**：兩個指標的關聯程度相近。
- **不一致案例逐一看**：
  - token 主導的 L0.E30：8 個決定性配對全對；int4 +1.1e-3、p4e1 +2.3e-3，正向損傷，BC 排序正確。
  - token 主導的 L0.E39：沒有決定性配對。
  - channel 主導的 L15.E33：效果都很小，5 / 5 正確。
  - channel 主導的 L2.E14：4 個「int4 有利」誤排。

### 5.6 已知案例 L1.E9 的重現

| 上一輪的誤排 | 本輪實測差 | 95% CI | 本輪 |
|---|---|---|---|
| int4 vs p6e1 | +3.5e-3 | [+1.9e-3, +5.1e-3] | **重現**（99% 也成立） |
| int6 vs p6e1 | +1.2e-3 | [+2.3e-5, +2.5e-3] | **重現**（99% 不成立） |
| int8 vs p6e1 | +3.6e-4 | [−4.9e-4, +1.2e-3] | 方向相同，不可判定 |
| int8 vs p8e1 | +2.5e-4 | [−5.9e-4, +1.1e-3] | 方向相同，不可判定 |

**4 個中重現 2 個。** BC 對 L1.E9 預測 p6e1 的成本（3.83）高於 int4（1.43），但實測 p6e1 幾乎沒有損傷（−2.2e-4）。

這與第 3.3 節的分解一致：L1.E9 的 BC Posit 成本幾乎全來自 ' the'／' a' 的 spike rows（p6e1 98%）。在確認資料上，spike rows 同樣占 p4e1 誤差能量的 96%、p8e1 的 99.8%，但 p6e1／p8e1 的實測損傷很小。

## 6. 確認資料上 tap 處的誤差（spike rows 用凍結門檻）

**int8 的誤差能量與損傷不一致**
- 在 token 強的 tap，int8 的誤差能量大部分是 spike rows 的 clipping：

  | tap | int8 誤差能量中 spike rows clipping 的占比 |
  |---|---|
  | L0.E39 | 91.5% |
  | L1.E49 | 55% |
  | L0.E30 | 35% |
  | L1.E23 | 32% |
  | L0.E46 | 20% |

- 但這些 tap 的 int8 實測 ΔNLL 都在 ±3e-4 以內。

**Posit 誤差集中在 spike rows 或 outlier channel**
- L6.E59：spike rows 占 p8e1 誤差能量 76%，outlier channel 占 77%。
- L1.E9：分別為 99.8% 與 99.7%。
- L15.E33：沒有 spike rows，但 outlier channel 占 95%（持續性大 channel）。

這些都是誤差能量的占比，不是實測損傷的占比。

## 7. 判定與下一輪建議

**問題 1：OLMoE 是否重現論文的 token spike？**
- 部分重現。
  - Down 輸入有 token-level spike；
  - 前段層（L1）最強；
  - 序列第一個位置的效應在沒有 BOS 的情況下仍然出現。
- 沒有重現的部分：
  - 後段層的集中；
  - 換行、標點等分隔符號的偏好。
- MoE 的特點是 spike 集中在少數 expert 與固定 channel。

**問題 2：spike 是否造成損傷？**
- p4e1：在 4 組配對上，token 強的損傷較大（已確認），但無法歸因於 spike rows 本身的誤差。
- int4：沒有一致差異。

**問題 3：spike 位置是否更容易誤排？**
- 新 tap 上沒有證據：Posit 與同位元排序全對。
- 唯一的誤排類型是「int4 有利」，也出現在沒有 spike 的 tap。

**問題 4：L1.E9 是否孤立？**
- 它的 Posit 高估誤排可以重現（2 / 4），並與 BC 成本被 spike rows 主導一致。
- 但同等極端的位置 0 expert 沒有被量測，所以**無法判定**是否孤立。

**問題 5：對 pipeline 的影響**
- q0 knee 已經把 L1.E9、L1.E11、L1.E18、L2.E30、L6.E59 的 Down activation 設為 f16，這些誤排是在低位元候選之間，不影響 q0 的配置。
- 本輪結果不支持修改 criterion。

**建議下一輪（需重新預先登記，使用新資料）**

1. **spike row 的 oracle 干預**
   - 對象：極端 spike expert（L1.E9、L1.E11、L1.E18、L2.E30、L6.E59），與同層的匹配控制 tap。
   - 做法：把 spike rows（凍結門檻）保留為 FP32，對照同數量的隨機 rows。
   - 格式：int4、int6、p4e1、p6e1。
   - 檢驗：
     - spike rows 是否承擔實測損傷；
     - 移除 spike rows 後，BC 的 Posit 高估是否消失。
2. **BC 獨立性近似的診斷**（不修改 criterion）
   - 在同一批 expert 上做一次 backward，比較 token 配對的 Σ_t g_tc² e_tc² 與 BC 的 G2_c/n·Σ_t(e_tc − μ_c)²。
   - 看 L1.E9 channel 915 上的 spike token，梯度是否明顯小於平均。
   - 注意：記憶中 Qwen3 的 paired variance 整體排序較差，這裡只作診斷。
3. **「int4 有利」誤排**（L6.E59、L2.E14）
   - 這與 spike 無關，屬於先前的一階效應問題。
   - 先前的一階驗證已經失敗，這裡不再提出。

## 8. 控制

| 項目 | 結果 |
|---|---|
| measure 的 base 與 `dr.forward` | 差 0.0 |
| 從第 0 層重算的 identity variant | 差 0.0 |
| 是否提前停止 | 否 |
| 輸入檔與模型檔 hash | 前後一致（`input_hashes_unchanged = true`） |
| BC 重算控制 | 見第 3.3 節：n 完全相同；T 的相對差中位數約 1e-7、最大 5.4e-4；S1 不逐位元 |
| 確認資料 | 不重複、近似重複已排除；排除 6074 篇歷史文章 |
| 修正 | 沒有 |

## 9. 計時

| 階段 | 秒 | 預算（分） | GPU 峰值 |
|---|---|---|---|
| plan | 32.8 | – | – |
| discovery | 64.4 | 10 | 6.40 GB |
| select | 20.1 | 4 | – |
| freeze | 0.4 | – | – |
| data | 67.1 | 5 | – |
| measure | 1223.4 | 25 | 4.64 GB |
| analyze | 56.7 | 5 | – |
| figures | 2.2 | 3 | – |

- GPU 合計 1287.8 s（21.5 分），在 40 分預算內。
- 測量的實際耗時接近事前估計（約 20 分）。

## 10. 限制

- **配對數量少**：只有 4 組匹配配對，P4 的推論只限這些 tap，不能推廣到所有 expert。
  - 一組跨層（L1.E23 / L0.E9）；
  - 一組位元類別不同（L6.E59 f16 / L6.E44 p8e1）。
- **隨機抽樣漏掉位置 0 expert**：最典型的位置 0 spike expert 沒有被量測；token 主導的不一致案例分歧也不大。
- **解析度**：新 tap 180 個配對中有 119 個不可判定；不可判定不算排對。同一 tap 的配對不是獨立案例。
- **discovery 的組成**：calibration（WikiText-2）只占 10 / 202 個窗口，其餘是上一輪的 WikiText-103 確認資料（已看過結果，只用於 discovery）。BC 的統計只來自 calibration。
- **門檻與指標**：
  - spike 門檻以 expert 的中位數為基準。
  - max／median 在 rows ≥ 512 的範圍內與覆蓋量關聯很小，但更小的 expert 不在比較範圍。
- **gate／up 與身分分析**：都是描述性的。「首次出現」與「位置靠前」重疊，無法分開。
- **路由**：只測 dynamic route，下游路由變化混在 ΔNLL 中。
- **沒有干預**：所有機制敘述都只是關聯。

## 11. 重現

```
cd experiments/mixed_precision
O=runs/olmoe_token_spike_diag_20261023
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_token_spike_diag.py
$CPU $S --stage plan --out-dir $O
$GPU $S --stage discovery --out-dir $O
$CPU $S --stage select --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage measure --out-dir $O
$CPU $S --stage analyze --out-dir $O
$CPU $S --stage figures --out-dir $O
```

各階段都拒絕覆寫既有輸出。

### 主要 hash（sha256）

| 檔案 | hash |
|---|---|
| `plan.json` | `bbba5fe4…4263` |
| `selected_taps.json` | `44184149…20f6` |
| `freeze.json` | `a7991bcf…0f71` |
| `tokens.npy` | `4762677c…f536` |
| `segments.json` | `64007e88…b41` |
| `analysis.json` | `7a38b97c…0904` |

discovery 輸出的 hash 都在 `freeze.json`，程式 hash 在 `plan.json` 與 `freeze.json`。

### 輸出檔

| 類別 | 檔案 |
|---|---|
| discovery | `discovery_experts.csv`（layer × expert，FP32 與 q0）、`discovery_channels.npz`、`discovery_windows.json`、`discovery_bc_control.json`、`layer_level_{fp32,q0}.json`、`spike_tokens_fp32.csv` |
| discovery 原始 records | `work/records_{fp32,q0}.npz`、`work/cal_bc_records.npz`，共 427 MB |
| 選點與資料 | `selected_taps.json`、`segments.json`、`tokens.npy` |
| 測量 | `measure.json` |
| 分析 | `analysis.json`、`ranking_pairs.csv`、`per_tap.csv`、`per_window.csv` |
| 計時 | `timing.json`、`timing.jsonl` |

### 圖表

| 檔案 | 內容 |
|---|---|
| `figures/layer_distributions.png` | 每層各 expert 的 R_token 與 R_channel 分布 |
| `figures/layer_level.png` | 層級 MoE 類比 R_token^layer（FP32 與 q0） |
| `figures/R_token_vs_R_channel.png` | 各 expert 的兩個指標，標出選中的 tap |
| `figures/coverage.png` | R_token、R_token@256 與覆蓋量的關係 |
| `figures/spike_rate_by_position.png` | spike 率隨窗口位置的變化 |
| `figures/confirm_dnll.png` | 各 tap 的實測 ΔNLL 與 95% CI |
