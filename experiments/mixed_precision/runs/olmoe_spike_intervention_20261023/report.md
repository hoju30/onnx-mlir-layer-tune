# OLMoE spike rows 量化誤差的因果干預

- 日期：2026-10-23
- 程式：
  - `run_olmoe_spike_intervention.py`（sha256 `50177ebc…70c2`），所有階段使用同一版本；
  - `wt_manifest_scan.py`（`75fa0bc6…119e`）。
- 預先登記：`plan.json`、`plan.md`（04:15:01），在選點與任何確認資料之前。
- 凍結：
  - `freeze.json`（04:15:10）：tap、門檻、格式、條件、seeds、BC 反事實；
  - `data_freeze.json`（04:16:19）：新文章。
- 本輪沒有修改 BC criterion、量化器、搜尋、knee 或 pipeline，沒有 fine-tune，也沒有加入 spike 懲罰。所有干預都只是診斷。

## 結論摘要

1. **L1.E9 的 Posit 高估有直接的干預證據（類別 B）。**
   - BC 預測：保護 L1.E9 的 spike rows，可去除 p6e1 成本的 99.7%（3.83 → 0.013）。
   - 實測：NLL(P) − NLL(F) = +1.3e-4，95% CI [−4.0e-4, +6.5e-4]，完全落在 BC 換算效果（以 int4 為錨，−7.6e-3）的 ±25% 範圍內。只量化 spike rows 的 D_S 也幾乎為 0（+5e-5）。
   - int4 的情況不同：保護 spike rows 讓損傷名目上下降 1.27e-3（約為全量化損傷的 44%，95% CI 排除 0）。但 p = 0.017，沒有通過重現族的 Holm 校正，依規則列為 E（無法判定）。
2. **位置 0 的 expert 不是同一種機制。**
   - **L1.E18**：p6e1 與 L1.E9 相同，BC 預測大降、實測接近 0（B）。但 int4 的 spike rows 誤差確實造成損傷（A）：保護後下降 4.9e-3，且優於保護同數量的隨機 rows（C2 −4.1e-3，Holm 顯著）。實測降幅大於 BC 換算值（比值 1.7）。
   - **L2.E30**：兩種格式的 spike rows 誤差都造成真實而且大的損傷（A）。
     - p6e1 的全部損傷 9.2e-3 都來自 spike rows（D_P ≈ 0，D_S ≈ D_F）。BC 的方向正確，換算後的幅度約高估 4.5 倍。
     - int4 保護後下降 4.2e-3，是 BC 換算值的 5 倍；而且 BC 認為保護隨機 rows 比保護 spike rows 更有效，實測正好相反。
   - **L1.E11**：兩種格式都沒有可量測的效果。int4 的 C1 ≈ 0，CI ±5e-4；BC 預測的變化也太小，B 不可評估。p6e1 無法判定。
3. **L6.E59 的「int4 改善 NLL」在新資料上重現（−2.75e-3），但與 spike 無關。** 保護 spike rows 後幾乎不變，改善來自一般 rows。
4. **非加性（類別 C）沒有出現。** 5 個重點 tap 的 D_F − D_P − D_S，95% CI 都含 0。L2.E30 int4 最接近（−5.5e-4 [−1.16e-3, +7e-5]）。
5. **BC 在格式排序上的錯誤，來自 spike rows 的 Posit 成本。**
   - 全量化（F）時，BC 在 L1.E9、L1.E18、L2.E30 都認為 p6e1 比 int4 貴，但實測都是 int4 較差（決定性）。
   - 保護 spike rows（P）後，三個 tap 的 BC 排序都與實測一致。
6. **下一輪值得收集逐 token 梯度。** 同一個 BC 公式在 L1.E9、L1.E18 高估 spike rows 的 Posit 成本，卻在 L1.E18、L2.E30 低估 spike rows 的 int4 重要性。這符合「spike token 的梯度與 G2/n 平均不同」的假說，但本輪只量了誤差，沒有逐 token 梯度，無法判定。
7. **本輪沒有測完整配置**，不能據此宣稱 knee 有改善。q0 knee 已經把這 5 個 expert 的 Down activation 設為 f16。

## 0. 資料隔離的修正與上一輪的更正

**原因（已核對）**
- `run_olmoe_selective_sensitivity.used_wt103_all()` 只依 manifest 的 source／split 欄位判斷來源。
- `runs/olmoe_act_outlier_diag_20261023/segments.json` 沒有這兩個欄位（宣告來源為 '?'），但 64 篇的 content_sha256 全部是 WT103 文章，所以沒有被排除。
- token-spike 輪因此抽到其中一篇：#39「Brushstrokes series」，token 窗口不同，但 content_sha256 相同。
- 舊函式就算從新目錄掃描，也只會經由 token-spike 的 manifest 排除這一篇；act_outlier 的其餘 63 篇仍會漏排。

**修正**
- 新模組 `wt_manifest_scan.used_wt103_verified()`：
  - 有 hash 的條目，一律以內容 hash 是否屬於 WT103 文章來判定；宣告的標籤只做交叉檢查，不一致就報錯。
  - 沒有 hash 的條目，必須能證明是 WikiText-2（標題屬於 WT2 文章，或有 WT2 字元位置且該 run 的 plan 註明 WikiText-2），否則報錯。
- 舊函式保持不變，以維持先前各輪記錄的 code hash。

**沒有 hash 的 6 個 manifest，都有證據證明是 WikiText-2**
- calibration_stability（55 個標題屬於 WT2；1 個由 WT2 字元位置判定，plan 註明 WT2）
- crosstext、dynamic_multipoint、dynamic_route、first_order_validation、route_decomposition（標題全部屬於 WT2）

WT2 文章本來就會依標題與內容 hash 從 WT103 抽樣池整批排除，所以這些 manifest 沒有漏洞。

**驗證**

| 項目 | 結果 |
|---|---|
| 舊掃描 / 驗證掃描排除篇數 | 6074 / 6138（舊集合是新集合的子集） |
| 舊掃描漏掉 | 64 篇，全部來自 act_outlier 的 manifest |
| 已知重複 | 驗證掃描會排除，舊掃描不會 |
| 其他近似重複（shingle ≥ 0.1） | 沒有 |

本輪使用驗證掃描，共排除 6233 篇（包括 token-spike 輪的 95 篇新文章）；新資料抽取時另有 8 篇因近似重複跳過。

**token-spike 輪的更正**
- 位置：`runs/olmoe_token_spike_diag_20261023/correction_20261023/`，並已在其 report.md 開頭加註。
- 原始輸出保持不變；report.md 原本的 hash 是 `ee5cab6c…e270`，記錄在 correction.json。
- 排除重複文章後以 95 篇重算（seed 不變），與你提供的參考值一致：

| 項目 | 96 篇（原） | 95 篇（更正） |
|---|---|---|
| p4e1 強弱配對差 | +0.001374，確認 | **+0.001344**，97.5% CI [+0.000615, +0.002072]，仍確認為正 |
| int4 強弱配對差 | 不可確認 | −0.000474，97.5% CI [−0.001162, +0.000183]，仍不可確認 |
| L1.E9 int4 vs p6e1 | 誤排重現 | +0.00355 [+0.00194, +0.00516]，**仍重現** |
| L1.E9 int6 vs p6e1 | 誤排重現 | +0.00126 [−0.00001, +0.00251]，**CI 含 0**，不再算重現 |
| L1.E9 上一輪誤排的重現數 | 2 / 4 | **1 / 4** |
| 新 tap 的 BC 正確 / 決定性 | 53 / 61 | 54 / 63（新增一個 L1.E37 int8 vs p6e1 誤排） |

## 1. 設計（預先登記）

### 1.1 tap

| tap | 角色 | spike 門檻（6 × discovery 中位數 s_t） | q0 act |
|---|---|---|---|
| L1.E9 | 已知案例（重現族） | 1.715 | f16 |
| L1.E11 | 位置 0（推廣族） | 1.354 | f16 |
| L1.E18 | 位置 0（推廣族） | 1.751 | f16 |
| L2.E30 | 位置 0（推廣族） | 4.950 | f16 |
| L6.E59 | int4 曾改善 NLL（推廣族） | 4.612 | f16 |
| L1.E37 | L1 對照（R_token 8.0；L1 唯一 ≤ p50 的 expert，上一輪也是弱 tap） | 2.071 | p8e1 |
| L2.E15 | L2 對照（R_token 6.7） | 2.385 | p8e1 |
| L6.E44 | L6 對照（R_token 7.5） | 4.909 | p8e1 |

**對照 tap**
- 規則：同層、R_token ≤ p50、log rows 最接近錨點、優先同 q0 位元類別。三個對照的位元類別都與重點 tap 不同，因為弱 tap 中沒有 f16 的。
- 選點只用 discovery 統計，不看任何 loss。

**選點規則在寫 plan 前修改過一次**
- smoke test 時發現：L1 唯一 R_token ≤ p50 的 expert（L1.E37）是上一輪的 tap。
- 修改：對照 tap 允許重用前幾輪的 tap；若某層沒有候選，改取該層 R_token 最低者。
- 這發生在本輪 plan 之前，沒有看過任何本輪資料。

### 1.2 格式與條件

**格式**：只用 int4 與 p6e1，分別對應已知的兩類誤排；不跑 int6／p4e1。

**條件**

| 代號 | 定義 |
|---|---|
| Z | tap 完全 FP32（等於 FP32 forward） |
| F | 全部 rows 量化 |
| P | spike rows 保留 FP32，其餘量化 |
| S | 只有 spike rows 量化 |
| R | 同數量的隨機一般 rows 保留 FP32 |

- R 使用 3 個固定 seeds（23341–3），在同一窗口、同一 expert 內抽樣；先對 seed 取平均，再做文章統計。
- 「保留 FP32」是把 Q(h) 換回 h，不刪 token、不設為 0。
- 所有條件都用原 calibration range。P 不會改善其他 rows 的量化間距，所以本輪量的是 spike rows 自身的誤差，不是 range 被撐大的效果。
- 對照 tap 只跑 F、P、S。

### 1.3 測量、BC 反事實與判讀

**測量**
- FP32 背景，一次只干預一個 expert 的 Down 輸入，dynamic route。
- 96 篇新 WT103 文章 × 3 窗口，共 81 個 variant。

**BC 反事實**
- 在 tap 的 calibration rows 上，把被保護 rows 的誤差設為 0，重算 S1、S2、mean、variance 與 bias。
- n、G2／GS 與正規化都不變；不從原分數直接減去舊的 spike 占比。

**主要對比**

| 對比 | 定義 | 問題 |
|---|---|---|
| C1 | NLL(P) − NLL(F) | 保護 spike rows 是否改善 |
| C2 | NLL(P) − NLL(R̄) | 是否比保護同數量的隨機 rows 更有用 |
| C3 | D_F − D_P − D_S | 兩部分誤差的整體非加性，不等於 Hessian 交叉項 |

**多重比較**
- 重現族：L1.E9 的 C1、C2 × 2 格式，共 4 個檢定。
- 推廣族：其他 4 個重點 tap，共 16 個檢定。
- 兩族各自做 Holm 校正，FWER 0.05，使用雙尾文章 bootstrap p 值。
- C3 與對照 tap 只報告 95% CI。

**判讀類別**

| 類別 | 條件 |
|---|---|
| A | C1 < 0 且 Holm 顯著 |
| D | C1 > 0 且 Holm 顯著 |
| B | 以同一 tap 的 int4 為錨，k = D_F(int4) ÷ T_F(int4)（只有 D_F(int4) 的 CI > 0 才定義）。E_BC = k·(T_P − T_F)，需 \|E_BC\| ≥ 5e-4，且 C1 的 95% CI 完全落在 ±0.25·\|E_BC\| 內 |
| C | C3 的 95% CI 排除 0 |
| E | A、B、D 都不成立：無法判定 |

## 2. 覆蓋量與控制

| tap | rows（確認資料） | spike rows | 有 spike 的窗口 | 有 spike 的文章 |
|---|---|---|---|---|
| L1.E9 | 5153 | 632 | 287 / 288 | 96 |
| L1.E11 | 9548 | 539 | 287 | 96 |
| L1.E18 | 11431 | 664 | 286 | 96 |
| L2.E30 | 2979 | 291 | 288 | 96 |
| L6.E59 | 5569 | 54 | 54 | 43 |
| L1.E37（對照） | 6562 | 7 | 7 | 7 |
| L2.E15（對照） | 3302 | 10 | 9 | 8 |
| L6.E44（對照） | 5719 | 5 | 2 | 2 |

**R 條件**：所有 tap、seed 都能在同一窗口內配對到足夠的一般 rows，不需要用其他窗口補足。

**控制**

| 檢查 | 結果 |
|---|---|
| Z 與原 FP32 forward（`dr.forward`）的差 | 0.0（逐位元相同） |
| 3 個 identity variant（自寫的 expert 迴圈，不干預）與 Z 的差 | 0.0 |
| 同一 tap 所有格式與條件的 spike mask（row-id checksum） | 相同 |
| R 的隨機 rows 跨格式 | 相同 |
| 被保護的 rows 數 | P、R 等於 spike rows；S 等於其餘 rows；F 為 0 |
| 被保護位置的殘餘誤差 | 0 |
| 每個 tap 的 range | 只有一個 |
| 輸入與模型檔 hash | 前後一致 |
| BC 反事實：calibration rows 數 | 與 cache 完全相同 |
| BC 反事實：T_F 與 cache 的 BC 成本 | 相對差 ≤ 3e-16，重現了原 BC 成本 |

## 3. 實測結果（nat／token，相對 Z；95% 文章 bootstrap CI）

### 3.1 重點 tap

| tap | 格式 | D_F | D_P | D_S | D_R | C1 = P − F | C2 = P − R̄ | C3 非加性 | 類別 |
|---|---|---|---|---|---|---|---|---|---|
| **L1.E9** | int4 | +2.86e-3 [1.25, 4.75]e-3 | +1.59e-3 | +1.54e-3 | +2.71e-3 | −1.27e-3 [−2.37e-3, −1.9e-4]，p 0.017，Holm ✗ | −1.12e-3 [−2.26e-3, −1e-5]，p 0.048，Holm ✗ | −2.7e-4 [−1.06e-3, +5.3e-4] | E |
| **L1.E9** | p6e1 | −3.9e-5 [−5.7, +4.9]e-4 | +8.7e-5 | +5.2e-5 | −7.7e-5 | +1.3e-4 [−4.0e-4, +6.5e-4] | +1.6e-4 [−3.5e-4, +6.7e-4] | −1.8e-4 [−5.6e-4, +2.0e-4] | **B** |
| L1.E11 | int4 | +2.59e-3 [1.01, 4.27]e-3 | +2.60e-3 | +4.9e-4 | +2.44e-3 | +1.3e-5 [−4.9e-4, +5.1e-4] | +1.6e-4 [−3.4e-4, +6.7e-4] | −5.0e-4 [−1.12e-3, +1.2e-4] | E（B 不可評估） |
| L1.E11 | p6e1 | +1.9e-4 [−3.3, +7.2]e-4 | +7.6e-5 | +1.2e-4 | +1.7e-4 | −1.1e-4 [−4.0e-4, +1.7e-4] | −8.9e-5 [−3.9e-4, +2.2e-4] | −4e-6 [−3.5e-4, +3.4e-4] | E |
| **L1.E18** | int4 | +1.70e-2 [1.38, 2.04]e-2 | +1.22e-2 | +4.87e-3 | +1.62e-2 | **−4.86e-3** [−6.54e-3, −3.25e-3]，Holm ✓ | **−4.07e-3** [−5.77e-3, −2.42e-3]，Holm ✓ | −1e-5 [−9.6e-4, +9.2e-4] | **A** |
| **L1.E18** | p6e1 | +3.0e-4 [−3.5, +9.8]e-4 | +2.3e-4 | +8.7e-5 | +1.6e-4 | −7.0e-5 [−5.3e-4, +3.7e-4] | +7.5e-5 [−3.6e-4, +4.9e-4] | −1.7e-5 [−5.3e-4, +5.1e-4] | **B** |
| **L2.E30** | int4 | +1.31e-2 [1.08, 1.54]e-2 | +8.91e-3 | +4.73e-3 | +1.23e-2 | **−4.18e-3** [−5.49e-3, −2.87e-3]，Holm ✓ | **−3.41e-3** [−4.76e-3, −2.07e-3]，Holm ✓ | −5.5e-4 [−1.16e-3, +7e-5] | **A** |
| **L2.E30** | p6e1 | **+9.15e-3** [7.49, 10.8]e-3 | **−1.1e-4** | **+9.19e-3** | +9.09e-3 | **−9.26e-3** [−1.09e-2, −7.65e-3]，Holm ✓ | **−9.20e-3** [−1.08e-2, −7.59e-3]，Holm ✓ | +7.7e-5 [−2.2e-4, +3.9e-4] | **A** |
| L6.E59 | int4 | **−2.75e-3** [−3.95, −1.55]e-3 | −2.84e-3 | +9.1e-5 | −2.76e-3 | −9.8e-5 [−2.3e-4, +2.3e-5] | −8.2e-5 [−2.5e-4, +7.9e-5] | +7e-6 [−6.4e-5, +7.4e-5] | E（沒有錨點） |
| L6.E59 | p6e1 | +7e-6 [−2.1, +2.3]e-4 | +4.4e-5 | −4.3e-5 | −9e-6 | +3.7e-5 [−1.8e-5, +9.4e-5] | +5.3e-5 [−3e-6, +1.1e-4] | +6e-6 [−3.4e-5, +4.4e-5] | E（沒有錨點） |

**Holm 結果**
- 重現族（L1.E9 的 4 個檢定）：沒有一個被拒絕。最小的 p 是 int4 C1 的 0.017，第一個門檻是 0.0125。
- 推廣族（16 個檢定）：拒絕 6 個，p 都是 bootstrap 下限 1e-4：
  - L1.E18 int4 的 C1、C2；
  - L2.E30 int4 的 C1、C2；
  - L2.E30 p6e1 的 C1、C2。
- 第 7 小的 p（L6.E59 p6e1 C2）是 0.067，超過門檻。
- 沒有任何類別 D，也就是保護 spike rows 後顯著變差的情況。

### 3.2 對照 tap

對照 tap 的 spike rows 很少（5–10 個），C1 都在 ±4e-5 以內，作為「干預機制本身沒有引入假效果」的負對照。

| tap | 格式 | D_F | C1 | C3 |
|---|---|---|---|---|
| L1.E37 | int4 | +1.06e-3 [−1e-4, +2.2e-3] | −7e-6 [−6.7e-5, +4.5e-5] | +1.2e-5 |
| L1.E37 | p6e1 | +1.6e-4 | −3.6e-5 [−9.0e-5, −2e-6] | +2.2e-5 |
| L2.E15 | int4 | +3.1e-4 | +3.1e-5 [−3.3e-5, +1.2e-4] | −7e-6 |
| L2.E15 | p6e1 | +7.4e-5 | +1.5e-5 [+8e-7, +3.5e-5] | −1.6e-6 |
| L6.E44 | int4 | +9.1e-4 [3.6e-4, 1.46e-3] | +3e-6 [0, +8e-6] | −2.9e-5 |
| L6.E44 | p6e1 | +3.2e-5 | +6e-6 [0, +1.7e-5] | +1.7e-6 |

**下游路由**：P 比 F 稍微減少路由變化（例如 L2.E30 p6e1 從 3.0% 降到 1.5%）。但這只是記錄，不能據此歸因於 router。

## 4. BC 反事實 vs 實測

### 4.1 保護 spike rows 去除了多少成本

| tap | 格式 | T_F | T_P／T_F | T_S／T_F | T_R̄／T_F | BC 換算的 E_BC（int4 錨） | 實測 C1 | C1 ÷ E_BC | 實測去除的 D_F 比例 |
|---|---|---|---|---|---|---|---|---|---|
| L1.E9 | int4 | 1.43 | 0.335 | 0.674 | 0.956 | −1.90e-3 | −1.27e-3 | 0.67 | 44% |
| L1.E9 | p6e1 | 3.83 | **0.003** | 0.997 | 1.000 | **−7.6e-3** | **+1.3e-4** | −0.02 | （D_F ≈ 0） |
| L1.E11 | int4 | 2.39 | 0.863 | 0.131 | 0.923 | −3.5e-4（< 5e-4） | +1.3e-5 | – | ≈ 0% |
| L1.E11 | p6e1 | 0.755 | 0.076 | 0.924 | 0.994 | −7.6e-4 | −1.1e-4 | 0.15 | （D_F ≈ 0） |
| L1.E18 | int4 | 9.51 | 0.833 | 0.164 | 0.955 | −2.84e-3 | −4.86e-3 | **1.71** | 29% |
| L1.E18 | p6e1 | 14.4 | **0.010** | 0.990 | 0.999 | **−2.55e-2** | **−7e-5** | 0.003 | （D_F ≈ 0） |
| L2.E30 | int4 | 17.9 | 0.936 | 0.064 | 0.888 | −8.4e-4 | −4.18e-3 | **4.99** | 32% |
| L2.E30 | p6e1 | 57.6 | 0.003 | 0.997 | 1.000 | −4.19e-2 | −9.26e-3 | 0.22 | **101%** |
| L6.E59 | int4 | 4.73 | 0.934 | 0.064 | 0.966 | 沒有錨點 | −9.8e-5 | – | （D_F < 0） |
| L6.E59 | p6e1 | 0.625 | 0.051 | 0.949 | 0.998 | 沒有錨點 | +3.7e-5 | – | （D_F ≈ 0） |

另見 `figures/bc_vs_measured.png`：缺少橘色長條的位置，表示 D_F 的 CI 含 0，沒有可量測的損傷可以去除。

**類別 B 對錨點不確定性的敏感度**（未預先登記，只作說明）
- 把錨點換成 D_F(int4) 的 CI 下限，L1.E9 p6e1 的容許範圍是 ±8.3e-4，L1.E18 p6e1 是 ±5.2e-3。
- 兩者的 C1 CI 仍然落在範圍內，所以 B 的判定不依賴錨點的點估計。

### 4.2 方向、排序與「P 優於 R」

**變化方向**
- BC 對所有 tap 都預測 P 比 F 好。
- 實測方向一致且決定性的有：L1.E9 int4（名目）、L1.E18 int4、L2.E30 int4、L2.E30 p6e1。
- 其餘實測都不可判定。

**int4 vs p6e1 的排序**

| tap | F 時 BC 的排序 | F 時實測 | F 是否一致 | P 時是否一致 |
|---|---|---|---|---|
| L1.E9 | p6e1 較貴 | int4 較差 +2.9e-3（決定性） | 否 | 是 |
| L1.E18 | p6e1 較貴 | int4 較差 +1.67e-2（決定性） | 否 | 是 |
| L2.E30 | p6e1 較貴 | int4 較差 +3.9e-3（決定性） | 否 | 是 |
| L1.E11 | int4 較貴 | 一致 | 是 | 是 |
| L6.E59 | int4 較貴 | int4 較好（−2.8e-3） | 否 | 否 |

- L6.E59 的不一致是「int4 有利」，P 之後仍然存在。
- 在 L1.E9、L1.E18、L2.E30，BC 的格式排錯完全來自 spike rows 的 Posit 成本：保護 spike rows 之後，排序就正確了。

**保護 spike rows 是否優於保護隨機 rows**
- BC 在 9 / 10 個案例認為 P 優於 R。
- 實測決定性的只有 3 個（L1.E18 int4、L2.E30 int4、L2.E30 p6e1，都 P 優於 R）。
- **L2.E30 int4 兩者不一致**：BC 認為保護隨機 rows 去除的成本更多（T_R̄ 15.9 < T_P 16.8），但實測保護 spike rows 明顯較好（C2 −3.4e-3）。

**非加性**
- BC 自己的非加性很小，都不到 T_F 的 1%。
- 實測的 C3 在重點 tap 上都不可判定，所以類別 C 沒有出現。

## 5. 回答

**問題 1：L1.E9 的 Posit 高估是否有直接干預證據？有（類別 B）。**
- 在預先登記的標準下，BC 預測：保護 L1.E9 的 spike rows，p6e1 的效果是 −7.6e-3（以 int4 換算）。
- 實測效果是 +1.3e-4，95% CI [−4.0e-4, +6.5e-4]，CI 夠窄，可以排除 BC 預測幅度的 25% 以上。
- 只量化 spike rows 的 D_S 也接近 0。
- 所以 BC 對這些 ' the'／' a' spike rows 的 Posit 誤差給了過高的局部成本。
- int4 的 spike rows 誤差則可能有實際影響（名目 −1.27e-3），但沒有通過 Holm 校正，不下結論。

**問題 2：位置 0 的 expert 是否出現相同現象？部分出現，不是單一機制。**
- **L1.E18**：p6e1 的現象與 L1.E9 相同（B）；但 int4 的 spike rows 誤差確實有損傷（A），且 BC 低估了它。
- **L2.E30**：相反，spike rows 的 Posit 誤差造成全部 9.2e-3 的 p6e1 損傷（A）。BC 方向正確、幅度偏高，但損傷是真的。int4 的 spike 保護效果也被 BC 低估約 5 倍，BC 還把隨機保護排在 spike 保護之前。
- **L1.E11**：兩種格式都沒有可量測的效果；int4 的 CI 窄（±5e-4）。

**問題 3：是否值得下一輪收集逐 token 梯度作診斷？值得。**
- 同一個 BC 公式在 L1.E9、L1.E18 高估 spike rows 的 Posit 成本，在 L2.E30 雖然方向正確但換算後高估，又在 L1.E18、L2.E30 低估 spike rows 的 int4 重要性。
- 這種依 expert、依格式而異的偏差，最直接的候選解釋是：spike token 的梯度與 G2_c/n 的平均假設不同。
- 本輪只控制了誤差，沒有逐 token 梯度，無法判定。
- 非加性沒有出現，所以「spike rows 與一般 rows 分開計算」的框架足以支撐這個診斷。
- 建議做法：在這 5 個 expert 上做一次 backward，比較 token 配對的 Σ_t g_tc² e_tc² 與 G2_c/n·Σ_t (e_tc − μ_c)²，並分 spike／一般 rows。只作診斷，不修改 criterion，並使用新的預先登記與新資料。

**問題 4：完整配置**
- 本輪只測單一 tap 的干預，沒有測完整配置，不能宣稱 knee 有任何改善。
- q0 knee 已經把這 5 個 expert 的 Down activation 設為 f16，所以這些現象不影響 q0。

## 6. 計時

| 階段 | 秒 | 預算（分） | GPU 峰值 |
|---|---|---|---|
| correct_prev | 59.4 | – | – |
| plan | 33.5 | – | – |
| prep（選對照與 BC 反事實） | 5.7 | 5 | 4.44 GB |
| data | 67.1 | 5 | – |
| measure | 1254.3 | 25 | 4.64 GB |
| analyze | 12.6 | 5 | – |
| figures | 0.4 | 3 | – |

- GPU 合計 1260 s（21.0 分），在 35 分預算內。
- 沒有提前停止，也沒有任何修正紀錄。

## 7. 限制

- **重點 tap 少**：只有 5 個重點 tap、每層 1 個對照。推廣族的結論只限這些 expert。
- **對照的匹配**：對照 tap 的 spike rows 很少，q0 位元類別也不同，只能當作負對照，不能當作同條件比較。
- **B 的錨點**：類別 B 依賴「以 int4 為錨」的尺度換算。在 L6.E59（int4 有利）沒有錨點，L1.E11 int4 的換算效果太小，兩者都無法評估。
- **spike 門檻**：沿用 6× 中位數，包含一些非位置 0 的 rows（L1.E11、L1.E18 約有一半的 spike rows 不在位置 0）。
- **背景與路由**：只用 FP32 背景與 dynamic route；路由變化沒有分解。
- **沒有梯度**：本輪沒有逐 token 梯度，「BC 獨立性近似」只是假說。

## 8. 重現

```
cd experiments/mixed_precision
O=runs/olmoe_spike_intervention_20261023
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_spike_intervention.py
$CPU $S --stage correct_prev --out-dir $O
$CPU $S --stage plan --out-dir $O
$GPU $S --stage prep --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage measure --out-dir $O
$CPU $S --stage analyze --out-dir $O
$CPU $S --stage figures --out-dir $O
```

各階段都拒絕覆寫既有輸出。

### 主要 hash（sha256 前 16 位）

| 檔案 | hash |
|---|---|
| `plan.json` | `d924402fc402dfe2` |
| `taps.json` | `2d3963365ba0f5d7` |
| `bc_counterfactual.json` | `cac29e449a6077b5` |
| `freeze.json` | `ced32619f571fc9a` |
| `tokens.npy` | `ab485bc8ea8e727d` |
| `segments.json` | `a2d0096da7a881e2` |
| `measure.json` | `28b3246d8e88cd66` |
| `analysis.json` | `000debd5fc3f10ae` |
| `correction_summary.json` | `b5ea1a4efb7b68a7` |
| token-spike 輪的 `correction_20261023/correction.json` | `a1cb2511c324eee0` |

### 輸出檔

| 類別 | 檔案 |
|---|---|
| 預先登記與凍結 | `plan.json`、`plan.md`、`freeze.json`、`data_freeze.json` |
| 選點與 BC 反事實 | `taps.json`、`bc_counterfactual.json` |
| 資料 | `segments.json`（含 source 欄位）、`tokens.npy` |
| 測量 | `measure.json`（每個 variant 的窗口 CE、row 計數、checksum、殘餘誤差、路由變化） |
| 分析 | `analysis.json`、`contrasts.csv`、`per_window.csv` |
| 圖表 | `figures/conditions.png`、`figures/bc_vs_measured.png` |
| 隔離修正 | `correction_summary.json`；完整版在 token-spike 輪的 `correction_20261023/` |
| 計時與紀錄 | `timing.json`、`timing.jsonl`、`logs/` |
