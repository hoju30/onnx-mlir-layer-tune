# OLMoE Down-input activation：spike 分組評分診斷（事後探索）

日期：2026-10-23。腳本：`run_olmoe_spike_group_diag.py`（sha256 `c499407d3d15e426…`）。計畫：`plan.json`／`plan.md`（18:10:37 寫入，早於任何分析輸出）。

## 性質與範圍

**這是事後探索。** 本輪使用的 tap、量測 loss 與排序，在前兩輪都已分析過：
- `olmoe_pair_gradient_diag_20261023`
- `olmoe_pair_generalization_20261023`

事先固定規則，也不會讓舊資料變成獨立確認資料。

本輪只用既有快取做 CPU 分析：
- 沒有新的模型推論、沒有新文章；
- 沒有搜尋配置；
- 沒有修改正式 BC 或 pipeline。

三組分開報告：

| 組 | 來源 | tap | 格式 | 量測 |
|---|---|---|---|---|
| mechanism_8 | pair_gradient_diag | 8 | int4、p6e1 | F 條件 |
| generalization_general | pair_generalization | 16 | 6 種 | 單 tap 量化 |
| generalization_spike | pair_generalization | 8 | 6 種 | 單 tap 量化 |

## 結論摘要

- **控制全部通過。**
  - T_old、T_pair 重現來源值，相對誤差 0。
  - 兩個恆等式成立，相對誤差 ≤ 1.4e-14。
  - 空組精確退化為原 BC。
- **相對原 BC（T_old），T_group 在可判定配對上沒有新增誤排。**
  - mechanism_8：0/4 → 2/4。
  - generalization_spike：23/30 → 25/30（主要配對 3/4 → 4/4）。
  - generalization_general：59/59 不變。
- **但修正全部落在 3 個已知 expert。**
  - L1.E9、L1.E18：機制輪的發現案例。
  - L9.E8：泛化輪 T_pair 唯一的修正案例。
  - 其他 29 個 tap 沒有任何排序改變。
- **T_group 在所有可判定配對上都不優於 T_pair。** 它的正確集合是 T_pair 的子集：沒有修正 T_pair 的錯誤，並失去 L2.E30。T_pair 的 5 個修正中保留 4 個。
- **L2.E30 的修正完全來自組內共變異**，被分組捨棄了。具體來說是 channel 565 上少數 rows 的 g² 與 r² 同時偏大：top-1 row 佔 23%，top-5 rows 佔 66%。
- **成本沒有優勢。** 每個 expert 的統計量（10 格式，float64）：分組約 172 KB，配對約 164 KB。前一輪實測，配對全部 Down input 只多 3.1 s。
- **建議：停止分組方向，不登記新 expert＋新文章的獨立確認。**
  - 預先登記停止規則的三個條件，字面上都沒有觸發。
  - 建議的理由見第 9 節：收益只在已知案例、被 T_pair 完全涵蓋，而 T_pair 本身的整體配置 A/B 兩次都未確認。
  - 不接入 pipeline。

## 1. 定義

### spike mask

- 每列分數：s_t = max_c |h_tc|，使用快取的 FP32 calibration rows（10 × 128 窗口）。
- 門檻：6 × median_t(s_t)。
- 分組：S = {s_t ≥ 門檻}，N = 其餘 rows。
- 所有格式共用同一 mask。
- 非空組 < 8 rows 時加低覆蓋旗標。

### 三種分數

共用全部 rows 的 μ_c，r = e − μ_c，組內不重新中心化；U = Σ_c GS_c μ_c² 不變。

- T_old = Σ_c mean_t(g²)·Σ_t r² + U（原 BC）
- T_group = Σ_c [mean_S(g²)·Σ_S r² + mean_N(g²)·Σ_N r²] + U
- T_pair = Σ_c Σ_t g² r² + U

### 分解（逐 channel；X = g²，Y = r²）

- between = V_group − V_old = (n_S n_N / n)(mean_S X − mean_N X)(mean_S Y − mean_N Y)
- within = V_pair − V_group = n_S Cov_S(X,Y) + n_N Cov_N(X,Y)（母體共變異）

### 排序

- 量測差 = NLL(f1) − NLL(f2)，取每篇文章 3 個窗口。
- 可判定 = 文章 bootstrap 95% CI 排除 0（10000 次，seed 23901）；三種方法共用同一判定。
- 主要配對 = 同位元 INT／Posit。mechanism_8 只有 int4 vs p6e1，當作它唯一的配對報告。
- proxy tie（相對差 < 0.02）單列，不算對。
- 同一 expert 的多個配對不是獨立樣本。

## 2. 控制（`controls.json`）

| 檢查 | 結果 |
|---|---|
| T_old 重現來源 mech.json | 最大相對誤差 0.0 |
| T_pair 重現來源 mech.json | 最大相對誤差 0.0 |
| between 恆等式 | 1.5e-15 |
| within 恆等式 | 1.4e-14 |
| 空組 = 原 BC | 是（精確） |
| 門檻非正／非有限 | 無 |

**mask 與 discovery 門檻的差異。** 舊實驗用 discovery median，本輪用 calibration median。32 個 tap 中有 27 個 mask 完全相同，其餘 5 個：

| 組 | tap | 本輪 spike rows | discovery 門檻 spike rows |
|---|---|---|---|
| mechanism_8 | L1.E11 | 24 | 27 |
| mechanism_8 | L1.E18 | 20 | 21 |
| general | L0.E28 | 2 | 1 |
| spike | L1.E50 | 0 | 1 |
| spike | L6.E9 | 9 | 8 |

依計畫，本輪不依 g、誤差、loss 或排序調整 mask，也不掃描其他倍數。

## 3. 覆蓋：分組在多少 tap 上能起作用

| 組 | 空組（= T_old） | 低覆蓋（1–7 rows） | 有效分組（≥ 8 rows） |
|---|---|---|---|
| mechanism_8 | 2（L2.E15、L6.E44） | 2（L6.E59 6、L1.E37 1） | 4（L1.E9 22、L1.E11 24、L1.E18 20、L2.E30 10） |
| general | 11 | 3（L0.E28 2、L1.E56 2、L4.E43 5） | 2（L2.E38 8、L5.E57 19） |
| spike | 5（L1.E50、L0.E44、L5.E58、L6.E57、L9.E33） | 1（L5.E60 1） | 2（L6.E9 9、L9.E8 13） |

- **spike 分層組的 8 個 expert 中，有 5 個在 calibration rows 上沒有任何 spike row。**
  - 分層用的是 discovery 文章，calibration 只有單一文章的 10 個窗口。
  - 這 5 個 tap 的 T_group 與 T_old 精確相同。分組能否起作用，取決於 calibration 是否剛好含有 spike。
- 一般組有 11/16 為空組，所以一般組不可能出現差異。

## 4. 分解（`decomposition.csv`、`scores.csv`）

非空組 tap-format 的 between／V_old：

| 組 | 非空 tap-format | between/V_old 中位數 | 範圍 | 負值 | 單一 channel > 50% |
|---|---|---|---|---|---|
| mechanism_8 | 12 | −0.27 | [−0.95, +0.46] | 8 | 7 |
| general | 30 | −0.006 | [−0.39, +0.02] | 23 | 5 |
| spike | 18 | −0.02 | [−0.93, +1.86] | 11 | 8 |

### 關鍵案例

| 組 | tap | spike rows | 格式 | V_old → V_group → V_pair | between（主導 channel、佔比） | within（主導 channel；top-1／top-5 rows） |
|---|---|---|---|---|---|---|
| mechanism_8 | L1.E9 | 22 | int4 | 1.40 → 0.73 → 0.64 | −0.67（c661，44%） | −0.10 |
| | | | p6e1 | 3.77 → 0.21 → 0.25 | −3.56（c915，97%） | +0.04（c915；15%／52%） |
| mechanism_8 | L1.E18 | 20 | int4 | 8.49 → 12.42 → 26.51 | +3.94（c488，67%） | +14.1（c131；12%／32%） |
| | | | p6e1 | 14.34 → 8.34 → 0.42 | −6.00（c627，99.5%） | −7.92（c627；19%／65%） |
| mechanism_8 | L2.E30 | 10 | int4 | 16.08 → 14.10 → 48.53 | −1.99（分散，c723 8%） | **+34.4（c565 87%；23%／66%）** |
| | | | p6e1 | 55.13 → 16.93 → 16.11 | −38.2（c742，99.8%） | −0.82 |
| spike | L9.E8 | 13 | int4 | 13.79 → 3.49 → 3.86 | −10.3（分散，c272 29%） | +0.37 |
| | | | p4e1 | 4.38 → 8.27 → 7.14 | +3.89（c3，76%） | −1.13（c3；14%／60%） |

**L1.E9、L2.E30（p6e1）：誤差偏大但梯度不大的 spike rows。**
- spike rows 上 Posit 的 r² 比 N 組大很多：L1.E9 約 380 倍，L2.E30 約 160 倍。
- 但在主導 channel，這些 rows 的 g² 並不比平均高。
- 原 BC 用全體平均的 g² 去乘這些誤差，因此高估了 p6e1。分組把高估去掉（between 為負，集中在單一 channel）。
- c915 就是 token-spike 輪已記錄的 L1.E9 ' the'／' a' channel。

**L9.E8：spike rows 的梯度大，而且 INT 與 Posit 的誤差落點相反。**
- 13 個 spike rows 的平均 g² 約為 N 組的 73 倍，各 channel 都是如此。
- int4 的 r² 在 spike rows 反而較小，所以 INT 的 between 為負，分散在許多 channel。
- p4e1 的 r² 在 spike rows 較大，集中在 c3，所以 Posit 的 between 為正。
- 這個修正來自組間差異，因此分組保留了它。

**L2.E30（int4）：修正來自組內共變異。**
- 少數 rows 在 c565 上同時有大 g² 與大 r²：top-1 row 佔 23%，top-5 rows 佔 66%。
- 這屬於 within 項，分組依定義把它捨棄。
- 結果 T_group 的 int4 只有 14.1，T_pair 是 48.5，排序方向相反。

## 5. 排序（`ranking_comparisons.csv`）

| 組 | 配對 | 可判定 | T_old 對 | T_group 對 | T_pair 對 | proxy tie（可判定） |
|---|---|---|---|---|---|---|
| mechanism_8（int4 vs p6e1） | 8 | 4 | 0 | 2 | 3 | 0 |
| general：主要 | 48 | 9 | 9 | 9 | 9 | 0 |
| general：全部 | 240 | 59 | 59 | 59 | 59 | 0 |
| spike：主要 | 24 | 4 | 3 | 4 | 4 | 0 |
| spike：全部 | 120 | 30 | 23 | 25 | 25 | 0 |

唯一的 proxy tie 是 spike 組 L0.E44 int8 vs p8e1，在 T_old 與 T_group 下出現。該配對不可判定，不影響計數。

### 5.1 T_group vs T_old（可判定配對）

| 組 | 修正 | 新增誤排 | 兩者皆對 | 兩者皆錯 | 修正所在 expert |
|---|---|---|---|---|---|
| mechanism_8 | 2 | 0 | 0 | 2 | L1.E9、L1.E18 |
| general | 0 | 0 | 59 | 0 | — |
| spike | 2 | 0 | 23 | 5 | L9.E8（int4–p4e1、int6–p4e1；都是主要配對） |

### 5.2 T_group vs T_pair（可判定配對）

| 組 | T_group 修正 | T_group 新增誤排 | 兩者皆對 | 兩者皆錯 |
|---|---|---|---|---|
| mechanism_8 | 0 | 1（L2.E30） | 2 | 1 |
| general | 0 | 0 | 59 | 0 |
| spike | 0 | 0 | 25 | 5 |

### 5.3 三種方法有差異、或全部排錯的可判定配對

量測差 = NLL(f1) − NLL(f2)，正值代表 f1 較差。

| 組 | tap | f1–f2 | 量測差 [95% CI] | T_old 差 | T_group 差 | T_pair 差 |
|---|---|---|---|---|---|---|
| mechanism_8 | L1.E9 | int4–p6e1 | +2.45e-3 [1.3e-3, 3.7e-3] | −2.40 ✗ | +0.49 ✓ | +0.36 ✓ |
| mechanism_8 | L1.E18 | int4–p6e1 | +1.24e-2 [9.4e-3, 1.6e-2] | −4.86 ✗ | +5.08 ✓ | +27.1 ✓ |
| mechanism_8 | L2.E30 | int4–p6e1 | +4.59e-3 [2.5e-3, 6.7e-3] | −39.7 ✗ | −3.46 ✗ | +31.8 ✓ |
| mechanism_8 | L6.E59（低覆蓋） | int4–p6e1 | −2.63e-3 [−3.9e-3, −1.5e-3] | +4.11 ✗ | +4.80 ✗ | +5.83 ✗ |
| spike | L9.E8 | int4–p4e1（主要） | −1.85e-2 [−2.1e-2, −1.6e-2] | +11.6 ✗ | −2.59 ✓ | −1.09 ✓ |
| spike | L9.E8 | int6–p4e1（主要） | −1.77e-2 [−2.0e-2, −1.5e-2] | +2.02 ✗ | −6.70 ✓ | −5.56 ✓ |
| spike | L9.E8 | int4–int6 | −8.4e-4 [−1.4e-3, −3.0e-4] | +9.59 ✗ | +4.10 ✗ | +4.46 ✗ |
| spike | L9.E8 | int4–int8 | −5.5e-4 [−1.1e-3, −8.1e-6] | +15.3 ✗ | +5.62 ✗ | +6.01 ✗ |
| spike | L9.E8 | int4–p6e1 | −7.9e-4 [−1.3e-3, −2.5e-4] | +15.6 ✗ | +4.36 ✗ | +5.82 ✗ |
| spike | L9.E8 | int4–p8e1 | −6.1e-4 [−1.2e-3, −3.2e-5] | +16.1 ✗ | +5.77 ✗ | +6.15 ✗ |
| spike | L5.E58（空組） | int6–p4e1 | +4.4e-4 [6.1e-7, 8.8e-4] | −0.332 ✗ | −0.332 ✗ | −0.332 ✗ |

- L9.E8 有 4 個次要配對三種方法全錯。量測顯示 int4 比 int6、int8、p6e1、p8e1 都好，差 5–8e-4。
  - 三種方法都是非負的二階分數，誤差較大的 int4 分數必然偏高，結構上排不出這種結果。
  - 這是否來自一階方向效應，本輪沒有檢驗。
- L6.E59 只有 6 個 spike rows（低覆蓋），三種方法都錯。

### 5.4 T_pair 的修正有多少被分組保留

T_pair 相對 T_old 的 5 個修正：

| 修正 | 是否保留 | 來源 |
|---|---|---|
| L1.E9 | 保留 | between 項 |
| L1.E18 | 保留 | between 項 |
| L9.E8 int4–p4e1 | 保留 | between 項 |
| L9.E8 int6–p4e1 | 保留 | between 項 |
| L2.E30 | 失去 | within 項（組內共變異） |

## 6. 逐窗口移除穩定性（`window_stability.csv`）

方法：移除 1 個 calibration 窗口，門檻固定為完整 calibration 的值，μ、U 與三種分數由剩餘 rows 重算。下表的翻轉指格式順序相對完整 calibration 改變。

| 組 | pair-window 數 | T_old 翻轉 | T_group 翻轉 | T_pair 翻轉 | 其中屬可判定配對 |
|---|---|---|---|---|---|
| mechanism_8 | 80 | 2（L1.E18 w6、L2.E30 w5） | 2（L2.E30 w2、w8） | 0 | 全部：old 2、group 2 |
| general | 2385 | 0 | 0 | 2（L5.E57、L15.E36） | 無 |
| spike | 1200 | 6（L0.E44 ×3、L6.E9 ×3） | 3（L0.E44 ×3，T_group 差僅 3e-5） | 3（L5.E60、L9.E33、L9.E8 w8） | pair 1（L9.E8 int4–p4e1） |

逐格式分數的相對變動（中位數／p90／最大）：

| 組 | T_old | T_group | T_pair |
|---|---|---|---|
| mechanism_8 | 6.9% / 23% / 74% | 8.0% / 27% / 61% | 7.4% / 25% / 59% |
| general | 7.4% / 20% / 64% | 7.3% / 20% / 64% | 7.1% / 21% / 64% |
| spike | 6.4% / 24% / 82% | 5.9% / 24% / 82% | 5.9% / 24% / 77% |

- **T_group 沒有展現穩定性優勢。**
  - mechanism_8：T_group 在 L2.E30 有 2 個窗口翻轉（因為它的差值 −3.46 接近 0）；T_pair 沒有翻轉。
  - general 與 spike：T_pair 在不可判定配對上多出少數翻轉。
- 分數的相對變動在三種方法間幾乎一樣：中位數 6–8%，最大 59–82%。
- 這只反映單一 calibration 文章內的窗口敏感度，不能證明跨文章穩定。

## 7. 成本（`timing.json`）

**CPU 分析時間**
- analyze 共 2.96 s：32 tap 的分數 0.17 s、窗口穩定性 1.59 s、排序 0.67 s。
- 這些是快取上的分析時間，不是 native 建表成本。

**每個 expert 的額外統計量（估計，1024 channel、10 格式、float64）**
- 分組：每 channel 需要 Σ_S g²，以及每格式的 Σ_S e、Σ_S e²，另加 n_S。共 1 + 2×10 = 21 陣列，約 172 KB。
- 配對：每格式的 Σ g²e²、Σ g²e，共 20 陣列，約 164 KB。
- 分組唯一的結構差異：前向誤差統計與反向梯度統計只需共用 row mask，不需逐 row 同時持有 g 與 e。但前一輪實測，配對統計全部 Down input × 10 格式只多 3.1 s（見 `olmoe_pair_knee_ab_20261023`），可節省的空間很小。

## 8. 五個問題

**1. 分組是否保留了 spike 相關的有效修正？**

部分保留。T_pair 的 5 個修正保留了 4 個：L1.E9、L1.E18、L9.E8 的兩個主要配對。這些都是組間效應：spike rows 的平均 g² 或 r² 與一般 rows 明顯不同。由 within 項造成的修正，分組依定義無法保留，這就是失去的 L2.E30：c565 上少數 rows 同時有大梯度與大誤差。

**2. 被捨棄的組內資訊，在哪些案例有用、哪些案例有害？**

- **有用：只有 L2.E30（int4 vs p6e1）。**
  - int4 的 within 為 +34.4，主要在 c565；top-5 rows 佔 66%。
  - 捨棄後 T_group 排錯，而且在 2 個窗口翻轉，顯示分數靠近 0、不穩。
- **有害：在可判定配對上沒有發現。**
  - 三組都沒有任何配對是 T_group 對、T_pair 錯。
  - within 項只在不可判定配對上，讓 T_pair 多出幾個窗口翻轉：L5.E57、L15.E36、L5.E60、L9.E33。
  - 另有可判定配對 L9.E8 int4–p4e1 的 1 個窗口翻轉。
- 這些證據都來自已分析過的案例。

**3. 分組是否真的比原 BC 好，而不只是比 T_pair 好？**

- 相對 T_old：可判定配對修正 4 個、新增誤排 0 個。
- 相對 T_pair：修正 0 個、新增誤排 1 個。所以分組並不是「只勝過 T_pair」，它其實從未勝過 T_pair。
- 但相對 T_old 的修正全部在 3 個已知 expert，而且這些案例正是提出 T_pair 與分組假設的依據。因此這不能算作分組比原 BC 好的獨立證據。

**4. 結果是否集中於已知的少數 expert 或低覆蓋組？**

- 集中於已知 expert：全部修正都在 L1.E9、L1.E18（機制輪發現案例）與 L9.E8（泛化輪 T_pair 唯一修正）。
- 不在低覆蓋組：這 3 個 tap 的 spike rows 分別為 22、20、13。
- 其餘 29 個 tap 沒有任何排序改變。其中 18 個是空組，精確等於原 BC：general 11、spike 5、mechanism 2。
- 唯一有判定的低覆蓋 tap 是 L6.E59，三種方法都錯。

**5. 是否值得進入新 expert＋新文章的獨立確認？**

不建議，理由見第 9 節。

## 9. 建議：停止分組方向

**字面判定。** 預先登記的停止規則有三個條件，字面上都沒有觸發：
- 改善的不是單一案例，而是 3 個 expert、4 個可判定配對；
- T_group 勝過 T_old，不是只勝過 T_pair；
- 相對 T_old 沒有新增誤排。

**仍建議停止，理由如下：**

1. **改善只發生在已知案例。** 3 個 expert 都是提出假設時已經看過的案例。在 32 個 tap、93 個可判定配對中，沒有任何未知案例被修正。這最接近停止規則所說「只是已知案例改善」的情形。
2. **被 T_pair 完全涵蓋。** T_group 的正確集合是 T_pair 的子集，還少了 L2.E30。
3. **上位方法的整體效益未確認。** T_pair 帶有全部這些修正，但兩次整體配置 A/B（knee、aggressive，各 600 篇新文章）都未確認整體效益，配對方向已依規則關閉。T_group 能修正的單 tap 排序更少。
4. **沒有成本或穩定性優勢。** 統計量大小相近；窗口穩定性沒有改善。
5. **適用範圍受 calibration 限制。** spike 分層組 8 個 expert 中有 5 個在 calibration 上沒有 spike rows，分組在這些 expert 上無法作用。

**處置**
- 不接入 pipeline，正式 BC 維持不變。
- 若日後要重啟，必須先登記新的 expert、新的文章與新的 calibration 來源，而不是在本輪資料上調整門檻或分組規則。

## 10. 重現與 hash

```
O=runs/olmoe_spike_group_diag_20261023
env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python run_olmoe_spike_group_diag.py --stage plan --out-dir $O
env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python run_olmoe_spike_group_diag.py --stage analyze --out-dir $O
```

**腳本與計畫**

| 檔案 | sha256 |
|---|---|
| 腳本 | `c499407d3d15e426d26aa315014ffd6b7b9ba08a4c91fd6bdf5706b8032e5b32` |
| plan.json | `4fb08de275cacf26336d7587f774ffe001e495ffa27e8c62a2c43b88c41995e2` |
| plan.md | `2ad796ca2c0839bd9bd24e9b6364d5aa1c8f54d148e3e5b42b9e4daffe234508` |

**輸出（sha256 前 16 碼）**

| 檔案 | sha256 |
|---|---|
| scores.csv | `293c28120918a9b2` |
| decomposition.csv | `cab759848ef378db` |
| ranking_comparisons.csv | `0a527c4866c6c6e1` |
| window_stability.csv | `bc1403220f5ac9f6` |
| controls.json | `5f8df5957fcecb1c` |
| summary.json | `1371804b102c63c7` |
| timing.json | `d3be73b6f8fb6c28` |

來源快取的 hash 記錄在 `plan.json["source_hashes"]`。

**沒有 amendment**：計畫寫入後，腳本、門檻、seed、資料都未變更。
