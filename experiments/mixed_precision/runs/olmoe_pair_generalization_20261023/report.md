# 原 BC 與逐位置配對評分在非誤排挑選 expert 上的排序（推廣與退步風險）

- 日期：2026-10-23
- 程式：`run_olmoe_pair_generalization.py`（sha256 `5ad9fd86…1697`），所有階段使用同一版本，沒有修正紀錄。
- 預先登記：`plan.json`、`plan.md`（15:21:32），在選點、梯度與任何確認資料之前。
- 凍結：
  - `freeze.json`（15:26:31）：選點、T_old／T_pair、全部 360 個配對的預測；
  - `data_freeze.json`（15:27:41）：新文章。
- **範圍**：只有 expert 的 Down 輸入 activation，不含 Gate／Up 共享輸入歸屬。本報告的結論都限於這個範圍。
- 正式 BC、量化器、range、搜尋、knee、pipeline 都不變。沒有刷新、調參、spike 懲罰、一階項，也沒有用 NLL 換算 proxy 尺度。

## 結論摘要

**判讀：C（依預先登記的規則）。保留原 BC，上一輪的改善視為局部的機制解釋。**

1. **一般樣本（16 個 expert，每層隨機 1 個）：配對沒有改變任何排序。**
   - 凍結的預測中，T_pair 與 T_old 對全部 240 個候選配對的方向完全相同。V_pair／V_old 的中位數在各格式為 1.00–1.11。
   - 58 個可判定配對上，兩者都 100% 正確；同位元配對 9 個可判定，兩者都 9 個全對。
   - 沒有修正，也沒有破壞，屬於「不變」。這是「預測完全相同」，不是統計上的等效。
2. **spike 分層樣本（4 強 + 4 弱）：方向有利，但證據不足。**
   - 360 個配對中，兩者預測不同的只有 8 個，全部在 spike 組；其中只有 2 個實測可判定。
   - 這 2 個都是 T_pair 修正、T_old 錯：都在 L9.E8，int4 vs p4e1 與 int6 vs p4e1；實測 p4e1 損傷 +1.8e-2。沒有任何破壞。
   - 同位元可判定配對只有 4 個（低於預先登記的 8），所以 spike 組的狀態是「不足」。
3. **預先登記的各個問題**
   - 同位元 INT／Posit 排序：一般組不變；spike 組多對 1 個（L9.E8 int4 vs p4e1），但不足以判定。
   - Posit4／Posit8 是否也有上一輪 Posit6 的現象：配對比值 < 1 的模式在 L6.E9（p4e1 0.36、p8e1 0.09）、L9.E33（約 0.5）等 expert 也出現，但沒有造成可判定的誤排。L9.E8 則相反，p4e1 的比值 > 1。
   - 是否犧牲 INT–INT、Posit–Posit：沒有，0 個破壞。
   - 一般 expert 是否因配對變差：沒有，預測完全相同。
4. **較嚴格門檻（99% CI）**：可判定的同位元配對變為一般組 4 個、spike 組 2 個，兩組都不足，相當於 D（證據不足）。結論依門檻落在 C 或 D 之間，但不論哪一個都不支持改用 T_pair。
5. **兩者都錯的 3 個配對**，都是 L9.E8 的 int4「有利擾動」：int4 的 ΔNLL 為 −5.9e-4，是本輪唯一 CI 小於 0 的格式。兩個分數都恆為正，原理上無法預測，已保留並分開解讀。
6. **成本**：配對統計的 streaming 累加（Σg²e²、Σg²e，搭配既有的 Σg²）在本輪 24 個 tap × 6 格式上只多 0.011 秒，約為誤差統計時間的 27%、forward + backward 的 0.12%。與逐元素計算的相對差 ≤ 3.8e-16。全模型的儲存估計約多 1.17 GB（**估計值**）。

## 1. 排除、合格與選點

**排除**
- 前幾輪做過損傷或干預測量的 49 個 expert（清單見 `excluded_experts.json`）。
- 來源包括：
  - fixed_route、first_order_validation、calibration_stability、crosstext 的候選或配對；
  - up_case、bc_numerics 報告中出現的 expert；
  - act_outlier、token_spike、spike_intervention、pair_gradient 四輪診斷的 tap。

**合格**
- 條件：
  - searched expert，有 Down 輸入的 act-error 與梯度 cache；
  - calibration rows ≥ 64，discovery rows ≥ 512；
  - R_token 有限，且 s_p50 > 0；
  - range 有限且 hi > lo；
  - cache 中 6 個候選的 S1／S2 與 G2／GS 都有限。
- 共 779 個合格。不合格的原因：已測量 49、覆蓋不足 189、無 cache 7。

**選點**（凍結於梯度之前；沒有使用任何 loss、誤排或 T_pair 資訊）

| 組 | 規則 | 結果 |
|---|---|---|
| 一般（16） | 每層從合格新 expert 隨機抽 1 個（seed 23601） | L0.E28、L1.E56、L2.E38、L3.E59、L4.E43、L5.E57、L6.E29、L7.E56、L8.E27、L9.E43、L10.E26、L11.E10、L12.E56、L13.E30、L14.E26、L15.E36；沒有空層 |
| spike 強（4） | R_token ≥ p90（16.4），隨機順序（seed 23602），每層最多 1 個 | L1.E50（R_token 19.9）、L5.E60（19.1）、L6.E9（41.1）、L9.E8（25.2） |
| spike 弱（4） | R_token ≤ p50（8.3），同層、log rows 最近 | L0.E44（**跨一層**：L1 沒有合格的弱 expert；log rows 差 0.10）、L5.E58、L6.E57、L9.E33 |

q0 下的 Down activation 格式：多數是 p8e1；L6.E9、L9.E8 為 f16，L0.E44 為 int8。

## 2. 方法與控制

**兩個分數**

| 分數 | 定義 |
|---|---|
| T_old | Σ_c G2_c/n·Σ_t r² + Σ_c GS_c μ_c²（原 BC） |
| T_pair | Σ_c Σ_t g² r² + 相同的偏差項 |

- r = e − μ；梯度語義與 BC cache 相同：sum-loss CE，route 與權重在 backward 中固定。
- 6 個候選：int4、int6、int8、p4e1、p6e1、p8e1。主要比較是 3 個同位元配對，其餘 12 個配對為次要。

**控制（`grad_controls.json`）：全部通過**

| 檢查 | 結果 |
|---|---|
| 梯度路徑 CE 與 `dr.forward` | 差 0.0 |
| 24 個 tap 的 G2／GS 與 cache | 逐位元相同 |
| 6 個候選的 S1／S2 與 cache | 逐位元相同 |
| rows 數與 cache 的 n | 相同 |
| T_old 與 cache 的 BC 成本 | 相對差 ≤ 3.6e-15 |
| 恆等式 V_pair,c − V_old,c = n·Cov(g², r²) | 最大相對誤差 1.5e-14 |
| streaming 累加與逐元素的 V_pair | 最大相對差 3.8e-16 |

**確認測量**

| 項目 | 結果 |
|---|---|
| 資料 | 96 篇新 WT103 文章 × 3 窗口；verified 掃描排除 6425 篇歷史文章；造訪 101 篇，5 篇因近似重複跳過 |
| 測量 | FP32 背景、一次量化一個 expert 的 Down 輸入、6 個格式、dynamic route |
| base 與 `dr.forward` | 差 0.0 |
| identity variant | 差 0.0 |
| 是否提前停止 | 否 |
| 輸入檔 hash | 前後一致 |

## 3. 凍結前的機制統計（calibration）

### 3.1 V_pair／V_old 在 24 個 tap 上的分布

| 格式 | 中位數 | 最小 | 最大 | < 2/3 的 tap 數 | > 1.5 的 tap 數 |
|---|---|---|---|---|---|
| int4 | 1.11 | 0.28 | 2.23 | 1 | 2 |
| int6 | 1.02 | 0.21 | 1.36 | 1 | 0 |
| int8 | 1.01 | 0.18 | 1.09 | 1 | 0 |
| p4e1 | 1.00 | 0.36 | 1.63 | 2 | 1 |
| p6e1 | 1.05 | 0.14 | 3.42 | 3 | 2 |
| p8e1 | 1.05 | 0.09 | 2.60 | 2 | 2 |

- 在一般 expert 上，比值幾乎都接近 1：位置相關很弱，所以兩個分數的預測相同。
- 偏離較大的 expert：
  - **L6.E9**（強 spike；p6e1 0.14、p8e1 0.09、p4e1 0.36、int4 2.23）：與上一輪 L1.E9 同型，Posit 被 BC 高估、INT 被低估。
  - **L9.E8**（強 spike；int4 0.28、int6 0.21、int8 0.18、p4e1 1.63）：方向相反，INT 被高估、p4e1 被低估。
  - **L5.E60**（p6e1 3.42、p8e1 2.60）。
  - **L9.E33**（Posit 約 0.5）。
- 主要 channel（V_old 最大的 channel）：在 L6.E9 占 p6e1 V_old 的 68%（比值 0.01），在 L9.E8 占 77%。在一般 expert 上，主要 channel 的占比多半 < 10%。

逐 tap 的完整統計見 `proxy_table.csv` 與 `mech.json`。

### 3.2 凍結的預測差異

- 360 個配對中，T_old 與 T_pair 符號不同的只有 8 個，全部在 spike 組：
  - 同位元 3 個：L6.E9 int4–p4e1、L9.E8 int4–p4e1、L0.E44 int8–p8e1；
  - 次要 5 個：L6.E9 int6–p4e1、L9.E8 int6–p4e1、L9.E8 int8–p6e1、L5.E60 int8–p6e1、L9.E33 int6–p4e1。
- proxy 平手：T_old 1 個，T_pair 0 個；兩個分數都用同一組配對評估。

## 4. 確認結果（實測可判定 = 文章 bootstrap 95% CI 排除 0）

### 4.1 各組摘要

| 組 | 配對集合 | 配對 | 可判定（覆蓋率） | T_old 對 | T_pair 對 | 修正 | 破壞 | 兩者皆錯 |
|---|---|---|---|---|---|---|---|---|
| 一般 | 同位元 | 48 | 9（19%） | 9 | 9 | 0 | 0 | 0 |
| 一般 | int4–p4e1 / int6–p6e1 / int8–p8e1 | 16 / 16 / 16 | 4 / 4 / 1 | 全對 | 全對 | 0 | 0 | 0 |
| 一般 | INT–INT | 48 | 14 | 14 | 14 | 0 | 0 | 0 |
| 一般 | Posit–Posit | 48 | 11 | 11 | 11 | 0 | 0 | 0 |
| 一般 | 跨家族 | 144 | 33 | 33 | 33 | 0 | 0 | 0 |
| 一般 | 全部 | 240 | 58（24%） | 58 | 58 | 0 | 0 | 0 |
| spike | 同位元 | 24 | **4（17%）** | 3 | 4 | 1 | 0 | 0 |
| spike | int4–p4e1 / int6–p6e1 / int8–p8e1 | 8 / 8 / 8 | 4 / 0 / 0 | 3 | 4 | 1 | 0 | 0 |
| spike | INT–INT | 24 | 6 | 5 | 5 | 0 | 0 | 1 |
| spike | Posit–Posit | 24 | 4 | 4 | 4 | 0 | 0 | 0 |
| spike | 跨家族 | 72 | 17 | 13 | 15 | 2 | 0 | 2 |
| spike | 全部 | 120 | 27（23%） | 22 | 24 | 2 | 0 | 3 |

另見 `figures/outcomes.png`。

### 4.2 主要指標與組別狀態

| 組 | 決定性同位元配對 | 淨值（修正 − 破壞） | expert bootstrap 97.5% CI | 狀態 | 99% 敏感度 |
|---|---|---|---|---|---|
| 一般 | 9 | 0 | [0, 0] | 不變 | 4 個 → 不足 |
| spike | 4 | +1 | [0, 3] | **不足**（< 8） | 2 個 → 不足 |

- 組內 Holm（主要配對）：一般組 0 個可判定；spike 組 1 個（L9.E8 int4–p4e1，T_pair 對、T_old 錯）。
- **expert 等權結果**

  | 組 | 有可判定配對的 expert | 平均準確率（T_old／T_pair） | 淨值為正 / 為負 |
  |---|---|---|---|
  | 一般 | 11 | 1.00／1.00 | 0 / 0 |
  | spike | 6 | 0.90／0.94 | 1（L9.E8）/ 0 |

- 合併數字只作描述：全部 85 個可判定配對中，T_old 對 80、T_pair 對 82，修正 2、破壞 0、兩者皆錯 3。不代表全模型平均。

### 4.3 修正與兩者皆錯的個案（都在 L9.E8）

| 配對 | T_old 差 | T_pair 差 | 實測（前 − 後） | 95% CI | 類別 |
|---|---|---|---|---|---|
| int4 vs p4e1 | +11.6 | −1.09 | −1.85e-2 | [−2.08e-2, −1.61e-2] | 修正（99% 也成立） |
| int6 vs p4e1 | +2.02 | −5.56 | −1.77e-2 | [−2.00e-2, −1.52e-2] | 修正（99% 也成立） |
| int4 vs int6 | +9.59 | +4.46 | −8.4e-4 | [−1.39e-3, −3.1e-4] | 兩者皆錯（int4 有利） |
| int4 vs p6e1 | +15.6 | +5.82 | −7.9e-4 | [−1.35e-3, −2.4e-4] | 兩者皆錯 |
| int4 vs p8e1 | +16.1 | +6.15 | −6.1e-4 | [−1.20e-3, −2e-5] | 兩者皆錯（99% 不成立） |

- L9.E8 的 p4e1 損傷很大（+1.8e-2）。BC 低估了它，配對修正了方向：V_pair／V_old 為 1.63。
- 同一個 expert 的 int4 讓 NLL 變好（−5.9e-4），兩個分數都預測 int4 最貴。這是負損傷，兩個非負的二次型分數原理上都無法表示。

### 4.4 其他預測不同但不可判定的配對

| 配對 | 實測差 | 95% CI |
|---|---|---|
| L6.E9 int4–p4e1 | +3.2e-4 | [−2.4e-4, +8.7e-4] |
| L6.E9 int6–p4e1 | −1.5e-4 | – |
| L0.E44 int8–p8e1 | −1.0e-5 | – |
| L5.E60 int8–p6e1 | −3.4e-5 | – |
| L9.E8 int8–p6e1 | −2.4e-4 | [−4.8e-4, +1.8e-5] |
| L9.E33 int6–p4e1 | +2.9e-5 | – |

都不算排對，也不算排錯。

## 5. 機制是否推廣（事後描述，不用來調整公式或建立門檻）

- **負共變異對應原 BC 的相對高估**：在 5 個 T_old 排錯的可判定配對中，T_old 高估的那個格式（實測較好、但 T_old 認為較貴），V_pair／V_old 5 / 5 小於 1，4 / 5 低於另一個格式。方向與假說一致，但樣本極少，而且大多來自 L9.E8。
- **配對差異大是否較容易修正**：依 |log(比值 a ÷ 比值 b)| 分組：

  | 區間 | 可判定配對 | 結果 |
  |---|---|---|
  | < 0.25 | 56 | 全部兩者皆對 |
  | 0.25–0.5 | 12 | 11 兩者皆對、1 兩者皆錯 |
  | 0.5–1 | 9 | 8 兩者皆對、1 兩者皆錯 |
  | ≥ 1 | 8 | 5 兩者皆對、**2 修正**、1 兩者皆錯 |

- 差異大的配對沒有出現破壞，但也只有 2 個修正。
- **一般 expert**：位置相關很弱（比值接近 1），配對不會改變預測，也就不可能改善或退步。這解釋了為何上一輪的效果只出現在極端 spike expert。
- 「只在某類 tap 有效」（例如極端 spike、主要 channel 占比高）是可以預先登記的下一輪假說；本輪沒有用它建立任何選擇規則。

## 6. 判讀

**依規則**
- 一般組狀態是「不變」（足夠的可判定配對，修正 = 破壞 = 0）。
- spike 組狀態是「不足」。
- 不符合 A、B、D，判讀為 **C**：保留原 BC，上一輪的結果視為局部機制解釋。
- 用較嚴格的 99% 門檻，兩組都不足，相當於 D。

**證據的意義**
- 在一般 Down 輸入 expert 上，逐位置配對沒有提供任何不同的排序資訊，原 BC 的可判定排序全對。
- 在 spike expert 上，配對改變了少數預測。可判定的 2 個都是修正，但個案集中在單一 expert（L9.E8），數量太少，不能宣稱改善，也不能宣稱等效。

**不做的事**
- 不在本輪事後加入「只用於 spike expert」之類的選擇規則。
- 不追加資料。
- 不宣稱算到 Hessian。
- 不宣稱單點排序與 knee 品質有關。
- 正式 pipeline 不變。

## 7. 成本

| 項目 | 本輪實測（24 個 tap × 6 格式，10 個 calibration 窗口） |
|---|---|
| 原 BC 統計：forward（全模型） | 4.13 s |
| 原 BC 統計：backward（到 L0） | 4.79 s |
| 原 BC 統計：G2／GS | 0.028 s |
| 原 BC 統計：S1／S2（6 格式） | 0.040 s |
| BC pass 的 GPU 峰值 | 4.60 GB |
| 新增配對累加 Σg²e²、Σg²e | **0.011 s**（約為 S1／S2 的 27%，fwd + bwd 的 0.12%） |
| 統計階段的 GPU 峰值 | 1.05 GB |
| 儲存：逐元素（只供本診斷） | 138 MB |
| 儲存：streaming 累加 | 2.36 MB（24 × 6 × 2 個 1024 維 float64） |
| 最終 loss 評估（144 個 variant，96 × 3 窗口） | 1598 s |

**全模型外推（估計值，未實測）**
- 1017 個 expert × 7168 欄 × 10 個格式 × 2 個陣列 × 8 B ≈ 1.17 GB，與現有 activation-error cache 同量級。
- 額外計算：在 backward 中對每個槽位與格式多做一次 Q(h) 與兩次加總。
- 本輪只量了部分 expert 的 Down 輸入，不能直接當作全模型的建立成本。

## 8. 計時與重現

| 階段 | 秒 | 預算（分） | GPU 峰值 |
|---|---|---|---|
| plan | 34.2 | – | – |
| select | 0.9 | 2 | – |
| grad | 17.2 | 8 | 4.60 GB |
| mech | 0.7 | 10 | – |
| data | 68.4 | 5 | – |
| measure | 1598.3 | 32 | 4.64 GB |
| analyze | 13.5 | 5 | – |
| figures | 0.5 | 3 | – |

GPU 合計約 26.9 分，在 42 分預算內。

```
cd experiments/mixed_precision
O=runs/olmoe_pair_generalization_20261023
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_pair_generalization.py
$CPU $S --stage plan --out-dir $O
$CPU $S --stage select --out-dir $O
$GPU $S --stage grad --out-dir $O
$CPU $S --stage mech --out-dir $O
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
| `plan.json` | `fd586218812ca2bd` |
| `excluded_experts.json` | `b9461e3593dc0d3a` |
| `selection.json` | `29e6ac51e31b4434` |
| `grad_controls.json` | `1ef009ffb31d0b38` |
| `mech.json` | `8bf89aba9efda93c` |
| `predictions.json` | `3aa293993413ac2c` |
| `proxy_table.csv` | `bb503d2a5186fdc3` |
| `freeze.json` | `bdbe5de0a6d1637f` |
| `tokens.npy` | `17cbd61ec8ff1081` |
| `segments.json` | `de7d30a29124bd9c` |
| `measure.json` | `cc993986cc33946c` |
| `analysis.json` | `630f66915d0dbaf3` |
| `comparisons.csv` | `f189398c62360401` |
| `per_article.csv` | `0bdb0829d9da3079` |

`work/tap*.npz`（132 MB）的 hash 都記錄在 `freeze.json`。

### 輸出檔

| 類別 | 檔案 |
|---|---|
| 預先登記與凍結 | `plan.json`、`plan.md`、`freeze.json`、`data_freeze.json` |
| 排除與選點 | `excluded_experts.json`、`selection.json`（含合格清單） |
| 梯度與控制 | `grad_controls.json`；逐元素資料在 `work/tap*.npz` |
| 機制統計與預測 | `mech.json`、`proxy_table.csv`、`predictions.json` |
| 資料與測量 | `segments.json`、`tokens.npy`、`measure.json` |
| 分析 | `analysis.json`、`comparisons.csv`（全部 360 個配對）、`per_tap.csv`、`per_article.csv`（逐篇 ΔNLL） |
| 圖表 | `figures/outcomes.png`、`figures/pair_ratio.png` |
| 計時與紀錄 | `timing.json`、`timing.jsonl`、`logs/` |
