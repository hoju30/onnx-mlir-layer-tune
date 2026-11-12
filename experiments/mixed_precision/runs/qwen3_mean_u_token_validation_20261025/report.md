# Qwen3-0.6B Down input：U_token 平均項的跨層與新文本驗證

日期：2026-10-25。
- 腳本：`run_qwen3_mean_u_token_validation.py`（sha256 `6589e960422d1108…`，與計畫時相同）。
- 計畫：`plan.json`／`plan.md`（02:57:22 寫入，之後沒有 amendment）。
- 規格：`reports/qwen3_mean_u_token_validation_prompt_20261025.md`。

## 首頁回答

**1. 新方法改了幾個選擇？**
- even selector（只看曲率項）：84 對中改了 **2 對**。
  - L24 int4–p4e1：int4 → p4e1。
  - L27 int6–p6e1：int6 → p6e1，即前輪已知的 6-bit 案例。
- 主要集合 P（L0–26 的 81 對）裡只改了 L24 這 1 對。
- signed selector（F + S）：改了 **3 對**，都在 P、都是 4-bit。
  - L9：int4 → p4e1。
  - L11：p4e1 → int4。
  - L12：int4 → p4e1。
- 校準分數沒有 tie。

**2. 修正幾對、新增幾對誤排？**

| 指標 | 集合 | 可判定 | 原方法排對 | 新方法排對 | 修正 | 新增誤排 |
|---|---|---|---|---|---|---|
| E(1/8)，even | P（81） | 70 | 69 | 70 | L24 int4–p4e1 | 0 |
| E(1/8)，even | 全 84 | 73 | 71 | 73 | L24、L27 int6–p6e1 | 0 |
| ΔL(+1)，signed | P（81） | 68 | 62 | 63 | L9 int4–p4e1 | 0 |
| ΔL(+1)，signed | 全 84 | 71 | 64 | 65 | L9 int4–p4e1 | 0 |

- 「可判定」指 raw 95% CI 不含 0；未定的配對仍留在 R 指標裡。
- Holm（E(1/8)，84 對一族）：72 對顯著，兩個修正都在其中，新增誤排 0。
- even 的 11 個未定配對全是 8-bit（P 內）。

**3. G_even 與 G_signed（P，nat/token，文章為單位，bootstrap 20000 次）**

| 指標 | 平均 | 95% CI | Bonferroni 97.5% CI | 判讀 |
|---|---|---|---|---|
| G_even（E(1/8)，實際損傷） | −1.22e-7 | [−1.34e-7, −1.09e-7] | [−1.36e-7, −1.07e-7] | 支持改善 |
| G_even 係數尺度（÷α0²） | −7.8e-6 | — | — | 偶次係數，不是部署時 NLL |
| G_signed（ΔL(+1)） | −3.34e-4 | [−3.97e-4, −2.69e-4] | [−4.06e-4, −2.59e-4] | 支持改善 |

兩者都是單層、人工平均偏移介入的 81 對平均，不是完整配置損傷，不能相加成 pipeline PPL 收益。

**4. 改善集中在哪裡？**
- G_even（P）**100% 來自單一配對 L24 int4–p4e1**。
  - 這不是已知失敗點（L2／L27），但只有一對。
  - 它是小差距配對：|ΔE(1/8)| 為 9.9e-6，P 的中位數是 3.9e-5。
  - 方向在文章間很一致：128 篇中 123 篇是 int4 較差。
- 全 84 對的 G_even 有 94% 來自 L27 int6–p6e1（已知案例）。
- G_signed（P）有 95.5% 來自 L9 int4–p4e1（128 篇中 113 篇 int4 較差）。
  - 另兩個改變的配對未定：L11 的點估計變差，L12 的點估計變好。
- 結論：改善真實、無新增誤排，但**廣度極窄**。U_token 與 U_old 在 82/84 的 even 選擇上相同；這輪的正面證據實質上是「它改動的少數配對都改對了」，不是「多數層的排序因它變好」。

**5. signed 正向 loss 也改善嗎？**
- 是。G_signed 在 95% 與 Bonferroni 97.5% CI 下都 < 0。
- 但只來自一對可判定的修正（L9 int4–p4e1）。
- 兩方法共同排錯的 signed 配對仍有 6 對，U 的改動完全沒有碰到它們：
  - P 內 5 對：L9 int6–p6e1、L13 int4–p4e1、L13 int6–p6e1、L16 int4–p4e1、L16 int6–p6e1；
  - 外加 L27 int4–p4e1。
- 這些配對多半是 ΔL(+1) 上 INT 較好、selectors 卻選 Posit，屬於一階／有限幅度問題，不是 U 的問題。

**6. 額外建表成本**（路徑 1，Python 診斷，不是 native latency）
- 比只算 U_old 多出約 **0.48 s**：快取寫 0.27 + 讀 0.16 + U_token 0.053。共同前向／反向本身是 1.06 s。
- 另需 **440 MB** 梯度快取（28 層 × 1280 × 3072 × float32）。
- 路徑 2（先 μ 再跑一次梯度 pass）估計多一次前向／反向，約 1.06 s，沒有實作。
- 詳見下方成本表。

## 性質與範圍

**做了什麼**
- Qwen3-0.6B，FP32 eager，全部 28 層的 Down input，每次只介入一層。
- 所有權重（含該層 Down）與其他 activation 都是 FP32，Down output 不量化。
- 介入方式：在新文章上把 x_l 換成 x_l + αμ_l,f，α ∈ {±1/8, ±1}。
  - μ 是原 calibration（10 × 128 窗口）每 channel 的平均量化誤差，新文章前已凍結。
  - 這是人工平均偏移，不是新文章的完整 Posit／INT QDQ。
- 每個介入都經過真實的 Down linear，即 W(x + αμ)，再跑完整 suffix（L0–26 含後續 attention），沒有用 y + αWμ。
- 規模：28 層 × 6 格式 × 4 幅度 = 672 個條件，加 FP32 baseline，共 256 個窗口（128 篇 × 2）。

**沒有做什麼**：沒有改 criterion、變異項、量化器、range、搜尋、knee 或 pipeline；沒有 HVP、完整量化、完整配置、weight-only、fine-tune 或 native。

**名詞**
- E(α) = [ΔL(+α) + ΔL(−α)]/2：固定幅度的偶次反應，含高階項，不是真正的 Hessian。這輪沒有用更小的 α 證明收斂。
- O(α) = [ΔL(+α) − ΔL(−α)]/2：有限幅度的奇次反應，不是純一階。

**凍結的分數**（原 calibration，μ 分母 1280）
- S_old = U_old / 2540，U_old = Σ_(s,c) μ_c²(Σ_p g)²。
- S_new = U_token / 2540，U_token = Σ_(s,p)(Σ_c g μ_c)²。
- F = F_b / 1270，F_b = Σ g μ。
- g 是每窗口 summed CE 的梯度。

## 選擇不同的配對：完整數字（nat/token，INT − Posit）

| 配對 | even：原 → 新 | signed：原 → 新 | ΔE(1/8) [95% CI] | ΔE(1) [95% CI] | ΔL(+1) [95% CI] | 判定 |
|---|---|---|---|---|---|---|
| L24 int4–p4e1 | int4 → **p4e1** | p4e1 → p4e1 | +9.88e-6 [+8.85e-6, +1.09e-5] | +6.33e-4 [+5.67e-4, +6.98e-4] | +7.92e-3 [+6.83e-3, +9.02e-3] | even 修正（Holm 0.0042） |
| L27 int6–p6e1 | int6 → **p6e1** | p6e1 → p6e1 | +1.62e-4 [+1.58e-4, +1.67e-4] | +1.04e-2 [+1.01e-2, +1.08e-2] | +2.60e-2 [+2.34e-2, +2.86e-2] | even 修正（Holm 0.0042） |
| L9 int4–p4e1 | p4e1 → p4e1 | int4 → **p4e1** | +3.98e-4 | +2.53e-2 | +2.58e-2 [+2.16e-2, +3.01e-2] | signed 修正 |
| L11 int4–p4e1 | p4e1 → p4e1 | p4e1 → **int4** | +1.12e-4 | +7.30e-3 | +7.3e-4 [−1.58e-3, +3.07e-3] | signed 未定（點估計變差） |
| L12 int4–p4e1 | p4e1 → p4e1 | int4 → **p4e1** | +1.47e-4 | +9.51e-3 | +1.95e-3 [−2.5e-4, +4.10e-3] | signed 未定（點估計變好） |

**兩個 even 修正**
- 在 E(1) 與 ΔL(+1) 上方向也一致：Posit 的平均偏移較不傷。
- signed selector 在這兩對原本就選對了，因為 F 已補上，所以 U 的改動在 signed 範圍內對它們沒有作用。

**校準分數**（INT／Posit）
- L24：S_old 0.0031／0.0079，S_new 0.0039／0.0034，S_new 差距約 15%。
- L27 int6：S_old 0.0123／0.0252，S_new 0.0184／0.0043。
- L9：S_new 0.0378／0.0004 比 S_old 0.0206／0.0002 更重罰 int4，壓過了 int4 的負 F（−0.0187）。

## 各位元、各層分布（P）

| 位元 | even 不同 | even 可判定 | 原／新排對 | signed 不同 | signed 可判定 | 原／新排對 | G_even | G_signed |
|---|---|---|---|---|---|---|---|---|
| 4-bit（27） | 1 | 27 | 26／27 | 3 | 23 | 20／21 | −3.66e-7 [−4.03e-7, −3.28e-7] | −1.00e-3 [−1.19e-3, −8.1e-4] |
| 6-bit（27） | 0 | 27 | 27／27 | 0 | 24 | 21／21 | 0 | 0 |
| 8-bit（27） | 0 | 16 | 16／16 | 0 | 21 | 21／21 | 0 | 0 |

- 全 84 對另含 L27：6-bit 修正 L27 int6–p6e1；L27 的 4-bit、8-bit 兩方法相同。
- even 誤排分布
  - 原方法只錯 L24 int4–p4e1 與 L27 int6–p6e1，新方法在 73 對可判定配對上全對。
  - 其餘 26 層的 even 排對數兩方法相同。
- even 未定：11 對，都是 8-bit，在 L0、3、5、6、9、11、12、13、14、15、16。
  - E(1/8) 差距約 1e-7，在雜訊內。
  - 它們的 E(1) 都可判定為 Posit 較好，兩方法也都選 p8e1。這是描述，E(1) 不是主要指標。
- Holm 與 raw 不一致只有 L1 int8–p8e1：raw 可判定，Holm 校正後 p 0.116。兩方法在這對選擇相同。

## 預先指定的補充集合

| 集合 | 對數 | even 不同 | G_even [95% CI] | signed 不同 | G_signed [95% CI] |
|---|---|---|---|---|---|
| P（L0–26，主要） | 81 | 1 | −1.22e-7 [−1.34e-7, −1.09e-7] | 3 | −3.34e-4 [−3.97e-4, −2.69e-4] |
| 全 84 | 84 | 2 | −2.05e-6 [−2.11e-6, −1.99e-6] | 3 | −3.22e-4 [−3.83e-4, −2.60e-4] |
| L27（3） | 3 | 1 | −5.41e-5 [−5.57e-5, −5.25e-5] | 0 | 0 |
| 去 L2／L27（78） | 78 | 1 | −1.27e-7 [−1.40e-7, −1.13e-7] | 3 | −3.46e-4 [−4.12e-4, −2.80e-4] |

- 去掉 L2／L27 的結果幾乎和 P 相同，因為 L2 的三對兩方法選擇相同。
- 全 84 的數值被 L27 int6 主導；依預先指定，它不取代主要結果。

## 奇次反應與凍結 F

- 比較的是新文章上的 O(1/8)/(1/8) 與凍結的 calibration F，共 168 個 layer × format。
- 差距中位數 4.9e-4，F 本身的 |F| 中位數 7.6e-4。
- 符號一致 129/168（77%）。
- F 在新文本上大致方向可轉移，但誤差與 F 同量級。O 是有限幅度奇次反應，不用來重選格式。
- signed 共同誤排的 6 對（見首頁第 5 點）說明一階／有限幅度問題仍在，與 U 的修改無關。

## 建表成本（Python 診斷，GPU FP32）

| 項目 | 時間／大小 |
|---|---|
| 共同 calibration 前向／反向（10 窗口，抓 28 層 X 與 g） | 1.06 s |
| 梯度快取寫入 | 0.27 s |
| 梯度快取讀回（μ 算好後） | 0.16 s |
| QDQ 與 μ（6 格式 × 28 層） | 0.31 s |
| U_old | 0.011 s |
| U_token | 0.053 s（約 U_old 的 5 倍，但絕對值小） |
| F | 0.053 s |
| 快取大小 | 440 MB（28 × 1280 × 3072 × float32） |
| 路徑 2 估計（先 μ 再跑一次梯度 pass） | 約 1.06 s（未實作） |
| calibration 峰值 GPU | 8.4 GB（含 672 條件的重播核對，不是 U_token 專屬） |
| 峰值 RAM | **未記錄**（缺口） |

- **路徑 1** 對 U_old 的新增成本約 0.48 s 加 440 MB 快取。
  - 快取隨 calibration tokens × Down 寬度 × 層數線性成長。
  - 對 OLMoE 等更大模型要另估。
- **native 的影響**：目前 `native_tables` 的梯度鏈與 ranges → errors 鏈並行，梯度 pass 不知道 μ。要用 U_token，只能二選一：
  - 保存逐 token 梯度（路徑 1）；或
  - 把梯度 pass 改到 errors 之後（路徑 2，序列化並多一次 pass）。
- 這一點是 pipeline 整合成本，不是 dot-product 的耗時。

**評估成本**
- 672 條件 × 256 窗口共約 50.5 分鐘：前 2 個 chunk 各 377 s，續跑 6 個 chunk 共 2274 s。
- 評估峰值 GPU 4.8 GB。
- analyze 89 s。

## 控制

**calibration**（`controls_calib.json`，全部通過）
- TF32 關閉、deterministic。
- Posit Universal LUT 驗證；calibration tokens 與 prepare 相同；權重是未修改的 checkpoint。
- 梯度驗證：grad forward 與 no-grad 前向差 0；g_x 與 g_y·W 相對差 0；position 127 梯度為 0。
- 重播 vs 完整前向：672 個 calibration 條件最大差 0（容差 1e-6）。
- α = 0 對照差 0。
- L27 重現前輪 U、U_token：相對差 ≤ 1.9e-16。F 同路徑 ≤ 2.2e-7（容差 1e-4）。
- L27 F 的梯度路徑差異另報，最大 4.1e-4 rel。
  - 來源：本輪用完整反向，前輪是 norm + head 子圖的 autograd。
  - 不影響 L27 的 signed 選擇，那兩對的 F + S 差距遠大於此。
- α = 1/8 的張量實際改變比例最低 0.99914（L11 p8e1）；rounding 殘差最大 4.5e-4 rel（L12 p8e1）。

**評估**（`controls_eval.json`）
- 8 個 chunk 在 L0、L13、L27 的 α = 0 重播與 baseline 差都是 0。
- 非有限值 0，矩陣完整。

**凍結鏈**
- plan → freeze（μ、scores、selectors、controls hashes，03:02:52）→ audit → data → data_freeze（tokens／segments／exclusions hashes，03:05:20）→ eval。
- eval 開始與每個 checkpoint 都核對程式、tokens、μ、selectors hashes。

## 資料

- 來源：128 篇全新 WikiText-103 train 文章，每篇 2 個連續不重疊的 128-token 窗口（每篇 254 targets）。
- 排除與抽樣規則
  - verified manifest 掃描排除 8745 篇歷史文章。
  - 歷史 Qwen token audit：4657 個窗口，無未解析來源。
  - WT2／近重複規則沿用前輪；任一窗口 WT2 5-gram overlap ≥ 50% 即拒收。
- seeds：順序 32101、起點 32102、bootstrap 32103，未與歷史衝突。
- 抽樣經過 162 篇，跳過如下：

| 跳過原因 | 篇數 |
|---|---|
| historical_qwen_overlap | 14 |
| near_duplicate_wt2 | 4 |
| near_duplicate_previous | 15 |
| window_wt2_overlap | 1 |

- 合格池 18650 篇。
- FP32 baseline 平均 3.945 nat/token。

## 揭露

**冒煙測試**（計畫前，02:43–02:57，scratchpad，已記入 plan.json）
- calibration 階段完整跑過：看過全部 calibration 分數、μ、selectors（含 2 個 even 差異）與控制。
- 第一版的 L27 F 重現失敗，原因是跨兩條梯度路徑比較。計畫前改成同路徑比較，路徑差另報。
- eval／analyze 只在 4 個 calibration 窗口上試跑，沒有新文章。

**評估中斷**
- 第一個 eval 程序在 03:18:11 寫完第 2 個 chunk 後被外部終止：沒有 Traceback，推測是前一段工作階段結束時連帶停掉的背景任務。
- 03:20:26 用同一指令續跑。前 2 個 checkpoint 通過 header hash 核對後重用。
- 中斷期間沒有讀取任何 chunk 的 loss，也沒有改動程式或計畫。
- `timing.json` 的 eval 項只記錄續跑的 6 個 chunk。

**沒有事後操作**：沒有擴樣本、換窗口、換 seed、改門檻或追加候選。

## 判讀（依規格第 8 節）

- **符合「支持作下一輪候選」的條件**
  - G_even 的 CI 支持改善，沒有新增誤排（raw 與 Holm 皆 0）。
  - P 內的收益不在已知的 L2／L27。
- **但收益只來自 1 對**（L24 int4–p4e1），不能說「多數層的平均曲率排序因 U_token 改善」。
  - 原 U 在 P 已經 69/70 排對，可改善的空間本來就只有這一對。
  - U_token 拿到了全部可判定的空間，同時沒有弄壞其他 69 對。
- **G_signed 也改善**，但同樣幾乎只來自 L9 一對；兩方法共同的 6 個 signed 誤排不受影響，一階／有限幅度問題保留。
- **整體判讀**：U_token 可作為平均曲率項的下一輪候選，屬於「不傷、且修正少數邊界配對」型證據，不直接換 pipeline。

**限制**
- CI 只反映文章抽樣；selectors 固定，沒有包含 calibration 批次的變異。
  - L24 的 S_new 差距只有約 15%，換一批 calibration 是否仍選 p4e1 未知。
  - CI 極窄是因為固定 μ 偏移在不同文章間的效果很一致，不代表選擇本身穩定。
- 介入是固定平均偏移，不是新文章的完整量化。
  - 前輪 L27 4-bit 顯示，完整誤差的排序由 residual 與交互項主導。
  - 平均項改善不等於完整 BC 或量化後 loss 改善。
- E(1/8) 不是局部 Hessian；這輪沒有做 α 收斂。
- 變異項、position 0、完整配置、native 都還沒有驗證。這輪即使通過，也只支持平均項候選。

依規格到此停止：不修改 criterion、不追加 HVP、不處理變異項、不做全模型配置或 native。

## 檔案

| 類別 | 檔案 |
|---|---|
| 計畫 | `plan.json`、`plan.md` |
| 校準 | `controls_calib.json`、`mu.npz`、`calibration_scores.json/csv`、`selectors.json`、`calibration_losses.npz` |
| 凍結 | `freeze.json`、`data_freeze.json` |
| 資料 | `audit.json`、`segments.json`、`tokens.npy`、`exclusions.json` |
| 評估 | `eval_dl.npz`、`controls_eval.json`、`work/eval_chunk00–07.npz` |
| 分析 | `stats.json`、`pairs.json/csv`、`per_article_losses.csv`、`per_window_losses.csv` |
| 時間 | `timing.json`、`timing.jsonl` |
| log | `logs/` |
