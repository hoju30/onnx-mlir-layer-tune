# Qwen3-0.6B L27 Down：BC 預測與實測誤差為何反轉

日期：2026-10-24。腳本：`run_qwen_l27_reversal_diag.py`（sha256 `8098d6adb91ba440…`）。計畫：`plan.json`／`plan.md`（15:46:35 寫入）。事後探索：`run_qwen_l27_reversal_posthoc.py` → `posthoc/weight_context.json`。

## 性質

這是單點診斷，不改 BC、搜尋、knee、range、量化器或 pipeline；用 GPU FP32 + QDQ，不宣稱 native 結果。

- **設定**：與 BC 的 activation 項 C 相同的情境，即除了 L27 Down 的 activation QDQ 以外，全部 FP32（包含 L27 Down 權重）。
- **拆解方式**：
  - 槽位：input x（SiLU(gate)·up）、output y、xy 兩者（配置語意）。
  - 位置：全部、position 0、其餘位置 1–127。
  - 誤差項：用 ±α 掃描把實測 ΔL 拆成奇次的一階部分 a、偶次的曲率部分 b。
- **in-sample BC**：在同一批窗口上用 BC 的原公式重算各項，與實測比對。
- **資料**：BC 自己的 10 個 calibration 窗口，以及 128 篇新 WT103 文章 × 2 窗口。新資料以文章 bootstrap 給 95% CI。

## 結論摘要

BC 表 C[L27 Down] 的值：int4 6482 < p4e1 7177；f16 0.0003 ≪ p8e1 160。兩個反轉的原因不同。

**1. int4 vs p4e1：真正的反轉，只在 input（x）槽位**

- 在 BC 自己的情境（全 FP32）下，x+y 實測是 p4e1 較好 0.403 nat/token（CI [0.390, 0.416]），與 BC 相反。
- 分槽位看：
  - y：實測 p4e1 好 0.292，BC 也判斷 p4e1 較好，正確。
  - x：實測 p4e1 好 0.253（int4 0.504、p4e1 0.251），BC 卻判斷 p4e1 比 int4 差約 4.7 倍。
  - x 這一項的錯，蓋過了 y 的正確判斷，導致 x+y 的總分排錯。
- 不是一階效應：實測的曲率部分 b 本身就排對（int4 0.434、p4e1 0.251）；一階部分小，而且方向相同。
- 是 BC 的曲率近似錯了，兩邊都錯：
  - p4e1 被高估約 5 倍（BC/2 = 1.24，實測 b = 0.25）。
  - int4 被低估（0.26 vs 0.43）。
- position 0 是主因之一，但不是全部：
  - p4e1 在 position 0 的 BC/2 是 0.53，實測 b 只有 0.004（約高估 150 倍）。
  - 拿掉 position 0 後，BC 仍排錯：其餘位置 p4e1 0.70 vs int4 0.26，其中 p4e1 的 0.70 主要來自偏差項 U。
- 用同樣梯度、但保留窗口內跨 channel 與跨位置相關性的「窗口層級經驗 Fisher」½Σ_w(g_w·e)²，排序是對的（int4 1.96 vs p4e1 0.56）。所以錯誤來自 BC 的分解假設（逐 channel 對角化、位置合併的 g²、只看 channel 平均的偏差項），而不是使用梯度本身。
- 在 BC 自己的 calibration 窗口上，x 槽位同樣反轉。所以這是公式近似誤差，不是 calibration 到新文本的轉移問題。

**2. f16 vs p8e1：BC 大幅高估 p8e1；在 BC 自己的情境下，實測差幾乎為 0**

- 全 FP32 情境下，x+y 實測差 +0.0006，CI [−0.0012, +0.0024]，無法與 0 區分。BC 卻預測 p8e1 的成本是 160，與 int8（674）同一量級。
- 分槽位看：
  - x：p8e1 反而降低 loss 0.0038（CI [−0.0051, −0.0026]）。
    - 效果幾乎全在 position 0，而且是一階效應：a = −0.0101，梯度預測 g·e = −0.0103，幾乎一模一樣。
    - BC 只有非負的二階項，表示不了這種有利的一階效應；同時它把 p8e1 在 position 0 的曲率高估約 8 倍（0.029 vs 0.0034）。
  - y：p8e1 略差（+0.0018），BC 方向正確，但幅度高估約 5 倍。
  - x 的好處與 y 的壞處加上交互作用 +0.0015，幾乎互相抵消。
- 事後探索（未預先登記）：把 L27 權重換成 A 的 int8，x+y 差仍是 +0.0006，CI [−0.0012, +0.0024]。所以前幾輪看到的 p8e1 優勢不是 int8 權重造成的。
- 前幾輪在不同文章、不同情境下的量測（只作描述，不能直接相減）：

  | 情境 | f16 − p8e1（正值 p8e1 較好） | 來源 |
  |---|---|---|
  | 其他層全依 A 量化 | +0.0085（native +0.0087） | 前兩輪完整配置 |
  | 前層依 A 量化 | +0.0047 | 前一輪 Q 背景 |
  | 全 FP32 | +0.0006（不顯著） | 本輪 |

  這個順序暗示 p8e1 的好處與「其他層已被量化」有關。但這是跨批次的描述，要確認需要在同一批文章上直接比較這三種情境。
- 不論哪種情境，BC 都是在 FP32 情境下計算的，看不到這種依賴情境的效應。

**3. 共同點**
- 兩個反轉都集中在 Down input（x）。
- BC 在 x 槽位的排序，對 4／6／8 bit 三組同位元對全部錯：int6–p6e1、int8–p8e1 在 x 槽位也是 BC 選 INT、實測 Posit 較好。只是在 6／8 bit，y 槽位的正確判斷佔優，總分才排對。
- L27 x 的 range 是 [−1682, 721]，position 0 有 massive activation。Posit 在大數值上的絕對誤差大，BC 用合併所有位置的平均 g² 去乘這些誤差，但實際 loss 對 position 0 的反應很小。

## 1. 主要表格（新文章；單位 nat/token；D 正值代表 Posit 較好）

### 1.1 逐槽位的成對結果

| 對 | 槽位 | 實測 D [95% CI] | BC 預測方向 | 槽位反轉 | 一階 D_a [CI] | 曲率 D_b [CI] | 判讀 |
|---|---|---|---|---|---|---|---|
| int4–p4e1 | x | +0.2530 [+0.2348, +0.2711] | int4 較好 | 是 | +0.0700 [+0.0579, +0.0823] | +0.1829 [+0.1732, +0.1927] | 曲率近似錯 |
| int4–p4e1 | y | +0.2923 [+0.2765, +0.3083] | p4e1 較好 | 否 | −0.2509 | +0.5433 | BC 對 |
| int4–p4e1 | xy | +0.4033 [+0.3903, +0.4163] | 表 C：int4 較好 | 是 | | | |
| f16–p8e1 | x | +0.0038 [+0.0026, +0.0051] | f16 較好 | 是 | +0.0096 [+0.0083, +0.0109] | −0.0057 [−0.0059, −0.0055] | 曲率方向對，一階項翻盤 |
| f16–p8e1 | y | −0.0018 [−0.0031, −0.0004] | f16 較好 | 否 | +0.0051 | −0.0068 | BC 對（幅度過大） |
| f16–p8e1 | xy | +0.0005 [−0.0012, +0.0024] | 表 C：f16 遠較好 | 未定（差 ≈ 0） | | | |
| int6–p6e1 | x | +0.1555 [+0.1457, +0.1654] | int6 較好 | 是 | +0.0361 | +0.1194 | 曲率近似錯 |
| int8–p8e1 | x | +0.0051 [+0.0021, +0.0080] | int8 較好 | 是 | −0.0135 | +0.0186 | 曲率近似錯 |
| int6–p6e1、int8–p8e1 | y | +0.856、+0.409 | Posit 較好 | 否 | | | BC 對；xy 也排對 |

calibration（BC 自己的 10 個窗口）的方向與上表完全相同，例如 x int4–p4e1：實測 +0.205，BC 判斷 int4 較好。

### 1.2 x 槽位各格式的分項（新文章）

| 格式 | 位置 | 實測 ΔL [CI] | 一階 a | g·e 預測 | 曲率 b | BC/2 | 其中 V | 其中 U | ½Σ_w(g_w·e)² |
|---|---|---|---|---|---|---|---|---|---|
| int4 | 全部 | +0.504 [+0.487, +0.521] | +0.070 | +0.147 | +0.434 | 0.264 | 0.295 | 0.234 | 1.957 |
| int4 | pos 0 | +0.0012 | +0.0006 | +0.0006 | +0.0006 | 0.0053 | 0.011 | 0.000 | 0.002 |
| int4 | 其餘 | +0.503 | +0.070 | +0.146 | +0.434 | 0.260 | 0.284 | 0.235 | 1.947 |
| p4e1 | 全部 | +0.251 [+0.240, +0.262] | +0.000 | +0.025 | +0.251 | 1.242 | 1.218 | 1.267 | 0.555 |
| p4e1 | pos 0 | −0.0104 | −0.014 | −0.066 | +0.0036 | 0.531 | 1.047 | 0.015 | 0.395 |
| p4e1 | 其餘 | +0.262 | +0.014 | +0.091 | +0.248 | 0.705 | 0.174 | 1.235 | 0.867 |
| p8e1 | 全部 | −0.0038 [−0.0051, −0.0026] | −0.0096 | −0.0097 | +0.0057 | 0.032 | 0.063 | 0.001 | 0.014 |
| p8e1 | pos 0 | −0.0067 | −0.0101 | −0.0103 | +0.0034 | 0.029 | 0.057 | 0.000 | 0.011 |
| p8e1 | 其餘 | +0.0029 | +0.0005 | +0.0006 | +0.0023 | 0.003 | 0.006 | 0.000 | 0.004 |
| f16 | 全部 | 0.0000 | 0 | 0 | 0 | 0.0000 | | | |

欄位說明：
- BC/2 = ½T_old：把 BC 視為對曲率 b 的預測；V、U 為 T_old 的兩項（未除以 2）。
- g·e：用梯度預測的一階效應，應對應實測的 a。
- p8e1 這種小誤差，g·e 與 a 幾乎相同；int4 這種大誤差則差約 2 倍，奇次高階項不可忽略。

### 1.3 可加性（xy 相對 x + y）

| 格式 | 交互作用 | 95% CI |
|---|---|---|
| int4 | −0.502 | [−0.519, −0.485] |
| p4e1 | −0.360 | [−0.375, −0.345] |
| p8e1 | +0.0015 | [−0.0000, +0.0030] |

低位元格式明顯次可加：x 被量化後，y 再量化增加的損傷較少。BC 把 x 與 y 的成本直接相加，因此會高估 4-bit 的總成本，但這不改變 int4 vs p4e1 的方向。

### 1.4 事後探索：L27 權重 FP32 vs int8（`posthoc/weight_context.json`）

| 權重 | f16−p8e1：x | y | xy |
|---|---|---|---|
| FP32 | +0.0038 [+0.0026, +0.0051] | −0.0018 [−0.0031, −0.0004] | +0.0006 [−0.0012, +0.0024] |
| int8（A） | +0.0040 [+0.0027, +0.0053] | −0.0014 [−0.0029, +0.0001] | +0.0006 [−0.0012, +0.0024] |

- int4−p4e1 在兩種權重下都約 +0.40。
- 只量化 int8 權重本身：ΔL −0.0008。

## 2. 控制（全部通過）

| 控制 | 結果 |
|---|---|
| 同路徑 FP32 重播 vs 完整前向 | 逐位元相同（calibration 與每個新資料 chunk） |
| α = 0 | 誤差 0 |
| α = 1、全遮罩 | 等於 QDQ |
| y | 等於 F.linear(h, W) |
| position 127 的梯度 | 0 |
| in-sample BC x+y（calibration）vs native 表 C[L27] | 最大相對差 2.5e-3（p4e1），容差 1e-2 |
| 其他 | Posit LUT 驗證；TF32 關閉；deterministic；calibration tokens 與 prepare 相同 |

## 3. 資料

- 128 篇新 WT103 文章 × 2 個不重疊窗口，沿用前一輪的抽樣規則。
- verified 掃描排除 8617 篇歷史文章，含前一輪 128 篇。
- 合格池 18778 篇，訪問 158 篇。跳過：歷史 Qwen 窗口重疊 18、近重複歷史 WT103 8、近重複 WT2 4。
- seed：順序 31101、起點 31102、bootstrap 31103。

## 4. 回答與限制

**回答**
- 反轉來自 input 還是 output？兩個案例都來自 input（x）；BC 的 output（y）判斷方向正確。
- int4 vs p4e1：BC 的二階近似本身錯了，而且在 BC 自己的資料上就錯。
  - position 0 的 massive activation 讓 p4e1 被嚴重高估。
  - 其餘位置的偏差項 U 也高估 p4e1，int4 則被低估。
  - 保留跨 channel 與跨位置相關性的窗口層級經驗 Fisher 能排對。
- f16 vs p8e1：
  - 在 BC 的 FP32 情境下，兩者實際上沒有可區分的差異，BC 卻把 p8e1 估到 int8 等級的成本。
  - x 槽位上 p8e1 在 position 0 有可由梯度精確預測的有利一階效應，BC 表示不了。
  - 前幾輪看到的 p8e1 優勢，只在其他層也被量化的情境下出現，FP32 情境的 BC 本來就看不到。

**限制**
- 單一模型、單一層。
- GPU QDQ；128 篇 Wikipedia 文章。
- 事後探索未預先登記。
- 「情境依賴」的比較跨了不同文章批次，尚未確認。

**本輪不做的事**：不提出或採用任何 criterion 修改。若要追，建議另行登記：
1. 在同一批文章上比較 FP32、前層 A、全 A 三種情境的 L27 f16 vs p8e1；
2. 把「窗口層級經驗 Fisher」與 BC 在 Down input 槽位跨層比較排序（只作診斷）。

## 5. 耗時與重現

| 階段 | 時間 |
|---|---|
| plan | 23 s |
| calib | 22 s |
| audit | 35 s |
| data | 105 s |
| eval（256 窗口 × 189 條件，GPU 峰值 5.1 GB） | 359 s |
| analyze | 3 s |
| 事後探索 | 62 s |

**冒煙測試（計畫前，在 scratchpad）**：calib 跑過原 calibration 窗口；eval／analyze 跑過重複填充的 calibration 窗口；已看過 calibration 端結果，設計未改。計畫寫入後腳本未變更，沒有 amendment；事後探索使用另一支腳本，不影響原計畫。

```
O=runs/qwen3_l27_reversal_diag_20261024
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$CPU run_qwen_l27_reversal_diag.py --stage plan --out-dir $O
$GPU run_qwen_l27_reversal_diag.py --stage calib --out-dir $O
$CPU run_qwen_l27_reversal_diag.py --stage freeze --out-dir $O
$CPU run_qwen_l27_reversal_diag.py --stage audit --out-dir $O
$CPU run_qwen_l27_reversal_diag.py --stage data --out-dir $O
$GPU run_qwen_l27_reversal_diag.py --stage eval --out-dir $O
$CPU run_qwen_l27_reversal_diag.py --stage analyze --out-dir $O
$GPU run_qwen_l27_reversal_posthoc.py --out-dir $O      # 事後探索
```

**hash（sha256 前 16 碼）**

| 檔案 | sha256 |
|---|---|
| 主腳本 | `8098d6adb91ba440` |
| 事後腳本 | `110a48aed5b53af6` |
| plan.json | `e0896a847a3421ba` |
| plan.md | `d5d0310b534183ef` |
| freeze.json | `87536a6fe4bf0219` |
| controls_calib.json | `3102a49fa009ba7e` |
| controls_eval.json | `06179075293e9d42` |
| calibration_scores.json | `3fbe091e21a81189` |
| new_scores.json | `3e7ad55110a97e14` |
| new_dL.npz | `29703d07d0ea41be` |
| tokens.npy | `720b1d8b514cd53d` |
| segments.json | `9506512bbffbccb0` |
| stats.json | `de95c5697e9a3383` |
| pairs.csv | `6efc18e1c177cf25` |
| conditions.csv | `9ae9740c10b7adc4` |
| posthoc/weight_context.json | `dadbf9fa1b67d4ea` |

圖：`fig_terms.png`，各槽位、各遮罩的實測 ΔL、a、b、BC/2 與 g·e。
