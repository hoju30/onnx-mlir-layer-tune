# Qwen3-0.6B L27 Down input：Posit 平均偏差項 U 的機制診斷

日期：2026-10-25。腳本：`run_qwen3_posit_mean_bias.py`（sha256 `cebf491b6a9af79b…`）。計畫：`plan.json`／`plan.md`（01:07:11 寫入）。規格：`reports/qwen3_posit_mean_bias_prompt_20261025.md` 與 2026-10-25 的需求。

## 性質與範圍

**做了什麼**
- 只在原 10 個 × 128-token calibration 窗口上做描述性機制診斷，沒有新文章、沒有泛化宣稱。
- 對象是 L27 Down input；所有權重（含 L27 Down）與其他 activation 都是 FP32，Down output 不量化。
- 格式：int4／p4e1 為主分析，int6／p6e1、int8／p8e1 為預先指定的對照。

**沒有做什麼**：沒有修改 criterion、變異項、量化器或 range；沒有搜尋、fine-tune 或 native 編譯。

**分解方式**
- e = Q(x) − x。
- μ_c = Σ_(s,p) e / 1280，以 float64 計算一次後固定。
- b = μ_c，施加在每個位置；r = e − b。
- 全部在 autograd 圖外建立並 detach。

**單位**：所有導數與 loss 先以 summed CE 計算，表格最後除以 1270 個 targets（nat/token）。U 類分數一律除以 2，以便與曲率 K_b 比較尺度，但重點是排序，不是數值是否接近。

**冒煙測試**：計畫前，在 scratchpad 用同一支腳本、同一批窗口完整跑過一次，已看到全部結果。之後程式、容差與設計都沒有改；正式量測與冒煙測試的 loss 逐位元相同。

## 首頁表：4-bit 主分析（nat/token）

| 項目 | INT4 | Posit4 | INT4 − Posit4 | 哪一方較大 |
|---|---|---|---|---|
| U/2（原 BC 平均偏差項） | 0.1179 | 0.5381 | −0.4202 | Posit4（4.6 倍） |
| U_elem/2 | 0.0368 | 0.1193 | −0.0825 | Posit4（3.2 倍） |
| U_token/2 | 0.0954 | 0.1631 | −0.0677 | Posit4（1.7 倍） |
| F_b（一階，g·b） | 0.0632 | 0.0637 | −0.0005 | 幾乎相同 |
| K_b（沿 b 的真實曲率，autograd HVP） | 0.0732 | 0.1003 | −0.0271 | Posit4（1.37 倍） |
| ΔL_b（只加平均偏移 b） | 0.1260 | 0.1377 | −0.0118 | Posit4（1.09 倍） |
| ΔL_r（只加剩餘誤差 r） | 0.3052 | 0.1330 | +0.1722 | INT4 |
| ΔL_e（完整量化） | 0.4605 | 0.2552 | +0.2054 | INT4（實測真值：Posit4 較好） |
| I_br = ΔL_e − ΔL_b − ΔL_r | +0.0294 | −0.0155 | +0.0449 | INT4 超可加，Posit4 部分互相抵消 |

- **HVP 驗證**：兩者的 K_b 都通過有限差分驗證，且小 α 曲線支持（E(α)/α² 收斂到 K_b）。
- **差距被放大的程度**：U 把平均偏移的 INT−Posit 差距放大到 K_b 的約 15 倍（−0.42 vs −0.027），ΔL_b 的約 36 倍（vs −0.012）。

## 6-bit／8-bit 對照（nat/token）

| 格式 | U/2 | U_elem/2 | U_token/2 | F_b | K_b | ΔL_b | ΔL_r | ΔL_e | I_br |
|---|---|---|---|---|---|---|---|---|---|
| int6 | 0.01233 | 0.00661 | 0.01842 | +0.01079 | 0.01415 | +0.02357 | 0.19602 | 0.21199 | −0.00760 |
| p6e1 | 0.02515 | 0.00774 | 0.00428 | −0.00804 | 0.00189 | −0.00602 | 0.05392 | 0.04665 | −0.00125 |
| int8 | 0.00185 | 0.00119 | 0.00486 | −0.00961 | 0.00321 | −0.00644 | 0.00711 | 0.00074 | +0.00007 |
| p8e1 | 0.00027 | 0.00016 | 0.00017 | −0.00043 | 0.00011 | −0.00033 | −0.00595 | −0.00640 | −0.00012 |

所有格式的 K_b 都通過有限差分與小 α 曲線驗證。

**各指標認為哪個格式的平均偏移較差**

| 對 | U/2 | U_elem/2 | U_token/2 | K_b（真曲率） | ΔL_b（實測） | ΔL_e（完整） |
|---|---|---|---|---|---|---|
| int4–p4e1 | Posit | Posit | Posit | Posit | Posit（差距小） | INT |
| int6–p6e1 | **Posit** | **Posit** | INT | INT | INT | INT |
| int8–p8e1 | INT | INT | INT | INT | **Posit** | INT |

## 回答

### 1. U 是否把 Posit 平均偏移本身排錯？

| 對 | 判讀 |
|---|---|
| **4-bit（主）** | **方向沒有排錯，但幅度嚴重高估。** 真實曲率 K_b、實測 ΔL_b 也都認為 Posit4 的平均偏移略差，與 U 同向。問題是 U 把差距放大約 15 倍（相對 K_b），使 BC 總分中 Posit4 的成本被大幅灌水。 |
| **6-bit** | **排錯（規則 A 成立）。** U/2 認為 p6e1 的平均偏移較差；但 K_b（0.0019 vs 0.0142）與 ΔL_b（−0.006 vs +0.024）都說 p6e1 的平均偏移傷害小得多，甚至降低 loss。K_b 已由有限差分與小 α 曲線確認。 |
| **8-bit** | U 與 K_b 的排序一致（int8 較差），但一階項 F_b 讓實測 ΔL_b 的排序翻轉：int8 的平均偏移在一階上降低 loss（F_b −0.0096）。屬規則 B。 |

### 2. 證據支持哪一種機制？

**跨位置的經驗聚合是 U 灌水的主要來源（三種位元都有）**
- U = Σ_(s,c)(Σ_p g)²μ²，相對 U_elem 多出跨位置的乘積項。對 Posit4，這一項佔 U 的 78%（U/2 0.538 vs U_elem/2 0.119）。
- L27 Down 之後只有逐位置運算，真實的 loss 在位置之間沒有曲率交互：本輪已驗證 K_all = K_P0 + K_Rest，以及 mean-only ΔL 可逐位置相加，誤差都在浮點精度內。所以這一項不是真實損傷。
- Posit 的平均誤差集中在少數大幅值 channel，這些 channel 的梯度在位置之間同號，跨位置乘積因此特別大：
  - Posit4 的 U 有 98% 落在前 10 個 channel；
  - 第 39 個 channel 單獨佔 48%（μ = −106）；
  - INT4 前 10 個 channel 佔 73%，μ 只有 ±10–24。
- 只保留同 token 內跨 channel 項、不含跨位置項的 U_token，在三組位元對上都與 K_b 同序；U 與 U_elem 在 6-bit 都排錯。這只是 10 個窗口上的描述性線索，不是唯一根因的證明。

**一階項**
- 4-bit：F_b 幾乎相同，不影響排序。
- 6-bit：p6e1 的 F_b 為負，有利。
- 8-bit：int8 的 F_b 為負，有利，並翻轉了 ΔL_b 的排序。
- 純非負的 U 無法表示這些 signed 效應。

**高階非線性**：α = 1 時，扣掉 F_b、K_b 之後的剩餘項 R 為 int4 −0.010、p4e1 −0.026，約佔 ΔL_b 的 8%–19%，不改變排序。小 α（≤ 1/8）時，R 小於 1e-4。

**平均與剩餘分量的非加性（補償）**
- 4-bit 完整損傷的 INT−Posit 差距是 +0.205，來源拆解：

  | 來源 | 貢獻 |
  |---|---|
  | 剩餘誤差 r | +0.172 |
  | 非加性 I_br | +0.045 |
  | 平均分量 | −0.012 |

  所以 Posit4 實際較好，主要是因為它的剩餘誤差（變異部分）傷害小很多，平均分量反而略不利。
- I_br 的方向：INT4 為正（超可加），Posit4 為負（部分抵消）。但 Posit4 的完整損傷（0.255）並不比 mean-only（0.138）小，不符合規格 C「平均損傷大、完整損傷卻較小」的典型樣態。只能說非加性擴大了差距，不能說平均項本身需要打折。
- 8-bit：int8 的平均偏移有利（−0.006），補償了大部分剩餘誤差的傷害（0.007 → 完整 0.0007）。

**position 0**：U 幾乎全在其餘位置（U0 只佔 Posit4 的 0.05%），F、K 也主要在其餘位置。position 0 不是本輪 U 問題的來源。

### 3. 仍未證明的部分

- 結果只來自 L27 與 10 個 calibration 窗口，沒有新文章、沒有其他層，也沒有重複樣本可做推論。
- 不知道去掉跨位置聚合（例如改用 U_token 型分數）能否在新資料與其他層改善排序、且不在原本排對的配對上造成新誤排。
- U 類分數與 ΔNLL 沒有數值尺度校準；比例只能作排序與相對幅度的線索。
- I_br 的來源（INT4 為何超可加、Posit4 為何部分抵消）沒有進一步拆解。
- channel 39 等大幅值 channel 的角色只做了描述，沒有因果介入。
- position 0 的變異項問題（前輪發現的另一個高估來源）本輪沒有處理；即使修正 U，也不代表整個 BC 已修好。

### 4. 下一個低成本修正應保留的資訊（候選，需另行凍結驗證）

1. **保留**：signed 的每 channel 平均誤差 μ，以及同 token 內跨 channel 的梯度耦合（Σ_c g_spc μ_c，逐 token 計算）。
2. **避免**：把跨位置的梯度和先相加再平方，即 (Σ_p g)²。
3. **保留**：signed 的一階資訊 F_b = Σ g·b，它在 6／8-bit 會改變平均分量的排序。
4. **保留**：剩餘誤差（變異）項。它決定了 4-bit 的實際排序，不能為了修 U 而犧牲。
5. **成本提示**：U_token 型分數需要在梯度 pass 時已知 μ（先跑誤差統計，或保存每列的 g·μ）。native pipeline 目前的梯度與誤差統計是分開的 pass，需要調整順序或多一次 pass。

即使這樣做，也只是候選。下一輪要另行凍結方案，並在新文章、其他層，以及原本排對的 INT/Posit 配對上驗證：既要改善排序，也不能增加新的誤排。

## 控制（`controls.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 完整 FP32 前向 vs suffix 重播 | 逐位元相同 |
| 只改 Down input | W_down 為 FP32（sha256 `2c392668…`）；y = linear(x′, W)；沒有 output QDQ |
| 重現前輪 calibration 的 full loss | 6 種格式差 0（`runs/qwen3_l27_reversal_diag_20261024/calibration_dL.npz`） |
| 重現前輪的 U | 相對差 0（calibration_scores.json） |
| 兩種 U 算法 | 相對差 ≤ 2e-16 |
| μ 與 r | r 的 channel 平均 ≤ 1e-6；b + r − e 最大 1.5e-5 |
| 重建差異 | x+e 與 Q(x) 完全相同；x+b+r 與 Q(x) 有浮點差（最大 6.1e-5，3–30% 元素不同），loss 差 ≤ 3.6e-5 summed CE／窗口，視為重建差異、不當機制 |
| g_x（autograd）vs g_y·W_down | 最大相對差 8.3e-6 |
| position 127 的 g 與 H_b | 都恰好為 0 |
| α = 0 | ΔL = 0 |
| α = 1 | 等於 mean-only |
| 位置可加性：U = U0 + UR + Ucross | ≤ 2e-16（Ucross 為負，保留） |
| 位置可加性：F、K | ≤ 4e-16 |
| 位置可加性：mean-only ΔL | 差 0 |
| HVP vs 梯度中心差分（h = 1/32、1/64、1/128） | 6 種格式的相鄰步長都收斂，FD(1/128) 與 HVP 一致（K 相對差 ≤ 0.04%），向量相對誤差 0.05%–3.6%，全表見 `raw.json` |
| 小 α 改變的元素比例 | ≥ 99.8%；rounding 相對殘差 ≤ 3e-5 |
| GPU 設定 | TF32 關閉、deterministic、float32 |
| 量化器 | INT scale／zp 在裝置上；Posit LUT 與 Universal 相符 |
| 非有限值 | 0 |

沒有做 float64 參考。

## 平均方向曲線（`analysis.json` 的 curve；`fig_mean_direction_curves.png`）

- 以 4-bit 為例：
  - α = 1/16 時，INT4 的 O(α)/α 為 0.0631、E(α)/α² 為 0.0733，與 F_b 0.0632、K_b 0.0732 吻合。
  - Posit4 為 0.0635／0.1005，對應 F_b 0.0637、K_b 0.1003。
- α 越大，剩餘項 R 越大；α = 1 時 Posit4 為 −0.026（+α）與 +0.051（−α）。
- 全部 11 個 α 都已報告。

## channel（`channels.csv`）

- 全部 3072 個 channel 的 μ（6 種格式）、GS、U_c，以及各位元的 INT−Posit 差都已保存。
- U 前 10 個 channel 的佔比：int4 73%、p4e1 98%、int6 36%、p6e1 97%、int8 26%、p8e1 96%。
- Posit 的 U 集中在 channel 39、47、81、222、386。
- 只作描述，沒有依 channel 的 loss 建立 mask。

## 耗時（`timing.json`；診斷成本，不是 native 建表 latency）

| 項目 | 時間 |
|---|---|
| FP32 前向與梯度 | 0.93 s |
| QDQ 與誤差統計 | 0.03 s |
| proxy 分數 | 0.01 s |
| HVP（6 格式 × 3 個方向） | 3.34 s |
| HVP 有限差分驗證 | 5.06 s |
| 介入與 α 曲線 | 8.31 s |
| run 總計 | 25.5 s |
| 分析 | 1.4 s |

GPU 峰值 5.4 GB。

## 重現與 hash

```
O=runs/qwen3_posit_mean_bias_20261025
env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python run_qwen3_posit_mean_bias.py --stage plan --out-dir $O --smoke-note <smoke note json>
env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python run_qwen3_posit_mean_bias.py --stage run --out-dir $O
env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python run_qwen3_posit_mean_bias.py --stage analyze --out-dir $O
```

**hash（sha256；腳本、計畫為完整碼）**

| 檔案 | sha256 |
|---|---|
| 腳本 | `cebf491b6a9af79b236d2ece96fe361eb8f875dd8000e0391085b38e4707e47a` |
| plan.json | `bc30011c9d2aaf3aa98b1aa02a9ddf131b12ce5b87e2221890475f5725742ebf` |
| plan.md | `a45611cecc836a9f79f6958e853e4215f237f9d2204f391e9dbb574463a24180` |

- 輸入 hash 記錄在 `plan.json["inputs"]`。
- 輸出檔案：controls.json、raw.json（U／U_elem／U_token 逐窗口、F／K 逐窗口、HVP-FD 全步長）、losses.npz（逐窗口 ΔL_b、ΔL_r、ΔL_e、x+e、x+b+r、全部 α、P0／Rest）、channels.csv、analysis.json、summary_table.csv、fig_mean_direction_curves.png、timing.json。
