# Qwen3-0.6B L2 Down input：原始量化誤差的 spike／一般 token 分區與有限幅度診斷

日期：2026-10-26。
- 腳本：`run_qwen3_raw_error_spike_diagnostic.py`。
- 計畫：`plan.json`／`plan.md`（03:58:51 寫入，沒有 amendment）。
- 規格：`reports/qwen3_raw_error_spike_diagnostic_prompt_20261026.md`。

**性質**
- 只做 L2 Down input 的事後機制診斷。
- B（recenter 輪的 128 篇）的完整量化 loss 先前已看過，本輪的分區重播是事後延伸，不是獨立確認或盲測。
- calibration 只有 10 個窗口，只作描述，不做 bootstrap。

## 首頁回答

### 表 1：六格式在 α = 1 的分區損傷與 raw proxy

只把指定分區換成量化值，其他維持 FP32。單位 nat/token：calibration 為 summed CE ÷ 1270；B 為每篇 3 窗口 ÷ 381 後對 128 篇取平均。

| 格式 | calib S-only | calib N-only | calib I_SN | calib 完整 | B S-only | B N-only | B I_SN | B 完整 | P_S raw | C_N raw | D raw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| int4 | 0.008 | 0.312 | +0.002 | 0.322 | 0.015 | 0.271 | +0.0005 | 0.286 | 23.2 | 1092.5 | 1115.7 |
| int6 | 0.014 | 0.312 | +0.000 | 0.327 | 0.015 | 0.271 | +0.0003 | 0.286 | 11.8 | 1092.5 | 1104.3 |
| int8 | 0.002 | 0.312 | −0.001 | 0.313 | 0.002 | 0.271 | +0.0003 | 0.273 | 1.7 | 1092.5 | 1094.2 |
| **p4e1** | **3.969** | 0.027 | −0.051 | **3.945** | **3.804** | 0.021 | −0.001 | **3.825** | 673.6 | 183.2 | 856.8 |
| p6e1 | 0.050 | 0.002 | −0.001 | 0.052 | 0.063 | 0.001 | +0.000 | 0.064 | 503.9 | 8.2 | 512.1 |
| p8e1 | −0.003 | −0.001 | 0.000 | −0.004 | 0.001 | −0.000 | −0.000 | 0.001 | 96.8 | 0.5 | 97.3 |

- I_SN = 完整 − S-only − N-only。
- B 的 128 篇中，p4e1 S-only 每篇都是正值；95% CI [3.72, 3.89]，屬事後描述性。
- 若把 proxy 換成 ÷2540 的曲率慣例尺度（不代表 Hessian）：
  - p4e1：P_S 0.265、C_N 0.072；
  - int4：P_S 0.009、C_N 0.430。

### 表 2：同位元對的實測損傷差分解（Posit − INT，正值代表 Posit 較傷）

分解式：D_all = D_S + D_N + D_int，數值恆等誤差 ≤ 9e-16。

| 配對 | 資料 | D_S | D_N | D_int | D_all |
|---|---|---|---|---|---|
| p4e1 − int4 | calibration | **+3.961** | −0.286 | −0.053 | +3.622 |
| p4e1 − int4 | B 平均 [事後 95% CI] | **+3.789** [3.705, 3.874] | −0.250 [−0.266, −0.233] | −0.001 [−0.015, +0.012] | **+3.538** [3.458, 3.619] |
| p6e1 − int6 | calibration | +0.036 | **−0.310** | −0.001 | −0.275 |
| p6e1 − int6 | B 平均 | +0.048 [0.043, 0.053] | **−0.269** [−0.287, −0.252] | −0.000 | −0.222 [−0.239, −0.205] |
| p8e1 − int8 | calibration | −0.005 | **−0.313** | +0.001 | −0.317 |
| p8e1 − int8 | B 平均 | −0.001 [−0.003, −0.000] | **−0.271** [−0.288, −0.254] | −0.000 | −0.272 [−0.290, −0.256] |

B 的文章符號比例：
- p4e1 − int4：D_S 與 D_all 在 128/128 篇都是正值。
- 6／8-bit：D_N 在 128/128 篇都是負值。

**回答：Posit4 多出的約 3.54 nat/token 出現在哪？**
- 幾乎全部出現在「只量化 spike 行」：B 上 D_S = +3.79。
- 被 INT 在一般 token 較大的損傷抵銷一部分（D_N = −0.25）；交互項約為 0。
- 一般 token 上 p4e1 只損失 0.02 nat/token，問題不在一般 token 的粗量化。

**回答：Posit6／8 為何沒有同樣的嚴重損傷？**
- 實際候選值集的 maxpos 分別是 p4e1 = 16、p6e1 = 256、p8e1 = 4096，spike 約 3000–3960。
- p6e1 把 spike 行中超過 256 的元素（約每行 6 個）削到 256，S-only 損傷只有 0.05–0.06。
- p8e1 的範圍涵蓋 spike，損傷約 0。
- 三個 INT 格式的 range 被 spike 撐到 [−3012, 3787]，一般 token 的輸入 100% 被量化成 0（calibration 與 B 都實測如此），損傷 0.27–0.31，三種位元相同。
- 所以 6／8-bit 由 INT 的一般 token 損傷主導，Posit 較好；4-bit 則由 p4e1 的 spike 行損傷主導，INT 較好。

### 表 3：有限幅度排序、F 的作用與數值狀態（calibration）

| 方向 | F（g·e_G） | 小 α 的 O/α | 小 α 的 E/α² | α = 1/2 | α = 1 | α = −1 | 判讀 |
|---|---|---|---|---|---|---|---|
| p4e1 S-only | 0.053 | 收斂到 0.053（α ≤ 1/64） | 約 0.14–0.15（α = 1/4 到 1/128 穩定） | 0.032 | **3.969** | −0.008 | 局部反應可靠；α = 1 是強烈的單側有限幅度跳升 |
| p6e1 S-only | 0.039 | 收斂到 0.039 | 約 0.08（穩定） | 0.022 | 0.050 | −0.007 | 平滑；α = 1 低於局部二次模型 |
| int4 S-only | 0.005 | 收斂到 0.0049 | 約 0.0034（穩定） | 0.003 | 0.008 | −0.002 | 局部反應可靠，量級小 |
| int4／6／8 N-only | 0.011 | 從 0.093 緩慢降到 0.011 | 由 0.22 隨尺度變小，α ≤ 1/128 轉負 | 0.087 | 0.312 | 0.126 | **局部曲率未定**（沒有穩定平台） |
| p4e1 N-only | 0.001 | 緩降到 0.001 | 小尺度轉負 | 0.007 | 0.027 | 0.013 | 量級小；局部曲率未定 |

**排序**
- p4e1 與 int4 的 S-only 在 α ≤ 1/2 的各個 α，損傷差都只有約 0.03 以下。
- 只有 α = 1 時，p4e1 才大到主導整個 4-bit 排序。
- F 的作用：p4e1 − int4 在 spike 分區的 F_S 差約 +0.048（完整 F_all 差 +0.038），與局部 E 同向，但量級只有 α = 1 實測差（3.62）的約 1%。F 不是翻序來源。

**數值狀態**
- 所有分區在 α = 1 的 x + e_G 重建都與 masked QDQ 逐位元相同，loss 差異也是 0。
- 小 α 的實際改變比例 ≥ 0.9997。
- 極小 α 時 E/α² 會受 FP32 消去誤差影響，這部分只報告、不解讀。

**目前 spike proxy 漏掉的是什麼？證據是否足夠？**
- **觀察到**：P_S 與 p4e1 spike 方向的局部偶次反應同量級（P_S/2540 = 0.265，局部 E/α² 約 0.15；局部 F + E 在 α = 1 約 0.20）。漏掉的是 α 介於 1/2 與 1 之間的**單側、大幅度跳升**：spike 從約 1900 再降到 16 時損傷從 0.03 跳到 3.97，而 spike 變大（α = −1）則無害。
- **證據足夠的部分**
  - 損傷位置（S-only 主導）：calibration 與 B 一致，B 的 128 篇全部同號。
  - 局部 proxy 與有限幅度反應不一致：calibration 曲線直接顯示，局部量經 F／O 收斂支持。
- **證據不足的部分**
  - 跳升的精確位置：grid 在 1/2 與 1 之間沒有點。
  - 觸發的是 spike channel 本身，還是同一行其他被削到 16 的 channel：沒有拆分 channel，本輪不新增遮罩。
  - 為何是 16 而不是 256：只有兩種格式、方向也不同。
  - 其他層是否有同樣現象：只做 L2。
- 不宣稱任何局部 proxy 都不可能捕捉 spike 損傷，也不宣稱已找到唯一根因。

## 設定

**範圍**
- Qwen3-0.6B L2 Down input x = SiLU(gate)·up；一次只改 L2 Down input。
- 所有權重、其他 activation、Down output 都是 FP32。
- 原 calibration、INT range（[−3011.6, 3787.0]）、Universal Posit LUT、RMSNorm eps 1e-6、loss。

**loss closure**
- 真實 Down、residual、L3–L27 的全部 attention／FFN、final norm、LM head、next-token CE。
- 使用 recenter 輪的評估器路徑：LM head 每次 16 窗口；B 每 chunk 32 窗口。

**遮罩**
- a = max_c\|x\|，門檻固定為 recenter 輪的 11.2965（6 × calibration median），S = {a ≥ 門檻}。
- calibration：S 為 10 行，全是 position 0，與 masks.npz 逐位元相同。
- B：用同一門檻建 S，共 405 行。
  - 384 行在 position 0，另 21 行分散（position 1 有 10 行，其餘零星）；
  - 每個窗口 1–3 行，沒有空組。

**方向與幅度**
- q = Q(x) 在 FP32 x 上算一次；e = q − x，不減平均。
- e_S = M_S·e、e_N = M_N·e，固定並 detach。
- α = 1 用 masked QDQ：S-only = where(S, q, x)、N-only = where(N, q, x)、all = q。
- calibration 的其他 α 用 x + αd，α ∈ {0, ±2⁻ᵏ, k = 0…10}。

**一階項**：在同一 closure 用 GPU 求一次 FP32 baseline 梯度 g，F_G = g·e_G。

**proxy**：D = C_N + P_S，在原 CPU calibration 快取上重現，分項與總分和 recenter 輪相對差為 0。

## 統計（只作解釋）

| 格式 | 一般 token：\|e\| 中位數 | 一般 token：輸出為 0 的比例 | spike 行：超過 maxpos 的元素 | spike 行：\|e\| 最大值（calib） |
|---|---|---|---|---|
| int4／6／8 | 0.047（= \|x\|，全部量化為 0） | 100%（calib 與 B 都實測） | —（spike 行也有 99.7–99.8% 輸出為 0） | 225／53／13 |
| p4e1 | 0.039 | 0 | 0.29%（calib 88／30720；B 0.29%） | 3771 |
| p6e1 | 0.006 | 0 | 0.20%（calib 60；B 0.19%） | 3531 |
| p8e1 | 0.001 | 0 | 0 | 1562 |

- p4e1 的正值集合是 {0.0625, 0.25, 0.5, 1, 2, 4, 16}。
- 一般 token 的 \|x\| 最大值為 10.8（calib）／11.1（B），都低於 16，所以 p4e1 不會在一般 token 上截斷。

## 控制（全部通過）

**環境與量化器**
- TF32 關閉、deterministic、FP32、eager、RMSNorm eps 1e-6。
- Posit LUT 與 Universal 不一致數為 0。
- 在 CPU 快取 h 上做的 GPU QDQ 與 recenter 的誤差 sha 逐位元相同（6 格式）。

**重播與介入位置**
- baseline 重播與無 hook 前向的差異為 0。
- 以 hook 替換 L2 Down input 的完整前向與重播的差異為 0（18 個 S／N／all 條件）。
- 只改 L2 Down input：W 不變、y = linear(x, W)。

**舊結果重現**
- calibration 遮罩（GPU x 與 CPU 快取）與 recenter 相同；D 分項與總分重現，相對差 0。
- calibration 的 all 與前輪變異診斷的 Q(x) loss 相同（差 0）。
- B 的 identity 與 all 和 recenter 輪的 L2 eval_ce 逐位元相同（差 0）。

**方向與梯度**
- e_S + e_N = e 逐位元成立；α = 0 與零方向都是 0；F_all = F_S + F_N（相對誤差 4.4e-16）。
- g_x = g_y·W（差 0）；position 127 的梯度為 0。
- 非有限值 0。

**只報告的項目**
- GPU x 與 CPU 快取 h：最大差 2.7e-3，為最大值的 7.1e-7。
- GPU g 與 CPU g：差為最大值的 1.2e-4。
- 兩者不混用：D 用 CPU，F 用 GPU。

## 成本（診斷成本，不是 native 建表）

| 項目 | 時間 |
|---|---|
| 模型載入 | 9.1 s |
| calibration prefix／baseline | 1.2 s |
| baseline backward（10 窗口，microbatch 1） | 0.7 s |
| QDQ 與統計 | 0.2 s |
| calibration 曲線（18 方向 × 22 α + 18 個 α = 1 masked） | 109 s |
| hook 完整前向核對 | 5.1 s |
| D 重現 | 2.5 s |
| B 分區重播（384 窗口 × 19 條件） | 215 s |
| analyze | 2.1 s |

峰值 GPU 6.6 GB（B）、峰值 RSS 5.4 GB。

## 揭露

**冒煙測試**（03:53–03:58，scratchpad，只跑 plan 與 calib，沒有 B）
- 第一次因控制檢查中的 numpy bool 無法寫入 JSON 而停止；改為轉成 Python bool／float，公式與容差都沒改。
- 第二次全部通過，並已看過：calibration 的 α = 1 分區損傷（即表 1 的 calibration 欄）、Posit 值集、GPU／CPU 差異、D 重現與時間。曲線與 F 有計算，但當時沒有檢視。
- 看過之後，問題、公式、遮罩、α grid、容差與判讀都沒有修改。
- 正式 calibration 的 loss 與冒煙逐位元相同。

**B 的性質**：B 的完整量化 loss 在 recenter 輪已看過，分區是在已知結果上的事後拆解。

依規格到此停止：不修改正式 BC 或 D，不新增格式、門檻或係數，不追加層或新文章，不搜尋配置、不改 Posit scaling、不做 fine-tuning 或 native。

## 下一個具體假說（只提出，未測試）

**假說：spike 行的評分改用路徑兩端梯度的梯形估計 T_S = ½[g(x) + g(x + e_S)]·e_S**

**理由**
- 本輪顯示 p4e1 的 spike 損傷是單側、在 α 接近 1 才出現的大跳升。
- 只用 g(x) 的局部量（F、P_S 或 HVP）在 α ≤ 1/2 都描述得很好，但看不到端點的反應。
- 在量化後的端點 x + e_S 再求一次梯度，仍屬於梯度式 criterion，不是前向量測雜訊。若端點附近 loss 急升，g(x + e_S)·e_S 會很大；若像 p6e1 一樣平滑，則接近局部值。

**成本**
- 只對有 spike 行的層、每格式多一次 backward（本批 28 層中只有 10 層有 spike 行）。
- 只需 spike 行的 g 與 e。

**最小檢驗**
1. 先在 L2 calibration 對 6 格式各求一次端點梯度，對照本輪的 S-only 實測（3.97、0.050、0.008…）。
2. 若排序與量級合理，再預先登記：在其他有 spike 行的層（L26、L27、L7 等）與新文章上驗證同位元選擇，並檢查不新增誤排。

**限制**：梯形估計在跳升非常窄時仍可能失準；本輪也還沒區分 spike channel 與同行其他 channel。

## 檔案

| 類別 | 檔案 |
|---|---|
| 計畫 | `plan.json`、`plan.md` |
| 控制 | `controls_calib.json`、`controls_replay.json` |
| calibration | `calib_losses.npz`、`calib_raw.json`（F、擾動、統計、proxy） |
| B | `replay_ce.npz`、`replay_statistics.json` |
| 分析 | `analysis.json`、`curves.csv`、`per_window.csv`、`per_article.csv`、`paired_decomposition.csv`、`proxy_components.csv` |
| 圖 | `fig_partition_curves.png` |
| 時間與 log | `timing.json`、`timing.jsonl`、`logs/` |
