# Qwen3-0.6B：前層 Down 的權重／activation 拆分，與 position 0 背景效應的 native 確認

日期：2026-10-26。
- 腳本：`run_qwen3_l27_down_source.py`。
- 計畫：`plan.json`／`plan.md`（沒有 amendment）。
- 前兩輪：`runs/qwen3_l27_context_attribution_20261026`、`runs/qwen3_l27_context_followup_20261026`。

**資料**：新抽 128 篇 WT103 文章。GPU 用每篇 2 窗口；native 用每篇第 1 個窗口（128 個）。

**單位與符號**
- D = NLL(L27 Down a = p8e1) − NLL(f16)，負值代表 p8e1 較好；nat/token 的貢獻量。
- 文章 bootstrap 20000 次，95% CI。
- **position 0 的效應不需要依位置設定格式**：L27 之後只有逐位置運算，所以整節點 p8e1 在 position 0 那一項的 CE 變化，就精確等於「只在 position 0 用 p8e1」的效果。

**沒有修改**：criterion、配置、pipeline。

## 首頁回答

### 1. native 確認：position 0 的背景效應成立，且與 GPU 一致

背景 U = 其他投影全 FP32；背景 F = 其他投影全依原 knee A。

| | native D_U | native D_F | **native 背景效應 D_F − D_U** | GPU（同 128 窗口）背景效應 | 後端差 | 逐篇相關／同號 |
|---|---|---|---|---|---|---|
| position 0 | −0.0066 | −0.0145 | **−0.0078** [−0.0099, −0.0057] | −0.0076 | −0.0003 [−0.0006, +0.0000] | 0.989／122 of 128 |
| 其他位置 | +0.0048 | +0.0053 | +0.0005 [−0.0013, +0.0023] | +0.0007 | −0.0002 | 0.43／91 |
| 全部 | −0.0019 | −0.0092 | −0.0073 [−0.0098, −0.0048] | −0.0069 | −0.0005 | 0.69／88 |

- **native 確認**：背景效應完全在 position 0；其他位置沒有背景效應。
- **與 GPU 的一致性**：position 0 的效應在兩個後端上幾乎相同。

**NLL 參照**（native）

| 模型 | NLL |
|---|---|
| U_f16 | 3.8612 |
| U_p8e1 | 3.8593 |
| F_f16（原 knee） | 3.9263 |
| F_p8e1 | 3.9171 |

### 2. 前面 Down 投影的貢獻來自權重量化，不是 activation 量化（GPU）

只讓 L0–26 的 27 個 down_proj 依 A 量化指定部分，其餘 FP32；看 position 0 背景效應（D_b,pos0 − D_U,pos0），4 個拆分 Holm 校正：

| 前面 Down 量化的部分 | position 0 背景效應 [95% CI] | Holm p |
|---|---|---|
| 權重與 activation 都量化（dn，參照） | −0.0035 [−0.0045, −0.0026] | — |
| **只量化權重**（dn_w） | **−0.0027** [−0.0037, −0.0016] | **< 1e-4** |
| 只量化 activation（x 與 y） | −0.0002 [−0.0004, +0.0001] | 0.40 |
| 只量化 activation input x | −0.0002 [−0.0004, +0.0001] | 0.40 |
| 只量化 activation output y | +0.0003 [+0.0000, +0.0005] | 0.10 |
| 全部投影依 A（F，參照） | −0.0080 [−0.0094, −0.0066] | — |

**加成性**
- dn −（權重 + activation）= −0.0007 [−0.0014, −0.0001]：有小交互，但權重本身已占約 75%。
- activation −（input + output）= −0.0003 [−0.0005, +0.0000]。
- 其他位置（rest）在所有拆分中都沒有背景效應，CI 都跨 0。

**結論**
- 前面 Down 投影造成 L27 position 0 補償效應的，**主要是它們的權重量化誤差**。
- 它們的 activation 量化（不論 input 或 output）幾乎沒有作用。

## 四輪合併的圖像

1. **現象**：只把原 knee 的 L27 Down activation 由 f16 改成 p8e1，native 品質改善約 0.009（前輪 native −0.0087、本輪 −0.0092），而且省資源。原 BC 預測相反。
2. **位置**：這個好處完全來自 position 0。p8e1 在其他位置一直是有害的（約 +0.005），與背景無關。
3. **背景依賴**（native 已確認）：在全 FP32 背景下，position 0 的好處較小（−0.0066），正好被其他位置的壞處抵消。其他層量化後，position 0 的好處加倍（−0.0145）。
4. **來源**：
   - 前面各層的 Down 投影貢獻約一半（前一輪 −0.0042，本輪 −0.0035），分散在各層深度；
   - 其中主要是權重量化；
   - 其餘來自其他投影與跨組交互，attention 單獨沒有效果。
5. **可能的機制**（線索，未證明）：position 0 的 massive activation 很大（L2 Down input 約 3000–3800）。前面 Down 的權重量化誤差乘上這些大值，會在 position 0 的 residual 留下較大的絕對誤差；L27 Down 在 position 0 的 p8e1 粗量化剛好部分抵銷這個累積誤差。

## 對 pipeline 的意義

- 這是**跨層、只在 position 0、由權重量化誤差觸發**的交互效應。
- 現行 BC 在 FP32 背景下逐節點估成本再相加，結構上無法表示，所以自動 knee 找不到這個收益。
- 如果要利用這個發現，可能方向如下（都超出目前 pipeline 的設計，需你決定）：
  - 對 sink token（position 0）單獨選格式（本輪顯示只在 position 0 用 p8e1 最有利）；
  - 在搜尋中加入「其他層已量化」背景下的評分（例如以 knee 配置為背景重算關鍵節點）；
  - 檢查前面 Down 權重量化對 position 0 residual 的影響，是否應以權重格式選擇本身來處理。

## 控制（全部通過）

**GPU**
- 在前一輪文章上，U、F、dn 的 f16／p8e1 與前一輪逐窗口完全相同。
- 前面 Down 的 activation 覆寫路徑與 LMReplay 原路徑逐位元相同（dn_a 背景，calibration 窗口）。
- TF32 關閉、deterministic。

**native**
- 重用的 F_f16、F_p8e1（L27 native confirm 的 A、C .so，hash 已核對）在 calibration 窗口 0–1 與前輪結果差 0。
- 新編譯的 U_f16 窗口 0 在另一個 process 重跑差 0。
- 新編譯模型的 196 個編譯條目都與配置相符。
- 逐位置 CE 加總等於窗口 CE。

## 成本與揭露

**時間**

| 項目 | 時間 |
|---|---|
| GPU 評估（14 條件 × 256 窗口） | 179 s |
| native 編譯（U_f16／U_p8e1） | 各約 32 s |
| native 評估：U_f16／U_p8e1 | 831 s／970 s |
| native 評估：F_f16／F_p8e1 | 1088 s／1132 s |
| native 評估合計（6 workers） | 約 67 分鐘 |

- native 評估超過平常約 15 分鐘的批次目標，也比我先前估計的約 35–50 分鐘長。

**資料**
- 抽樣跳過 44 篇：歷史 Qwen 重疊 19、近重複舊文章 22、近重複 WT2 3。
- 排除 9385 篇已用文章；seed 26102651／52／53，均未用過。

**冒煙測試**
- 只跑 plan、GPU 控制（前一輪已看過的文章）、native 編譯與控制（calibration 窗口 0–1），沒有看新文章。
- 看完後沒有修改設計。

**限制**
- 權重／activation 拆分只有 GPU 結果；native 只確認了 position 0 的背景效應。
- 前面 Down 的權重拆分沒有細到單層，不知道是否集中在 L2 這類 massive activation 產生層。
- 補償機制仍是假說。

到此停止，沒有修改 criterion、配置或 pipeline。

## 檔案

| 類別 | 檔案 |
|---|---|
| 計畫 | `plan.json`、`plan.md` |
| 控制與凍結 | `gpu_controls.json`、`native_controls.json`、`freeze.json` |
| 編譯 | `builds.json`、`configs/`、`native/` |
| 資料 | `segments.json`、`tokens.npy`、`data_freeze.json` |
| 評估 | `gpu_ce_pos.npz`、`native_ce_pos.npz`、`native_results/` |
| 分析 | `stats.json` |
| 時間與 log | `timing.json`、`timing.jsonl`、`logs/` |
