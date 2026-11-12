# OLMoE：一次 proxy refresh 收益的新文本確認

日期：2026-10-22
腳本：`run_olmoe_proxy_refresh_confirm.py`（plan → data → control → eval → analyze）
輸出：本目錄

**本輪的範圍**
- 固定 `runs/olmoe_proxy_refresh_20261022` 已凍結的 q0（static BC）和 q1（refresh 一次），在另一批獨立的新文章上評估。
- 沒有重建 proxy、沒有重新搜尋；calibration、ranges、量化器、配置、router 都沒有改動，也沒有 fine-tuning。
- 本輪只檢驗**評估文本的抽樣不確定性**，不能推論到不同的 calibration、其他資源點或其他資料集。
- 測的是 GPU 上的 QDQ 行為，不代表 native kernel 的結果。

**流程**
- `plan.json`／`plan.md` 在抽文章和計算任何新 loss 之前寫定，沒有修訂。
- 每個階段用的都是同一份腳本（sha `5cb8ce5e…`）。

## 結論

**refresh 的改善在 600 篇獨立新文章上得到確認，但幅度很小：每 token 約 0.0012 nat，相當於 q0 損傷的 3.6%，PPL 只差 0.12%。等價判定沒有成立，因為 CI 下界只比 −0.002 多出一點。**

**主要終點**（600 篇文章 × 4 個窗口；每篇文章的 NLL = 四個窗口的 summed CE／508）

| | 平均 | 95% CI | 文章級 SD |
|---|---:|---|---:|
| **Δ = NLL(q1) − NLL(q0)** | **−0.00121** | **[−0.00219, −0.00024]** | 0.0121 |

**判定**（依預先規則）
- **A：方向**——**確認 refresh 較好**（CI 上界 −0.00024 < 0）。
- **B：等價**——**未確認**（CI 下界 −0.00219 < −0.002）。
- A 和 B 是分開的判定。這次 A 成立而 B 不成立；B 只差一點，但依規則不得放寬。
- 描述：600 篇中，q1 較好的有 316 篇，較差的有 284 篇。

| | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| FP32 dynamic | 2.95550 | 19.211 | — |
| q0 | 2.98950 | 19.876（+3.46%） | +0.03400 [+0.03237, +0.03565] |
| q1 | 2.98829 | 19.852（+3.33%） | +0.03278 [+0.03118, +0.03443] |

PPL 由平均 NLL 取 exp 得到，不是平均各窗口的 PPL。

**和上一輪並列**（只作為歷史結果，不合併）

| | 文章數 | Δ | 95% CI |
|---|---:|---:|---|
| 上一輪（`olmoe_proxy_refresh_20261022`） | 128 | −0.00196 | [−0.00415, +0.00017] |
| **本輪（確認）** | 600 | **−0.00121** | **[−0.00219, −0.00024]** |

- 方向一致。上一輪的點估計落在本輪的 CI 內。
- 本輪的效果量比上一輪小，符合上一輪 128 篇時的不確定性。

## 回答三個問題

**1. 上一輪觀察到的改善，是否在獨立的新文章上重現？**
- 是，方向重現而且達到確認標準。
- 效果量比上一輪小：從 −0.0020 變成 −0.0012。

**2. 效果量和不確定性，是否足以支持付出額外的 offline 成本？**
- 從「降低 offline 時間」這個研究目標來看，**目前不足以支持整合**。
- 收益只有 q0 損傷的約 3.6%：PPL 從 19.876 降到 19.852，約 −0.12%。95% CI 允許的範圍是 0.0002 到 0.0022 nat/token。
- 換算成資源（推論）：依先前同一條 BC frontier 上 aggressive（R 0.595）和 knee（R 0.707）的 dynamic 損傷差（約 0.020 nat/token 對 ΔR 0.11），0.0012 的收益大約等於多給 1% 的 R。
- 代價是 offline 時間大約翻倍：上一輪單次量測為基準約 104 秒、額外約 108 秒，本輪沒有重新量測。
- 先前的研究顯示，同一個量化配置在不同後端（GPU、CPU、native）之間有約 0.001–0.002 nat/token 的實作差異（見 knee Gate/Up 的研究）。0.0012 的收益和這個量級相近，在 native kernel 上是否仍存在，需要另外驗證。
- **顯著不等於值得**：結論是「方法有小而真實的效果，但成本效益偏弱」，不建議整合為預設。

**3. 是否值得進入 sensitivity-only／error-only 消融？**
- **只在附加條件下值得**：
  - 消融的目的是保留收益、降低成本。但收益只有約 0.0012，而兩個配置之間的文章級 SD 約 0.012。
  - 要確認「部分更新」不比完整更新差超過 0.0005，需要約 2000 篇以上的文章。這在評估端的成本很高。
  - 可行的目標只有「判斷部分更新是否明顯失去全部收益」。要測，就要把這個目標預先登記。
- **更直接的省時方向**（推論）：上一輪的額外成本中，約 45 秒是沒有分項計時的 checkpoint 和快取 I/O，比 backward（28 秒）或 activation 誤差（18 秒）都大。不論採用哪種更新方式，先降低 I/O 才是最大的省時空間。
- **建議順序**：
  1. 先決定這個約 3.6% 的損傷改善，是否符合「低 offline 成本」的目標；
  2. 若要繼續，先在第二個資源點（例如 knee）用已凍結的流程重複「static 對 refresh 一次」，確認效果不是只在 aggressive 點出現；
  3. 最後才做消融，並同時處理 I/O。

**沒有確認的事**：其他資源點、其他 calibration、其他模型或資料集、native kernel 上的效果，以及多次 refresh 的效果。

## 方法（`plan.json`）

**固定的配置**
- q0：`runs/olmoe_proxy_refresh_20261022/configs/q0.json`（sha256 `2fe46627745c…`）。
- q1：`runs/olmoe_proxy_refresh_20261022/configs/q1.json`（sha256 `346fd805dfcb…`）。
- 兩者都和上一輪 `freeze.json` 的 hash 一致。

**固定的量化**
- ranges：`runs/olmoe_multipoint_20261015/ranges.json`，和上一輪相同。
- 量化器：`dev_quant(corrected=True)`。

**評估方式**
- FP32、q0、q1 都使用 `run_olmoe_dynamic_route.forward` 的正常 dynamic route：每個 token 由模型自己選 expert 和 routing weight，不餵回 FP32 的路由或比例。
- GPU FP32，TF32 關閉；每批 128 個窗口，最後一批是剩下的 96 個；每窗口 127 個預測 token。

**統計**：10,000 次 paired article bootstrap，所有條件共用同一組文章索引，seed 22211。

## 資料（`segments.json`、`dedup_checks.json`）

**來源與抽樣**
- Salesforce/wikitext wikitext-103-raw-v1 的 train，sha256 已核對。
- 嚴格文章邊界；每篇單獨 tokenize。
- 依隨機順序（seed 22201）逐篇處理，窗口起點用 seed 22202。

**排除**（在抽樣前全部固定）

| 理由 | 篇數 |
|---|---:|
| 標題或內容和 WikiText-2 重疊 | 613 |
| WikiText-103 內部重複 | 464 |
| 歷史 OLMoE 用過的 WikiText-103 文章（依內容 hash） | 674 |

674 篇的組成：route_mass_restore 103、route_dynamic_prop 242、precision_protection 的 selection 32 與 final 169、proxy_refresh 128。

**掃描過的歷史 manifest**
- WikiText-103 來源（依內容 hash 排除）：
  - `olmoe_proxy_refresh_20261022/segments.json`
  - `olmoe_route_dynamic_prop_20261018/segments.json`
  - `olmoe_route_mass_restore_20261018/segments.json`
  - `olmoe_route_precision_protection_20261019/{selection,final}_segments.json`
- WikiText-2 來源（整份 WikiText-2 排除即涵蓋）：
  - segments 檔：calibration_stability、crosstext、dynamic_multipoint、dynamic_route、first_order_validation、route_decomposition。
  - 窗口檔：stage 1 plan、multipoint plan、solver_regression plan、search_final C、numerics B。
- 完整清單和 hash 在 `plan.json` → `data.manifests_scanned`、`wt2_window_files`。

**結果**
- 看了 622 篇：接受 600 篇；跳過 20 篇近重複（≥10% 的 5-gram 出現在 WikiText-2、歷史文章或已選文章中）、2 篇不足 512 token。
- 和歷史文章的重疊為 0；600 篇彼此不重複。
- 已接受文章的最大 shingle 重疊：和 WikiText-2 為 9.5%，和歷史文章為 9.9%，彼此之間為 7.6%。
- 每篇的內容 hash、窗口位置和 tokens hash 都已保存。

## 控制（`checks.json`；全部通過）

| 控制 | 結果 |
|---|---|
| 程式控制：上一輪看過的窗口 0–127，只作控制，不納入統計 | FP32、q0、q1 的 CE 和上一輪逐位元相同（差 0） |
| 新資料第一批（128 個窗口）重跑 q0、q1 | CE、最終 hidden state、路由都逐位元相同 |
| 沒有用 FP32 路由替換 | q0 有 30.5%、q1 有 29.9% 的 (token, 層) expert 集合和 FP32 不同 |
| CE 有限值；router 未被替換；hook 只在 expert 投影 | 是 |
| 配置與輸入檔 hash 前後一致 | 是（q0、q1、ranges、資料檔、上一輪的紀錄檔） |
| 上一輪的原始控制檔 | `static_checks.json`、`refresh_checks.json`（pass）、`eval.json` 的控制（pass）都已核對，不依賴彙整的 `checks.json` |

## 成本（只計研究驗證成本）

| 項目 | 時間 |
|---|---|
| 資料準備（去重、tokenize、近重複檢查） | 59.6 秒（CPU；峰值 RSS 3.4 GB） |
| 程式控制（FP32 9.6 秒、q0 20.6 秒、q1 20.6 秒） | 57.4 秒 |
| 評估：FP32 | 123.3 秒 |
| 評估：q0、q1 | 各約 269 秒 |
| 重跑控制（2 × 20.7 秒） | 41.4 秒 |
| 評估 stage 合計 | 710.9 秒（峰值 GPU 7.7 GB，CPU RSS 15.1 GB，含 mmap 的 checkpoint） |
| 分析 | 0.3 秒 |
| plan | 21.7 秒 |

這些是研究驗證的成本，**不是** proxy-refresh 方法的 offline 建置成本。後者（基準約 104 秒、額外約 108 秒）是上一輪的單次量測，本輪沒有重新量測。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_proxy_refresh_confirm_20261022      # 新實驗請改用新目錄
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_proxy_refresh_confirm.py
../../venv/bin/python $S --stage plan --out-dir $O
../../venv/bin/python $S --stage data --out-dir $O
$GPU $S --stage control --out-dir $O       # 約 1 分鐘
$GPU $S --stage eval --out-dir $O          # 約 12 分鐘
../../venv/bin/python $S --stage analyze --out-dir $O
```

## 檔案

- **預先登記**：`plan.json`、`plan.md`、`freeze_manifest.json`、`input_hashes_before.json`。
- **資料**：`segments.json`、`tokens.npy`、`dedup_checks.json`、`data_freeze.json`。
- **控制**：`control.json`、`checks.json`。
- **結果**：`eval.json`、`routes_{fp32,q0,q1}.npz`（expert ID）、`per_window.csv`、`per_article.csv`、`analysis.json`。
- **時間**：`timing.json`、`timing.jsonl`；`logs/`。
