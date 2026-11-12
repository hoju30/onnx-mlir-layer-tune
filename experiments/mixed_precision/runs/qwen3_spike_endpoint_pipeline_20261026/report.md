# Qwen3-0.6B：spike 端點敏感度候選 E 的機制確認與完整 native pipeline A/B

日期：2026-10-26。
- 腳本：`run_qwen3_spike_endpoint_pipeline.py`（sha256 `3fdb7ea7a7fa…`，見 amendment）。
- 計畫：`plan.json`／`plan.md`（04:43:47 寫入；04:58 一筆 amendment）。
- 規格：`reports/qwen3_spike_endpoint_pipeline_prompt_20261026.md`（使用者授權在機制確認後接完整 pipeline）。

**性質**
- 這是「混合建表＋完整配置 native 評估」：E 的端點項由 GPU PyTorch 計算，模型由 onnx-mlir 編譯與推論。新統計並非 compiler-native。
- 正式 BC、量化器、MILP、knee 與 pipeline 預設都沒有改動。

## 首頁回答

### 1. L2 的大損傷出現在哪段幅度、哪些元素分區？

**幅度**：只量化 spike 行（S-only），沿 α 從 x 移向 Q(x)（calibration，nat/token）：

| α | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.925 | 0.95 | 0.975 | 0.99 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p4e1 | 0.032 | 0.027 | 0.020 | 0.023 | 0.032 | 0.068 | 0.099 | 0.258 | 0.975 | 2.562 | **3.969** |
| p6e1 | 0.022 | 0.019 | 0.012 | 0.008 | 0.010 | 0.015 | 0.019 | 0.026 | 0.036 | 0.044 | 0.050 |
| int4／int6／int8 | 0.003／0.006／0.001 | | | | | | | | | | 0.008／0.014／0.002 |
| p8e1 | −0.002 | | | | | | | | | | −0.003 |

- p4e1 的損傷在 α ≥ 0.9 之後才急升，主要變化落在 α 0.95–1 之間。最大值 channel 在 α = 0.9、0.95、0.975、0.99、1 時分別約為 394、205、110、54、16，換算方式是 x + α(16 − x)。
- 這是離散 grid 上觀察到的急升，不稱為數學不連續。
- α = 0.5 與 1 的數值與前輪逐位元相同。

**元素分區**（α = 1，只對指定元素套 q，其餘維持 FP32）：

| 格式 | maxpos | H：每行最大的 1 個元素（10 個，全在 channel 55） | O：S 中 \|x\| > maxpos | R：S 其餘 | 完整 S | I_OR = S − O − R |
|---|---|---|---|---|---|---|
| p4e1 | 16 | 0.026 | **4.020**（88 個元素，約每行 9 個） | −0.000 | 3.969 | −0.051 |
| p6e1 | 256 | 0.021 | 0.049（60 個元素） | 0.001 | 0.050 | +0.000 |
| p8e1 | 4096 | 0.005 | 空集合 | −0.003 | −0.003 | 0 |

**已確認**
- p4e1 的損傷由「超出 maxpos 的元素」承擔（O 的 4.02 約等於 S 的 3.97）。
- 只截斷每行最大的那一個元素（H，channel 55），只損失 0.026。

**仍無法唯一歸因**
- 88 個元素中，是哪一部分（或需要多少個）同時被削才觸發損傷：沒有拆分 channel，本輪也不新增遮罩。
- 觸發的是元素被削後的絕對大小，還是 spike 行整體被縮小的比例。
- 是否與其他層的機制相同。

### 2. 端點平方梯度 E 是否修正 L2 4-bit、保住 6/8-bit？（calibration 診斷，不是最終品質）

L2 只看 x 項（raw proxy）的同位元選擇：

| 配對 | 實測（前輪新文章完整量化） | B（GPU 原 BC） | D（新分解，無端點） | E（新分解 + 端點） |
|---|---|---|---|---|
| int4–p4e1 | INT 較好 | int4 ✔ | p4e1 ✘ | int4 ✔ |
| int6–p6e1 | Posit 較好 | int6 ✘ | p6e1 ✔ | **int6 ✘** |
| int8–p8e1 | Posit 較好 | int8 ✘ | p8e1 ✔ | p8e1 ✔ |

**E 修正了 4-bit，但在 6-bit 改錯，8-bit 保住**
- E 對 p4e1 的端點梯度平方極大：Pend 為 6.3e8，P0 只有 674。
- 對 p6e1 也被放大 35 倍（504 → 17476），因此 6-bit 又選回 int6。
- 也就是說，E 用端點梯度的量級拉開差距，並不是在量測 loss。

**另報：½(g0 + g1)·e_S 對 S-only 實測的對照**（只作診斷，不是搜尋成本）

| 格式 | ½(g0 + g1)·e_S | S-only 實測 |
|---|---|---|
| int4 | 0.0084 | 0.0084 |
| int6 | 0.0145 | 0.0142 |
| int8 | 0.0021 | 0.0021 |
| p8e1 | −0.0036 | −0.0032 |
| p6e1 | 0.36 | 0.050 |
| p4e1 | **−5.2** | +3.97 |

- 在平滑的情況幾乎準確；在跳升情況（p4e1）符號都錯了。

**L2 的 x 項修正對完整配置沒有作用**
- 正式的 activation 格式 a 同時量化 Down 的 input x 與 output y。L2 Down 的 y 項主導 Posit：

| 格式 | C_y,old | 完整 C（A） | 完整 C（E） |
|---|---|---|---|
| p8e1 | 6.66e5 | 8.73e5 | 6.66e5 |
| p6e1 | 3.97e6 | — | — |
| f16 | — | 0.17 | 0.13 |

- 因此所有配置的 L2 Down 都是 int8／f16。L27 也一樣，所有配置都是 f16。

### 3. 原 BC 與新候選的配置各改了哪些 w/a？

| 配置 | 相對參考的差異 | size／BOPs／R |
|---|---|---|
| A_auto | 重現歷史 knee（hash `36205a06…`） | 0.759821／0.877976／0.877976 |
| E_auto | 與 A_auto 差 7 個投影 | **0.761012**／0.877976／0.877976（weight bytes 較多） |
| A_cap、B_cap | **與 A_auto 完全相同** | 同 A_auto |
| D_cap、E_cap | **兩者完全相同**；與 A_auto 差 7 個投影 | 0.759226／0.877976 |
| A_aggr | A frontier 上 R = 0.7071（√(0.5 × 0.878) ≈ 0.663 的最近點）；與 A_auto 差 151 個投影 | 0.684821／0.707068 |
| E_aggr | 與 A_aggr 差 6 個投影 | 0.684821／0.707068 |

**改動細節**
- E_auto 相對 A_auto：
  - L1 gate（int7/p8e1 → int6/f16）、L6 k（int7 → int6）；
  - **L11 Down（a p8e1 → p6e1）**；
  - L15 q（int5 → int6）、L21 v（int6 → int7）、L21 o（int5 → int6）；
  - **L26 Down（int6/f16 → int7/p8e1）**。
- D_cap = E_cap 相對 A_auto：L1 gate、L6 k、L8 v（a p8e1 → f16）、L11 Down、L21 o、L22 gate（int6 → int5）、L26 Down。
- E_aggr 相對 A_aggr：L7 up 與 L8 gate（a p6e1 → p8e1）、**L7 Down、L10 Down（int5/p8e1 → int6/p6e1）**、L18 Down 與 L20 Down（int6 → int5）。

**讀法**
- 有 spike 行的 Down 中，真正改變選擇的是 L11、L26（以及 aggr 的 L7、L10、L18、L20）。
- 原因是新分解去掉了全域 μ 的假象，Posit 的 x 項變小，搜尋於是在其他投影上重新分配資源。
- 端點更新（E 對 D）沒有改變同上限的配置；後端橋接（B 對 A）也沒有。

### 4. 新文章 native 的 NLL／PPL 與資源是否改善？與 GPU 是否一致？

資料：128 篇新文章，每篇 1 個窗口，nat/token。

| 配置 | native NLL | native PPL | 相對 FP32 | GPU NLL | GPU 相對 FP32 |
|---|---|---|---|---|---|
| FP32 | 3.9604 | 52.48 | — | 3.9604 | — |
| A_auto（= A_cap = B_cap） | 4.0289 | 56.20 | +0.0685 | 4.0342 | +0.0738 |
| E_auto | 4.0320 | 56.38 | +0.0716 | 4.0306 | +0.0702 |
| D_cap = E_cap | 4.0360 | 56.60 | +0.0756 | 4.0351 | +0.0746 |
| A_aggr | 4.0988 | 60.27 | +0.1384 | 4.0973 | +0.1369 |
| E_aggr | 4.0900 | 59.74 | +0.1296 | 4.0917 | +0.1312 |

**主要比較**（native；文章 paired bootstrap 20000 次）

| 比較 | 平均 | 95% CI | Bonferroni 97.5% CI | 判讀 |
|---|---|---|---|---|
| **P1：E_cap − A_auto**（同兩個資源上限，E_cap 確實沒有超過） | **+0.0071** | [+0.0027, +0.0115] | [+0.0021, +0.0122] | **較差**（CI 全為正） |
| **P2：E_auto − A_auto**（無預算自動 knee） | +0.0031 | [−0.0016, +0.0077] | [−0.0023, +0.0085] | **未定**；非劣性也未成立（上界 > δ = 0.001）；E_auto 還多用了 0.16% weight bytes |

**次要家族**（native，Holm 校正）

| 比較 | 平均 [95% CI] | Holm p |
|---|---|---|
| E_aggr − A_aggr | −0.0088 [−0.0159, −0.0015] | 0.052（未過） |
| B_cap − A_auto | 0（配置相同） | 1 |
| D_cap − B_cap | +0.0071 [+0.0027, +0.0115] | **0.006** |
| E_cap − D_cap | 0（配置相同） | 1 |

**GPU 對照**
- P1 = +0.0009 [−0.0042, +0.0057]，P2 = −0.0036 [−0.0091, +0.0018]，E_aggr − A_aggr = −0.0056 [−0.0128, +0.0015]，三者 CI 都跨 0。
- 兩後端的逐篇效應相關只有 0.19–0.26，同號 69–81/128。
- native − GPU 的效應差：P1 +0.0062 [+0.0002, +0.0123]、P2 +0.0067 [+0.0004, +0.0131]。
- 也就是說，這些約 0.005 量級的配置差異，在兩個後端上的效果並不一致；依計畫以 native 為主要證據。

**回答**
- 同資源上限下，新候選選出的配置在 native 上**明顯較差**（PPL 56.20 → 56.60）。
- 自動 knee 未定，而且多用 weight bytes。
- 只有激進點出現改善跡象，但沒有通過預先指定的多重比較校正，GPU 也不支持。

### 5. 多花的建表時間換到多少？

| 項目 | 時間 |
|---|---|
| 原 native 建表（重用；當初實測） | 1074 s（CPU） |
| E 的必要增量：遮罩 + g0（GPU，100 窗口 backward） | 0.01 + 4.5 s |
| E 的必要增量：端點 backward（100 個 (層, 格式) × 10 窗口） | 44.7 s（每條件中位數 0.49 s，範圍 0.13–0.70 s） |
| E 的必要增量：分項統計 | 0.75 s |
| B 的後端橋接：重跑原 native 統計取 C_y | 264 s（實驗用，不是 E 的必要成本） |
| 搜尋：A_auto／E_auto（冷啟動） | 26 s／63 s |
| 上限求解（每個） | 0.9–1.0 s |
| 編譯：每個量化模型 | 約 40 s |
| 編譯：FP32 | 27 s |
| 編譯：posit runtime | 17 s |
| 研究評估：native（6 模型 × 128 窗口，6 workers） | 5369 s |
| 研究評估：GPU | 109 s |

**完整 offline 成本**（建表 + 搜尋 + 選定模型編譯）

| 方法 | 計算 | 合計 |
|---|---|---|
| A | 1074 + 26 + 39 | 約 1139 s |
| E | 1074 + 約 50（GPU 增量）+ 63 + 40 | 約 1227 s（**+8%**） |

- E 的數字還需要另一個 GPU PyTorch 環境。
- 若做成 compiler-native，需要約 1000 次額外的 suffix forward／backward（100 個條件 × 10 個 calibration 輸入）。以原 native grads 每個輸入完整一次約 67–108 s 粗估，就算只跑部分 suffix，也要數小時 CPU 時間，不是低成本。這是推估，未實測。
- **成本與收益**：沒有任何主要品質收益；同上限甚至變差。不符合「低 offline 成本換品質」的目標。

**建議**：目前不值得做 compiler-native 的新統計整合。正式預設不更動。

## 控制

**phase 1**
- 重播與無 hook 前向的差異為 0；遮罩與 recenter 相同。
- α = 0.5、1 的 S-only 與前輪逐位元相同；α = 0 為 0。
- g_x = g_y·W；position 127 的梯度為 0。

**GPU 統計**
- 28 層遮罩都與 recenter 相同（10 層有 spike 行，18 層回退成 A）。
- L2 的 B／D／E 分項與 phase 1 相同。
- g 的檢查同上。

**native x／y**
- 重跑原 grads 與 errors6 統計後，Down 的 C_x,old + C_y,old 等於 table.json（相對差 0）。
- GPU 與 native 的 x-slot 統計差異：G2 2e-4、GS 8e-4、S1 2.3e-3、S2 6.8e-4（相對最大值），只描述。

**表**
- A 等於 native table.json。
- B／D／E 只改 10 個 Down 的 C；其他 C 與全部 B 項逐位元相同。

**搜尋**
- A_auto 完整重現歷史 knee（hash、size、BOPs、R）。
- 所有上限求解的 status 都是 0，也都在上限內。

**native 控制**
- 196 個編譯條目都與配置相符；A_auto 的 compile env 與前輪相同。
- calibration 窗口 0–1：FP32 與 A_auto 和前輪 native 結果差 0；A 窗口 0 在另一個 process 重跑差 0。
- GPU 與 native 在這兩個窗口的差異（只報告）：FP32 1.3e-4、量化配置 2.6–7.3 summed CE。

**評估**：6 個模型的 128 個窗口都完整，每個 worker 峰值 RSS 約 3.1 GB。

## 揭露

**seed**：規格建議的 26102611 已是前輪（原始誤差診斷）的 bootstrap seed，計畫前改用 26102621／22／23，記錄在 plan.json。

**冒煙測試**（04:34–04:43，scratchpad；只跑 plan、phase1、gpu_stats、native_xy、tables、search，沒有新文章、沒有編譯或評估）
- 看過的內容：
  - phase 1 的全部數字（即上方表 1、表 2）；
  - 10 個 spike 層的 GPU 分項；
  - native x + y 的重現；
  - 搜尋出的配置與差異（E_auto 差 7、D_cap = E_cap、B_cap = A_auto、aggr 點）。
- 看過之後沒有修改公式、表、搜尋規則、比較配置、統計或容差。
- 正式的 phase1、GPU 分項與表都和冒煙逐位元相同。

**amendment（04:58）**
- controls 階段在 CPU 環境呼叫了 GPU LMReplay，用於只報告的 FP32 GPU 對照，因而在所有 native 控制都跑完之後停止；當時尚未凍結、也未抽資料。
- 該對照移到 pytorch_eval（`controls_gpu.json`）；native 控制與其他設計不變，已完成的 native 控制結果重用。

**其他**
- native 評估約 90 分鐘，遠超平常約 15 分鐘的批次目標，因為規模是規格指定的。
- 新文章抽樣：經過 161 篇，跳過 33 篇（歷史 Qwen 重疊 15、近重複舊文章 12、近重複 WT2 6），排除 9001 篇已用文章。

## 已證明／未定的界線

**已證明**
- 在本批 calibration 上：L2 p4e1 的 spike 損傷集中在超出 maxpos 的元素，並在 α 接近 1 時急升。
- 單一最大元素不足以重現損傷。
- E 改變了 L2 的 x-only 排序，但完整配置不受影響。
- 在 128 篇新文章上：同上限的 E／D 配置 native 較差（CI 全為正）。

**未定**
- E_auto 對 A_auto。
- 激進點（Holm 未過，GPU 不支持）。
- 兩後端效應差異的來源。
- p4e1 損傷的精確觸發元素與門檻。

到此停止：正式 BC、D、E、量化器、MILP、knee 與 native lowering 都沒有更動，沒有追加層、文章、係數或配置搜尋。

## 檔案

| 類別 | 檔案 |
|---|---|
| 計畫 | `plan.json`、`plan.md`（含 amendment） |
| phase 1 | `phase1.json`、`phase1_losses.npz` |
| GPU 統計 | `gpu_components.json`、`cost_components.csv`、`gpu_slot_stats.npz`、`endpoint_timing.json` |
| native 統計 | `native_xy.json` |
| 表 | `table_A/B/D/E.json`、`tables_controls.json` |
| 搜尋 | `search/*.json`、`search_summary.json`、`configs/*.json` |
| 編譯 | `builds.json`、`native/*/`（.so、constants、compile_env） |
| 控制與凍結 | `controls_native.json`、`controls_gpu.json`、`freeze.json` |
| 資料 | `audit.json`、`segments.json`、`tokens.npy`、`data_freeze.json` |
| 評估與分析 | `native_results/`、`native_ce.npz`、`pytorch_ce.npz`、`stats.json`、`per_article_nll.csv` |
| 時間 | `timing.json`、`timing.jsonl` |
