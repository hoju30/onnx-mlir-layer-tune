# 本週進度：GEMQ 啟發的 proxy 更新與 activation-spike 評分改動

報告日期：2026-10-28。實驗範圍：2026-10-22–2026-10-27。

**主講只有兩件事，建議六頁、約 10–15 分鐘。**

1. 在 OLMoE 上，借鑑 GEMQ 的量化工作狀態更新，測試原 BC 的完整與部分 refresh。
2. 參考 activation-spike 論文，觀察 Down input 的 spike，測試保護、分組與同位置配對，最後檢查完整配置收益。

報告採「論文線索 → 我們改了什麼 → 數據與決策」的順序。兩條主線都是借鑑後的實驗，不標成兩篇論文的完整復現。詳細的一階、L26/L27 輸出端評分及最新 BC 可靠性研究，留在[完整備查版](weekly_progress_20261028.md)，本次主講不展開。

各表的 ΔNLL 都是表中方法相對其基準的差，負值代表較好；不同實驗用不同文章，不能直接相減或加總。CI 跨 0 表示未確認方向，不表示等效。模型評估以 GPU FP32＋QDQ 為主，native 結果另外標明；BOPs 為資源 proxy。

## 第 1 頁｜方向一：量化後，FP32 上收集的 proxy 是否仍適用？

**論文線索**

GEMQ 結合全模型 expert 位元分配、router 微調與漸進量化，使用先前量化並微調的模型重新估計重要性。它主要研究 weight-only quantization。[GEMQ，§4](https://arxiv.org/html/2605.23078v1)

**我們提出的實驗問題**

原 BC 在 FP32 狀態收集 sensitivity 和候選誤差。但量化後，activation、梯度與路由可能改變；更新一次統計，能否在相同資源上限選到更好的格式？

```text
FP32 模型 → 原 BC 成本表 → 搜尋配置 q0
                                ↓
                  在 q0 上重跑原校準資料
                                ↓
                    更新指定的 BC 統計
                                ↓
                  相同資源上限重搜一次 → q1
                                ↓
                       新文章比較 q0 / q1
```

- 模型：OLMoE；forward 使用自身 dynamic route。
- 沿用 BC 公式與 INT／FP／Posit 候選，做一次同預算 refresh。
- 固定 calibration、ranges、量化器與 router 參數。更新 sensitivity 需要 backward，並非只多一次 inference。
- 含 sensitivity 的實驗：backward 固定當次路由、QDQ 使用 identity STE；沒有 optimizer 更新。
- 沒有採用 GEMQ 的完整 expert-level 評分、跨預算漸進流程或把 router 訓練串入 refresh。

**口稿**

「第一個方向來自 GEMQ：量化後模型的工作狀態會改變，所以 FP32 上取得的重要性不一定仍適合。我們保留原 BC，先選出 q0，再在 q0 上重新跑校準資料、更新統計，在相同預算重新選一次。這樣可以單獨檢查更新統計的價值，也能量它額外增加多少離線成本。」

來源：[本地完整 refresh 實驗](../runs/olmoe_proxy_refresh_20261022/report.md)。

## 第 2 頁｜實際改動：更新哪些資訊，才能保留收益並省時間？

**七種更新版本，static BC 作共同基準**

| 版本 | 更新的資訊 | 目的 |
|---|---|---|
| B：完整 refresh | 全節點 weight／activation sensitivity＋activation error | 確認完整更新的收益 |
| C：sensitivity-only | 全節點 weight／activation sensitivity；誤差沿用 static | 省候選 activation 誤差計算 |
| D4：≤4-bit sensitivity | 只更新 q0 中低 bit 投影的 weight／activation sensitivity | 減少統計收集與運算 |
| D5：≤5-bit sensitivity | 將 D4 的投影範圍擴到 ≤5-bit | 檢查更新範圍較小的影響 |
| E：error-only | 全節點只更新 activation 誤差的平均／變異 | 更新階段省去 backward |
| E5：≤5-bit error | 只更新 D5 同一組投影的 activation error | 再減少候選誤差計算 |
| W：weight sensitivity-only | 全節點只更新 weight sensitivity；activation 整項沿用 | 分離 weight 項更新的作用 |

**講清楚三件事**

1. 低 bit 節點定義為 q0 的 weight 或 activation 位元數 ≤門檻。更新選中投影的所有候選成本，最終格式仍由搜尋決定。
2. 原始權重與候選量化器固定，候選 ΔW = Q_f(W)−W 不變，完整 B 也重用 static weight error。
3. D4／D5 仍要反傳整個模型，省的是部分梯度統計與後續計算。E／E5 只有 refresh 階段不需 backward，最初 static BC 仍需梯度。

Knee 的 D4 更新 169／3051 個投影，D5 更新 1512／3051；aggressive D4 是 1161／3051。不同資源點的更新範圍相差很大。

**口稿**

「完整更新之後，我們拆兩個問題。第一，應該更新敏感度，還是量化誤差？第二，需要全模型更新，還是只更新低 bit？所以分別做了 sensitivity-only、error-only、四 bit 和五 bit 子集，最後再單獨測權重敏感度。這些版本都是改統計的來源，再重新選格式；沒有改 BC 的數學形式或訓練模型參數。」

來源：[aggressive D4](../runs/olmoe_selective_sensitivity_20261022/report.md)、[knee D4](../runs/olmoe_knee_selective_sensitivity_20261022/report.md)、[C](../runs/olmoe_knee_all_sensitivity_20261022/report.md)、[D5](../runs/olmoe_knee_low5_sensitivity_20261022/report.md)、[E](../runs/olmoe_knee_error_only_20261022/report.md)、[E5](../runs/olmoe_knee_low5_error_only_20261022/report.md)、[W](../runs/olmoe_knee_weight_sensitivity_20261023/report.md)。

## 第 3 頁｜GEMQ 啟發實驗的結果：aggressive 有收益，knee 尚不足以採用

每輪以下主要結果各使用 600 篇新文章，保持原 router 的 dynamic route，GPU QDQ 評估。

| 資源點／方法 | 相對 static BC 的 ΔNLL | 區間與判讀 | 額外成本 |
|---|---:|---|---|
| Aggressive，完整更新 B | **−0.00121** | 95% CI [−0.00219, −0.00024]，確認小幅改善 | 前輪單次量測約 **108 秒**；原流程約 104 秒 |
| Aggressive，低 bit sensitivity D4 | **−0.00173** | 95% CI [−0.00264, −0.00080]，確認改善 | **39.9 秒**，同輪完整 B 為 **80.6 秒** |
| 預設 knee，D4 | +0.000001 | 95% CI [−0.00063, +0.00065]，未確認改善 | 較省時，但收益沒有重現 |
| 預設 knee，只更新 weight sensitivity W | −0.00021 | 校正 CI [−0.00098, +0.00056]，未確認改善 | **66.1 秒** |

- Aggressive 完整更新：PPL **19.876 → 19.852**，改善約 0.12%。
- Aggressive D4：相對完整更新的預登記非劣性成立（界線 +0.0005 nat/token），同輪額外時間約減半。
- 正常 knee 的 C、D4、D5、E、E5、W，尚未確認穩定優於 static BC；完整 B 的小收益在不同新文章批次也不總能確認。
- 各輪絕對秒數與品質不可直接橫比；39.9 對 80.6 秒才是同輪計時。

**決策**：證明量化後更新有可能帶來收益，但目前幅度小、依資源點而異，未設為正式預設。

**口稿**

「結果不是所有更新都沒用。在 aggressive 點，完整更新有小幅收益；只更新低 bit 的敏感度也確認有效，而且更新時間減半。但把相同做法帶到預設 knee，收益沒有重現，後續擴大到五 bit 或拆成其他更新方式，也沒有穩定改善。因此我們保留這個研究結果，但還沒有把額外更新成本放進預設 pipeline。」

來源：[完整更新跨文本確認](../runs/olmoe_proxy_refresh_confirm_20261022/report.md)、[D4 aggressive](../runs/olmoe_selective_sensitivity_20261022/report.md)、[D4 knee](../runs/olmoe_knee_selective_sensitivity_20261022/report.md)、[W 與 knee 多輪彙整](../runs/olmoe_knee_weight_sensitivity_20261023/report.md)。Router 微調的獨立試驗見備用頁 A。

## 第 4 頁｜方向二：GLU activation spikes 會如何影響 BC？

**論文線索與原方法**

論文觀察到部分早／晚層 Down input 的大 activation 集中在少數 token。它提出 QFeM：讓選定模組的 input activation 保持高精度；QFeP：以未量化模型預算 prefix KV cache，隔離 spike。這兩項沒有提出修改 BC 的公式。[Activation-spike 論文，§3–4](https://arxiv.org/html/2405.14428v1)

**我們延伸的問題**

```text
Gate → SiLU ─┐
             × → Down input h → Down projection
Up ──────────┘
                 ↑
        觀察 spike、量化誤差與對應梯度
```

\[
h_{t,c}=\operatorname{SiLU}(\mathrm{Gate}(x)_t)_c\,\mathrm{Up}(x)_{t,c}
\]

- 大 activation 是否真的造成大的 NLL 損傷？
- BC 的 channel 平均敏感度，是否代表 spike 發生位置的敏感度？
- 保留 spike／位置資訊後，能否改善候選排序及完整配置？

**本地觀察**

- OLMoE 的 Down input：1017 個搜尋 expert 中，**993 個**至少有一個依本地門檻判定的 outlier channel。
- 這個 993／1017 是 channel outlier 統計，不能直接稱為 993 個都有有害的 token spike。
- BC 使用 per-channel 統計，但 INT activation 使用 per-tensor scale；這是不同粒度。Posit 在本實作直接用 LUT，不套 INT 的 scale。

**口稿**

「第二個方向來自 GLU activation spike 的論文。我們把焦點放在 Gate 與 Up 相乘後、送進 Down 的 activation。論文的處理是高精度保護或 prefix；我們進一步問，這種集中在少數位置的誤差，會不會讓 BC 的 channel 平均敏感度不適用。因此我們先做保護介入，再把 spike 與位置資訊放進評分。」

來源：[本地 outlier 診斷](../runs/olmoe_act_outlier_diag_20261023/report.md)。後續 token-spike 的更正與獨立介入見[spike intervention](../runs/olmoe_spike_intervention_20261023/report.md)。

## 第 5 頁｜實際改動：保護 spike、分組評分、同位置配對

| 嘗試 | 具體改動 | 想驗證什麼 |
|---|---|---|
| Spike 保護／單獨量化 | spike rows 保留 FP32；或只量化 spike rows，並與保護同數量隨機 rows 比較 | spike 自身誤差是否造成真實損傷 |
| Spike 分組 | spike／一般 rows 各自計算平均 g²，再與各組誤差平方和配對 | 兩群敏感度不同是否被 channel 彙整掩蓋 |
| 同位置配對 | 把每個位置的 g² 與該位置的中心化誤差平方相乘，再加總 | 誤差與敏感度的相關性是否影響排序 |

記號：t 為送到該 expert 的 token row，c 為 channel，e = Q(h)−h，μ 為每 channel 誤差均值，r = e−μ。

原變異項與候選配對項：

\[
V_{\mathrm{BC}}=\sum_c\operatorname{mean}_t(g_{t,c}^2)\sum_t r_{t,c}^2,
\qquad
V_{\mathrm{pair}}=\sum_c\sum_t g_{t,c}^2r_{t,c}^2.
\]

分組法位在兩者之間：只區分 spike 與一般兩群。這兩個評分候選都保留原平均偏差項；沒有改模型推論為 per-token 量化器。

**局部數據支持**

- 在選定的 Posit6 spike 案例，**88–98% 的誤差落在 spike rows**，但部分 channel 的 spike 梯度較一般位置低 **1.8–33 倍**。
- 96 篇新文章的 4 個預定主要比較：原 BC **1／4** 排對，同位置配對 **4／4**。
- 這些 expert 來自已知失敗案例；新文章確認局部現象，但不能據此宣稱對一般 expert 也有效。

**口稿**

「我們實際改的是統計配對方式。原本一個 channel 用同一個平均敏感度，去乘全部位置的誤差；新方法保留位置對應，或至少分成 spike 與一般兩群。在部分 Posit6 案例，大部分誤差剛好落在梯度較低的位置，因此原 BC 會高估。配對後四個主要比較從只對一個變成全部對，但因為這些是已知失敗 expert，還必須檢查能否推廣，以及完整配置有沒有收益。」

來源：[保護介入](../runs/olmoe_spike_intervention_20261023/report.md)、[同位置配對機制與新文章](../runs/olmoe_pair_gradient_diag_20261023/report.md)、[spike 分組診斷](../runs/olmoe_spike_group_diag_20261023/report.md)。分組診斷使用已看過的快取，屬事後探索，不能當另一批獨立確認。

## 第 6 頁｜Spike 改動的結果：局部有效，完整配置收益未成立

**先看能否推廣**

- OLMoE 另選 16 個一般 expert：原 BC 與配對法在 **58 個可判定配對都正確**，排名沒有改變。
- 另選 spike 分層組：兩個確認的修正都集中在同一 expert；主要同 bit 的可判定樣本不足，未支持普遍替換。
- Qwen3 的 28 層 Down-input 測試：原 BC、spike 分組、同位置配對都為 **74／78**，主要同 bit 誤排沒有被修正。

**再看完整配置**

| 模型／實驗 | 新方法−原 BC 的 ΔNLL | 區間與結論 |
|---|---:|---|
| OLMoE knee，同位置配對 | −0.000341 | 95% CI [−0.00100, +0.000308]，未確認改善 |
| OLMoE aggressive，同位置配對 | −0.000207 | 95% CI [−0.00101, +0.000609]，未確認改善 |
| Qwen3，spike 分解＋端點平方梯度 E，相同資源上限 | **+0.0071** | Native 校正 CI [+0.0021, +0.0122]，變差 |

- OLMoE 兩輪各為 600 篇新文章；Qwen E 為 128 篇新文章，新增統計在 GPU 算，完整模型由 native 執行。E 是後續不同的評分候選，不能當成同位置配對的同一方法。
- OLMoE 全 Down 配對建表只多約 **3.4 秒**，但 knee 的新成本表觸發 MILP，從輸入到配置約 **97 → 369 秒**；aggressive 則仍由 LP 路徑接受。建表增量小，不保證搜尋成本小。

**本週兩件事的結論**

| 方向 | 已取得的結果 | 目前決策 |
|---|---|---|
| GEMQ 啟發的 refresh | Aggressive 有小幅收益，低 bit 更新可省時；正常 knee 未有穩定收益 | 不設為預設 |
| Spike 啟發的評分 | 部分已知局部誤排可修正；一般化與完整配置收益不足 | 保留原 BC，保留機制證據 |

**口稿**

「最後用完整配置判斷是否值得整合。OLMoE 在正常和 aggressive 兩個點都沒有確認收益；Qwen 的分組和配對也沒有修正主要錯誤，後續端點候選在同資源 native 評估還變差。這不表示 spike 沒有影響，而是知道 spike 在哪裡，仍不足以直接設計一個通用評分。這週兩條方向都保留了有用的局部結果，但正式 pipeline 目前維持原 BC。」

來源：[OLMoE 新 expert 泛化](../runs/olmoe_pair_generalization_20261023/report.md)、[OLMoE knee 完整配置](../runs/olmoe_pair_knee_ab_20261023/report.md)、[OLMoE aggressive 完整配置](../runs/olmoe_pair_aggressive_ab_20261023/report.md)、[Qwen3 單點](../runs/qwen3_down_spike_diag_20261023/report.md)、[Qwen3 E 的完整 native 評估](../runs/qwen3_spike_endpoint_pipeline_20261026/report.md)。

## 備用頁 A｜Router 微調與 GEMQ 的關係

有另做固定 q0、只訓練 router 20 步的試驗：新 600 篇文章，ΔNLL **−0.08243**，CI [−0.08556, −0.07925]；額外時間 **342.6 秒**。

這是獨立訓練實驗，不是上述 refresh 的步驟，也沒有做「微調後再刷新」的整套流程。缺少 FP32＋相同 FT 與非 WikiText 對照，不能把全部收益歸給 BC 或量化修復。主講聚焦編譯器分析與格式搜尋，這項只在追問時補充。

來源：[router-only FT](../runs/olmoe_router_ft_20261022/report.md)。

## 備用頁 B｜有沒有試過高精度保護？

有，但本地測的是選定介入或固定配置替換，不是完整復現 QFeM 的選模組流程，也沒有做 QFeP 的 prefix KV cache 方法。

- OLMoE：spike rows 保留 FP32、只量化 spike rows、保護同數量隨機 rows，都是診斷。固定原 ranges，因此主要隔離 rows 自身誤差，不同於重新縮小 INT range。
- Qwen3：原 knee 的 L2／L27 Down 都是 FP16。固定其他格式，分別改為 Posit8，input／output 同時改。
- GPU 新 512 篇文章：改 L2 增加 NLL 約 **0.305**；改 L27 降低約 **0.0085**。高精度保護的必要性依位置與背景不同。
- L27 後續在另一批新 128 篇文章取得 native 確認：NLL **−0.00868**，CI [−0.01179, −0.00555]，BOPs proxy 少 **0.814%**。這是單一固定配置的成功，還不是自動找出這類位置的規則。

來源：[OLMoE 介入](../runs/olmoe_spike_intervention_20261023/report.md)、[Qwen3 FP16 保護](../runs/qwen3_down_fp16_protection_20261024/report.md)、[L27 native 確認](../runs/qwen3_l27_native_confirm_20261024/report.md)。

## 備用頁 C｜Spike 分組的公式與實作邊界

本地分組定義：s_t = max_c |h_tc|，spike 集合 S 使用校準資料的 6×median(s_t) 門檻，N 為其餘 rows。這是本地診斷規則，不宣稱是論文指定的通用門檻。

\[
V_{\mathrm{group}}=\sum_c\left[
\operatorname{mean}_{t\in S}(g_{t,c}^2)\sum_{t\in S}r_{t,c}^2+
\operatorname{mean}_{t\in N}(g_{t,c}^2)\sum_{t\in N}r_{t,c}^2
\right].
\]

三者都使用共同的全域 channel 均值 μ、相同平均偏差項；group 不在組內重新中心化。

- 分組保留兩群平均差異，但遺漏組內 g² 與 r² 的相關性。
- 同位置配對可用 Σg²e²、Σg²e 與既有統計 streaming 累加，不必永久存下所有逐 token 張量。
- 增加評分粒度不等於改用 per-token scale，也不等於執行時將 spike token 分配不同 datatype。
- 分組在已看過資料上的收益沒有超過配對法，且校準資料沒出現 spike 時會退化回原 BC。

來源：[spike 分組](../runs/olmoe_spike_group_diag_20261023/report.md)、[配對完整建表](../runs/olmoe_pair_knee_ab_20261023/report.md)。

## 開場口稿（約 30 秒）

「這週主要報告兩個方向。第一是參考 GEMQ，在 OLMoE 量化後重新取得 BC 統計，並拆成只更新 sensitivity、error 和低 bit 子集，評估品質與離線成本。第二是參考 GLU activation-spike 的觀察，測試 spike 保護、分組以及梯度與誤差的同位置配對。兩個方向都有局部正面結果，但還沒有取得穩定的預設 pipeline 收益，因此目前保留原 BC。」

## 投影片標題可直接使用

1. GEMQ 啟發：在量化工作狀態更新 BC proxy
2. 更新策略：Sensitivity／Error／Low-bit 子集
3. OLMoE 結果：品質收益與 offline 成本
4. Activation spikes：論文觀察與我們的問題
5. Spike-aware 改動：保護、分組、同位置配對
6. 排序與完整配置結果：是否值得整合？
