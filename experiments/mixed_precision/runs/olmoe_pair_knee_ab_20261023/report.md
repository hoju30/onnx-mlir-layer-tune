# OLMoE 完整配置 A／B：原 BC vs 只替換 Down 輸入 activation 的逐位置配對評分

> **下界更正（2026-10-23）**
> 本報告把 B 寫成「MILP，gap 0，已證明最佳」，這不正確：gap 0 與 dual bound 18.784755200 屬於兩個預算都收緊 1 ppm 的重試問題，不能證明原始預算問題已達最佳。
> 原始預算的 MILP 第一次求解得到的解在 BOPs 上超出 4.5e-07（相對值），屬於不可行。
> 以原始預算的 LP 下界 18.784720089 計，B 最終解的差距是 3.51e-05（相對 1.9e-06）；A 的對應差距是 0.0055。
> 結論不變。詳見 `correction_lower_bound_20261023.json`（含原始 hash）。


- 日期：2026-10-23
- 程式：`run_olmoe_pair_knee_ab.py`
  - plan 到 configs 階段：sha256 `ed3bbec8…`；
  - freeze 起：`9fe1411c…`，只修了一行路徑；見第 9 節的修正紀錄。
- 預先登記：`plan.json`、`plan.md`（16:15:21），在產生任何新表、新配置或評估資料之前。
- 凍結：
  - `freeze.json`（16:29:22）：表、配置、程式、hash；
  - `data_freeze.json`（16:32:55）：600 篇新文章。
- 正式 pipeline 的預設仍是原 BC。這是獨立的實驗分支：沒有 fine-tune、沒有刷新 proxy、沒有新的 frontier 或 knee。

## 結論摘要

**主要結果：改善未確認，保留原 BC（依預先登記的決策規則）。**

1. **配置有改變。** 在 A 的精確資源上限下，B 與 A 共有 43 個投影不同，分屬 43 個 expert：

   | 分類方式 | 細項 |
   |---|---|
   | 改變內容 | 只改 activation 24、只改 weight 16、兩者都改 3 |
   | 投影類型 | Down 27、Up 10、Gate 6 |
   | 層別 | 前段 11、中段 27、後段 5 |

   - A 中有 3 個極端 spike expert 的 Down 輸入原本是 f16，B 降為 p8e1：L1.E11、L6.E59、L6.E9。
   - 省下的預算主要用在 p6e1 → p8e1（9 個）與 weight int5 → int6（12 個）。
2. **完整配置的品質差異未確認**（600 篇新文章 × 4 窗口）：
   - Δ = NLL(B) − NLL(A) = **−3.41e-4**，95% CI **[−1.00e-3, +3.08e-4]**，CI 含 0。
   - B 較好的文章 314 篇、較差的 286 篇。
   - NLL（PPL）：FP32 2.95384（19.179）、A 2.97028（19.497）、B 2.96993（19.491）。
   - B 的量化損傷點估計比 A 少 2.1%，但未確認。依規則，「沒有顯著變差」不能當成值得接入的證據。
3. **控制全部通過。**
   - 重建的 A 表與原 static 表逐位元相同；A 表的搜尋重現 q0_knee。
   - B 的 weight 列與非 Down 的 activation 列都與 A 逐位元相同。
   - streaming 配對值與直接逐元素值的相對差 ≤ 7.2e-16。
   - 評估端：A 在舊批次上的程式控制、B 的重跑都逐位元一致。
4. **搜尋公平性。** A 與 B 用同一套規則，但走到不同分支：
   - A：LP 四捨五入解直接被接受，與 LP 下界差 0.0055，在容許值內，但沒有證明最佳。
   - B：LP 四捨五入與下界差 0.020，超過容許值，改走修正版 MILP（gap 0，已證明最佳）。
   - A 配置在 B 表上的目標值高於 B 的解，所以預先登記的近似處理規則沒有觸發。
   - 兩者都在資源上限內，B 幾乎用滿 BOPs（餘量 8.4e6，A 為 3.05e9）。
5. **offline 成本：主要增加來自搜尋，不是配對統計。**

   | 項目 | A | B（從頭） | B（增量） |
   |---|---|---|---|
   | 建表 | 92.2 s | 95.7 s | 17.7 s |
   | 其中配對統計 | – | 3.1 s | 3.1 s |
   | 剪枝＋求解 | 4.2 + 0.8 s | 4.2 + 268.7 s | 4.2 + 268.7 s |
   | 從輸入到配置 | 97 s | 369 s | 291 s |

   - 全部 1017 個 Down 輸入 × 10 格式的配對統計只多 3.1 s，B − A 的建表時間中位數差 3.4 s（+3.7%），峰值記憶體不變。
   - B 表在這個資源點上 LP 無法直接接受，MILP 求解花了 269 s。這是 B 表在此點的求解性質，不一定在其他資源點重現。
6. **建議**
   - 不接入預設。
   - 若要繼續，下一輪應預先登記其他資源點與自動 knee，並考慮搜尋時間；不能用本批評估結果去修改配置後重新報告。
   - 不宣稱算到 Hessian，也不宣稱 ONNX-MLIR native 整合或低位元 kernel 加速。

## 1. A／B 定義與固定條件

| 項目 | A | B |
|---|---|---|
| 成本表 | 原 static BC（`runs/olmoe_proxy_refresh_20261022/table_static.json`） | 只把 1017 個搜尋 expert 的 Down 輸入（down_proj x，h = SiLU(g)·u）變異項由 Σ_c G2_c/n·Σ_t r² 換成 Σ_c Σ_t g²r² |
| 不變的部分 | – | 偏差項 Σ_c GS_c μ_c²、Down 輸出項、Gate／Up 的 x／y 項與共享輸入歸屬、全部 weight 列、量化器、range、calibration、資源模型、合法性、3051 個搜尋節點 |
| 候選 | 完整網格：int4–int8、p4e1、p6e1、p8e1、f16、bf16，以及 FP32（成本 0） | 同 A |
| 配置 | q0_knee（`runs/olmoe_knee_selective_sensitivity_20261022/configs/A.json`） | 在 A 的精確上限搜尋一次 |

- **精確上限**（直接讀 `knee/knee.json`）：max_bytes = 4524359631.556117，max_bops = 46648327679632.125。
- **梯度語義**（與 static BC 相同）：
  - FP32 calibration（10 × 128），sum-loss next-token CE；
  - route 與權重在 no-grad forward 中記錄、backward 時固定；
  - g、h、e 來自同一次逐層重算的 forward。
- **streaming 計算**：每個 expert 與格式累加 Σ_t g²e²、Σ_t g²e，再與 G2、μ = S1/n 組成 V_pair,c = Σg²e² − 2μ_cΣg²e + μ_c²G2_c。
  - 使用 float64，不做任何截斷，不保存逐 token 張量。
  - 實作上另做一次 QDQ（沒有與既有誤差計算融合），所以量到的時間是上限。

## 2. 控制

| 控制 | 結果 |
|---|---|
| 重建的 A 表（r1、r2）與 static 表 | 逐位元相同；B 建表中的 A 部分也相同 |
| 增量版 Binc 的 G2／GS、S1／S2、n 與 static cache | 全部逐位元相同 |
| B 表 weight 列與 A | 逐位元相同 |
| B 表非 Down 的 activation 列與 A | 逐位元相同 |
| B 表的有限性與完整性 | 全部有限，格式完整；1017 個 Down 列中 1009 個改變 |
| Down 列拆項 | A 列 = vx_old + vy，誤差 0（逐位元）；B 列 = vx_pair + vy；拆項存於 `work/runs/build_B_r1/down_terms.json` |
| B（r1、r2）、Binc（r1、r2）的重複一致性 | 逐位元相同 |
| Binc 與完整 B 的 Down 列 | 相對差 ≤ 4.1e-16 |
| 預先指定的直接逐元素驗證（L1.E9、L2.E30、L9.E8、L0.E28、L15.E36） | 相對差 ≤ 7.4e-16 |
| A 表的搜尋 | 重現 q0_knee（config hash 相同；目標值 18.297319102902886，與 knee 記錄相同） |
| 評估：A 程式控制（weight-sensitivity 輪的 128 個窗口） | CE 差 0.0 |
| 評估：B 重跑前 128 窗口 | CE 差 0，hidden 128 / 128 逐位元相同，routes 相同 |
| 評估：有限性與 hook 審核 | 通過 |
| 輸入檔與模型檔 hash | 前後一致 |

**停止規則被觸發一次（已調查，見修正紀錄）**
- 有 18 個（expert, 格式）組合的 streaming 變異項總和為負，最小 −1.5e-21。
- 全部來自 4 個只有 1 個 calibration row 的 expert：L6.E41、L7.E28、L14.E28、L15.E51。
- 這時真正的變異項恰好為 0：直接逐元素值為 0.0，原 BC 也為 0。
- 加上偏差項後，vx_pair 與 vx_old 逐位元相同，成本不受影響。
- 其餘 10,152 組的 streaming 與直接值最大相對差為 7.2e-16。
- 依 plan 不做數值保護，成本不修改。診斷程式與結果在 `diagnostics/`。

## 3. 搜尋

| | A | B |
|---|---|---|
| 路徑 | LP → 四捨五入＋修復被接受 | LP 四捨五入不被接受 → MILP 第一次超出 BOPs 4.5e-7 → 1-ppm 重試可行 |
| 目標值（自身表） | 18.297319 | 18.784755 |
| LP 下界 | 18.291816 | 18.784755（MILP 證明最佳） |
| gap | 0.0055（容許值 ≈ 0.018，接受） | 0（四捨五入時為 0.020，超過容許值 0.0188） |
| 保留的選項（剪枝後） | 70153 | 70128 |
| 剪枝／求解時間 | 4.2 s／0.8 s | 4.2 s／268.7 s |
| size 用量 | 97.66% | 97.75% |
| BOPs 用量 | 99.9935% | 99.99998% |
| 預算餘量（bytes／BOPs） | 1.057e8／3.05e9 | 1.017e8／8.4e6 |

**交叉評分**（不同表的絕對值不可比較）

| 表 | A 配置 | B 配置 |
|---|---|---|
| A 表 | 18.2973 | 18.4065 |
| B 表 | 18.8671 | 18.7848 |

- 各表上自己的解都比較低，所以近似處理規則沒有觸發，也沒有搜尋受限的情況。
- **公平性限制**：A 的 LP 解沒有被證明最佳（gap 0.0055，約 0.03%），B 的解已證明最佳。兩者遵守同一套預先登記的規則；這個差異是規則分支造成的。

## 4. 配置差異（`config_diff.json`）

**改變的投影：43 個，分屬 43 個 expert**

| 分類 | 細項 |
|---|---|
| 改變內容 | 只改 activation 24、只改 weight 16、兩者都改 3 |
| 投影類型 | down_proj 27、up_proj 10、gate_proj 6 |
| 層別 | 前段（L0–L1）11、中段 27、後段（L14–L15）5 |
| 強 spike expert（R_token ≥ discovery p90 = 16.4） | 9 個 |

**格式轉換**

| 類型 | 轉換（數量） |
|---|---|
| activation | p6e1 → p8e1（9）、p8e1 → int8（5）、int7 → p8e1（3）、p6e1 → int7（3）、f16 → p8e1（3）、p8e1 → p6e1、int8 → int7、int7 → int8、int6 → int7（各 1） |
| weight | int5 → int6（12）、int6 → int7（3）、int4 → int5（2）、int6 → int5、int5 → int4（各 1） |

**A 中原本為 f16 的 Down 輸入**
- B 降為 p8e1：L1.E11、L6.E59、L6.E9。
- 其他 f16 仍保留：L1.E9、L1.E18、L2.E30、L9.E8 等。
- 全部 activation 中，f16 從 35 個變為 32 個。

**整體格式分布（A → B）**

| 項目 | A | B |
|---|---|---|
| activation p8e1 | 2188 | 2197 |
| activation p6e1 | 506 | 495 |
| activation int8 | 62 | 67 |
| activation int7 | 210 | 211 |
| activation f16 | 35 | 32 |
| weight int5 | 1343 | 1333 |
| weight int6 | 1389 | 1397 |
| weight int7 | 125 | 128 |
| weight int4 | 169 | 168 |

完整清單見 `config_diff.json` 與 `figures/transitions.png`。

**資源**
- A：M_rel 0.69059，B_rel 0.70706。
- B：M_rel 0.69121，B_rel 0.70711。
- 兩者都在上限內。

## 5. 新資料評估（600 篇 × 4 窗口 × 128 token）

**資料**
- 使用 `wt_manifest_scan.used_wt103_verified`，排除 6521 篇歷史文章（含所有 discovery、確認與 calibration 相關文章；calibration 屬 WikiText-2，整批排除）。
- 跳過：近似重複 62 篇、太短 7 篇。
- 與歷史文章的 content hash 交集為 0（已直接核對）。

**主要結果**

| 指標 | 值 |
|---|---|
| Δ = NLL(B) − NLL(A)，文章平均 | **−3.41e-4** |
| 95% 文章 bootstrap CI | **[−1.00e-3, +3.08e-4]**，含 0 → **改善未確認** |
| 文章間標準差 | 8.13e-3 |
| B 較好／較差的文章數 | 314／286 |

**各自的結果**

| 條件 | NLL（token 加權） | PPL | 相對 FP32 的損傷（文章平均，95% CI） |
|---|---|---|---|
| FP32 | 2.953840 | 19.1795 | – |
| A | 2.970275 | 19.4973 | 1.644e-2 [1.525e-2, 1.771e-2] |
| B | 2.969934 | 19.4906 | 1.609e-2 [1.493e-2, 1.733e-2] |

- 絕對改善 3.4e-4 nat／token（點估計）；量化損傷的相對改變 −2.1%（點估計，未確認）。
- 文章分布見 `figures/per_article_delta.png`。
- **規模參照**：記憶中，不同後端之間完整配置的實作雜訊約 1e-3 到 2e-3。本輪的比較使用同一後端與逐位元控制，所以這個參照不直接適用，只用來說明效果量級很小。

**路由診斷**（不是主要比較）

| 項目 | A | B |
|---|---|---|
| 實際路由的 BOPs proxy（相對 int8 × fp32 計數） | 0.67778 | 0.67807（+0.04%） |
| 路由到未搜尋 expert 的 token | 169,307 | 169,408 |

- FP32 下路由到未搜尋 expert 的 token 為 167,116。
- B 與 A 的路由集合在 16.6% 的（token, 層）上不同。
- 搜尋預算相同，不代表 dynamic route 下的實際工作量完全相同；這裡兩者差 0.04%。

## 6. 完整 offline 成本

### 6.1 建表（2 次交錯；中位數）

GPU 為 RTX 3060；資料都在 page cache 中；建表之間的負載約 1.0–1.8。

| 類別 | A（原 BC） | B（從頭） | Binc（增量，重用 static 表與 cache） |
|---|---|---|---|
| 載入與準備 | 10.1 | 10.1 | 10.3 |
| forward（routing） | 0.8 | 0.8 | 0.8 |
| backward | 1.6 | 1.6 | 1.5 |
| weight 敏感度 | 27.2 | 27.3 | – |
| weight 誤差 | 35.7 | 35.7 | – |
| activation 梯度統計 | 3.1 | 3.2 | 0.6 |
| activation 誤差 | 12.3 | 12.4 | – |
| **配對統計（新增）** | – | **3.1** | **3.1** |
| 組表＋寫檔 | 0.3 | 0.4 | 0.7 |
| **method 合計** | **92.2** | **95.7** | **17.7** |
| GPU／CPU 峰值 | 3.82／14.4 GB | 3.82／14.4 GB | 3.10／15.6 GB |

- 全部 1017 個 Down 輸入 × 10 個格式的配對統計是 3.1 s。這是本輪的實測，沒有沿用前幾輪 24 個 tap 的 0.011 s 去外推。
- B − A 的建表時間中位數差 +3.4 s（+3.7%）；峰值記憶體不變。
- **儲存**：B 表 2.28 MB，與 static 表相同；Down 拆項診斷檔 0.84 MB；沒有逐 token 張量。若把配對累加量寫入 cache 供未來重用，估計約 1.17 GB（約為 activation-error cache 的大小；估計值，未實測）。

### 6.2 搜尋與從輸入到配置

| 項目 | A | B |
|---|---|---|
| 剪枝 | 4.2 s | 4.2 s |
| 求解 | 0.8 s（LP） | 268.7 s（LP → MILP → 1-ppm 重試） |
| 從輸入到配置，從頭（建表中位數＋剪枝＋求解） | 97.2 s | 368.6 s |
| 從輸入到配置，增量 | – | 290.6 s |

- 從頭的數字是各階段實測值相加（建表為中位數、搜尋為單次），不是一次完整 pipeline 的實測。
- 若從頭的 pipeline 還要自己選上限（自動 knee frontier），需另加 frontier 時間；前幾輪 A 約 80 s，本輪沒有執行。

### 6.3 最終評估（另列）

- FP32 123.7 s、A 268.9 s、B 268.7 s、B 重跑 20.8 s；GPU 峰值 7.7 GB。
- 這是量化模擬的 PyTorch 推論時間，不代表低位元 kernel 的速度。

## 7. 判讀

**依預先登記的決策規則**
- CI 含 0：改善未確認，保留原 BC。
- 不把「沒有顯著變差」或負的點估計當成接入證據。

**資源與搜尋**
- 兩者都合法，上限完全相同，搜尋規則也相同，所以這不是「比較受限」的情況。
- 但 A 的 LP 解未證明最佳（gap 約 0.03%），而 B 是 MILP 的最佳解。這可能讓 B 在自己的表上稍佔優勢；由於 CI 含 0，這不改變結論。

**為什麼局部改善沒有轉成可確認的收益**
- 預先登記的比較只回答「是否改善」。
- 描述上，配對主要只改變極端 spike expert 的 Down 輸入相對成本。B 把 3 個 f16 降為 p8e1，省下的預算分散到 40 個投影的小幅升級。
- 整體改變很小（43 / 3051 個投影），加上文章間差異大（標準差 8e-3），在 600 篇文章上無法分辨這樣的效果。
- 前幾輪顯示配對在一般 expert 上不改變排序，所以本輪的小變動與預期一致。

**下一步（若要繼續）**
- 需要新的預先登記，在其他資源點與自動 knee 上檢驗。
- 也要把 B 表在某些資源點可能需要 MILP（本輪 269 s）的搜尋成本納入考量。
- 不能用本批評估結果去修改配置後重新報告。

## 8. 計時（各階段）

| 階段 | 秒 |
|---|---|
| plan | 35.1 |
| build A r1 / B r1 / Binc r1 / A r2 / B r2 / Binc r2（process wall） | 95.0 / 98.3 / 22.1 / 94.7 / 98.3 / 21.8 |
| search（含兩次剪枝與求解） | 280.1 |
| configs | 0.1 |
| freeze | 0.3 |
| data | 213.9 |
| eval | 712.3 |
| analyze | 14.0 |
| figures | 0.6 |

GPU 合計：建表約 7.2 分，評估約 11.9 分。

## 9. 修正紀錄（`amendments.json`）

1. **16:25，搜尋之前：停止規則被觸發。**
   - 18 組 streaming 變異項總和為負，經診斷是單一 calibration row 的消去殘差，成本不受影響。
   - 依 plan 不做數值保護，繼續執行。
   - 診斷程式與輸出在 `diagnostics/`；script sha 不變。
2. **16:29，freeze 階段，任何評估資料之前。**
   - freeze 因相對路徑的 `relative_to()` 出錯，在寫出任何檔案之前就中止。
   - 修正為 `resolve()` 後重跑。表、配置與 plan 都沒有改變。
   - script sha：`ed3bbec8…` → `9fe1411c…`。

## 10. 重現

```
cd experiments/mixed_precision
O=runs/olmoe_pair_knee_ab_20261023
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
S=run_olmoe_pair_knee_ab.py
$CPU $S --stage plan --out-dir $O
for vr in "A 1" "B 1" "Binc 1" "A 2" "B 2" "Binc 2"; do set -- $vr; $GPU $S --stage build --variant $1 --rep $2 --out-dir $O; done
$GPU $O/diagnostics/negative_total_diagnosis.py $O        # stop-rule diagnosis (optional; does not change costs)
$CPU $S --stage search --out-dir $O
$CPU $S --stage configs --out-dir $O
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage eval --out-dir $O
$CPU $S --stage analyze --out-dir $O
$CPU $S --stage figures --out-dir $O
```

各階段都拒絕覆寫既有輸出。

### 主要 hash（sha256 前 16 位）

| 檔案 | hash |
|---|---|
| `plan.json` | `bb20be6bbae0cfe7` |
| `search.json` | `9bef2feecff64580` |
| `config_diff.json` | `8c45b9e3edc72b83` |
| `configs/A.json` | `de285bcc3beda683`（= q0_knee） |
| `configs/B.json` | `d89038eb71c7eb11` |
| `tables_B.json` | `0e77ca315f737f7e` |
| `freeze.json` | `0ee0975d28559af2` |
| `tokens.npy` | `1f5fe3e9549de0f7` |
| `segments.json` | `c870e23cb649ad18` |
| `eval.json` | `12b4b78826b6cc77` |
| `analysis.json` | `f846329a1925b7d8` |
| `per_article.csv` | `4341f245b1b2d557` |
| `amendments.json` | `333a36e0b75287ac` |
| `diagnostics/negative_total_diagnosis.json` | `09e7d68446b86e71` |

各建表輸出的 hash 都在 `freeze.json`。

### 輸出檔

| 類別 | 檔案 |
|---|---|
| 預先登記與凍結 | `plan.json`、`plan.md`、`freeze.json`、`data_freeze.json` |
| 成本表 | A = static 表（路徑見 plan）；`tables_B.json`；`work/runs/build_*/`（各次建表的表、Down 拆項、run.json） |
| 配置與搜尋 | `configs/A.json`、`configs/B.json`、`search.json`（含完整求解紀錄）、`logs/` 內的 HiGHS 記錄、`config_diff.json` |
| 資料與評估 | `segments.json`、`tokens.npy`、`eval.json`、`routes_{fp32,A,B}.npz` |
| 分析 | `analysis.json`、`per_article.csv`、`per_window.csv` |
| 圖表 | `figures/transitions.png`、`figures/per_article_delta.png` |
| 計時與紀錄 | `timing.json`、`timing.jsonl`、`amendments.json`、`diagnostics/` |
