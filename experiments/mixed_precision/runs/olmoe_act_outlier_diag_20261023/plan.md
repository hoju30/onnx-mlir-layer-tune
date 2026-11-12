# 預先登記：OLMoE activation outlier 與 BC 排序誤差（診斷）

時間：2026-10-23T01:35:30+0800。以 plan.json 為準。

## 不預設因果
- 分開回答：outlier 是否存在、是否主導量化誤差與實測 NLL 損傷、BC 是否因此排錯。
- 不修改 criterion、不 fine-tune、不重選 knee、不改成 weight-only、不加入新量化方法；干預只是診斷 oracle。

## 量化粒度（程式確認）
- INT activation：每個實體 QDQ 位置（expert × 投影 × x／y 槽）一組 scalar scale／zero-point，來自 FP32 calibration 的 [min, max]（含 0）；是 per-tensor，不是 per-channel scale。
- Posit activation：直接對原值做 LUT 捨入，沒有 scale 也沒有 range；INT 的 range／clipping 機制不套用到 Posit。
- BC 使用逐 channel 的 G2、GS 與誤差統計（per-channel 統計 ≠ per-channel scale）。
- 實體位置：gate.x、up.x（同一輸入 xe 的兩份副本、range 相同；xe 的統計只算一次）、gate.y、up.y、down.x、down.y。BC 對 gate.x 與 up.x 都用 xe 的 total gradient；選中的 x tap 另算單一路徑梯度作診斷。

## Outlier 指標（門檻固定）
- 逐 channel：max|a|、min、max、RMS、mean、|a| 的 p99（rows ≥ 100）、max 是否出現在 position 0。
- 跨 channel：R_cross = max_c max|a_c| ÷ median_c max|a_c|；channel 內：max|a_c| ÷ RMS_c（RMS 近零者排除）。
- outlier channel：max|a_c| ≥ 6.0 × median_c max|a_c|。token：前 1% rows 與 position-0 rows 的 Σa² 占比。
- 狀態：FP32 與 q0，calibration 窗口；每個張量報告 expert token 數；FP32 rows < 64 不參與選點。

## 誤差
- 候選：int4, int6, int8, p4e1, p6e1, p8e1。逐張量：MSE、平均誤差、變異；INT clipping 比例與 clipped 誤差占比；outlier channel、前 1% channel、前 1% token、position-0 的誤差能量占比。
- BC 分解：各實體 slot 的變異項 Σ G2/n·(S2 − n·μ²) 與偏差項 Σ GS·μ²，以及 outlier channel 的占比（預測成本占比 ≠ 實測損傷占比）。

## 選點（不使用任何 BC 誤排資訊）
- 6 種 tap 類型 × 強（R_cross ≥ 同類 90 百分位）／弱（≤ 中位數），每格 2 個，共 24 個。
- 每類隨機抽 2 個強 tap（seed 23121），各配一個同類弱 tap：優先同 q0 activation 位元類別（≤6／7–8／≥16），再取 log rows 最近者；不重複使用 expert。
- tap 的 outlier channel 只由 FP32 discovery（calibration）決定。

## 資料切分
- discovery = 原 calibration（BC 也用它）；確認 = 凍結後才抽的 64 篇新 WikiText-103 文章 × 3 窗口（排除 6074 篇歷史文章）。

## 主測量與排序
- FP32 背景、一次只量化一個實體 tap、dynamic route；ΔNLL = NLL(tap) − NLL(FP32)；tap 之前的層重用 FP32 結果（逐位元控制）；記錄下游路由變化與確認資料上的 INT clipping。
- 每個 tap 15 個候選配對：BC 排序 vs 實測排序；決定性配對 = 實測差的 95% 文章 bootstrap CI 排除 0；BC 平手 = 相對差 < 0.02；INT-INT、Posit-Posit、跨家族、同位元分開；報告實測近零配對數與每個 tap 的 Spearman。
- outlier 與誤排的關聯：強 vs 弱的誤排率（以 tap 為單位 bootstrap）；R_cross 與誤排率的 Spearman（描述）。
- q0 背景：4 個代表案例的投影層級 activation 格式替換（int6、p6e1，若等於現行格式則改 int8／p8e1）；與 static 表的成本差比較；這是完整配置中的邊際效果，與 FP32 單點結果分開報告。

## 干預（只有觸發才做）
- 觸發（任一）：T1 for int4 / int6, the outlier channels carry >= 50% of the tap error energy (FP32 discovery) in at least half of the strong taps；T2 median over the matched (strong, weak) pairs of measured int4 dNLL(strong) / dNLL(weak) >= 2 (pairs with both dNLL > 0)；T3 decisive-pair misranking rate strong - weak >= 0.10 with a 95% tap-bootstrap CI excluding 0。
- 案例：至少有 1 個 outlier channel 的強 tap；int4、int6（Posit 沒有 range，不做）。
- I1 outlier channel 誤差 bypass；I2 相同數量的隨機一般 channel bypass；I2b 最接近中位數幅度的 channel bypass；I3 在 I1 之上，其餘 channel 用不含 outlier channel 的 discovery range 重新校準。
- I1 對 I2／I2b：outlier channel 自身誤差的損傷；I3 對 I1：outlier 撐大共用 range 的損傷；兩者分開解讀。

## 統計與預算
- 10000 次文章 bootstrap（seed 23111）；強弱比較以 tap 為單位（seed 23112）。
- 預算（分鐘）：{'discovery': 5, 'select': 2, 'pathgrad': 3, 'data': 4, 'measure': 25, 'q0local': 5, 'intervene': 12, 'analyze': 3, 'total_gpu': 60}；主測量超過預算則在目前層組結束後停止，只分析已完成的 tap。看到確認結果後不改候選、tap 或門檻。
