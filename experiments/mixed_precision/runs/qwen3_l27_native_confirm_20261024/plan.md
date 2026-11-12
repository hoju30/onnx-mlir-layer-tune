# 計畫：Qwen3-0.6B L27 Down activation 的 native 獨立確認

時間：2026-10-24T02:11:14.875269+08:00。以 plan.json 為準；寫入後的程式修改須記錄 amendment。

- 問題：只把 L27 Down 的 activation 由 f16 改成 p8e1（input 與 output 都量化），在 onnx-mlir 編譯模型、新文章上是否重現前輪 PyTorch QDQ 的品質改善。
- 不改 criterion、不搜尋、不重選 knee、不 fine-tune、不換 calibration、不改 pipeline；不測 L2、不加 Posit6 或其他候選；FP32 非必要條件。
- 配置：A、C 直接取自前輪（檔案 sha256 與前輪相同；hash A 36205a0624a3…、C d83946b8ec55…）；唯一差異 /model/model/layers.27/mlp/down_proj/MatMul 的 a：f16→p8e1，w 都是 int8；L2 Down 仍為 f16。
- 資源（原成本函式重算並與前輪逐值核對）：A size 0.759821、BOPs 0.877976、R 0.877976；C size 0.759821、BOPs 0.870833、R 0.870833；搜尋範圍總 BOPs -0.814%，weight bytes 不變。BOPs 是 proxy。
- 編譯：Krnl O3、clang O0、constants to file、MLIR bytecode、MIXEDP_WEIGHT_PER_CHANNEL=1、native_eps_fix 的 model.mlirbc 與 ranges；由配置重新產生 MIXEDP_NODE_FORMATS，須與前輪 compile_env.json 逐字相同（196 個條目逐一核對）。保留 .so、constants、編譯紀錄與 hash。
- 控制：config diff 只有 L27 Down a；calibration 窗口 0–1 的 native A／C summed CE 須重現前輪（≤ 1e-08）；A 的窗口 0 在另一個新 process 重跑（≤ 1e-08）。失敗先定位，不放寬容差。
- 執行：4 個隔離 worker，每個 process 只載入一個 .so，OMP／MKL／BLAS threads = 1；float32 logits 串流進 run_native_bc_pipeline.cross_entropy（float64），127 個位置；逐窗口 checkpoint，標頭含 tokens／.so／constants／程式 hash、配置與窗口範圍，續跑必須相符；截斷的最後一行丟棄重算；不跳過任何窗口；記錄 VmHWM。
- 資料：128 篇新 WT103 文章 × 1 個 128-token 窗口（Qwen3 tokenizer）；verified 掃描排除 8361 篇（含 spike 診斷 128 篇、前輪 512 篇）；排除 WT2 重複／近重複、歷史 Qwen 窗口重疊、本輪內部近重複；歷史來源無法解析即停止；seed：順序 29101、起點 29102；控制窗口不計入正式評估；不與前輪 512×3 合併做主要推論。
- 主要統計：每篇 NLL = summed CE / 127；d_native = C − A；paired 文章 bootstrap 20000 次（seed 29103），雙尾 95% CI；δ = 0.001。
  上界 < 0 → 支持 native 品質改善；CI 全正且上界 < δ → 小幅變差但在容許內；上界 < δ（其他）→ 支持非劣、改善未確認；下界 > δ → 超過容許；其餘 → 未定，保留 A。CI 跨 0 不等於等效；未顯著不等於沒有收益。
- 次要：同一批窗口的 PyTorch QDQ A／C（修正後量化器、TF32 關閉、deterministic、FP32）；b = d_native − d_pytorch 的平均與 95% CI、兩後端效應相關、逐文章符號一致率。不改變主要判讀，也不據差異宣稱某後端有錯。
- 檢定力背景（估計，非保證）：80% 檢定力可偵測約 0.0035（native calibration SD）至 0.0042（PyTorch 逐窗口 SD）nat/token；前輪 native 在 calibration 上只有 −0.0018 ± 0.0045，若 native 效應約 −0.002，確認改善的機率低。樣本數固定，不依途中結果增減。
- 決策：即使 native 成功，只記為已選定案例的獨立確認，不改正式配置或 criterion；未確認則保留原配置，不加樣本、不以其他層替代。
- 計畫前觀察：只讀前輪報告與輸出；另在 scratchpad（不在 runs/）做冒煙測試：建 A／C，跑控制階段、續跑與鎖定測試，以及 native／PyTorch 在 calibration 窗口 0–9 上的評估（前輪皆已量過）；以其他 seed（999201／999202）測試 audit／data，抽 4 篇未評估。設計未改，只加入「同一 checkpoint 只允許一個寫入者」的鎖。沒有評估任何新文章。

