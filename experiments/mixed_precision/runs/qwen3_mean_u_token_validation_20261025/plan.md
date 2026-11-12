# 計畫：U_token 平均項的跨層與新文本驗證

時間：2026-10-25T02:57:22+08:00。以 plan.json 為準；寫入後的修改須記 amendment。

- 範圍：Qwen3-0.6B 全部 28 層 Down input，一次一層；權重與其他 activation 全 FP32；Down output 不量化；int4／p4e1、int6／p6e1、int8／p8e1，84 個同位元對。
- 結構：只有 L27 之後是逐位置運算；L0–26 之後有 attention，可能有真實跨位置交互。本輪檢驗候選能否轉移，不預設它正確。
- calibration（原 10×128）：e = Q(x)−x；μ = Σe/1280；g = 每窗口 summed CE 對 Down input 的梯度；U_old = Σ_(s,c)μ²(Σ_p g)²；U_token = Σ_(s,p)(Σ_c gμ)²；F_b = Σgμ；S_old = U_old/2540、S_new = U_token/2540、F = F_b/1270。
- selector（新文章前凍結）：每對內 argmin S_old、argmin S_new、argmin F+S_old、argmin F+S_new；tie（|ΔS| ≤ 1e-12·max(1,|S|)）選 INT 並列出。
- 成本路徑：採路徑 1（共同前向／反向保存梯度，μ 算好後重讀）；路徑 2（先 μ 再跑梯度 pass）只作估計。native 目前梯度 pass 與誤差 pass 並行、梯度 pass 不知道 μ。
- 新文本：128 篇新 WT103 × 2 窗口；沿用前輪抽樣規則（verified 掃描排除 8745 篇、歷史 Qwen 窗口、WT2 窗口 ≥ 50% 拒收）；seed {'order': 32101, 'start': 32102, 'bootstrap': 32103}。
- 介入：x_l + αμ（α = [0.125, -0.125, 1.0, -1.0]），經真實 Down linear 與含 attention 的完整 suffix；672 個條件 + FP32 baseline。
- 主要 A：G_even = mean_P[E(q_new_even) − E(q_old_even)]，α0 = 1/8，P = L0–26 的 81 對；95% CI；另報 /α0² 係數尺度。次要 B：G_signed（ΔL(+1)）；若作共同結論另報 Bonferroni 97.5% CI。補充：84 對、L27 3 對、去 L2／L27 的 78 對。
- 統計：文章為單位（兩窗口平均），paired bootstrap 20000 次（seed 32103）；逐對 INT−Posit 的 E(1/8)、E(1)、ΔL(+1) 與 95% CI；E(1/8) 84 對 Holm；修正／新增誤排；各位元、各層分布；O(1/8)/(1/8) vs 凍結 F；tie 清單。
- 控制：TF32 關閉、deterministic、量化器驗證；正確 layer／slot、無 output QDQ；calibration 全部條件「重播 vs 完整前向」≤ 1e-06；α=0；α=1/8 rounding；L27 重現前輪 U／U_token／F（≤ 0.0001）；checkpoint hash。
- 判讀與停止：依規格第 8 節；完成後停止，不改 criterion、不做 HVP、變異項、完整配置或 native。
- 冒煙測試：scratch smoke 2026-10-25 02:43-02:57 (outside runs/): the calibration stage was run in full (all calibration scores, mu, selectors with 2 differing even picks, controls seen); its first version failed the L27 F reproduction because it compared F across two gradient paths, fixed before this plan by a same-path comparison (F path difference reported); eval / analyze ran on 4 calibration windows only (no new article); eval estimate about 50 min for 256 windows

