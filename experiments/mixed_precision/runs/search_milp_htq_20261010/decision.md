# Pipeline 決策建議

以本次五模型、相同 native BC 表、11×11 格式與共同固定選點政策的實驗，建議採用 **MILP 粗搜＋局部獨立資源補搜＋三維 Pareto** 作為主要搜尋流程。HTQ beam=200 可保留為重視最短搜尋時間的比較模式；目前沒有理由以 beam=2000 作預設。

## 支持證據

- 對 HTQ beam=2000：MILP 搜尋中位耗時快 3.0～6.0 倍。自動選點的 MobileNetV2、EfficientNet-B0 accuracy 分別高 8.5、5.7 個百分點，GPT-2 PPL 由 69.96 降至 67.86；這三項配對 95% bootstrap 區間都不含零。
- ResNet18 的自動 accuracy 高 0.8 個百分點，但區間 [-0.7, 2.2] 包含零。BERT accuracy 同為 92.20%，不能宣稱 MILP 在這兩個模型有確定 accuracy 優勢。
- 對 HTQ beam=200：HTQ 搜尋更快，每模型約 1.1～5.2 秒；MILP 約 4.3～25.3 秒。MILP 多花約 3.1～20.2 秒，MobileNetV2、EfficientNet accuracy 分別增加 3.7、4.6 個百分點，GPT-2 PPL 降低 2.95；這三項配對區間也不含零。ResNet18、BERT 差異區間包含零。
- 使用 HTQ beam=2000 的實際 size/BOPs 作兩項上限，額外求解 MILP 後，MobileNetV2、EfficientNet、GPT-2 的改善仍存在。因此自動選點的上述優勢不能完全歸因於使用更多資源。資源一致診斷的 BERT accuracy 則由 92.20% 降至 91.74%，區間包含零；BC 最小化不保證 accuracy 最高。
- MILP 三次搜尋在五模型都選出完全相同配置。HTQ 大 beam 五模型均產生三個不同配置。accuracy 依事先協議只評估第一輪，沒有用測試結果挑選有利的一輪。

## 可以決定什麼，還不能決定什麼

可以將 MILP 定為下一版的主要搜尋 backend；它在此實作中兼具相對大 beam HTQ 的速度優勢，以及三個模型較明確的模型品質優勢。HTQ 小 beam 的速度確實較快，應保留這個事實，不能概括成「MILP 比所有 HTQ 設定都快」。

本次比較的是整套搜尋與共同選點政策，沒有單獨證明局部補搜或三維 Pareto 各自帶來多少 accuracy 提升。最終選點使用解析端點正規化的三軸等權理想點距離，與舊版 max-resource＋二維 knee 不同；這次不能被解讀為已證明它比舊版選點政策更好。

三次計時不能替代多次 accuracy 實驗。配對信賴區間反映固定配置在本次測試樣本上的差異，不涵蓋 HTQ 搜尋的跨次變動、重新抽取校準集的變動，或新模型的表現。

本次未切換正式 `mp_auto.py`，評估資料也沒有回饋給搜尋或選點。後續若要改變選點政策，應用獨立的驗證資料或新測試集評估，避免反覆用這份測試集調參。

完整數字、資源與限制見 [report.md](report.md)，逐筆核驗見 [independent_validation.json](independent_validation.json)。
