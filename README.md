# Statistical-Practice

## 國道瓶頸資料專案：已確認研究設定（2026-11-04）

- **正式壅塞 Y：** 使用龍潭—大溪北向的四支 VD（K69.510、K65.990、K63.990、K61.670；不納入 K61.000），以代表區段長度權重計算加權壅塞比例；局部／大範圍門檻 `c` 尚待確認。
- **OD 分群：** 2024 年建立分群、2025 年外部測試。比較 K-means／K-medoids／GMM／Sparse K-means，`K=2,3,4,5`，**預測最佳 K 以保留時序的壅塞預測表現判定，不以 Silhouette 直接定案。**
- [四站 VD 壅塞 Y 與最佳預測 K：完整研究規範](10_OD-analysis/FOUR_VD_Y_AND_PREDICTIVE_K_PROTOCOL.md)
- [OD 分群 Python 程式與使用說明](10_OD-analysis/README.md)

- [四站 VD 的 c=0.25／0.50／0.75 本機比較程式與指令](10_OD-analysis/FOUR_VD_C_RUNBOOK.md)（2026-11-04：程式與合成資料測試完成，真實 VD 結果待本機執行）

## V3_LITE：2024–2025 週末與連假 OD／VD

- [V3 定義、日期與四站 VD 缺值規則](10_OD-analysis/V3_LITE_HOLIDAY_RUNBOOK.md)

### EDA（V3_LITE）

- [已完成的 V3 EDA 統整：OD 長尾、車流時間分布、週末 vs 連假、車種／距離、天氣與資料限制](10_OD-analysis/V3_LITE_HOLIDAY_RUNBOOK.md#g-eda-v3_lite2026-11-05)
- [OD-pair 進階 EDA 程式](10_OD-analysis/v3_od_quick_eda.py)：逐 OD × 2024/2025 × WEEKEND/LONGHOLIDAY 的 24 小時比例、假日差異、車種、距離及跨年穩定性（已完成 2024／2025 全量 EDA）。
- 已確認 V3 共 **16,068,709** 筆、**230** 個研究假日；2024 OD 每組 ≥300 筆時選入 **1,493** 組，涵蓋 2025 年約 **95.07%** 旅次。其他明細與未完成項目見上方 EDA。

- [**V3 OD-pair EDA 實際發現與 64 組分群 pilot**](10_OD-analysis/V3_OD_CLUSTER_PILOT_20261105.md)：比較四個特徵組、K-means／K-medoids／GMM／Sparse K-means、K=2–5；含每群特徵解讀、分群分數與下游 LR／XGBoost／LightGBM 評估設計。**尚未完成壅塞預測**。
- [OD 分群結果：全部候選的分數、群組特性與 OD 標籤](10_OD-analysis/results/v3_od_cluster_pilot/v3_method_scores.csv)（其餘 CSV 連結位於上述研究紀錄）。
