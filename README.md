# Statistical-Practice

## 國道瓶頸資料專案：已確認研究設定（2026-11-04）

- **正式壅塞 Y：** 使用龍潭—大溪北向的四支 VD（K69.510、K65.990、K63.990、K61.670；不納入 K61.000），以代表區段長度權重計算加權壅塞比例；局部／大範圍主分析門檻 `c=0.50`（2026/11/12）。
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

## 網站與完整程式封存

- [網站首頁](index.html)／[V3 EDA 與 OD 分群 Pilot](v3_lite.html#v3-eda)／[OD 分群比較](od_clustering.html)／[15 分鐘壅塞預測實驗設計](machine_learning.html)
- [2026-11-05 本機 `20_MODEL_ML.zip` 整理後的程式與小型 EDA 結果](20_MODEL_ML/README.md)：原程式保留於獨立目錄，不覆蓋 `10_OD-analysis/` 的主要研究版本。
- GitHub Pages 的來源需設為 `main` 分支 `/(root)`，才能由同一分支同步網站與程式。O 預測 Pilot 數值績效已完成，2025 屬探索性回測。


## 2026-11-12 第一次研究進度報告

- [完整研究內容（35 頁簡報文字整理）](reports/2026-11-12_summary.html)｜[報告頁](reports.html)
- OD10 當期辨識及 OD 特徵比較已完成；O 歷史需求模型以時間／VD 為 Baseline，Top20＋OTHER 的 P／P+Q 為額外歷史需求，三演算法共九組。15／30 分鐘比較及成本敏感、Rejection 已完成 Pilot。
- LightGBM Baseline 的 Log loss 0.1613；K-means 目前只有描述性 O 分群，未完成下游預測比較；2025 為探索性回測，2026 留作獨立驗證。
- 原始報告 PDF／PPTX 歸檔位置：`reports/2026-11-12_report.pdf`、`reports/2026-11-12_report.pptx`（原檔上傳後報告頁會自動顯示下載入口）。
