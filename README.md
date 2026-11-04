# Statistical-Practice

## 國道瓶頸資料專案：已確認研究設定（2026-11-04）

- **正式壅塞 Y：** 使用龍潭—大溪北向的四支 VD（K69.510、K65.990、K63.990、K61.670；不納入 K61.000），以代表區段長度權重計算加權壅塞比例；局部／大範圍門檻 `c` 尚待確認。
- **OD 分群：** 2024 年建立分群、2025 年外部測試。比較 K-means／K-medoids／GMM／Sparse K-means，`K=2,3,4,5`，**預測最佳 K 以保留時序的壅塞預測表現判定，不以 Silhouette 直接定案。**
- [四站 VD 壅塞 Y 與最佳預測 K：完整研究規範](10_OD-analysis/FOUR_VD_Y_AND_PREDICTIVE_K_PROTOCOL.md)
- [OD 分群 Python 程式與使用說明](10_OD-analysis/README.md)
