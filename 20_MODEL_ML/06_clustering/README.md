# 分群方法報告動畫 V3

這一版專門為「明天報告」調整。

主講四種：
1. K-means
2. Hierarchical clustering
3. DBSCAN
4. GMM

K-medoids 與 HDBSCAN 改成補充，不再放在主動畫。

## 主要修改

### K-means
- 清楚畫 assignment -> centroid move -> reassignment。
- 讓觀眾看懂「為什麼不是只分一次」。

### Hierarchical
- 左邊逐步累積連線。
- 右邊同時顯示 dendrogram。
- 在剩 3 群時明確顯示「在這裡切成 3 群」。
- 教學動畫使用 Single linkage；正式分析若採 Ward，要另說明 Ward 準則不同。

### DBSCAN
- 取代 HDBSCAN 當主講。
- 明確顯示：
  1. raw data
  2. epsilon neighborhood
  3. core points
  4. density expansion
  5. noise
- 更適合第一次接觸 clustering 的觀眾。

### GMM
- 仍保留 Gaussian ellipse，因為 2D Gaussian 的等密度線自然是橢圓。
- 但重點改成 soft membership：
  - 圈出一個邊界點
  - 顯示屬於各 component 的機率
  - 再讓 mean / covariance 更新

## 安裝

```powershell
pip install numpy scipy matplotlib scikit-learn pillow
```

## 執行

```powershell
python clustering_report_v3.py
```

輸出在：

```text
animations\
01_kmeans.gif
02_hierarchical.gif
03_dbscan.gif
04_gmm.gif
```
