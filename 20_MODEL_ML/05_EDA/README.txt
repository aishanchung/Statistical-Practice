OD K-selection dashboard
========================

目的
----
先不要預設 K=2 或 K=3。

固定同一批 OD：
    Trips >= 100

掃描：
    K = 2, 3, ..., 10

快速比較：
1. KMeans
2. Cosine KMeans
3. Agglomerative Ward
4. Gaussian Mixture Model

等 K 的合理範圍縮小之後，
再跑較慢的：
- KMedoids
- KShape
- DTW
- HDBSCAN（本身不需 K）


為什麼先固定 threshold=100？
------------------------
因為我們前面已經發現 threshold 本身會改變資料組成。

這一支程式的目的只有一個：

「在同一批 OD 下，K 到底選多少比較合理？」

不要把 threshold effect 和 K effect 混在一起。


輸入
----
D:\統計實務\Data\OD_ANALYSIS_N3_N_2024_2025_V2\
01_od_summary_all.parquet


輸出
----
D:\統計實務\Data\OD_K_SELECTION_N3_N


主要結果
--------
00_run_settings.csv

01_k_scan_metrics.csv
    各 Method × K：
    - WithinClusterSSE
    - Silhouette
    - DaviesBouldin
    - CalinskiHarabasz
    - cluster size
    - GMM BIC/AIC

02_assignments_all_k.parquet
    每組 OD 在各方法、各 K 的 cluster assignment。

03_cluster_profiles_all_k.csv
    各方法、各 K、各 cluster 的 24h profile。

04_stability_resample_detail.csv
    每一次 80% resampling 的 ARI。

05_stability_by_k.csv
    每個 Method × K 的 stability summary。

06_k_selection_full_table.csv
    把 metrics + stability 合併成一張總表。

07_k_recommendation_summary.csv
    每個方法分別列：
    - ElbowK
    - BestSilhouetteK
    - BestDaviesBouldinK
    - BestCalinskiHarabaszK
    - BestStabilityK
    - GMM BestBIC / BestAIC

08_cluster_size_detail.csv
    看 K 增加之後有沒有出現極小 cluster。


圖
--
plots\01_elbow_within_cluster_sse.png
plots\02_silhouette_by_k.png
plots\03_davies_bouldin_by_k.png
plots\04_calinski_harabasz_by_k.png
plots\05_stability_ari_by_k.png
plots\06_smallest_cluster_od_share.png
plots\07_smallest_cluster_trip_share.png
plots\08_runtime_by_k.png
plots\09_gmm_bic_aic.png


K 不要只靠 Elbow
-----------------
正式判斷請一起看：

1. Elbow
   找 SSE 開始趨平的位置。

2. Silhouette
   越高越好。

3. Davies-Bouldin
   越低越好。

4. Calinski-Harabasz
   越高越好。

5. Resampling stability
   每次抽 80% OD 重新分群。
   Median ARI 越接近 1 越穩。

6. Cluster size
   如果 K 增加後只是一直切出非常小的群，
   要判斷它是真實特殊需求，還是 over-clustering。

7. GMM
   額外看 BIC / AIC，越低越好。


重要
----
不同方法的 WithinClusterSSE 數值不可直接比大小。

應該看「同一方法內 K 增加時的曲線轉折」。

GMM 的主要 model-selection 指標是 BIC / AIC，
不是 SSE。


Stability 的意思
----------------
例如完整資料 K=3。

每次抽 80% OD，
在抽出的 OD 上重新做 K=3，
然後拿：

    完整資料對這些 OD 的 cluster assignment

和

    抽樣後重新得到的 assignment

計算 Adjusted Rand Index。

如果 Median ARI：
    0.95
代表 K=3 非常穩。

如果：
    0.50
代表 K=3 對樣本組成很敏感。


下一步
------
例如結果發現：

KMeans:
    elbow = 3
    silhouette = 3
    DB = 3

CosineKMeans:
    silhouette = 3

Agglomerative:
    silhouette = 2 or 3

GMM:
    BIC = 3 or 4

而 stability 顯示：
    K=2,3 高
    K>=4 明顯下降

那麼下一輪只需要跑：

    KMedoids K=2,3,4
    KShape   K=2,3,4
    DTW      K=2,3,4

而不需要把慢方法從 2 跑到 10。
