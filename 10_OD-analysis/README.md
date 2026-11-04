# N3 北向 OD 需求分群（Python）

這套程式會讀取 `V2_LITE` 拆分後的 `part_*.parquet`，以**一組 `(GantryO, GantryD)` 為一個分群個體**，不會把 4,500 萬筆旅次直接送進分群演算法。DuckDB 只掃描所需欄位，pandas 只接收 OD 彙總資料，因此不必一次把所有原始旅次載入記憶體。

已用 Google Drive 目錄中兩個真實檔案測試：`part_0000.parquet`（2024，500,000 筆）與 `part_0090.parquet`（2025，150,592 筆）。**這只是功能測試，並不是完整 2024–2025 年的分群結果。** 正式執行要在本機準備全部 91 個 `part_0000.parquet`～`part_0090.parquet`。

## 0. 已確定的研究決策（2026-11-04）

- **正式壅塞標籤 Y 採四站 VD：** K69.510（4.17%）、K65.990（46.00%）、K63.990（36.00%）、K61.670（13.83%）；**K61.000 不納入主 Y**。四站加權壅塞比例為 \\(C_t^{(w)}=\\sum_jw_j\\mathbf{1}(V_{j,t}<40)\\)，正式標籤為 Y=0 未壅塞、Y=1 局部壅塞、Y=2 大範圍壅塞。大範圍門檻 \\(c\\) **尚待正式確認**。
- **OD 資料已經過路段篩選**，研究前提為所有 OD 旅次完整經過龍潭—大溪北向研究路段；不再做「只經過部分路段」的 OD 篩選。
- **Silhouette 最佳 K ≠ 壅塞預測最佳 K。** 下一階段固定 `time_vehicle` 特徵，比較 K-means、K-medoids、GMM、Sparse K-means 各 `K=2,3,4,5`（共 16 組）；用 2024 年內向前時間驗證決定方法/K，2025 年留作最後測試，與「無 OD」及「總 OD 車流」基準比較。
- 四站缺值處理、2025 防洩漏、選 K 判準與待完成的五分鐘候選模型輸出，請見 **[四站 Y 與預測最佳 K 研究規範](FOUR_VD_Y_AND_PREDICTIVE_K_PROTOCOL.md)**。此文件是後續變更主設定時的單一參考。

## 1. 安裝

建議 Python 3.10 或更新版本。Windows PowerShell：

```powershell
cd D:\統計實務\M06B_project
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 10_OD-analysis\requirements.txt
```

若已經有可用的 `.venv`，不用重建，直接安裝依賴即可。

## 2. 先執行快速測試

將 `--input` 改成你儲存 91 個 Parquet 分檔的資料夾；以下路徑僅供示範：

```powershell
.\.venv\Scripts\python.exe 10_OD-analysis\od_clustering.py `
  --input "D:\統計實務\Data\V2_LITE_parts" `
  --output "D:\統計實務\M06B_project\outputs\od_clustering_quick" `
  --min-trips 100 --k-min 2 --k-max 5 `
  --methods kmeans gmm sparse `
  --feature-sets time time_vehicle time_vehicle_distance `
  --restarts 2 --threads 4 --memory-limit 4GB
```

確認快速測試可以跑完後，再跑正式比較：

```powershell
.\.venv\Scripts\python.exe 10_OD-analysis\od_clustering.py `
  --input "D:\統計實務\Data\V2_LITE_parts" `
  --output "D:\統計實務\M06B_project\outputs\od_clustering_full" `
  --min-trips 100 --k-min 2 --k-max 10 `
  --methods kmeans kmedoids gmm sparse `
  --feature-sets time time_vehicle time_vehicle_distance time_vehicle_distance_volume `
  --restarts 3 --threads 4 --memory-limit 4GB `
  --five-minute both
```

`--five-minute both` 會多掃描原始 Parquet，產生 5 分鐘群組需求。全量執行時間受硬碟速度、CPU、記憶體限制而異。第一次建議先不加這個參數，核對分群後再跑。

程式會依 `BottleneckTime` 篩選訓練年和測試年；**不以檔案序號判定年份**，因為單一分檔內日期可能不連續。

## 3. 設定與特徵

| 設定 | 預設值 | 說明 |
|---|---|---|
| `--train-year` / `--test-year` | 2024 / 2025 | 不使用 2025 年資料估計 2024 年群組 |
| `--min-trips` | 100 | 訓練年 OD 總筆數門檻 |
| `--thresholds` | 30、100、300、500、1000 | 門檻對次年需求涵蓋率的敏感度比較 |
| `--k-min` / `--k-max` | 2 / 10 | 群數候選區間 |
| `--smooth` | 10 | 少量旅次時的日內／車種占比平滑強度 |
| `--sparse-l1` | 2.5 | Sparse K-means 的 L1 權重限制 |
| `--exclude-date` | 無 | 已確認的來源異常日可明確排除，例如 `--exclude-date 2024-01-16`；預設不任意刪除異常日期 |
| `--gantry-map` | 無 | 可選 CSV，欄位為 `GantryID,latitude,longitude`，才會啟用空間特徵組合 |

資料欄位已依實檔核對：`GantryO`、`GantryD`、`BottleneckTime`、`DetectionTimeD`、`DayType`、`TripLength`、`VehicleType`。車種代碼預設 `31,32,41,42,5`；其他代碼歸為 `OTHER`。可用 `--vehicle-types` 調整。

每個 OD 產生的特徵：

- **時間：** 平日／週末各 6 個四小時區段的條件占比，以全體需求型態進行平滑。
- **日期：** 週末與平日平均每日需求比（log），有連假觀測時另建立連假需求比。
- **車種：** 各車種占比，使用全體車種比例作為平滑先驗。
- **距離：** `TripLength` 近似四分位數、中位數與 IQR 的 log 轉換。
- **需求規模：** 平均每日旅次量的 `log1p` 轉換；有單獨不含規模的特徵組合可比較。
- **空間：** 只有提供門架地理對照表時，加入起終點經緯度。不把門架代碼當成數值距離。

特徵先裁切極端 1%／99% 值，再做標準化，各特徵區塊按欄位數開平方根調整權重，避免時間欄位僅因數量較多就支配分群。

比較的方法為 K-means、K-medoids（交替式 medoid 更新，非 PAM swap 版）、GMM（對角共變異數）、Sparse K-means（交替更新群組與稀疏特徵權重）。目前全為轉換後數值特徵，因此 K-medoids 使用歐氏距離；若後續引入原生類別特徵，再另行比較 Gower 距離。

## 4. 輸出檔案

| 檔名 | 用途 |
|---|---|
| `od_yearly_counts.csv` | 每組 OD 在訓練／測試年的旅次數 |
| `threshold_coverage.csv` / `.png` | 30、100、300 等門檻與測試年需求涵蓋率 |
| `od_aggregates_train.csv` | 訓練年原始彙總計數 |
| `od_features_train.csv` | 訓練年可解釋特徵表 |
| `method_scores.csv` / `method_comparison.png` | 不同方法、特徵組合、群數的 Silhouette、重複實驗 ARI、GMM BIC |
| `sparse_feature_weights_all.csv` | 各組 Sparse K-means 的特徵權重，包含零權重 |
| `od_cluster_labels.csv` | 探索性候選模型對每組 OD 的群組標籤 |
| `cluster_profiles.csv` / `.png` | 每群 OD 數、旅次數與原始特徵的平均值、群組熱圖 |
| `summary.json` / `run_config.json` | 候選模型與參數紀錄 |
| `five_min_bottleneck.csv` | 可選，依 `BottleneckTime` 的**回溯型**五分鐘群組需求 |
| `five_min_completed.csv` | 可選，依 `DetectionTimeD` 的完成旅次五分鐘群組需求 |

程式會以平均 Silhouette 加上重複初始化穩定性 ARI 的探索性分數，並優先要求最小群組至少占 1% OD，選出一個**待解讀候選模型**。這不代表已經找到對未來壅塞預測最好的方法。仍需檢視群組特性，並在後續 M0–M2 預測實驗中比較增額價值。GMM BIC 僅用於 GMM 內部的模型比較。

## 5. 重要的預測限制

- `five_min_bottleneck.csv` 使用事後完整 OD 資訊，**只適合關聯／回溯分析**。若在時間 (t) 預測 (t+15) 分鐘的壅塞，不能使用尚未完成、尚不知道 `GantryD` 的旅次。
- `five_min_completed.csv` 使用 `DetectionTimeD`，是較保守的時間代理，但仍需確認資料實際到達模型的延遲。
- OD 分群本身不使用 VD 速度與降雨；這些應留給後續壅塞模型，避免結果解釋混淆。
- 沒有被 2024 門檻選入的 OD，以及 2025 新出現的 OD，會在五分鐘彙總時歸為 `OTHER`。不要把 `OTHER` 當成 HDBSCAN 的「離群值」。
- 2025 年目前驗證的是**需求涵蓋率與群組需求總量**，還不是完整的 2025 型態穩定性或壅塞預測績效。
- 未儲存個別車牌／匿名車輛 ID 或原始路徑文字到程式輸出及 Git。

## 6. 基本檢查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 10_OD-analysis\tests -v
```

若 `threshold_coverage.csv` 的 100 筆門檻與先前網站 EDA 的約 3,103 組／2025 年 97.51% 不同，先檢查：91 個分檔是否齊全、是否為同一個北向版本、2024／2025 的時間篩選與 QC 日期設定是否一致。不要直接把差異當成演算法錯誤。
