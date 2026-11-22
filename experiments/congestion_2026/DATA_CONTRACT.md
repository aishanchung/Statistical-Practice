# 資料介面與重現條件

本專案不附原始研究資料。clone 後可閱讀程式、方法、彙總成果與圖表；要重跑真實訓練，須提供下列本機來源。所有路徑由 config 或環境變數指定，不能用隨機資料代替缺項。

## 主來源

| config key | 相對 data_root 的預設位置 | 核心時間與角色 |
|---|---|---|
| v3 | V3_LITE/V3_LITE.parquet | DetectionTimeO 出發、BottleneckTime 通過，抵達日期母體 |
| m06b | M06B_HOLIDAYS_2024_2025/M06B_WEEKEND_LONGHOLIDAY_2024_2025.parquet | DetectionTimeO、DetectionTimeD，出發日期母體 |
| vd | VD_N3_N_LITE/VD_N3_N_2024_2025_LITE.parquet | TargetTime、VDID、Speed、Occupancy、VDType |
| labels | V3_LITE/V3_Y_5MIN.parquet | Time5、DayType、Y、n_valid_vd、C_weighted |
| calendar | M06B_HOLIDAYS_2024_2025/_logs/holiday_calendar.csv | SourceDate（YYYYMMDD）、互斥 DayType |

M06B 使用 GantryO、VehicleType、TripInformation；VehicleID、出發／到達時間與門架組合用於既有旅次稽核，不作預測特徵。詳細實際 schema、日期範圍、筆數及版本指紋見公開 [input_manifest_portable.csv](../../reports/2026-11-22/tables/input_manifest_portable.csv)。CSV schema 儲存格可含換行，讀取應使用標準 CSV parser，不以文字行數當資料列數。

原始與月彙總的統計口徑須先核對；來源名稱存在不代表已完整觀測，不能直接使用檔名推定母體。原始 CSV 候選的 17,544 檔數只代表 metadata 盤點，不代表所有列已稽核可用。

## 保留相容性的 legacy_root 依賴

| 相對 legacy_root 路徑 | 用途與必要結構 |
|---|---|
| pilot_2024_2025/features.py | aggregate_counts、probability、read_counts、history_before、estimate_history 原函式 |
| pilot_2024_2025/model.py | make_model、predict 原函式 |
| pilot_2024_2025/results/monthly/*.parquet | 出發日 depart_day、o、kind、slot、lag、n；lag>=0 表示通過且有可用延遲 |
| pilot_2024_2025/results/cohort_completion.csv | depart_day、latest_end，用於完成時間＋1日的歷史可用性檢查 |
| o_features_grouping_20261111_v2/features.py | build_table：同出發母體 O 特徵、比值與支持度 |
| o_features_grouping_20261111_v2/groups.py | geometry：訓練期組成與標量轉換 |
| o_features_grouping_20261111_v2/results/checkpoint_manifest.json | 每月 month、path 及來源驗證資訊；path 須指向當前機器實際有效 checkpoint |
| 上述 manifest 所列詳細月 Parquet | level=0 的 o/kind/hour/vehicle/n/n_b；level=1 的 travel_seconds/n_b；data_window 標示用途 |
| horizon_15_30_20261111_v1/results/features_h15.csv、features_h30.csv | origin、target、day、slot、kind、y、n_valid_vd，以及既有 cal_/vd_ 欄位 |

此外 audit 會盤點舊九組評估、逐筆機率與來源程式。`local_source_root` 指向舊 M06B_project 供版本比較，不應改成新的輸出位置。搬移專案時應另建新的 config、run_id，並更新舊 manifest 的路徑副本，保留原 manifest 不動。

reference 中逐字摘錄的函式附來源 SHA256，供方法審查。它們沒有連同私有月資料，也沒有改成取代舊流程的全新模型；不可宣稱公開 repository 已自帶完整真實資料重建流程。

## 逐筆預測介面（僅本機）

`predictions.parquet` 每列代表一個模型在一個預測起點的輸出，主鍵為 `(fold, horizon, algorithm, configuration, prediction_time)`：

| 欄位 | 定義 |
|---|---|
| prediction_time | 發出預測時間 t |
| target_time | t+h，不能拿來錯當警報發出時間 |
| horizon | 15 或 30，分鐘 |
| day、kind | 目標研究日期、日型 |
| y | 事後目標真值，0/1/2 |
| p0、p1、p2 | 原始模型三類機率，有限且合計1 |
| predicted | argmax 的三分類，不是成本決策 |
| fold、algorithm、configuration | 實驗識別與模型配置 |
| Y_version | 正式標籤版本 |
| history_cutoff | 該列歷史 N/P/Q 的估計截止 |
| vd_feature_available_at | VD 可用時間假設，須<=prediction_time |
| evaluation_type | 2024_temporal_validation 或 2025_exploratory |

`decision_predictions.parquet` 額外保留 method、decision、cost_scenario；R以−1表示。不能用 decision 替換原機率，也不能用 R 當第四類 y。

## 共同比較表介面

`metrics_by_fold.csv` 按 fold/horizon/algorithm/configuration/segment 保存n、各類指標、訓練秒數與評估性質；segment為ALL、WEEKEND或LONGHOLIDAY。`metrics_by_class.csv` 展開class=0/1/2及support。`confusion.csv` 的actual為列、predicted為欄。逐折平均與合併OOF指標請分開讀取。

`kmeans_cluster_assignments.csv` 包含o、group、fold、grouping、history_before、complete_months、N_all、N_B。同一G編號僅在該折／該K有意義，不能直接把不同折G00視為同一交通群；跨群比較應比membership或特徵，不按群編號硬接。

`event_detection_metrics.csv` 明列風險集、共同起點、事件數、可評估事件、命中／漏報、clean事件、提前量、已確認誤報／研究日與截尾警報數。事件數及日分母不同時不能只比百分比。

## Dashboard 匯出與安全

先執行研究流程的 export，產生 outputs/<run_id>/dashboard_export；Dashboard只讀小型匯出，不直接載入巨大原始M06B。`dashboard/config.yaml` 指向匯出資料，缺檔會顯示缺項。介面沒有DEMO或隨機機率fallback。

Git忽略原始Parquet、逐筆機率、模型、cache、outputs與本機設定。公開 reports 僅包含研究彙總及指定案例截圖；完整月Q表、逐旅次及本機絕對路徑不公開。

## 自動測試範圍

純函式契約測試可檢查Y與機率等數學規則；涉及真實checkpoint、共同樣本、來源不變、Dashboard頁面者需要本機研究資料／匯出。Playwright截圖另外需要安裝chromium。本輪本機20項pytest通過不代表任一缺資料的乾淨機器可直接重現全部測試。
