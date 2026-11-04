# V3_LITE：週末／連假 OD + 四支 VD，日期與缺值稽核

本研究版本以國道 3 號北向龍潭→大溪為對象，原 V2 原檔不修改。`build_v3_lite_holiday.py` 產生新的 **V3_LITE_HOLIDAY**：

- 日期：2024–2025 年、`DayType ∈ {WEEKEND, LONGHOLIDAY}`；不納入 `WEEKDAY`、`STANDALONEHOLIDAY` 作為主要分析樣本。**保留假日完整 24 小時**供落後項、事件計算；`is_focus_window` 指示 14:00 ≤ 時間 < 20:00 的主要模型時間窗。
- VD：只納入 `K69.510`（0.0417）、`K65.990`（0.4600）、`K63.990`（0.3600）、`K61.670`（0.1383）。`K61.000` 不進入 V3 或 Y。
- VD 原始欄位：`TargetTime,VDID,Speed,Occupancy,VDType`；負速度、缺速度與缺測站列分開標註。四站缺一不可產生正式 `Y`；沒有資料不是 `Y=0`。
- `C_weighted = Σ_j w_j 1(Speed_j < 40)`；四站有效時，`Y=0`（`C=0`）、`Y=1`（`0<C<c`）、`Y=2`（`C>=c`），預設 `c=0.50`。如需比較門檻，重新執行 `--c 0.25`、`--c 0.75`，輸出到不同資料夾。

## A. Windows PowerShell：取得程式

Windows 上沒有 `git` 指令時，直接下載 GitHub 原始碼。**注意 Git 路徑是 `10_OD-analysis`（連字號），你本機程式夾是 `10_OD analysis`（空格）。**

```powershell
cd "D:\統計實務\M06B_project"
New-Item -ItemType Directory -Force ".\10_OD analysis" | Out-Null
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/aishanchung/Statistical-Practice/main/10_OD-analysis/build_v3_lite_holiday.py" `
  -OutFile ".\10_OD analysis\build_v3_lite_holiday.py"
```

使用原本 `.venv`，依賴為 `duckdb pandas numpy`。不需要重新安裝 scikit-learn。

## B. 先快速重算缺值分布（稽核模式）

```powershell
& ".\.venv\Scripts\python.exe" ".\10_OD analysis\build_v3_lite_holiday.py" `
  --od-input "D:\統計實務\Data\V2_LITE_WEATHER\V2_parts" `
  --vd-input "D:\統計實務\M06B_project\07_VD\VD_N3_N_2024_2025_LITE.parquet" `
  --output "D:\統計實務\M06B_project\outputs\V3_LITE_holiday_audit" `
  --audit-only `
  --threads 4 --memory-limit 4GB
```

`--audit-only` **仍會完整掃描 OD 日期並核對所有四站預期五分鐘格**，但不輸出旅次級 V3 與 OD-pair 五分鐘量，省磁碟與時間。

## C. 確認日期後，正式產生 V3

```powershell
& ".\.venv\Scripts\python.exe" ".\10_OD analysis\build_v3_lite_holiday.py" `
  --od-input "D:\統計實務\Data\V2_LITE_WEATHER\V2_parts" `
  --vd-input "D:\統計實務\M06B_project\07_VD\VD_N3_N_2024_2025_LITE.parquet" `
  --output "D:\統計實務\M06B_project\outputs\V3_LITE_holiday_full" `
  --c 0.50 `
  --threads 4 --memory-limit 4GB
```

時間視硬碟及 91 個 OD 分檔容量而定；掃描 4,500 萬多筆 OD 時不會一次載入 pandas。最費時／佔空間的是 `V3_LITE_OD_TRIPS.parquet`。只需要聚合 OD 與四站資料時加 `--skip-trip-level`。

## D. 缺值與原始日期稽核輸出

| 輸出 | 重點 |
|---|---|
| `V3_calendar_all_dates.csv` | 2024–2025 **所有應有日曆日期**；DayType、來源、OD 該日是否沒有旅次、是否納入 V3 |
| `V3_calendar_focus_dates.csv` | 真正納入分析的日期清單，含週末無 OD 旅次的日期 |
| `OD_daytype_counts.csv`、`OD_daytype_conflicts.csv` | 同一日期各個 DayType 的筆數及衝突（不默默當成一致） |
| `OD_dates_needing_calendar_review.csv` | OD 完全沒資料的平日、DayType 衝突日、週末標錯日；**優先人工檢查** |
| `VD_missing_summary.csv` | 依年、WEEKEND / LONGHOLIDAY、全日／14–20；**四站有效率、單／雙／三／四站同時缺失率** |
| `VD_missing_by_date.csv` | 精確到日期的缺值比例與缺一支至缺四支的五分鐘格數 |
| `VD_missing_by_date_station.csv` | 每日**每支 VD 公里位置**的 288／72 預期格、有效／不存在／無效速度／缺值％ |
| `VD_missing_by_station_summary.csv` | 依站、年、DayType、範圍的缺值比例 |
| `VD_missing_by_month_station.csv` | 依月、站、DayType、範圍的缺值比例 |
| `VD_missing_runs_by_station.csv` | **缺測位置＋連續起訖時間＋分鐘數＋原因**，涵蓋完全沒有列的時間 |
| `VD_missing_runs_joint.csv` | **同時缺失哪幾支 VD**、連續多久，完整區分單站、多站、全站 |
| `VD_duplicate_station_time.csv` | 重複測站時間檢核；若有重複，程式會停止以免模糊平均 |
| `V3_focus_date_coverage.csv` | 逐日期：有效 Y 的 5 分鐘格比例、所有四站同缺、OD 零旅次格比例 |
| `V3_LITE_VD_4SITES.parquet` | 四站 × 全部預期五分鐘格；缺值保留 `MissingStatus` |
| `V3_Y_5MIN.parquet` | 四站嚴格的 Y、C、有效站數、14–20 標記、同格 OD 旅次數（僅診斷） |
| `V3_LITE_OD_TRIPS.parquet` | 正式模式：篩選過的原 OD 旅次；移除 K61.000 相關速度／佔有率欄位 |
| `V3_OD_5MIN_BY_PAIR.parquet` | 正式模式：依 BottleneckTime/起訖門架/車種每五分鐘旅次數 |
| `V3_run_config.json` | 參數與空間權重、時間窗、分析日期數 |

注意：VD 預期資料分母為 **每個假日每站 288 個 5 分鐘格**，14:00–20:00 為 **每站 72 格**；不使用「實際存在的資料列數」當分母。`VD_missing_summary.csv` 中 `missing_pct` 分母為四站所有格，`strict_Y_unavailable_pct` 分母為時間格，意義不同。

`ROW_ABSENT` = 應有測站／時間資料列不存在；`INVALID_SPEED` = 列存在但速度負值或空值。兩者目前都保留為無法計算 Y，不做速度補值。`VDType` 原值保留，可後續辨別官方缺值標記。**某個五分鐘格 OD=0 筆不能直接認定為 OD 缺檔**；OD 沒有預期的固定旅次數基準。

## E. 重要缺檔限制及日曆修正

`DayType` 存在於 OD V2，不存在於 VD 原檔。程式先對每個日期統計 OD 的標籤，以最多票為主，平票時參考週末規則；無 OD 的六、日仍納入 WEEKEND。但**某個平日如果 OD 也完全缺檔，它是否原屬 LONGHOLIDAY 無法從 V2 推回來**，在 `OD_dates_needing_calendar_review.csv` 會警告。這時請提供經官方日期核對的 CSV：

```csv
Date,DayType
2024-02-09,LONGHOLIDAY
2024-02-10,LONGHOLIDAY
```

以上僅為格式示例，日期請另行核對原始行政機關行事曆、假日實際調整。執行時加上：

```powershell
  --calendar-csv "D:\統計實務\Data\confirmed_holiday_calendar_2024_2025.csv"
```

同一天如果出現多種 OD `DayType`，原筆數全部保留於報表；CSV 覆寫日曆標籤優先，讓全日分類一致。所有輸出只放在指定 `--output`；**不會刪除或改動 V2 原始檔**。

## F. 預測／研究倫理

1. V3 日期篩選與原本對**全年 OD** 做出的探索性分群是兩個階段。若要針對假日重新分群，需另訓練及比較群組穩定性；不能說全年最佳 K 就是假日預測最佳 K。
2. 正式驗證仍建議 2024 訓練、2025 測試，絕不可利用 2025 的 Y 反推挑選 2024 模型所用的 `c` 或分群規則。
3. `BottleneckTime` 5 分鐘 OD 歸屬通常是**事後回溯**；在真正預測未來 Y 時不可偷用尚未完成旅次的 OD 資訊。
