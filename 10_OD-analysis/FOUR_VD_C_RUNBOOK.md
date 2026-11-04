# 四站 VD：先比較 c=0.25／0.50／0.75（可直接執行）

本頁執行 [四站 Y 與選 K 正式規範](FOUR_VD_Y_AND_PREDICTIVE_K_PROTOCOL.md) 的**第一階段**。此程式**只比較四站壅塞標籤的門檻 c**，尚未跑 16 組 OD 分群預測，不會宣稱已選出最佳預測 K。

## Windows PowerShell

請先確認本機資料路徑（示例為先前的 VD 檔案位置），然後在專案根目錄執行：

```powershell
cd "D:\統計實務\M06B_project"

# 1. 先列出 VD 資料的實際欄名與測站代碼；不會生成結果檔。
& ".\.venv\Scripts\python.exe" ".\10_OD-analysis\four_vd_c_sensitivity.py" `
  --input "D:\統計實務\M06B_project\07_VD\VD_N3_N_2024_2025_LITE.parquet" `
  --inspect

# 2. 如果欄位可自動辨識，再跑 2024 的主要比較（並產出 2025 描述性結果）。
& ".\.venv\Scripts\python.exe" ".\10_OD-analysis\four_vd_c_sensitivity.py" `
  --input "D:\統計實務\M06B_project\07_VD\VD_N3_N_2024_2025_LITE.parquet" `
  --output "D:\統計實務\M06B_project\outputs\four_vd_c_sensitivity" `
  --train-year 2024 `
  --c-values 0.25 0.50 0.75 `
  --threads 4 --memory-limit 4GB
```

若欄名無法自動辨識，依 `--inspect` 的輸出加上 `--time-col`、`--site-col`、`--speed-col`（長表），或 `--time-col` 和 `--wide-speeds COL69 COL65 COL63 COL61`（寬表；**依指定順序**）。若 VD 站碼只有不含里程的 ID，再提供 `--station-map` CSV：

```csv
raw_id,kilometer
實際站碼A,69.510
實際站碼B,65.990
實際站碼C,63.990
實際站碼D,61.670
```

若同一 VD、同一時段有多行、各行車流量不同，可加 `--flow-col Volume`（將 `Volume` 換成實際欄名）做流量加權速度；預設對重複列取平均，請先確認資料是已聚合到站點或仍分車道。

**缺失代碼檢查：** 請先確認 VD 原始資料中的 `VD_ABSEN`／`OFFICIAL_NA`／資料缺檔是否已轉成速度 NA。程式預設負速度為無效，但 **0 km/h 保留為可能的真實速度**；若來源文件明確指出某數字（如 `999`）代表缺失，可加 `--missing-speed-values 999`，不要未經核實把 0 當缺值。

本程式會將所有時間向下取到五分鐘，假設無時區標記的時間是臺灣當地時間。若有時區標記則會轉換為臺灣時間。

## 產生的結果

| 輸出 | 說明 |
| --- | --- |
| `c_yearly_comparison.csv` | 2024／2025 各 c 的有效五分鐘筆數、缺失筆數、Y0／Y1／Y2 次數和比例、Y2 連續事件數和長度 |
| `c_monthly_comparison.csv` | 各月三類 Y 的比例；檢查季節性和少數類別問題 |
| `four_vd_y_timeseries.csv` | 每五分鐘四站速度、`C_weighted`、`Y_c025`／`Y_c050`／`Y_c075`（缺失為 NA） |
| `spatial_pattern_counts.csv` | 四站壅塞與否的 16 種空間型態之出現次數，便於解釋門檻意義 |
| `c_comparison_train.png` | 2024 年各 c 的三類比例堆疊圖 |
| `run_config.json` | 站點、四站權重、c、資料路徑及缺失處理紀錄 |

### 選 c 的研究準則

1. 先以 **2024 年**比較三類是否有足夠樣本、`Y=2` 連續事件數和中位長度、月分間是否過度不均；2025 年只能在定案後描述／驗證，**不要用 2025 來選 c**。
2. `c=0.25` 容許 **只有 K65.990（權重 0.46）或 K63.990（權重 0.36）壅塞就判定 Y=2**，與「大範圍」的空間概念可能衝突；即使此門檻使樣本比較平衡也不能直接採用。
3. `c=0.50` 表示約半數加權代表長度壅塞；`c=0.75` 比較嚴格，可能使 Y=2 變少。**優先評估 0.5 的可解釋性**，再用 0.25、0.75 做敏感度比較。
4. **缺值不當 0。正式 Y 依有效 VD 重新正規化權重**，至少有一站有效就計算；只有四站全缺才設 `Y=NA`。後續 KNN／線性插值須獨立標明「短補」或「全補」版本，再確認結論是否改變。
5. 若缺失站比例很高，不能光看有效資料的 Y 比例就推論整年度壅塞率；同時檢查 `Y_valid_bins`、`complete_four_vd_bins`、`weight_coverage` 與 `missing_bins`，因為單站有效的 Y 空間代表性有限。

本工具使用既有 OD requirements 內的 `duckdb, numpy, pandas, matplotlib`，不必新增依賴。內附測試：

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s "10_OD-analysis\tests" -p "test_four_vd_c_sensitivity.py" -v
```

拿到 `c_yearly_comparison.csv` 和 `spatial_pattern_counts.csv` 後，再決定是否正式選 `c=0.5`，並啟動 4 種分群方法 × K=2..5 的完整預測比較。該比較需要以同樣的 Y 計算規則、同樣的 2024 時序驗證折及 2025 留出測試，詳見研究規範。
