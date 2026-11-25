# 二元壅塞 Pilot 執行與續跑

既有 `.venv`、月需求 NPQ 快取、`grouping_pilot_v1`、`speed_occupancy_eda_v1` 與 `workpackages_0ABCDE_v1` 是必要輸入；公開 repository 不包含私有快取。程式遇到缺檔或來源改變會停止，沒有隨機資料替代。原始三分類輸出不動，也不掃原始 M06B。

從 repository 根目錄執行：

```powershell
Set-Location 'D:\統計實務\Statistical-Practice'
# Smoke：F1、ALL；再執行同一命令即可驗證中斷續跑。
.\experiments\congestion_2026\run_binary_congestion_pilot.ps1 -Smoke
# 正式執行或接續現有結果，自動辨識既有 manifest 及來源 checkpoint。
.\experiments\congestion_2026\run_binary_congestion_pilot.ps1
# 若要保留舊結果另跑一版，指定不存在的新目錄。
.\experiments\congestion_2026\run_binary_congestion_pilot.ps1 -OutputDir '.\experiments\congestion_2026\outputs\binary_congestion_pilot_new'
# 完成後僅重建驗收、中文報告與ZIP，不重新訓練。
.\.venv\Scripts\python.exe -m experiments.congestion_2026.finalize_binary_congestion_pilot
```

本次正式目錄是 `outputs/binary_congestion_pilot`。本機 `binary_congestion_pilot_prevalidation` 保留初版訓練完成的 checkpoint 與當時 source_script.py；最終程式核對相同輸入 SHA256、舊模型重現紀錄及逐格鍵後繼承60個已訓練二元模型。事件時間精度與 Bootstrap 抽樣日分母修正後，重新計算全部評估。初版及驗收前版本不當成研究結果。最終程式執行全量原有模型稽核在先，二元訓練／載入在後。中斷時已完成 cell 以原子檔案保存，重跑會跳過已完成模型；完整評估表在評估結束後寫出。

若直接使用 Python 接續這次正式目錄：

```powershell
.\.venv\Scripts\python.exe -u -m experiments.congestion_2026.run_binary_congestion_pilot --resume --reuse-models-from experiments/congestion_2026/outputs/binary_congestion_pilot_prevalidation
.\.venv\Scripts\python.exe -m experiments.congestion_2026.finalize_binary_congestion_pilot
```

`--reuse-models-from` 只供已核對的相同輸入模型遷移，必須保留來源 manifest、source_script.py、audit、cells、models；不允許用不同資料冒充舊模型。一般新目錄不需要此參數。任何模型參數、來源或主程式變動請使用新的輸出目錄，不能把不相容版本混進原 checkpoint。

指標內容：正負類各別Precision/Recall/F1及support、正類AP、PR曲線、混淆矩陣、漏報、誤報、每日誤報格數、原Y1/Y2漏報率、五種時段／日型。Z0主分析及Z1 strict/available均另列共同樣本。B01 threshold=-1 代表原三分類硬決策；其他threshold為分數切點。2024 comparison是折別算術平均，n為平均折樣本數，n_total另列總數；pooled OOF是另一種彙總，不可混用。

門檻固定0.2至0.8、間距0.05；precision>=0.80的門檻只由2024 pooled OOF作開發探索並於2025前凍結。未執行折內歷史門檻選擇，所以選後2024績效不稱無偏驗證。2025一直是探索性回測。B04為加權分數，未經機率校準。配對區間為條件於已訓練模型的探索性日期區塊Bootstrap，未含重訓或選模不確定性。

`finalize_binary_congestion_pilot`會再驗16項交付條件、重算補充的B03對B02及權重對B03配對區間、產出中文findings與分享ZIP。逐格預測、PR座標、模型、完整manifest與事件明細僅本機。ZIP只含彙總、圖表與Python程式；PowerShell入口在repository內。公開報告時使用：

```powershell
.\.venv\Scripts\python.exe -m experiments.congestion_2026.finalize_binary_congestion_pilot --publish-dir reports/2026-11-25/binary_congestion_pilot
```

請先檢查彙總後再自行提交所需檔案，不要 `git add` 私有來源或完整 outputs。本次使用者已授權更新GitHub研究進度。
