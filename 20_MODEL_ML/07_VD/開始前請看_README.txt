VD 長缺漏 11 方法驗證｜一鍵啟動包（2026-10-29）

1. 解壓縮 ZIP 到例如 D:\VD_長缺漏驗證\，不可直接在 ZIP 內執行。
2. 安裝 Python 3（勾選 PATH，Windows 可使用 py -3）。
3. 雙擊「安裝並執行_長缺漏驗證.bat」。它會自動安裝 Python 套件、首次從你的 GitHub 專案抓取四支分析模組，再執行 6 / 24 / 72 / 151.5 小時的測試。
4. 結果出現在 outputs\vd_long_gap_validation\long_gap_all_methods.csv。

ZIP 已包含：
* VD_N3_N_2024_2025_LITE.parquet（五站 VD 原始資料；約 5.4 MB）
* run_vd_long_gap_validation.py（長缺漏測試流程）
* prepare_vd_sources.py（自動抓取 GitHub 分支的四支既有模型模組）
* Windows 啟動與檢查用 .bat 檔
* 既有完整操作說明

重要：因為研究模型程式以 GitHub 為單一來源，四支既有模組不是事先封裝在 ZIP 內；第一次執行需要連網，抓取後會保留本機檔案，以後不需要再次下載。這次並沒有先完成長缺漏模型評分，不應把程式包視為已驗證的長缺漏成績。

若只想測 6 和 24 小時：在解壓縮資料夾開啟 PowerShell，執行：
py -3 prepare_vd_sources.py
py -3 run_vd_long_gap_validation.py VD_N3_N_2024_2025_LITE.parquet --only-hours 6 24

GitHub 來源：
https://github.com/aishanchung/Statistical-Practice/tree/%E8%B3%87%E6%96%99%E6%AC%84%E4%BD%8D%E7%9B%B8%E9%97%9C%E6%96%87%E4%BB%B6/analysis

驗證原始資料 SHA256 與來源方法版本，請由研究團隊另存實際下載版本；程式使用 2024 年訓練、2025 年人工遮罩、同時遮罩 1 至 5 支 VD。
