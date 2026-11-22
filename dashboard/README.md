# 國三北向短期壅塞預警歷史回放

這是 **Historical Replay / Offline Evaluation**。不是即時監控，沒有即時VD或即時O串流。所有機率來自已完成的真實模型，不提供混用的DEMO模式。

完成共同實驗的`--stage export`後，在repository根目錄執行 `python -m streamlit run dashboard/app.py`，或雙擊`dashboard/start_dashboard.bat`。

設定在`dashboard/config.yaml`，data_dir相對設定檔所在目錄，可透過`CONGESTION_DASHBOARD_CONFIG`指定另一設定。缺少匯出資料時顯示缺項，不產生隨機機率。

首頁可選歷史日期、五分鐘起點、15/30分鐘、模型與argmax／Balanced／Rejection。最近可觀測C及Y採t−10分鐘資料，介面標示來源及假設發布時間。這與事後精確C(t)不同，不偷用預測時尚未發布的觀測。

Traffic History只顯示當時可取得的VD。O Historical Demand可選P或P+Q，後者是替代需求建構，不能相加。Prediction Comparison只顯示已完成配置，2024與2025分開。Event Replay可選真實事件、拖動五分鐘時點，成功與漏報案例保存在examples表。

「顯示事後答案」預設關閉。開啟後的真實未來Y明示事後驗證，永不回填原始機率。R代表待複核，不能算成功預警。

每張主要圖可下載CSV，PNG以相同Plotly trace數值重新輸出科學圖，不啟動瀏覽器子程序。自動測試與畫面截圖使用Playwright，`python -m playwright install chromium`。海報截圖保留在本機outputs中的dashboard_screenshots。

繁體中文字型以Microsoft JhengHei顯示。部署與公開前請保留資料來源、Y版本、訓練截止日與探索性標記，不上傳敏感旅次或大型逐筆預測。
