# 統計實務工作包0–E

[完整研究定義與公式](METHODS_zh-TW.md)｜[資料介面與重現條件](DATA_CONTRACT.md)｜[已完成結果與公開圖表](../../reports/2026-11-22/README.md)。僅clone不含原始資料或legacy checkpoint，請先核對必要依賴。

沿用本機M1B_O_PILOT的數學定義與函式，主要預測為t+30分鐘，15分鐘作對照。正式Y使用四站有效權重重新正規化，速度<40，c=.50。2025為探索性回測，不是新獨立測試。未使用完整OD作提前預測特徵。

## Windows執行

從GitHub專案根目錄執行。

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/congestion_2026/requirements.txt -r dashboard/requirements.txt
.\.venv\Scripts\python.exe -m experiments.congestion_2026.run --stage audit
.\.venv\Scripts\python.exe -m experiments.congestion_2026.run --stage features
.\.venv\Scripts\python.exe -m experiments.congestion_2026.run --stage train
.\.venv\Scripts\python.exe -m experiments.congestion_2026.run --stage analyze
.\.venv\Scripts\python.exe -m experiments.congestion_2026.run --stage export
.\.venv\Scripts\python.exe -m pytest experiments/congestion_2026/tests dashboard/tests -q
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

也可使用 `--stage all`。已完成profile與每個模型cell均可續跑。設定更動請使用新run_id，程式拒絕把不同設定混入既有checkpoint。程式不會重新抓取原始M06B，不會改寫舊成果，不會自動push GitHub。

## 資料路徑

`config/default.json`路徑相對repository根目錄，不含個人電腦絕對路徑。預設同層`Data`、`M1B_O_PILOT`及`M06B_project`。可複製config傳入`--config`，或用環境變數`CONGESTION_DATA_ROOT`、`CONGESTION_LEGACY_ROOT`、`CONGESTION_LOCAL_SOURCE_ROOT`、`CONGESTION_OUTPUT_ROOT`。

原始Parquet、逐筆預測、模型、快取與dashboard匯出資料都在.gitignore。`outputs/<run_id>`本機保留所有CSV、PNG/SVG、run_config、input_manifest、experiment_registry、predictions.parquet及run_summary.md。只有審閱後的小型彙總CSV及研究圖才適合另外複製到公開reports。不要直接git add大型outputs。

## 重用範圍

- `legacy_functions`從已核對原始Python AST逐字載入指定函式，避免執行舊模組的OUT寫入動作。
- 原P為零延遲歷史預期通過量，P+Q替換為延遲卷積抵達量。兩者不是疊加特徵。
- O分群重用既有log規模、arcsin√p、Hellinger時段／車種、log延遲幾何。只比較無r，早期連假不足不借用未來資料。
- B0已具有P與Q pooling。B1是有限增強N平滑與P強度，B2另增加daytype Q，不宣稱從無平滑變成首次平滑。
- 模型重用原Logistic與LightGBM factory，固定小型參數。XGBoost舊結果可重算，這一輪不擴大其新實驗。

## 時間與可用性

2024Q1暖機，三個expanding驗證窗為6–8、9–10、11–12月。模型只使用4月後訓練格，最後訓練target+30分鐘embargo早於驗證prediction_time。這比舊版未明列embargo的單一Q4驗證更嚴格，所以新舊數字分開。

N/P/Q訓練特徵逐月只回看已完成旅次加1日緩衝。驗證／2025Pattern在訓練截止凍結。群別與標準化是fold訓練期擬合的transformer，不能用驗證統計；對原始訓練樣本採共同fold座標以保持模型欄位一致，不宣稱群別於每個歷史訓練日已固定。詳細月cache無逐日欄，故群特徵僅納入「整月所有旅次皆成熟」的月份，可能保守排除最新月份，實際月份保存在cluster assignments。

VD可用延遲採10分鐘保守代理，輸入t−10/t−25，遮罩RECOVERED_P1。尚未取得真實營運發布時間證據，不能宣稱已驗證實際即時部署。

新O使用預先固定OTHER，因沒有訓練出發需求，預期需求為0，不用驗證資料估計。這會漏掉新生O需求，需報限制。分群前逐O算P/Q，分群後只加總需求並檢查守恆。

## 事件與決策

所有方法使用同一正式Y事件清單，至少3格持續、前格可觀測非Y2。Clean需前30分完整觀測無Y2。警報只用原始機率、固定成本或2024驗證門檻。R不算正確分類或成功預警，複核另列。R成本沒有定義，因此不比較包含R的全人口成本優越性。

新事件風險集使用最新可取得VD狀態，與舊current_state_known評估不同。不同horizon事件窗固定future30，共同起點敏感度需另列，不能用不同樣本指標直接宣稱完全公平。

## 檔案結構

`config/` 設定、`data/`唯讀稽核／介面、`features/`逐O與聚合、`models/`重用factory及checkpoint、`evaluation/`分類／事件／需求診斷、`tests/`可用性契約、`outputs/`忽略的本機結果。Dashboard獨立於研究計算，只讀小型標準匯出。

## 驗收與續跑保護

完成版本保存 `sealed_provenance.json`，記錄程式、小型來源 SHA256、大型來源大小與修改時間、套件版本。續跑會先核對來源；程式、資料或設定變更時須使用新 `run_id`，不能混用已封存 checkpoint。大型檔採 metadata fingerprint，並非全檔內容雜湊。

事件不確定性以連續研究日期區塊重抽樣，同日及相鄰日 onset 不拆成獨立事件。未完整觀測警報後事件窗且沒有已知命中的 episode 保留為截尾，不能視為誤報或成功。`event_bootstrap.csv` 同時提供事件 Recall 與已確認誤報／日的區間。

最後驗收見本機輸出 `acceptance_report.md`，研究結論見 `research_conclusions_zh.md`。現有原始逐日 CSV 是非假日母體候選，尚未完成其完整性及發布時間稽核，不能直接當作已可用的 N/P/Q 輸入。

## 2026-11-25 二元壅塞 Pilot

[實際成果與圖表](../../reports/2026-11-25/binary_congestion_pilot/findings_zh-TW.md)｜[獨立執行與續跑說明](BINARY_PILOT_zh-TW.md)。使用既有快取、LightGBM、PQ、h30、五種代表性O分組；三分類保留，Z0為主、折內歷史Q95標籤為敏感度。門檻網格預先固定，2024 OOF選門檻僅開發探索；2025不參與選模或門檻調整。

## 2026-11-25 三分類器二元補充

[完整研究結果、六個問題及圖表](../../reports/2026-11-25/binary_classifier_comparison/findings_zh-TW.md)｜[執行、續跑、參數與私有依賴](BINARY_CLASSIFIER_COMPARISON_zh-TW.md)。固定原Pilot Z0、B0_PQ、h30、五組O配置與時間切分，新增Logistic/XGBoost，LightGBM完全重用。B01/B02共用三分類模型，不以Macro-F1選模，2025仍為探索性回測；公開報告不含逐格預測或原始旅次。
