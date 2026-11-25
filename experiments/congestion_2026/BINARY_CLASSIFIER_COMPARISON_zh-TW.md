# 二元分類器比較 執行與資料介面

本輪接續commit `12ef9da`，固定Z0、B0_PQ、h30、ALL/TOP20/K05/GMM05/TS10，2024三折及2025探索回測。分類器為Logistic、XGBoost及不重訓的既有LightGBM。B01/B02共用一個三分類模型，B03直接二元，B04正類權重2/4、負類1。160個訓練checkpoint包含12個舊Logistic三分類機率重現與148個新模型；另讀LightGBM原有100份預測，共300個策略／折別／分組預測配置。

## Windows執行與中斷續跑

```powershell
Set-Location 'D:\統計實務\Statistical-Practice'
# 小規模F1/ALL，重跑同一命令會驗證並沿用checkpoint。
.\experiments\congestion_2026\run_binary_classifier_comparison.ps1 -Smoke
# 正式執行或續跑，自動偵測manifest。
.\experiments\congestion_2026\run_binary_classifier_comparison.ps1
# 只續跑模型，不重算既有評估與圖表。
.\experiments\congestion_2026\run_binary_classifier_comparison.ps1 -Stage train
# 只重算評估或交付。
.\experiments\congestion_2026\run_binary_classifier_comparison.ps1 -Stage analyze
.\experiments\congestion_2026\run_binary_classifier_comparison.ps1 -Stage finalize
# 改參數或來源請指定新的目錄，不混入舊checkpoint。
.\experiments\congestion_2026\run_binary_classifier_comparison.ps1 -OutputDir '.\experiments\congestion_2026\outputs\binary_classifier_comparison_new'
```

Python等價指令：

```powershell
.\.venv\Scripts\python.exe -u -m experiments.congestion_2026.run_binary_classifier_comparison --resume
.\.venv\Scripts\python.exe -u -m experiments.congestion_2026.analyze_binary_classifier_comparison
.\.venv\Scripts\python.exe -m experiments.congestion_2026.finalize_binary_classifier_comparison
.\.venv\Scripts\python.exe -m pytest experiments/congestion_2026/tests -q
```

新目錄第一次直接Python執行可省略`--resume`。每個模型以原子檔案保存模型／預測，再寫checkpoint；中斷時只有完整且雜湊一致的checkpoint會重用，不完整cell會重建。來源、訓練程式、環境版本及超參數封存於manifest；有變動則拒絕混跑。評估是可重算衍生成果，只寫此新目錄，不寫舊Pilot。wrapper會重算評估，時間較只載入模型長；不會重新掃原始旅次。

## 必要私有輸入

公開clone不附研究資料。需保有`outputs/workpackages_0ABCDE_v1`的run_config、逐折h30 features/meta及npq陣列快取、`grouping_pilot_v1/cluster_assignments.csv`、已驗收`binary_congestion_pilot`的manifest/audit/cells、`speed_occupancy_eda_v1/candidate_Y_grid.parquet`、既有行事曆與設定引用的legacy model依賴。可用的`o_group_q_B0_v2/cells`之Logistic三分類機率會先重現再重用；不存在者直接按相同設定建立三分類模型。缺少必要來源會報錯，不產生假資料替代。

本機XGBoost不在.venv頂層套件，而在原研究`pilot_2024_2025/.model_deps`；沿用同一.venv解譯器與此既有3.4.1套件，未變更來源環境。若其他機器沒有該套件，請於其既有.venv安裝：

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments/congestion_2026/requirements_binary_classifier.txt
```

套件來源版本與原生DLL雜湊記錄在本機manifest。版本變動需另開輸出目錄。

## 固定參數及理由

Logistic沿用median imputation＋缺失指示＋StandardScaler，L2、C=1、lbfgs、max_iter=2000、seed=20261122。前處理僅fit訓練資料，權重只傳分類器。若出現不收斂則停止，不用外部驗證選solver或C。

XGBoost使用舊研究小型候選160棵、深度3、min_child_weight=8、eta=.05、subsample=.9；固定colsample=1、reg_lambda=10、reg_alpha=0、hist CPU、max_bin=256、4執行緒及同seed。深度與小學習率控制複雜度，較強L2收縮葉節點權重；沒有調參、early stopping或外部eval_set。min_child_weight為Hessian權重門檻，不等同最少樣本數。參考[XGBoost官方參數](https://xgboost.readthedocs.io/en/stable/parameter.html)。

LightGBM只讀原Pilot分數與硬決策，核對全部逐格鍵、Y、樣本及固定門檻指標；沒有另擬合LightGBM。所有分類器原始輸入欄位同一；Logistic可因缺失指示展開增加維度，不能把相同原始特徵誤寫成相同展開欄數。

## 評估與重現邊界

固定門檻0.5；B01 threshold=-1代表原argmax合併。門檻網格0.20–0.80、間距0.05；2024 pooled OOF選precision>=0.80下最大recall僅屬開發探索，於新增2025模型擬合前凍結。2025不調門檻、不選最佳模型；沒有折內歷史選門檻後的無偏驗證。額外precision 0.80–0.82窄帶僅便於比較相近工作點，無可行者不補猜。加權分數未校準，不稱真實壅塞機率。

2024 comparison為三折指標算術平均，n為平均折樣本數、n_total為總樣本數；pooled_metrics與配對Bootstrap是合併OOF，不可混用。2024合併16642格、2025共33423格；F3缺失約27.30%。2024連假只有F1有驗證樣本，缺樣本不寫成0績效。全日、14–20、其他、WEEKEND/LONGHOLIDAY分層皆有CSV。每日誤報格數以目標研究日期有有效樣本者為分母，不是誤報事件數或每24小時標準化率。

paired_effect為同一批樣本、同一組日期區塊重抽300次，報候選減參照。含跨分類器、B03−B02、B02−B01、ALL加權−B03及其他分組−ALL；AP採含ties的精確加權計算並與sklearn核對。區間條件於已訓練模型、未含重訓／選門檻變異，也無多重比較校正。current_state_strata用事後Y(t)作描述，未知保持未知，不當成部署閘門或確認的新發事件。

Logistic係數全表含fold、group、strategy、標準化係數、填補後訓練均值／SD及截距。係數是其他欄位固定下的每訓練SD條件log-odds關聯；缺失指示亦被標準化。高共線性、正則化、群需求組成與不同fold mapping使單一係數不宜作因果解釋；不同fold的G00不可硬接同一群。

## 交付檔案

獨立目錄`outputs/binary_classifier_comparison`含manifest、模型、逐格預測、checkpoint及完整衍生CSV；result ZIP與公開reports只放彙總、圖、程式，不含parquet、joblib、VehicleID或原始旅次。19組圖同時有300dpi PNG及向量PDF；PR座標僅保留在本機cells。圖表彙總口徑見figure_catalog.csv，六個研究問題的實際回答見findings_zh-TW.md。

如需重建可公開彙總：

```powershell
.\.venv\Scripts\python.exe -m experiments.congestion_2026.finalize_binary_classifier_comparison --publish-dir reports/2026-11-25/binary_classifier_comparison
```

本輪不修改原Word計畫書，不擴K或Y，也不建立新窗口／事件模型。
