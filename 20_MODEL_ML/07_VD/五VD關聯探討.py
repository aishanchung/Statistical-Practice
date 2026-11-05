import pandas as pd
import numpy as np

# 1. 讀取 VD 資料
df = pd.read_parquet(
    "D:/統計實務/Data/V2_LITE_WEATHER/N3_Daxi_Longtan_N_2024_2025_V2_LITE_WEATHER.parquet"
)

# 2. 確認時間格式
df["TargetTime"] = pd.to_datetime(df["TargetTime"])

# 3. 保留原始速度，建立有效速度欄位
df["Speed_clean"] = df["Speed"].where(
    df["Speed"] >= 0, np.nan
)

# 4. 檢查是否有重複的測站、時間組合
assert not df.duplicated(
    ["TargetTime", "VDID"]
).any(), "發現重複的時間與 VDID"

# 5. 轉換成寬表：每支 VD 一個欄位
speed_wide = df.pivot(
    index="TargetTime",
    columns="VDID",
    values="Speed_clean"
).sort_index()

# 6. 計算五支 VD 兩兩之間的 Pearson 相關係數
corr_matrix = speed_wide.corr()

print(corr_matrix.round(3))