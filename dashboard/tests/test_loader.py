"""Isolated availability tests, no simulated research probabilities."""
import numpy as np
import pandas as pd
from dashboard.data_loader import snapshot,snapshot_state,settings

def test_dashboard_no_future_VD():
    frame=pd.DataFrame({'site':['69.510']*2,'observed_time':pd.to_datetime(['2025-01-01 00:00','2025-01-01 00:05']),'available_at':pd.to_datetime(['2025-01-01 00:10','2025-01-01 00:15']),'speed':[100,10]})
    z=snapshot(frame,'2025-01-01 00:10');assert z.speed.iloc[0]==100
def test_missing_dashboard_configuration_is_explicit():assert settings()['positioning']=='Historical Replay / Offline Evaluation'
