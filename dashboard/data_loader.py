import os,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import streamlit as st

HERE=Path(__file__).resolve().parent
def settings():
    path=Path(os.environ.get('CONGESTION_DASHBOARD_CONFIG',HERE/'config.yaml'))
    cfg=yaml.safe_load(path.read_text(encoding='utf8'));cfg['data_dir']=(path.parent/Path(os.path.expandvars(cfg['data_dir']))).resolve();return cfg
@st.cache_data(show_spinner=False)
def load_bundle(directory,signature):
    root=Path(directory)
    required=['predictions.parquet','vd.parquet','actual_y.parquet','metadata.json','historical_demand.parquet','metrics.csv','events.csv']
    missing=[f for f in required if not (root/f).exists()]
    if missing:raise FileNotFoundError('缺少真實研究匯出檔案：'+', '.join(missing)+'。請先執行研究管線 --stage export。')
    return {'predictions':pd.read_parquet(root/'predictions.parquet'),'vd':pd.read_parquet(root/'vd.parquet'),'y':pd.read_parquet(root/'actual_y.parquet'),'demand':pd.read_parquet(root/'historical_demand.parquet'),'metrics':pd.read_csv(root/'metrics.csv'),'events':pd.read_csv(root/'events.csv',parse_dates=['onset','end']),'event_metrics':pd.read_csv(root/'event_metrics.csv'),'examples':pd.read_csv(root/'examples.csv',parse_dates=['event_time']),'metadata':json.loads((root/'metadata.json').read_text(encoding='utf8'))}
def bundle():
    cfg=settings();root=cfg['data_dir']
    if not (root/'metadata.json').exists():st.error('尚無可用的真實研究匯出資料。請閱讀README執行匯出，不會以隨機資料替代。');st.stop()
    try:return load_bundle(str(root),(root/'metadata.json').stat().st_mtime_ns)
    except (FileNotFoundError,ValueError) as error:st.error(str(error));st.stop()
def snapshot(vd,t):
    """Latest published row per site, without backfilling invalid readings."""
    z=vd[vd.available_at.le(pd.Timestamp(t))].sort_values('observed_time').groupby('site',as_index=False).tail(1)
    return z.sort_values('site')
def snapshot_state(vd,t,weights):
    z=snapshot(vd,t).set_index('site').reindex(list(weights))
    # A stale record is not treated as a present observation.
    current=pd.Timestamp(t)-pd.Timedelta(minutes=10)
    fresh=z.observed_time.eq(current)
    z['speed']=z.speed.where(fresh);z['occupancy']=z.occupancy.where(fresh)
    speed=z.speed.to_numpy(float)
    valid=np.isfinite(speed);w=np.array(list(weights.values()))
    if not valid.any():return z.reset_index(),np.nan,np.nan,0.
    c=((speed<40)&valid).dot(w)/valid.dot(w)
    return z.reset_index(),float(c),0 if c==0 else 2 if c>=.5 else 1,float(valid.dot(w))
def rows_at(data,choice,t=None):
    z=data['predictions'];t=pd.Timestamp(t if t is not None else choice['time'])
    row=z[z.algorithm.eq(choice['algorithm'])&z.configuration.eq(choice['configuration'])&z.horizon.eq(choice['horizon'])&z.prediction_time.eq(t)].copy()
    row['decision']=row[choice['method']]
    return row
