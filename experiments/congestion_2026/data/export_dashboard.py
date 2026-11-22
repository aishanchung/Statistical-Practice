import json,logging
from pathlib import Path
import duckdb
import pandas as pd
from ..common import *
from .contracts import spatial_y

def run(cfg):
    out=outdir(cfg);dest=out/'dashboard_export';dest.mkdir(exist_ok=True)
    pred=pd.read_parquet(out/'decision_predictions.parquet')
    keys=['prediction_time','target_time','algorithm','configuration','horizon']
    pred=pred.drop_duplicates(keys+['method'])
    # One probability row, three separate frozen decisions. Keep the dashboard
    # export compact instead of repeating p0/p1/p2 three times.
    wide=pred.drop(columns=['method','decision']).drop_duplicates(keys).set_index(keys)
    decisions=pred.pivot(index=keys,columns='method',values='decision')
    wide.join(decisions).reset_index().to_parquet(dest/'predictions.parquet',index=False)
    with duckdb.connect() as con:
        vd=con.execute(f"SELECT TargetTime AS observed_time,regexp_extract(VDID,'69[.]510|65[.]990|63[.]990|61[.]670',0) AS site,Speed AS speed,Occupancy AS occupancy,VDType AS source_status FROM read_parquet({lit(Path(cfg['data_root'])/cfg['inputs']['vd'])}) WHERE regexp_matches(VDID,'69[.]510|65[.]990|63[.]990|61[.]670')").df()
    vd['available_at']=vd.observed_time+pd.Timedelta(minutes=cfg['availability']['vd_delay_minutes'])
    vd['valid_input']=vd.speed.ge(0)&~vd.source_status.isin(cfg['availability']['exclude_vd_types'])
    vd['occupancy']=vd.occupancy.where(vd.occupancy.ge(0)&vd.valid_input);vd['speed']=vd.speed.where(vd.valid_input)
    vd.to_parquet(dest/'vd.parquet',index=False)
    y=pd.read_parquet(Path(cfg['data_root'])/cfg['inputs']['labels']).rename(columns={'Time5':'time','Y':'y','DayType':'kind'})
    # Independent reconstruction is an audit only. Never replace formal Y.
    with duckdb.connect() as con:
        formal=con.execute(f"SELECT TargetTime AS time,regexp_extract(VDID,'69[.]510|65[.]990|63[.]990|61[.]670',0) AS site,Speed AS speed FROM read_parquet({lit(Path(cfg['data_root'])/cfg['inputs']['vd'])}) WHERE regexp_matches(VDID,'69[.]510|65[.]990|63[.]990|61[.]670')").df()
    sx=formal.pivot(index='time',columns='site',values='speed').reindex(index=y.time,columns=list(cfg['Y']['weights'])).to_numpy(float)
    _,rebuilt,_=spatial_y(sx,list(cfg['Y']['weights'].values()))
    mismatch=~np.isclose(rebuilt,y.y.to_numpy(float),equal_nan=True)
    save_json({'rows':len(y),'mismatches':int(mismatch.sum()),'formal_Y_not_modified':True,'input_status_mask_separate_from_outcome_reconstruction':True},out/'Y_reconstruction_audit.json')
    if mismatch.any():raise ValueError('Formal Y reconstruction differs. Stop export and inspect version.')
    y[['time','y','C_weighted','n_valid_vd','weight_coverage','kind']].to_parquet(dest/'actual_y.parquet',index=False)
    demands=pd.read_parquet(out/'cache/features_EXP2025_h30.parquet');demands=demands[demands.split.eq('evaluation')]
    cols=['target_time','history_cutoff']+[c for c in demands if c.startswith(('M1__','M2__','M3__','M4__','M5__','B0_PQ__'))]
    demands[cols].drop_duplicates('target_time').to_parquet(dest/'historical_demand.parquet',index=False)
    for source,target in [('model_comparison_all.csv','metrics.csv'),('onset_event_list.csv','events.csv'),('event_detection_metrics.csv','event_metrics.csv'),('historical_examples.csv','examples.csv'),('cluster_profile.csv','cluster_profile.csv'),('rejection_workload.csv','rejection_workload.csv')]:
        save_csv(pd.read_csv(out/source),dest/target)
    save_json({'config':cfg,'run_config':json.loads((out/'run_config.json').read_text(encoding='utf8')),'frozen_selection':json.loads((out/'frozen_selection.json').read_text(encoding='utf8')),'positioning':'Historical Replay / Offline Evaluation','live_feed':False,'feature_availability':'VD observed time + 10 minutes (proxy assumption), RECOVERED_P1 excluded','Y_version':cfg['Y']['version'],'prediction_truth_is_posthoc':True,'actual_Y_not_prediction_input':True,'training_cutoff':'2025-01-01','source_predictions_sha256':digest(out/'decision_predictions.parquet')},dest/'metadata.json')
    logging.info('Dashboard export ready: %s',dest)
