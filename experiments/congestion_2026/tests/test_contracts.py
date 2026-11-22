"""Synthetic fixtures only for software tests, never exported as research data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from experiments.congestion_2026.data.contracts import *
from experiments.congestion_2026.common import read_only_guard,load_config,outdir
from experiments.congestion_2026.evaluation.events import onset_events,alert_episodes,evaluate_events

def test_future_feature_rejected():
    with pytest.raises(ValueError):assert_available(pd.Series(['2024-05-01 00:01']),pd.Series(['2024-05-01 00:00']))
def test_asof_feature_accepted():assert_available(pd.Series(['2024-05-01']),pd.Series(['2024-05-01']))
def test_all_missing_Y_stays_NA():
    c,y,n=spatial_y([[np.nan]*4],[.0417,.46,.36,.1383]);assert np.isnan(y[0]) and n[0]==0
def test_one_valid_station_cannot_be_Y1():
    for k in range(4):
        for speed in [0,39.99,40,100]:
            x=np.full((1,4),np.nan);x[0,k]=speed
            c,y,_=spatial_y(x,[.0417,.46,.36,.1383]);assert y[0] in [0,2]
def test_partial_weight_renormalization():
    c,y,n=spatial_y([[np.nan,30,100,np.nan]],[.0417,.46,.36,.1383]);assert c[0]==pytest.approx(.46/.82) and y[0]==2
def test_exact_40_is_not_congested():
    c,y,n=spatial_y([[40]*4],[.0417,.46,.36,.1383]);assert y[0]==0
def test_probability_rules():
    assert_probabilities([[.2,.3,.5]])
    for p in [[[.2,.3,.6]],[[0,np.nan,1]],[[-.1,.2,.9]]]:
        with pytest.raises(ValueError):assert_probabilities(p)
def test_samples_and_targets_must_match():
    a=pd.DataFrame({'prediction_time':[1],'target_time':[2],'y':[1],'fold':['F1']});b=a.copy();b['target_time']=3
    with pytest.raises(ValueError):assert_same_samples(a,b)
def test_target_embargo_rejects_overlap():
    a=pd.DataFrame({'target_time':pd.to_datetime(['2024-05-31 23:50'])});b=pd.DataFrame({'prediction_time':pd.to_datetime(['2024-06-01'])})
    with pytest.raises(ValueError):assert_train_validation(a,b,30)
def test_original_output_collision_rejected(tmp_path):
    source=tmp_path/'raw';source.mkdir()
    with pytest.raises(ValueError):read_only_guard([source],[source/'overwrite.parquet'])
def test_event_contiguity_and_missing_boundary():
    f=pd.DataFrame({'time':pd.date_range('2024-01-01',periods=8,freq='5min'),'kind':'WEEKEND','y':[0,2,2,2,np.nan,2,2,2]})
    e=onset_events(f);assert len(e)==2 and e.qualifying.tolist()==[True,False]
def test_episode_merge():
    f=pd.DataFrame({'prediction_time':pd.to_datetime(['2024-01-01 00:00','2024-01-01 00:05','2024-01-01 00:20'])})
    assert len(alert_episodes(f,[True,True,True]))==2
def test_group_mass_conservation():
    demand=np.array([[1.,2.,3.],[4.,5.,6.]]);mapping=np.array([[1.,0.],[0.,1.],[1.,0.]])
    assert np.allclose((demand@mapping).sum(1),demand.sum(1))
def test_real_saved_predictions_no_future_history():
    cfg=load_config();out=outdir(cfg)
    if not (out/'predictions.parquet').exists():pytest.skip('Run pipeline first')
    p=pd.read_parquet(out/'predictions.parquet');assert_available(p.history_cutoff,p.prediction_time);assert_probabilities(p[['p0','p1','p2']])
    for (fold,h),g in p.groupby(['fold','horizon']):
        ref=None
        for _,cell in g.groupby(['algorithm','configuration']):
            cell=cell.sort_values('prediction_time')
            if ref is None:ref=cell
            else:assert_same_samples(ref,cell)
def test_original_files_remain_unchanged():
    from experiments.congestion_2026.common import digest
    out=outdir(load_config())
    if not (out/'input_manifest.csv').exists():pytest.skip('Audit not available')
    for _,r in pd.read_csv(out/'input_manifest.csv').iterrows():
        if not r['exists']:continue
        p=Path(r.path);assert p.stat().st_size==r.bytes and p.stat().st_mtime_ns==r.mtime_ns,p
        if isinstance(r.sha256,str) and len(r.sha256)==64:assert digest(p)==r.sha256,p
def test_config_declares_primary_30():
    cfg=load_config();assert cfg['primary_horizon_minutes']==30 and cfg['Y']['c']==.5 and cfg['availability']['embargo_minutes']>=30
def test_actual_saved_group_demand_conservation():
    out=outdir(load_config())
    for p in (out/'cache').glob('features_*.parquet'):
        frame=pd.read_parquet(p);total=frame.filter(regex='^M1__').sum(axis=1)
        for method in ['M2','M3','M4','M5']:
            assert np.allclose(total,frame.filter(regex='^'+method+'__').sum(axis=1),rtol=3e-6,atol=1e-4),p
def test_future_missing_outcome_is_censored_not_false():
    timeline=pd.DataFrame({'time':pd.date_range('2024-01-01',periods=17,freq='5min'),'kind':'WEEKEND','y':[0,np.nan]+[0]*10+[2]*5})
    events=onset_events(timeline)
    frame=pd.DataFrame({'prediction_time':pd.to_datetime(['2024-01-01']),'target_time':pd.to_datetime(['2024-01-01 00:30'])})
    metric,detail,episodes=evaluate_events(frame,events,[True],[True],outcome_timeline=timeline)
    assert metric['total_alarm_episodes']==1 and metric['censored_alarm_episodes']==1 and pd.isna(episodes.false_episode.iloc[0])
