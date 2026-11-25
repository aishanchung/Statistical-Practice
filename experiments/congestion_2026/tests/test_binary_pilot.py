import numpy as np
import pandas as pd
from experiments.congestion_2026.run_binary_congestion_pilot import events, labels, SITES

def test_event_duration_is_identical_for_nanosecond_and_microsecond_parquet_times():
    times=pd.date_range('2024-06-01',periods=6,freq='5min')
    for unit in ['ns','us']:
        e=events(times.as_unit(unit),[0,1,1,np.nan,1,0])
        assert len(e)==2
        assert [r['minutes_observed'] for r in e]==[10,5]
        assert e[0]['right_censored'] and e[1]['left_censored']

def test_unknown_speed_is_never_assigned_negative_and_history_is_fold_only():
    times=pd.to_datetime(['2024-01-01','2024-02-01','2024-06-01','2024-06-02','2024-06-03'])
    grid=pd.DataFrame({'time':times,'Z_original':[0,0,np.nan,1,0]})
    for s in SITES:
        grid['speed_'+s]=[90,90,np.nan,20,90]
        grid['occ_'+s]=[10,10,np.nan,11,99]
    z,threshold=labels(grid,'2024-06-01')
    assert threshold==[10.]*4
    assert np.isnan(z['Z0'][2]) and np.isnan(z['Z1_strict'][2]) and np.isnan(z['Z1_available'][2])
    assert z['Z1_strict'][3]==1

def test_missing_gaps_split_events_and_clean_onset_requires_full_history():
    t=pd.date_range('2024-01-01',periods=8,freq='5min')
    a=events(t,[0,0,0,0,0,0,1,1]);assert a[0]['clean_30m'] and a[0]['minutes_observed']==10
    a=events(t.delete(3),[0,0,0,0,0,1,1]);assert not a[0]['clean_30m']
