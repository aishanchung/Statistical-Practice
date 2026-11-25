import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from experiments.congestion_2026.analyze_binary_classifier_comparison import weighted_ap_batch,date_blocks,bootstrap_arrays

def test_weighted_ap_matches_sklearn_with_tied_scores_and_omitted_blocks():
    y=np.array([0,1,1,0,1,0]);scores=np.array([.8,.8,.5,.5,.2,.1]);codes=np.array([0,0,1,1,2,2]);weights=np.array([[1,1,1],[2,0,1],[0,3,1]])
    result=weighted_ap_batch(y,scores,codes,weights)
    for i,w in enumerate(weights):assert np.isclose(result[i],average_precision_score(y,scores,sample_weight=w[codes]))

def test_date_blocks_use_calendar_gaps_in_both_parquet_time_units():
    times=pd.DatetimeIndex(['2024-06-01','2024-06-01','2024-06-02','2024-06-08'])
    for unit in ['ns','us']:
        code,days=date_blocks(times.as_unit(unit));assert code.tolist()==[0,0,0,1];assert days.tolist()==[2,1]

def test_bootstrap_daily_false_alarm_denominator_counts_repeated_dates():
    a=pd.DataFrame({'y':[0,1,0,1],'score':[.9,.9,.1,.9],'hard':[1,1,0,1]})
    codes=np.array([0,0,1,1]);weights=np.array([[1,1],[2,0]])
    values=bootstrap_arrays(a,'B03',codes,weights,np.array([1,1]));assert values[0,4]==.5;assert values[1,4]==1.
