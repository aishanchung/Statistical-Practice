import numpy as np
import pandas as pd

def onset_events(timeline,min_slots=3):
    z=timeline.sort_values('time').drop_duplicates('time').set_index('time')
    y=z.y;consecutive=y.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    severe=y.eq(2);starts=severe&~(severe.shift(fill_value=False)&consecutive)
    runs=starts.cumsum();records=[]
    for _,part in z.loc[severe].groupby(runs.loc[severe]):
        t=part.index[0];end=part.index[-1];prior=y.get(t-pd.Timedelta(minutes=5),np.nan);following=y.get(end+pd.Timedelta(minutes=5),np.nan)
        preceding=y.reindex(pd.date_range(t-pd.Timedelta(minutes=30),t-pd.Timedelta(minutes=5),freq='5min'))
        records.append({'event_id':t.strftime('%Y%m%d_%H%M'),'onset':t,'end':end,'duration_minutes':len(part)*5,'slots':len(part),'left_censored':pd.isna(prior),'right_censored':pd.isna(following),'qualifying':len(part)>=min_slots and pd.notna(prior) and prior!=2,'clean_onset':len(preceding)==6 and preceding.notna().all() and not preceding.eq(2).any(),'kind':part.kind.iloc[0]})
    return pd.DataFrame(records)
def alert_episodes(frame,trigger):
    z=frame.loc[np.asarray(trigger)].sort_values('prediction_time')
    if z.empty:return pd.DataFrame(columns=['episode_id','start','end','trigger_slots','trigger_times'])
    group=z.prediction_time.diff().ne(pd.Timedelta(minutes=5)).cumsum()
    return pd.DataFrame([{'episode_id':i,'start':g.prediction_time.iloc[0],'end':g.prediction_time.iloc[-1],'trigger_slots':len(g),'trigger_times':g.prediction_time.tolist()} for i,g in z.groupby(group)])
def evaluate_events(frame,events,trigger,eligible,window=30,outcome_timeline=None,min_slots=3):
    """Only hard Y2 alerts count as event warnings. R is handled separately."""
    x=frame.copy();x['eligible']=np.asarray(eligible);x['alarm']=np.asarray(trigger)&x.eligible
    onsets=events.loc[events.qualifying&events.onset.ge(x.prediction_time.min())&events.onset.le(x.prediction_time.max()+pd.Timedelta(minutes=window))].copy()
    details=[];triggers=x.loc[x.alarm,'prediction_time']
    for _,event in onsets.iterrows():
        start=event.onset-pd.Timedelta(minutes=window)
        risk=x.loc[x.eligible&x.prediction_time.ge(start)&x.prediction_time.lt(event.onset)]
        hit=triggers.loc[triggers.ge(start)&triggers.lt(event.onset)]
        details.append({**event.to_dict(),'evaluable':not risk.empty,'warned':not hit.empty,'first_alert':hit.min() if len(hit) else pd.NaT,'lead_minutes':(event.onset-hit.min()).total_seconds()/60 if len(hit) else np.nan})
    detail=pd.DataFrame(details);eps=alert_episodes(x,x.alarm)
    qualify_times=onsets.onset.to_numpy(dtype='datetime64[ns]')
    false=[];censored=[]
    future_complete=pd.Series(True,index=x.index)
    if outcome_timeline is not None:
        truth=outcome_timeline.set_index('time').y
        for offset in range(1,window//5+min_slots):
            future_complete &= truth.reindex(pd.DatetimeIndex(x.prediction_time)+pd.Timedelta(minutes=offset*5)).notna().to_numpy()
    complete_by_origin=dict(zip(x.prediction_time,future_complete))
    for _,ep in eps.iterrows():
        times=np.array(ep.trigger_times,dtype='datetime64[ns]')
        hit=bool(len(qualify_times) and ((qualify_times[None,:]>times[:,None])&(qualify_times[None,:]<=times[:,None]+np.timedelta64(window,'m'))).any())
        complete=all(complete_by_origin[t] for t in ep.trigger_times)
        unknown=not hit and not complete
        censored.append(unknown);false.append(pd.NA if unknown else not hit)
    eps['false_episode']=false
    eps['outcome_censored']=censored
    if 'target_time' in x:
        days_by_origin=dict(zip(x.prediction_time,x.target_time.dt.normalize()))
        eps['research_day']=eps.start.map(days_by_origin)
    # Cohort membership is the target research holiday. Early-night origins may
    # fall on a nonholiday Friday; do not create extra "research days" from them.
    day_field='target_time' if 'target_time' in x else 'prediction_time'
    days=x.loc[x.eligible,day_field].dt.normalize().nunique()
    n=len(detail);clean=detail.clean_onset if n else pd.Series(dtype=bool)
    metric={'events':n,'evaluable_events':int(detail.evaluable.sum()) if n else 0,'event_hits':int(detail.warned.sum()) if n else 0,'event_misses':int((~detail.warned).sum()) if n else 0,'event_recall':detail.warned.mean() if n else np.nan,'clean_events':int(clean.sum()),'clean_hits':int(detail.loc[clean,'warned'].sum()) if n else 0,'clean_recall':detail.loc[clean,'warned'].mean() if clean.sum() else np.nan,'median_lead_minutes':detail.loc[detail.warned,'lead_minutes'].median() if n else np.nan,'lead_p25':detail.loc[detail.warned,'lead_minutes'].quantile(.25) if n else np.nan,'lead_p75':detail.loc[detail.warned,'lead_minutes'].quantile(.75) if n else np.nan,'total_alarm_episodes':len(eps),'false_alarm_episodes':int(eps.false_episode.sum()),'research_days':days,'total_episodes_per_day':len(eps)/days if days else np.nan,'false_episodes_per_day':eps.false_episode.sum()/days if days else np.nan,'trigger_slots':int(x.alarm.sum())}
    metric['censored_alarm_episodes']=int(eps.outcome_censored.sum())
    metric['scorable_alarm_episodes']=len(eps)-metric['censored_alarm_episodes']
    if len(eps)>0 and metric['scorable_alarm_episodes']==0:metric['false_episodes_per_day']=np.nan
    metric['false_rate_interpretation']='confirmed false episodes/day; unknown future outcomes retained as censored, not counted as false or correct'
    return metric,detail,eps
