import json,logging,gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from ..common import *
from .metrics import evaluate
from .events import onset_events,evaluate_events,alert_episodes
from .plots import plt,save,COLORS

PROB=['p0','p1','p2']
def paired_intervals(base,alt,cfg):
    assert np.array_equal(base.prediction_time,alt.prediction_time)
    days=pd.DatetimeIndex(base.prediction_time.dt.normalize().unique()).sort_values()
    block=pd.Series((days.to_series().diff().dt.days.fillna(0)>1).cumsum().to_numpy(),index=days)
    bid=base.prediction_time.dt.normalize().map(block).to_numpy();units=np.unique(bid)
    indices=[np.flatnonzero(bid==u) for u in units]
    rng=np.random.default_rng(cfg['seed']);values=[]
    y=base.y.to_numpy(int);p=base[PROB].to_numpy();q=alt[PROB].to_numpy()
    for _ in range(cfg['selection']['bootstrap_reps']):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(units),len(units))])
        if not (y[ix]==2).any():continue
        values.append(average_precision_score(y[ix]==2,q[ix,2])-average_precision_score(y[ix]==2,p[ix,2]))
    return np.quantile(values,[.025,.975]),len(units)

def classification_tables(cfg,pred):
    out=outdir(cfg);metrics=pd.read_csv(out/'metrics_by_fold.csv')
    save_csv(metrics,out/'model_comparison_all.csv')
    comparisons=[]
    for (fold,h,alg),g in pred[pred.configuration.isin(['M0','M1','M2','M3','M4','M5'])].groupby(['fold','horizon','algorithm']):
        for left,right in [('M0','M1'),('M1','M2'),('M1','M3'),('M1','M4'),('M1','M5'),('M2','M3'),('M2','M4'),('M2','M5')]:
            a=g[g.configuration.eq(left)].sort_values('prediction_time');b=g[g.configuration.eq(right)].sort_values('prediction_time')
            ma,_=evaluate(a.y.to_numpy(int),a[PROB].to_numpy());mb,_=evaluate(b.y.to_numpy(int),b[PROB].to_numpy())
            for metric in ['log_loss','macro_f1','average_precision_2','recall_1','recall_2']:
                comparisons.append({'fold':fold,'horizon':h,'algorithm':alg,'baseline':left,'candidate':right,'metric':metric,'paired_delta':mb[metric]-ma[metric],'n':len(a)})
    # Preserve contiguous-holiday dependence. Only the primary LGB h30 comparison
    # receives bootstrap intervals; other point estimates remain explicitly so.
    v=pred[pred.fold.ne('EXP2025')&pred.horizon.eq(30)&pred.algorithm.eq('lightgbm')]
    for left,right in [('M0','M1'),('M1','M2'),('M1','M3'),('M1','M4'),('M1','M5'),('M2','M3'),('M2','M4'),('M2','M5')]:
        a=v[v.configuration.eq(left)].sort_values('prediction_time');b=v[v.configuration.eq(right)].sort_values('prediction_time')
        interval,units=paired_intervals(a,b,cfg)
        comparisons.append({'fold':'2024_pooled_OOF','horizon':30,'algorithm':'lightgbm','baseline':left,'candidate':right,'metric':'average_precision_2','paired_delta':average_precision_score(a.y==2,b.p2)-average_precision_score(a.y==2,a.p2),'lo95':interval[0],'hi95':interval[1],'bootstrap_units':units,'n':len(a)})
    save_csv(pd.DataFrame(comparisons),out/'incremental_gain_vs_baselines.csv')
    selection=pd.read_csv(out/'selection_2024_only.csv');s=selection[selection.algorithm.eq('lightgbm')].sort_values('configuration')
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(s.configuration,s.average_precision_2,color=COLORS[0]);ax.set_ylim(0,1);ax.set_ylabel('Y2 Average Precision');ax.set_title('A3　2024三折時序驗證，h=30，LightGBM\n各折AP2平均，所有分組使用相同樣本');save(fig,out,'fig_A3_model_comparison')
    delta=pd.DataFrame(comparisons);z=delta[delta.fold.eq('2024_pooled_OOF')]
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(z));ax.errorbar(x,z.paired_delta,yerr=np.vstack([z.paired_delta-z.lo95,z.hi95-z.paired_delta]).clip(min=0),fmt='o',color=COLORS[0],capsize=4);ax.axhline(0,color='grey',lw=.8);ax.set_xticks(x,z.candidate+'−'+z.baseline);ax.set_ylabel('Δ AP2（候選減基準）');ax.set_title('A4　2024 OOF增額價值，h=30 LightGBM\n95%日期連續假日區塊bootstrap區間');save(fig,out,'fig_A4_incremental_gain')
    cluster=pd.read_csv(out/'cluster_profile.csv');z=cluster[cluster.fold.eq('F1')&cluster.grouping.eq('K05')]
    fig,axes=plt.subplots(1,3,figsize=(12,3.7));axes[0].bar(z.group,z.O_count,color=COLORS[0]);axes[0].set_ylabel('O個數');axes[1].bar(z.group,z.passing_share,color=COLORS[0]);axes[1].set_ylabel('瓶頸旅次占比');axes[2].bar(z.group,z.lag_median_mean,color=COLORS[0]);axes[2].set_ylabel('O行車時間中位數之群平均（分）');fig.suptitle('A1　K=5訓練歷史群特徵，F1截止2024-06-01\n未通過瓶頸O仍保留，低量群解讀需保守');save(fig,out,'fig_A1_o_cluster_profiles')
    fig,ax=plt.subplots(figsize=(10,4))
    for _,r in z.iterrows():ax.plot(range(24),[r[f'pi_B_{h:02d}'] for h in range(24)],label=r.group)
    ax.set_xlabel('出發小時');ax.set_ylabel('群內瓶頸旅次比例');ax.legend();ax.set_title('A2　K=5逐O估計後聚合的歷史需求型態\nF1訓練歷史，無通過量群不繪組成');save(fig,out,'fig_A2_group_demand_patterns')

def npq_diagnostics(cfg):
    from ..features.engine import Engine,KINDS
    out=outdir(cfg);engine=Engine(cfg);con=engine.con
    nrows=[];prows=[];qrows=[];demandrows=[];support=[]
    for fold in cfg['folds']:
        cutoff=fold['start'];profile=engine.profile(cutoff);origins=profile['origins'].tolist();oi={o:i for i,o in enumerate(origins)}
        cond=engine.condition(cutoff)
        counts=engine.old['read_counts'](con,cond)
        ptab,_=engine.old['probability'](counts,1,cfg['npq']['P_strength'],origins)
        # Outcomes are read strictly for held-out diagnostics after profiles freeze.
        val=con.execute(f"SELECT o,kind,slot,depart_day,sum(n) AS n,coalesce(sum(n) FILTER(WHERE lag>=0),0) AS n_b FROM m WHERE depart_day>=DATE {lit(cutoff)} AND depart_day<DATE {lit(fold['end'])} GROUP BY ALL").df()
        actual=val.groupby(['o','kind','slot'])[['n','n_b']].sum()
        val_by_ok={key:g for key,g in val.groupby(['o','kind'])}
        val_hours=val.assign(hour=val.slot//12).groupby(['o','kind','hour'])[['n','n_b']].sum()
        empty_val=val.iloc[:0]
        for kind in KINDS:
            days=[d for d,k in engine.calendar.items() if k==kind and pd.Timestamp(cutoff)<=d<pd.Timestamp(fold['end'])]
            if not days:continue
            for o in origins:
                hist=counts[counts.o.eq(o)&counts.kind.eq(kind)]
                observed=val_by_ok.get((o,kind),empty_val)
                support.append({'fold':fold['id'],'o':o,'kind':kind,'train_n':hist.n.sum(),'train_N_B':hist.passing.sum(),'validation_days':len(days),'history_before':cutoff})
                for version in ['B0','B1']:
                    estimate=profile[f'{version}_N_{kind}'][:,oi[o]]
                    z=observed.groupby('slot').n.sum().reindex(range(288),fill_value=0).to_numpy()/len(days)
                    nrows.append({'fold':fold['id'],'o':o,'kind':kind,'version':version,'mae_mean_daily_5min':np.mean(abs(estimate-z)),'rmse_mean_daily_5min':np.sqrt(np.mean((estimate-z)**2)),'bias_mean_daily_5min':np.mean(estimate-z),'actual_total':z.sum()*len(days),'predicted_total':estimate.sum()*len(days),'n_validation_days':len(days)})
                for hour in range(24):
                    ix=(o,hour,kind)
                    p=float(ptab.loc[ix,'p']) if ix in ptab.index else np.nan
                    observed_hour=val_hours.loc[(o,kind,hour)] if (o,kind,hour) in val_hours.index else None
                    nn=observed_hour.n if observed_hour is not None else 0;kk=observed_hour.n_b if observed_hour is not None else 0;train=hist[hist.hour.eq(hour)]
                    prows.append({'fold':fold['id'],'o':o,'hour':hour,'kind':kind,'train_n':train.n.sum(),'train_passes':train.passing.sum(),'p_estimated':p,'validation_n':nn,'validation_passes':kk,'p_observed':kk/nn if nn else np.nan,'binary_log_loss':-(kk*np.log(max(p,1e-12))+(nn-kk)*np.log(max(1-p,1e-12)))/nn if nn else np.nan,'low_train_support':train.n.sum()<300})
        qt=con.execute(f"SELECT o,kind,slot//12 AS hour,lag,sum(n) AS n FROM m WHERE lag>=0 AND {cond} GROUP BY ALL").df()
        for (o,hour,kind),g in qt.groupby(['o','hour','kind']):
            g=g.sort_values('lag');cum=g.n.cumsum()/g.n.sum()
            quant={str(q):int(g.loc[cum.ge(q),'lag'].iloc[0])*5 for q in [.1,.5,.9]}
            for _,r in g.iterrows():qrows.append({'fold':fold['id'],'o':o,'hour':hour,'kind':kind,'lag_minutes':int(r.lag)*5,'count':int(r.n),'probability_empirical':r.n/g.n.sum(),'support':g.n.sum(),'p10':quant['0.1'],'median':quant['0.5'],'p90':quant['0.9'],'history_before':cutoff,'right_censor_rule':'completed cohort plus one-day latency; no completed trip treated as negative before maturity'})
        # Arrival outcomes include all holiday-departure cohorts, including B;
        # they exclude A nonholiday departures absent from this data source.
        arrivals=con.execute("SELECT depart_day+((slot+lag)//288)::INTEGER AS day,(slot+lag)%288 AS slot,sum(n) AS actual FROM m WHERE lag>=0 GROUP BY ALL").df()
        arrivals.day=pd.to_datetime(arrivals.day)
        frame=pd.read_parquet(out/'cache'/f'features_{fold["id"]}_h30.parquet');frame=frame[frame.split.eq('evaluation')]
        z=frame.merge(arrivals,on=['day','slot'],how='left');z['actual']=z.actual.fillna(0)
        for configuration in ['M2','B0_PQ','B1_P','B1_PQ','B2_PQ']:
            expected=z.filter(regex='^'+configuration+'__').sum(axis=1)
            for kind in ['ALL',*KINDS]:
                keep=np.ones(len(z),bool) if kind=='ALL' else z.kind.eq(kind)
                if not np.any(keep):continue
                e=expected[keep]-z.loc[keep,'actual']
                demandrows.append({'fold':fold['id'],'configuration':configuration,'kind':kind,'n':int(np.sum(keep)),'mae':e.abs().mean(),'rmse':np.sqrt((e**2).mean()),'bias':e.mean(),'outcome_population':'holiday-departure trips arriving in study date; excludes A','forecast_type':'frozen training historical pattern'})
        # Collinearity is a diagnostic on training rows only, never a feature.
        train=pd.read_parquet(out/'cache'/f'features_{fold["id"]}_h30.parquet');train=train[train.split.eq('training')]
        cc=train.filter(regex='^cal_|^M1__').corr();save_csv(cc.reset_index(),out/f'demand_time_correlation_{fold["id"]}.csv')
    save_csv(pd.DataFrame(support),out/'npq_sample_support.csv');save_csv(pd.DataFrame(nrows),out/'n_forecast_diagnostics.csv');save_csv(pd.DataFrame(prows),out/'p_calibration_by_origin.csv');save_csv(pd.DataFrame(qrows),out/'q_delay_distribution.csv');save_csv(pd.DataFrame(demandrows),out/'historical_demand_error.csv')
    p=pd.DataFrame(prows);top=p.groupby('o').validation_passes.sum().nlargest(8).index
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for ax,o in zip(axes.flat,top):
        z=p[p.o.eq(o)&p.fold.eq('F1')]
        for kind,g in z.groupby('kind'):ax.plot(g.hour,g.p_estimated,label=kind)
        ax.set_title(o);ax.set_ylim(0,1);ax.set_xlabel('出發小時')
    axes.flat[0].legend(fontsize=7);fig.suptitle('B1　F1歷史P，訓練截止2024-06-01；零通過及低頻樣本另列CSV');save(fig,out,'fig_B1_p_by_origin')
    q=pd.DataFrame(qrows);fig,ax=plt.subplots(figsize=(10,4))
    for o in ['03F0846N','03F0961N','01F1621N']:
        z=q[q.o.eq(o)&q.fold.eq('F1')&q.hour.eq(8)].groupby('lag_minutes')['count'].sum();z=z/z.sum();ax.plot(z.index,z.values,label=o)
    ax.set_xlim(0,180);ax.set_xlabel('延遲（分鐘，圖僅顯示≤180，估計未截尾）');ax.set_ylabel('條件比例');ax.legend();ax.set_title('B2　08時出發的Q經驗分布，F1訓練歷史');save(fig,out,'fig_B2_q_delay_curves')
    dem=pd.DataFrame(demandrows);fig,ax=plt.subplots(figsize=(9,4));z=dem[dem.kind.eq('ALL')].pivot(index='fold',columns='configuration',values='rmse');z.plot.bar(ax=ax);ax.set_ylabel('抵達需求RMSE（輛／5分鐘）');ax.set_title('B3　2024驗證期需求誤差，固定訓練Pattern\n實際量限假日出發旅次，非研究假日出發A類未涵蓋');ax.legend(fontsize=8);save(fig,out,'fig_B3_expected_vs_actual_demand')
    engine.con.close();del qrows;gc.collect()

def decisions(cfg,pred):
    out=outdir(cfg)
    base=pred[pred.algorithm.eq('lightgbm')&pred.configuration.eq('M0')&pred.horizon.eq(30)]
    validation=base[base.fold.ne('EXP2025')];test=base[base.fold.eq('EXP2025')]
    # C1 is a decision change only. Preserve argmax2 and a p2>=.4 safety guard.
    original,_=evaluate(validation.y.to_numpy(int),validation[PROB].to_numpy())
    choices=[]
    for threshold in [.15,.2,.25,.3,.35,.4,.45,.5]:
        p=validation[PROB].to_numpy();d=p.argmax(1);guard=(d==2)|(p[:,2]>=cfg['decision']['y2_protection_probability']);d[(p[:,1]>=threshold)&~guard]=1
        m,_=evaluate(validation.y.to_numpy(int),p,d)
        if m['recall_2']>=original['recall_2']-.01:choices.append((m['f1_1'],threshold))
    threshold=max(choices)[1]
    save_json({'C1_y1_threshold':threshold,'Y2_protection':cfg['decision']['y2_protection_probability'],'selection':'2024 OOF Y1 F1 with Y2 Recall >= original-.01','2025_used':False},out/'selected_C1_rule.json')
    comparison=[];condition_rows=[];coverage_rows=[];transition_rows=[];timeline=pd.read_parquet(Path(cfg['data_root'])/cfg['inputs']['labels']).rename(columns={'Time5':'time','Y':'y','DayType':'kind'})
    truth=timeline.set_index('time')
    # Observed Y1 runs include missing/gap boundaries; duration is observed only.
    one=truth.y.eq(1);cont=truth.index.to_series().diff().eq(pd.Timedelta(minutes=5));runs=(one&~(one.shift(fill_value=False)&cont)).cumsum()
    duration=truth.loc[one].groupby(runs.loc[one]).size()*5
    run_duration=pd.Series(np.nan,index=truth.index);run_duration.loc[one]=runs.loc[one].map(duration)
    for split,z in [('2024_validation',validation),('2025_exploratory',test)]:
        p=z[PROB].to_numpy();d=p.argmax(1);guard=(d==2)|(p[:,2]>=cfg['decision']['y2_protection_probability']);d[(p[:,1]>=threshold)&~guard]=1
        for method,dec in [('C0_original',p.argmax(1)),('C1_threshold',d)]:
            m,mat=evaluate(z.y.to_numpy(int),p,dec);comparison.append({'split':split,'method':method,**m,'Y0_to_Y1':int(((z.y.to_numpy()==0)&(dec==1)).sum()),'Y2_to_Y1':int(((z.y.to_numpy()==2)&(dec==1)).sum())})
        for model in ['C2_w15','C2_w20']:
            w=pred[pred.configuration.eq(model)&pred.horizon.eq(30)&(pred.fold.eq('EXP2025') if split.startswith('2025') else pred.fold.ne('EXP2025'))]
            m,mat=evaluate(w.y.to_numpy(int),w[PROB].to_numpy());comparison.append({'split':split,'method':model,**m,'Y0_to_Y1':int(mat[0,1]),'Y2_to_Y1':int(mat[2,1])})
        joined=z.join(truth[['n_valid_vd','C_weighted']],on='target_time',rsuffix='_outcome')
        joined['outcome_duration_minutes']=joined.target_time.map(run_duration)
        joined['outcome_prev_y']=truth.y.reindex(pd.DatetimeIndex(joined.target_time)-pd.Timedelta(minutes=5)).to_numpy()
        joined['prediction']=joined[PROB].to_numpy().argmax(1);joined['target_hour']=joined.target_time.dt.hour
        joined['duration_bin']=pd.cut(joined.outcome_duration_minutes,[0,5,15,30,60,10000],labels=['5','10–15','20–30','35–60','>60'])
        joined['near_c050']=np.abs(joined.C_weighted-.5)<=.10
        for field in ['kind','target_hour','duration_bin','near_c050']:
            for condition,g in joined[joined.y.eq(1)].groupby(field,observed=True):
                condition_rows.append({'split':split,'condition':field,'value':str(condition),'n_Y1':len(g),'predicted0':int(g.prediction.eq(0).sum()),'predicted1':int(g.prediction.eq(1).sum()),'predicted2':int(g.prediction.eq(2).sum()),'recall_Y1':g.prediction.eq(1).mean()})
        for n in [1,2,3,4]:
            g=joined[joined.n_valid_vd.eq(n)];y1=g[g.y.eq(1)]
            if n==1 and len(y1):raise ValueError('Formal Y1 impossible with one valid VD')
            coverage_rows.append({'split':split,'target_valid_VD':n,'n':len(g),'n_Y1':len(y1),'Y1_recall':y1.prediction.eq(1).mean() if len(y1) else np.nan,'Y1_mathematically_possible':n>=2})
        for prev,g in joined[joined.y.eq(1)].groupby('outcome_prev_y'):
            transition_rows.append({'split':split,'previous_observed_y':prev,'current_y':1,'n':len(g),'Y1_recall':g.prediction.eq(1).mean()})
        prior=truth.y.reindex(truth.index-pd.Timedelta(minutes=5)).to_numpy()
        transition_times={'onset_Y1':truth.index[(truth.y.eq(1)&pd.notna(prior)&(prior!=1)).to_numpy()], 'end_Y1':truth.index[(truth.y.ne(1)&truth.y.notna()&(prior==1)).to_numpy()]}
        lookup=z.drop_duplicates('target_time').set_index('target_time')
        for transition,dates in transition_times.items():
            for offset in [-15,-10,-5,0,5,10,15]:
                matched=lookup.reindex(dates+pd.Timedelta(minutes=offset)).dropna(subset=['p0'])
                if matched.empty:continue
                transition_rows.append({'split':split,'transition':transition,'offset_minutes':offset,'n':len(matched),'p0_mean':matched.p0.mean(),'p1_mean':matched.p1.mean(),'p2_mean':matched.p2.mean(),'Y1_fraction':matched.y.eq(1).mean()})
    comp=pd.DataFrame(comparison)
    for split,g in comp.groupby('split'):
        base=g[g.method.eq('C0_original')].iloc[0]
        for field in ['Y0_to_Y1','Y2_to_Y1']:
            comp.loc[g.index,'delta_'+field]=g[field]-base[field]
    save_csv(comp,out/'y1_model_comparison.csv');save_csv(pd.DataFrame(condition_rows),out/'y1_error_breakdown.csv');save_csv(pd.DataFrame(coverage_rows),out/'y1_by_vd_coverage.csv');save_csv(pd.DataFrame(transition_rows),out/'y1_by_congestion_transition.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,z,title in [(axes[0],validation,'2024 OOF'),(axes[1],test,'2025探索性')]:
        _,m=evaluate(z.y.to_numpy(int),z[PROB].to_numpy());ax.imshow(m,cmap='Blues');ax.set_title(f'C1　{title}原Baseline，n={len(z):,}');ax.set_xlabel('預測Y');ax.set_ylabel('真實Y')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(m[i,j]),ha='center',va='center',color='white' if m[i,j]>m.max()/2 else 'black')
    save(fig,out,'fig_C1_confusion_matrices')
    z=pd.DataFrame(condition_rows);z=z[z.split.eq('2025_exploratory')&z.condition.eq('target_hour')]
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(z.value,z.recall_Y1,color=COLORS[0]);ax.set_ylim(0,1);ax.set_xlabel('目標小時');ax.set_ylabel('Y1 Recall');ax.set_title('C2　2025探索性回測，Y1各小時辨識\n僅錯誤診斷，不依2025時段改選模型');save(fig,out,'fig_C2_y1_error_by_condition')
    fig,ax=plt.subplots(figsize=(8,5))
    for split,g in comp.groupby('split'):
        ax.scatter(g.precision_1,g.recall_1,label=split)
        for _,r in g.iterrows():ax.annotate(r.method,(r.precision_1,r.recall_1),fontsize=8)
    ax.set_xlabel('Y1 Precision');ax.set_ylabel('Y1 Recall');ax.legend();ax.set_title('C3　局部壅塞的Precision／Recall取捨');save(fig,out,'fig_C3_y1_recall_precision_tradeoff')
    return comp

def event_analysis(cfg,pred):
    out=outdir(cfg);timeline=pd.read_parquet(Path(cfg['data_root'])/cfg['inputs']['labels']).rename(columns={'Time5':'time','Y':'y','DayType':'kind'})
    events=onset_events(timeline,cfg['decision']['onset_min_slots']);save_csv(events,out/'onset_event_list.csv')
    sensitivity=[]
    for minimum in [1,3,6]:
        e=onset_events(timeline,minimum);sensitivity.append({'min_slots':minimum,'year':2025,'onsets':int((e.qualifying&e.onset.dt.year.eq(2025)).sum()),'clean_onsets':int((e.qualifying&e.clean_onset&e.onset.dt.year.eq(2025)).sum())})
    save_csv(pd.DataFrame(sensitivity),out/'onset_definition_sensitivity.csv')
    frame_by_h={h:pd.read_parquet(out/'cache'/f'features_EXP2025_h{h}.parquet').set_index('prediction_time') for h in [15,30]}
    metrics=[];details=[];episodes=[];work=[];classification=[];costsensitivity=[];decision_exports=[]
    specs=pred[pred.configuration.isin(['M0','M1','M2','M3','M4','M5'])].groupby(['algorithm','configuration','horizon'])
    common_origins={}
    horizon_common=[]
    for (alg,config),g in pred[pred.configuration.isin(['M0','M1','M2','M3','M4','M5'])&pred.fold.eq('EXP2025')].groupby(['algorithm','configuration']):
        left=g[g.horizon.eq(15)];right=g[g.horizon.eq(30)]
        common=pd.Index(left.prediction_time).intersection(pd.Index(right.prediction_time))
        common_origins[(alg,config)]=common
        for h in [15,30]:
            s=g[g.horizon.eq(h)&g.prediction_time.isin(common)];m,_=evaluate(s.y.to_numpy(int),s[PROB].to_numpy())
            horizon_common.append({'algorithm':alg,'configuration':config,'horizon':h,'cohort':'common_prediction_origins_both_future_targets_valid',**m})
    save_csv(pd.DataFrame(horizon_common),out/'horizon_common_metrics.csv')
    for (alg,config,h),z in specs:
        v=z[z.fold.ne('EXP2025')].sort_values('prediction_time');test=z[z.fold.eq('EXP2025')].sort_values('prediction_time')
        if test.empty:continue
        test=test[test.prediction_time.isin(common_origins[(alg,config)])]
        cost=np.array(cfg['decision']['balanced_cost']);vp=v[PROB].to_numpy();vdec=(vp@cost).argmin(1)
        # Fixed validation-only coverage constraint, reducing unsafe accepted Y0.
        eligible=[]
        for theta in [0.,.5,.6,.7,.8,.85,.9,.92,.94,.96,.98]:
            reject=(vdec==0)&(vp[:,0]<theta);coverage=1-reject.mean()
            if coverage>=cfg['decision']['coverage_floor']:
                gain=int(((v.y.to_numpy()!=0)&reject).sum());eligible.append((gain,coverage,theta))
        theta=max(eligible)[2]
        p=test[PROB].to_numpy();d0=p.argmax(1);d1=(p@cost).argmin(1);r=(d1==0)&(p[:,0]<theta);d2=d1.copy();d2[r]=-1
        source=frame_by_h[h].reindex(test.prediction_time)
        speed=source[[f'vd_{site.replace(".","_")}_speed_lag10' for site in cfg['Y']['weights']]].to_numpy(float)
        from ..data.contracts import spatial_y
        _,state,_=spatial_y(speed,list(cfg['Y']['weights'].values()));risk=np.isfinite(state)&(state!=2)
        identity={'algorithm':alg,'configuration':config,'horizon':h,'split':'2025_exploratory','cohort':'common_prediction_origins_h15_h30','gate':'latest_available_VD_t_minus10','validation_reject_threshold':theta}
        for method,decision in [('D0',d0),('D1',d1),('D2',d2)]:
            m,detail,ep=evaluate_events(test,events,decision==2,risk,cfg['decision']['event_window_minutes'],outcome_timeline=timeline,min_slots=cfg['decision']['onset_min_slots'])
            metrics.append({**identity,'method':method,**m})
            details.append(detail.assign(**identity,method=method));episodes.append(ep.drop(columns='trigger_times').assign(**identity,method=method))
            accepted=decision>=0
            cls,_=evaluate(test.y.to_numpy(int),p,decision)
            realized=cost[test.y.to_numpy(int)[accepted],decision[accepted]]
            work.append({**identity,'method':method,'n':len(test),'coverage':accepted.mean(),'rejection_count':int((~accepted).sum()),'rejection_true_Y2':int(((~accepted)&test.y.eq(2).to_numpy()).sum()),'risk_wrongly_accepted_as_Y0':int(((test.y.to_numpy()!=0)&(decision==0)).sum()),'Y2_recall_all':cls['recall_2'],'Y2_precision':cls['precision_2'],'accepted_mean_relative_cost':realized.mean(),'full_population_cost':realized.mean() if accepted.all() else np.nan,'R_cost_defined':False,'total_review_episodes_full_cohort':len(alert_episodes(test,(decision==2)|(decision==-1))),'R_episodes_full_cohort':len(alert_episodes(test,decision==-1)),'total_review_episodes_riskset':len(alert_episodes(test,((decision==2)|(decision==-1))&risk)),'R_episodes_riskset':len(alert_episodes(test,(decision==-1)&risk)),'eligible_origin_count':int(risk.sum()),'eligible_research_target_days':m['research_days'],'review_episodes_per_eligible_day':len(alert_episodes(test,((decision==2)|(decision==-1))&risk))/m['research_days'] if m['research_days'] else np.nan})
            export=test.copy();export['method']=method;export['decision']=decision;export['cost_scenario']='balanced';decision_exports.append(export)
        for name,cm in {'balanced':cost,**{k:np.asarray(v) for k,v in cfg['decision']['cost_sensitivity'].items()}}.items():
            decision=(p@cm).argmin(1);m,_,_=evaluate_events(test,events,decision==2,risk,window=cfg['decision']['event_window_minutes'],outcome_timeline=timeline,min_slots=cfg['decision']['onset_min_slots'])
            costsensitivity.append({**identity,'scenario':name,'relative_cost_mean':cm[test.y.to_numpy(int),decision].mean(),**m})
    save_csv(pd.DataFrame(metrics),out/'event_detection_metrics.csv');save_csv(pd.concat(details,ignore_index=True),out/'event_level_results.csv');save_csv(pd.concat(episodes,ignore_index=True),out/'alert_episodes.csv');save_csv(pd.DataFrame(work),out/'rejection_workload.csv');save_csv(pd.DataFrame(costsensitivity),out/'cost_sensitivity.csv')
    pd.concat(decision_exports,ignore_index=True).to_parquet(out/'decision_predictions.parquet',index=False)
    metric=pd.DataFrame(metrics);z=metric[metric.algorithm.eq('lightgbm')&metric.configuration.eq('M0')]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,field,label in [(axes[0],'event_recall','事件Recall'),(axes[1],'median_lead_minutes','成功事件提前量中位數（分）')]:
        z.pivot(index='horizon',columns='method',values=field).plot.bar(ax=ax);ax.set_ylabel(label);ax.set_xlabel('預測horizon（分）')
    fig.suptitle('D1　15／30分預警，2025探索性回測\n相同future30事件窗，最近可取得VD作風險集');save(fig,out,'fig_D1_15_vs_30_minutes')
    fig,ax=plt.subplots(figsize=(9,4))
    for _,r in z.iterrows():ax.scatter(r.false_episodes_per_day,r.event_recall);ax.annotate(f'h{r.horizon}-{r.method}',(r.false_episodes_per_day,r.event_recall),fontsize=9)
    ax.set_xlabel('誤報episode／可評估研究日');ax.set_ylabel('事件Recall');ax.set_title('D2　LightGBM Baseline 警報取捨，2025探索性');save(fig,out,'fig_D2_recall_false_alarm_tradeoff')
    w=pd.DataFrame(work);w=w[w.algorithm.eq('lightgbm')&w.configuration.eq('M0')&w.horizon.eq(30)]
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(w.method,w.coverage,color=COLORS[0]);axes[0].set_ylim(0,1);axes[0].set_ylabel('Coverage');axes[1].bar(w.method,w.review_episodes_per_eligible_day,color=COLORS[1]);axes[1].set_ylabel('Y2或R複核episode／日');fig.suptitle('D4　成本與拒絕決策，2025探索性\nR不計為成功預警，完整人口R成本尚未定義');save(fig,out,'fig_D4_cost_rejection_comparison')
    # Resample dates in contiguous study-day blocks. Several onsets on one date
    # may be rapid recurrences, so do not assume independent onset fragments.
    ci=[];all_details=pd.concat(details,ignore_index=True);all_episodes=pd.concat(episodes,ignore_index=True)
    research_dates=pd.DatetimeIndex(timeline.loc[timeline.time.dt.year.eq(2025),'time'].dt.normalize().unique()).sort_values()
    block_ids=(research_dates.to_series().diff().dt.days.fillna(0)>1).cumsum().to_numpy()
    date_to_block=pd.Series(block_ids,index=research_dates)
    for (alg,c,h,m),g in all_details.groupby(['algorithm','configuration','horizon','method']):
        daily=pd.DataFrame({'day':research_dates,'block':block_ids}).set_index('day')
        gd=g.assign(day=g.onset.dt.normalize()).groupby('day').agg(events=('event_id','size'),hits=('warned','sum'))
        eps=all_episodes[all_episodes.algorithm.eq(alg)&all_episodes.configuration.eq(c)&all_episodes.horizon.eq(h)&all_episodes.method.eq(m)]
        epd=eps.groupby('research_day').agg(total_episodes=('episode_id','size'),false_episodes=('false_episode','sum'))
        daily=daily.join(gd).join(epd).fillna(0);daily['days']=1
        unit=daily.groupby('block')[['events','hits','total_episodes','false_episodes','days']].sum()
        rng=np.random.default_rng(cfg['seed']);ix=rng.integers(0,len(unit),(cfg['selection']['bootstrap_reps'],len(unit)))
        sums=unit.to_numpy(float)[ix].sum(1);recall=np.divide(sums[:,1],sums[:,0],out=np.full(len(sums),np.nan),where=sums[:,0]>0)
        false_rate=sums[:,3]/sums[:,4]
        ci.append({'algorithm':alg,'configuration':c,'horizon':h,'method':m,'event_recall':g.warned.mean(),'lo95':np.nanquantile(recall,.025),'hi95':np.nanquantile(recall,.975),'false_episodes_per_day_lo95':np.quantile(false_rate,.025),'false_episodes_per_day_hi95':np.quantile(false_rate,.975),'unit':'consecutive research-date block; onset fragments on same/adjacent dates retained together','bootstrap_blocks':len(unit)})
    save_csv(pd.DataFrame(ci),out/'event_bootstrap.csv')
    return events,all_details

def examples(cfg,pred,events,details):
    out=outdir(cfg);timeline=pd.read_parquet(Path(cfg['data_root'])/cfg['inputs']['labels']).rename(columns={'Time5':'time','Y':'y'})
    import duckdb
    with duckdb.connect() as con:
        vd=con.execute(f"SELECT TargetTime AS time,regexp_extract(VDID,'69[.]510|65[.]990|63[.]990|61[.]670',0) AS site,Speed AS speed FROM read_parquet({lit(Path(cfg['data_root'])/cfg['inputs']['vd'])}) WHERE regexp_matches(VDID,'69[.]510|65[.]990|63[.]990|61[.]670')").df()
    z=details[details.algorithm.eq('lightgbm')&details.configuration.eq('M0')&details.horizon.eq(30)&details.method.eq('D0')]
    chosen=[]
    for outcome in [True,False]:
        c=z[z.warned.eq(outcome)].sort_values(['clean_onset','onset'],ascending=[False,True]).head(1)
        if len(c):chosen.append((('success' if outcome else 'miss'),c.iloc[0].onset))
    one=timeline[timeline.y.eq(1)&timeline.time.dt.year.eq(2025)].sort_values('time')
    if len(one):chosen.append(('Y1',one.time.iloc[0]))
    records=[]
    for name,t in chosen:
        start=t-pd.Timedelta(minutes=60);end=t+pd.Timedelta(minutes=60)
        v=vd[vd.time.between(start,end)];truth=timeline[timeline.time.between(start,end)]
        probs=pred[pred.algorithm.eq('lightgbm')&pred.configuration.eq('M0')&pred.horizon.eq(30)&pred.fold.eq('EXP2025')&pred.prediction_time.between(start,end)]
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
        for site,g in v.groupby('site'):axes[0].plot(g.time,g.speed.where(g.speed.ge(0)),label=site)
        axes[0].set_ylabel('VD速度（km/h）');axes[0].legend(ncol=4)
        axes[1].step(truth.time,truth.y,where='post');axes[1].set_ylabel('事後真實Y');axes[1].set_yticks([0,1,2])
        for k in range(3):axes[2].plot(probs.prediction_time,probs[f'p{k}'],label=f'p{k}')
        axes[2].set_ylabel('t發出的機率');axes[2].legend(ncol=3);axes[2].set_xlabel('歷史時間，機率目標為t+30分')
        for ax in axes:ax.axvline(t,color=COLORS[1],ls='--')
        fig.suptitle(f'{name} 真實案例　{t}，2025探索性回測\n選例規則為最早乾淨成功／漏報，Y1為最早觀測例；VD及Y為事後診斷')
        namefile='fig_C4_example_y1_event' if name=='Y1' else ('fig_D3_event_timeline' if name=='success' else 'fig_D3_missed_event_timeline')
        save(fig,out,namefile);records.append({'case':name,'event_time':t,'selection_rule':'earliest clean outcome if present, otherwise earliest observed; no manually manufactured values','figure':namefile})
    save_csv(pd.DataFrame(records),out/'historical_examples.csv')

def report(cfg):
    out=outdir(cfg);rank=pd.read_csv(out/'selection_2024_only.csv');a=rank[rank.algorithm.eq('lightgbm')].sort_values('average_precision_2',ascending=False)
    b=pd.read_csv(out/'historical_demand_error.csv');bc=b[b.kind.eq('ALL')].groupby('configuration')[['rmse','mae']].mean()
    cm=pd.read_csv(out/'y1_model_comparison.csv');em=pd.read_csv(out/'event_detection_metrics.csv');em=em[em.algorithm.eq('lightgbm')&em.configuration.eq('M0')]
    text=['# 統計實務工作包0–E研究結果','',f'資料來源為GitHub main及本機已核對checkpoint。本次配置{cfg["version"]}，主要h30、h15對照。Y維持四站有效權重重正規化、<40 km/h與c=.50。2025僅為探索性回測。','','## 工作包0　資料與共同規格','','297 O為兩年V3瓶頸抵達假日旅次，341 O為早期全部假日出發母體，85個早期O沒有觀測瓶頸通過。原九組逐筆機率可重算，未找到原訓練模型權重檔。新實驗保存joblib。','新框架重用原P/Q、O特徵轉換與模型factory，加入至少30分鐘embargo，並以prediction_time切分。因此新表不能直接與舊33,423樣本分數混成同一實驗。','原始資料及舊輸出唯讀。所有新cache、模型、機率與大型檔均Git忽略。','','## 工作包A　無r分群與預測增益','',a.to_markdown(index=False),'','此表為2024三折的AP2平均。正式候選只按2024選擇，AP2差≤.005時優先較少欄位。各折重新估計群特徵、變換與Kmeans，群別只用訓練期成熟完整月份。N/P/Q訓練輸入逐月回看，驗證期凍結。幾何ARI另列，與預測增益分開。','增額AP2的日期區塊bootstrap區間見incremental_gain_vs_baselines.csv，不挑選有利月份。','','## 工作包B　需求估計與分類效果分開','',bc.to_markdown(),'','B0是原本已包含P分層pooling與O×hour Q平滑的方法，不能把B1描述成首次加入平滑。B1僅增加N對O小時父型態的3日平滑及P先驗300改600。B2再加入daytype Q，pseudo=200，沒有無限制細分五分鐘Q。候選為事前固定的小型比較。','N/P診斷與抵達需求真值限完整假日出發旅次，A類非假日出發仍缺完整母體，尤其午夜不可宣稱全部真實需求。歷史資料僅在旅次結束加1日後估計。Q保留所有歷史已觀測延遲質量，未額外截尾；未見過的超長延遲仍可能低估，必須保留此限制。','Q是否改善抵達需求，以historical_demand_error.csv判斷；是否改善壅塞分類，以npq_ablation_metrics.csv的2024驗證判斷，兩者不可互相替代。新增需求與時間欄位的訓練相關表只能評估重複型態，不能證明VD已完全包含所有O資訊。','','## 工作包C　局部壅塞','',cm.to_markdown(index=False),'','只有一站有效時C只能為0或1，Y1在數學上不可能，診斷表保留n_Y1=0、Recall=NA。目標站數與C是事後錯誤診斷，沒有回填為預測特徵。C1保留原機率，只改決策。C2為Y1權重1.5或2。C3二階段未執行，避免在尚未完成C1/C2穩定評估前擴大模型。','C1/C2是否替換原模型只看2024改善與Y2保護，不能因2025Y1 Recall較高而反向選擇。','','## 工作包D　事件、成本與複核','',em.to_markdown(index=False),'','新風險集以t−10分鐘最近可取得VD狀態排除已觀測Y2，與舊報告current_state_known協定不同。R只計入複核負擔，不計為成功預警。D2不壓掉D1的Y2硬警報，因此硬警報事件Recall相同。','所有事件清單相同，onset需前格非Y2且至少3格持續。Clean要求前30分鐘完整可觀測且無Y2。缺失與資料邊界保留censor旗標。15/30分的未來30分事件窗一致，共同起點敏感度另列，不把exact horizon Y2與within-window onset混為同標籤。','Balanced成本是相對研究損失。拒絕R的成本尚未定義，故D2不提供全人口成本優越宣告。複核episode與真正人工處理次數不同，需營運流程驗證。','','## 工作包E　歷史回放展示','','Dashboard只讀精簡匯出資料，明示Historical Replay / Offline Evaluation。t當下輸入採最近可取得VD，標示來源時間及10分鐘假設延遲。未來真值須開啟事後答案選項。排行榜只讀已完成機率與指標。','','## 限制與下一步','','未取得同口徑2026資料，不提供正式新獨立驗證。VD發布延遲仍為保守代理，尚未串接即時系統。原始模型權重缺失但訓練程式與逐筆機率完整。A類缺失、極端延遲、新O零預期量fallback、連假事件數有限，仍需獨立情境驗證。','本次受控比較可判斷預測增益，不能作因果結論。若CI跨零或實務差異小，保留較簡單的Baseline/ALL/Top20，並用2026凍結設定獨立驗證。']
    (out/'run_summary.md').write_text('\n'.join(text),encoding='utf-8-sig')

def run(cfg):
    out=outdir(cfg);pred=pd.read_parquet(out/'predictions.parquet')
    if not (out/'incremental_gain_vs_baselines.csv').exists():classification_tables(cfg,pred)
    logging.info('A comparison tables complete or reused')
    if not (out/'historical_demand_error.csv').exists():npq_diagnostics(cfg)
    logging.info('N/P/Q held-out diagnostics complete or reused')
    ablation=pd.read_csv(out/'metrics_by_fold.csv');ablation=ablation[ablation.configuration.isin(['M0','M2','B0_PQ','B1_P','B1_PQ','B2_PQ'])&ablation.algorithm.eq('lightgbm')]
    save_csv(ablation,out/'npq_ablation_metrics.csv')
    z=ablation[ablation.fold.ne('EXP2025')&ablation.segment.eq('ALL')&ablation.horizon.eq(30)].groupby('configuration').average_precision_2.mean()
    fig,ax=plt.subplots(figsize=(9,4));z.plot.bar(ax=ax,color=COLORS[0]);ax.set_ylabel('Y2 Average Precision');ax.set_title('B4　N/P/Q消融，2024三折平均AP2，h30 LightGBM\n固定Top20+OTHER，分組與其他特徵不變');save(fig,out,'fig_B4_npq_ablation')
    decisions(cfg,pred);events,detail=event_analysis(cfg,pred);examples(cfg,pred,events,detail);report(cfg)
    from .supplement import run as supplement
    supplement(cfg)
