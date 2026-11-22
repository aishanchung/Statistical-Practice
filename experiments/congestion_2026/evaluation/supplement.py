"""Additional diagnostics and concise Chinese conclusions. No model fitting."""
import json,logging
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from ..common import *
from .analysis import paired_intervals

def run(cfg):
    from ..features.engine import Engine,KINDS
    out=outdir(cfg)
    if (out/'q_calibration_by_origin.csv').exists() and (out/'npq_paired_uncertainty.csv').exists():
        write_conclusions(cfg,pd.read_csv(out/'q_calibration_by_origin.csv'))
        return
    engine=Engine(cfg);calibration=[];day_errors=[]
    for fold in cfg['folds']:
        condition=engine.condition(fold['start']);profile=engine.profile(fold['start']);origins=profile['origins'].tolist();index={o:i for i,o in enumerate(origins)}
        train=engine.con.execute(f'SELECT o,kind,slot//12 AS hour,lag,sum(n) AS n FROM m WHERE lag>=0 AND {condition} GROUP BY ALL').df()
        validation=engine.con.execute(f"SELECT o,kind,slot//12 AS hour,lag,sum(n) AS n FROM m WHERE lag>=0 AND depart_day>=DATE {lit(fold['start'])} AND depart_day<DATE {lit(fold['end'])} GROUP BY ALL").df()
        maxlag=int(train.lag.max());global_hist=np.bincount(train.lag.to_numpy(int),weights=train.n,minlength=maxlag+1);global_hist/=global_hist.sum()
        train_o={o:g for o,g in train.groupby('o')};train_oh={key:g for key,g in train.groupby(['o','hour'])};train_okh={key:g for key,g in train.groupby(['o','kind','hour'])}
        empty=train.iloc[:0]
        def hist(g):return np.bincount(g.lag.to_numpy(int),weights=g.n,minlength=maxlag+1)
        for (o,kind,hour),g in validation.groupby(['o','kind','hour']):
            qo=hist(train_o.get(o,empty))+cfg['npq']['Q_strength']*global_hist;qo/=qo.sum()
            q0=hist(train_oh.get((o,hour),empty))+cfg['npq']['Q_strength']*qo;q0/=q0.sum()
            observed_train=train_okh.get((o,kind,hour),empty)
            q2=hist(observed_train)+cfg['npq']['B2_Q_kind_strength']*q0;q2/=q2.sum()
            lag=g.lag.to_numpy(int);n=g.n.to_numpy(float);total=n.sum()
            true_median=float(g.sort_values('lag').loc[lambda f:f.n.cumsum().ge(total/2),'lag'].iloc[0])*5
            for name,probs in [('B0',q0),('B2',q2)]:
                assigned=np.zeros(len(g));inside=lag<=maxlag;assigned[inside]=probs[lag[inside]]
                predicted_median=int(np.searchsorted(np.cumsum(probs),.5))*5
                calibration.append({'fold':fold['id'],'version':name,'o':o,'hour':hour,'kind':kind,'train_O_hour_kind_support':observed_train.n.sum(),'validation_support':total,'mean_log_loss':float(-np.sum(n*np.log(np.maximum(assigned,1e-12)))/total),'median_train_minutes':predicted_median,'median_validation_minutes':true_median,'absolute_median_error_minutes':abs(predicted_median-true_median),'wasserstein_minutes':wasserstein_distance(np.arange(maxlag+1)*5,lag*5,u_weights=probs,v_weights=n),'validation_count_beyond_train_maxlag':int(n[~inside].sum()),'training_maxlag_minutes':maxlag*5,'low_train_support':observed_train.n.sum()<200})
        val=engine.con.execute(f"SELECT o,kind,slot,depart_day,sum(n) AS n FROM m WHERE depart_day>=DATE {lit(fold['start'])} AND depart_day<DATE {lit(fold['end'])} GROUP BY ALL").df()
        for (o,kind),g in val.groupby(['o','kind']):
            if o not in index:continue
            days=sum(k==kind and pd.Timestamp(fold['start'])<=d<pd.Timestamp(fold['end']) for d,k in engine.calendar.items())
            for version in ['B0','B1']:
                expected=profile[version+'_N_'+kind][:,index[o]].astype(float);e=expected[g.slot.to_numpy(int)];y=g.n.to_numpy(float)
                mae=(days*expected.sum()+np.sum(np.abs(y-e)-e))/(days*288)
                rmse=np.sqrt(max(0,(days*np.sum(expected**2)+np.sum(y*y-2*y*e))/(days*288)))
                day_errors.append({'fold':fold['id'],'o':o,'kind':kind,'version':version,'n_calendar_days':days,'n_daily_cells':days*288,'mae_daily_5min':mae,'rmse_daily_5min':rmse,'actual_total':y.sum(),'predicted_total':expected.sum()*days})
    engine.con.close();q=pd.DataFrame(calibration);save_csv(q,out/'q_calibration_by_origin.csv');save_csv(pd.DataFrame(day_errors),out/'n_daily_forecast_diagnostics.csv')
    support=pd.read_csv(out/'npq_sample_support.csv')
    summaries=[]
    for (fold,kind),g in support.groupby(['fold','kind']):summaries.append({'fold':fold,'kind':kind,'origins':len(g),'origins_zero_passages':int(g.train_N_B.eq(0).sum()),'origins_under_30_passages':int(g.train_N_B.lt(30).sum()),'total_departures':g.train_n.sum(),'total_passages':g.train_N_B.sum()})
    save_csv(pd.DataFrame(summaries),out/'npq_support_summary.csv')
    p=pd.read_parquet(out/'predictions.parquet',filters=[('algorithm','=','lightgbm'),('horizon','=',30),('fold','!=','EXP2025')]);uncertainty=[]
    for left,right in [('M2','B1_P'),('B0_PQ','B1_PQ'),('B0_PQ','B2_PQ'),('M0','B1_P'),('M0','B2_PQ')]:
        a=p[p.configuration.eq(left)].sort_values('prediction_time');b=p[p.configuration.eq(right)].sort_values('prediction_time')
        interval,units=paired_intervals(a,b,cfg)
        from sklearn.metrics import average_precision_score
        uncertainty.append({'baseline':left,'candidate':right,'metric':'AP2','paired_delta':average_precision_score(a.y==2,b.p2)-average_precision_score(a.y==2,a.p2),'lo95':interval[0],'hi95':interval[1],'bootstrap_holiday_blocks':units,'selection_period':'2024 OOF only'})
    save_csv(pd.DataFrame(uncertainty),out/'npq_paired_uncertainty.csv')
    write_conclusions(cfg,q)

def write_conclusions(cfg,q):
    out=outdir(cfg);ranking=pd.read_csv(out/'selection_2024_only.csv');a=ranking[ranking.algorithm.eq('lightgbm')].sort_values('configuration')
    b=pd.read_csv(out/'npq_ablation_metrics.csv');b=b[b.fold.ne('EXP2025')&b.horizon.eq(30)&b.segment.eq('ALL')].groupby('configuration')[['average_precision_2','macro_f1','log_loss']].mean()
    demand=pd.read_csv(out/'historical_demand_error.csv');demand=demand[demand.kind.eq('ALL')].groupby('configuration')[['mae','rmse']].mean()
    event=pd.read_csv(out/'event_detection_metrics.csv');event=event[event.algorithm.eq('lightgbm')&event.configuration.eq('M0')]
    c=pd.read_csv(out/'y1_model_comparison.csv');r=pd.read_csv(out/'rejection_workload.csv');r=r[r.algorithm.eq('lightgbm')&r.configuration.eq('M0')&r.horizon.eq(30)]
    qsum=q.groupby('version').apply(lambda g:pd.Series({'weighted_Q_logloss':np.average(g.mean_log_loss,weights=g.validation_support),'weighted_median_absolute_error_minutes':np.average(g.absolute_median_error_minutes,weights=g.validation_support),'weighted_Wasserstein_minutes':np.average(g.wasserstein_minutes,weights=g.validation_support)}),include_groups=False)
    save_csv(qsum.reset_index(),out/'q_calibration_summary.csv')
    report=['# 2026/11/22 統計實務研究進度與海報結果摘要','','## 已完成範圍','','GitHub main `98b742f759bdf30e444f073107dbe8bd35e3c3d5`及本機來源完成盤點。保留舊成果，未推送GitHub。正式Y獨立重建66,240格，差異0。原九組逐筆機率核心分數重算相符，舊模型權重檔未找到；本輪120個模型均另存權重、機率、設定與時間。','297個O是V3兩年瓶頸抵達假日母體，341個O是第一季完整假日出發母體，後者含85個零瓶頸通過O。範圍差異不能當資料錯誤，也不可直接把V3分子除以M06B分母。','','## A　O分群有無預測增益','','主要h30 LightGBM，以下為2024三折指標平均，所有配置同折同樣本。','',a[['configuration','average_precision_2','macro_f1','log_loss','n_features']].to_markdown(index=False),'','K=5的平均AP略高，但與Baseline差約.004，小於預先指定.005實務近似範圍。依較少欄位規則，LightGBM與Logistic都保留M0 Baseline。各項增額AP的日期區塊95%區間皆跨零，本輪不能宣稱Kmeans已有穩定預測優勢。ARI僅描述幾何，不作預測結論。','','## B　需求估計改善與分類改善不同','','抵達需求誤差為各折平均，單位輛／五分鐘，實際量限假日出發母體，不包含A類非假日出發。','',demand.to_markdown(),'','P+Q明顯降低歷史抵達需求RMSE，但改善後P/Q在AP、Macro-F1與Logloss並未一致勝過Baseline。B0本來就有P與Q pooling；B1/B2是有限增強，不是首次加入平滑。','',b.to_markdown(),'','Q的真正驗證期診斷如下，低頻O、小時、daytype支持度與訓練最大延遲外的旅次另列CSV。','',qsum.to_markdown(),'','P的總體樣本多，但O×小時條件的低頻格仍可能不穩。Q只以通過旅次估計，條件樣本更少。用分層Q可減少部分延遲誤差，是否穩健仍需結合按日期bootstrap與低支持格診斷；不能只憑某幾個O下結論。','本輪僅有需求與時間特徵的相關診斷，尚不足以證明VD已完全包含O資訊，也不足以把效果有限歸因為突發需求。當日即時O流量尚缺，兩項解釋都維持待驗假說。','','## C　Y1改善需要保護Y2','','以下2024為選參集合結果，未做外層巢狀獨立驗證；2025為凍結規則後的探索性回測。','',c[['split','method','precision_1','recall_1','f1_1','recall_2','delta_Y0_to_Y1','delta_Y2_to_Y1']].to_markdown(index=False),'','C1只改決策、不改機率，可提高Y1 F1且保留原Y2 Recall，但會增加Y0誤判Y1。C2的類別權重明顯降低Y2 Recall，因此不建議替換原模型。保留原始多類機率，C1作偏重局部壅塞的補充策略，待核定額外誤報代價後再決定正式採用。只有一支有效VD時Y1不可能，不能歸因為模型學不會。','','## D　15／30分鐘與警報代價','','同一未來30分事件窗、相同雙horizon有效起點、最新可取得VD(t−10)風險集。研究日分母以目標研究假日計，不額外把前晚的星期五算研究日。與11/12的current_state_known協定不同，故不直接混合命中數。','',event[['horizon','method','events','event_hits','clean_events','clean_hits','median_lead_minutes','false_episodes_per_day','total_episodes_per_day']].to_markdown(index=False),'','111個onset中只有23個前30分鐘完全無Y2的clean onset。必須區分新壅塞與短間隔再次進入Y2。15分鐘分類較準，30分鐘在同一事件窗提供較長提前量。成本決策增加命中並提高誤報，沒有無代價的Recall改善。','',r[['method','coverage','rejection_true_Y2','risk_wrongly_accepted_as_Y0','total_review_episodes_riskset','review_episodes_per_eligible_day']].to_markdown(index=False),'','D2降低錯誤宣稱暢通，但不增加Y2硬警報命中，R須複核。尚未定義R處理成本及實際人力流程，不能宣稱比D1營運成本更低。PNG包含2025/1/31 15:40成功事件與2025/2/15 12:30漏報事件，依固定規則挑選，沒有製造案例。','','## E　展示系統','','五頁Streamlit Dashboard使用真實保存機率。可選模型、日期、五分鐘、h15/h30及D0/D1/D2，可下載CSV/PNG，未來真值須開啟「事後答案」。明示Historical Replay / Offline Evaluation。VD及C/Y當前顯示以實際可用t−10資料為準，不偽裝成未發布的C(t)。','','## 下一步','','1. 主模型暫保留Baseline，O需求／K5作候選研究，不依2025改選K。','2. 補齊非假日出發A母體或獨立評估午夜偏差，建立真實資料發布時間。','3. 固定成本與人工複核作業規範，再判斷D1/D2是否值得部署。','4. 取得同口徑2026資料後，以凍結設定作正式獨立驗證。','5. 若進一步開發C3或新Q，不與本輪已固定結果混成同一模型。']
    (out/'research_conclusions_zh.md').write_text('\n'.join(report),encoding='utf-8-sig')
    with (out/'run_summary.md').open('a',encoding='utf8') as f:f.write('\n\n詳見 research_conclusions_zh.md 的需求與分類分離結論、日期區塊不確定性及人工採用限制。\n')
    with (out/'research_conclusions_zh.md').open('a',encoding='utf8') as f:
        f.write('\n\n## 額外可用性與截尾說明\n\n本機找到2024–2025逐日原始M06B CSV候選，含非研究假日。尚缺經完整性與可取得時間稽核、可直接供N/P/Q使用的非假日出發彙總，並非本機完全沒有原始候選。依本輪保持B0母體的控制變因要求，未把未稽核原始候選混入模型。\n\n事件警報只依當時機率與可觀測VD發出。若警報後的真實Y不足以完整確認事件窗，且沒有已知合格onset命中，該episode保留為outcome_censored，既不算誤報也不算正確。表中誤報率為已確認誤報／研究日，同時保留censored_alarm_episodes，不能把未觀測未來當作無事件。\n\nB2的Q距離與中位數誤差略降，條件Logloss並未下降。分布估計改善呈現指標差異，不能宣稱所有Q誤差都改善。\n')
if __name__=='__main__':
    cfg=load_config();setup(cfg);run(cfg)
