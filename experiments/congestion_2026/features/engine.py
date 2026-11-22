import logging,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from ..common import *
from ..data.contracts import assert_available

KINDS=('WEEKEND','LONGHOLIDAY');SLOTS=288
class Engine:
    def __init__(self,cfg):
        self.cfg=cfg;self.out=outdir(cfg);self.legacy=Path(cfg['legacy_root']);self.cache=self.out/'cache';self.cache.mkdir(exist_ok=True)
        self.con=duckdb.connect(config={'threads':cfg['threads'],'memory_limit':cfg['memory_limit']})
        paths=sorted((self.legacy/'pilot_2024_2025/results/monthly').glob('*.parquet'))
        self.con.execute('CREATE VIEW m AS SELECT * FROM read_parquet(['+','.join(map(lit,paths))+'])')
        self.completion=pd.read_csv(self.legacy/'pilot_2024_2025/results/cohort_completion.csv',parse_dates=['depart_day','latest_end'])
        self.con.register('completion',self.completion)
        cal=pd.read_csv(Path(cfg['data_root'])/cfg['inputs']['calendar'],dtype={'SourceDate':str})
        self.calendar=dict(zip(pd.to_datetime(cal.SourceDate,format='%Y%m%d'),cal.DayType))
        self.old=legacy_functions(cfg,'pilot_2024_2025/features.py',['aggregate_counts','probability','read_counts','history_before','estimate_history'])
        self.detail_manifest=json.loads((self.legacy/'o_features_grouping_20261111_v2/results/checkpoint_manifest.json').read_text(encoding='utf8'))
        self.timing=[];self.assignments=[];self.profiles=[];self.stability=[]
    def condition(self,cutoff):return self.old['history_before'](str(pd.Timestamp(cutoff)))
    def profile(self,cutoff):
        tag=pd.Timestamp(cutoff).strftime('%Y%m%d');file=self.cache/f'npq_{tag}.npz'
        if file.exists():return dict(np.load(file,allow_pickle=False))
        logging.info('Estimating legacy N/P/Q and controlled variants before %s',cutoff)
        condition=self.condition(cutoff)
        counts=self.old['read_counts'](self.con,condition)
        origins=sorted(counts.o.unique());idx={o:i for i,o in enumerate(origins)}
        _,b0,q=self.old['estimate_history'](self.con,condition,origins,1,self.cfg['npq']['P_strength'],self.cfg['npq']['Q_strength'])
        # estimate_history is reused verbatim with one column per O, then grouping.
        # Its appended OTHER column is always zero for training-observed origins.
        for a,b in b0.values():
            if not np.allclose(a[:,-1],0) or not np.allclose(b[:,-1],0):raise ValueError('Unexpected legacy OTHER mass')
        maxlag=int(q.lag.max());q_count=np.zeros((len(origins),24,maxlag+1))
        q_count[q.o.map(idx),q.hour,q.lag]=q.n
        global_q=q_count.sum(axis=(0,1));global_q/=global_q.sum()
        qo=q_count.sum(1)+self.cfg['npq']['Q_strength']*global_q[None,:];qo/=qo.sum(1,keepdims=True)
        q0=q_count+self.cfg['npq']['Q_strength']*qo[:,None,:];q0/=q0.sum(2,keepdims=True)
        prob,_=self.old['probability'](counts,1,self.cfg['npq']['B1_P_strength'],origins)
        prob=prob.p.unstack(['kind','bucket'])
        demand=self.con.execute(f'SELECT o,kind,slot,sum(n) AS n FROM m WHERE {condition} GROUP BY ALL').df()
        day_counts=self.con.execute(f'SELECT kind,count(DISTINCT depart_day) AS n FROM m WHERE {condition} GROUP BY kind').df().set_index('kind').n
        arr=np.zeros((2,len(origins),SLOTS))
        for k,kind in enumerate(KINDS):
            z=demand[demand.kind.eq(kind)];arr[k,z.o.map(idx),z.slot]=z.n
        pooled=arr.sum(0)/day_counts.sum()
        hour_parent=pooled.reshape(len(origins),24,12).mean(2).repeat(12,axis=1)
        qkind=self.con.execute(f'SELECT o,kind,slot//12 AS hour,lag,sum(n) AS n FROM m WHERE lag>=0 AND {condition} GROUP BY ALL').df()
        result={'origins':np.asarray(origins),'cutoff':np.asarray(str(cutoff))}
        for k,kind in enumerate(KINDS):
            result['B0_P_'+kind]=b0[kind][0][:,:-1].astype('float32')
            result['B0_PQ_'+kind]=b0[kind][1][:,:-1].astype('float32')
            result['B0_N_'+kind]=(arr[k]/float(day_counts[kind])).T.astype('float32')
            n=(arr[k]+self.cfg['npq']['B1_N_pseudo_days']*hour_parent)/(day_counts[kind]+self.cfg['npq']['B1_N_pseudo_days'])
            prob_hour=prob[kind].reindex(columns=range(24)).to_numpy().repeat(12,1)
            expected=n*prob_hour
            # Q B2 adds daytype conditioning with explicit shrinkage to B0 O/hour.
            z=qkind[qkind.kind.eq(kind)]
            q2=np.zeros_like(q0);q2[z.o.map(idx),z.hour,z.lag]=z.n
            q2+=self.cfg['npq']['B2_Q_kind_strength']*q0;q2/=q2.sum(2,keepdims=True)
            for version,distribution in [('B1',q0),('B2',q2)]:
                arrivals=np.zeros((SLOTS+maxlag,len(origins)))
                for o in range(len(origins)):
                    for hour in range(24):
                        slot=hour*12;arrivals[slot:slot+12+maxlag,o]+=np.convolve(expected[o,slot:slot+12],distribution[o,hour])
                if not np.allclose(expected.sum(),arrivals.sum(),rtol=1e-8):raise ValueError('N/P/Q convolution mass not conserved')
                result[version+'_P_'+kind]=expected.T.astype('float32')
                result[version+'_PQ_'+kind]=arrivals.astype('float32')
                result[version+'_N_'+kind]=n.T.astype('float32')
        np.savez_compressed(file,**result)
        mature=self.completion[self.completion.latest_end+pd.Timedelta(days=1)<pd.Timestamp(cutoff)]
        self.timing.append({'cutoff':cutoff,'history_last_departure':mature.depart_day.max(),'feature_available_at':mature.latest_end.max()+pd.Timedelta(days=1),'history_days':len(mature),'origins':len(origins),'maxlag_slots':maxlag,'complete_trip_extra_delay_days':1})
        save_csv(pd.DataFrame(self.timing),self.out/'history_timing_audit.csv')
        return result
    def group_table(self,cutoff,fold):
        cutoff=pd.Timestamp(cutoff)
        # Detailed monthly checkpoints do not retain departure dates. Reuse only
        # full months whose last completed trip plus latency is before cutoff.
        month_complete=self.completion.groupby(self.completion.depart_day.dt.strftime('%Y%m')).latest_end.max()+pd.Timedelta(days=1)
        months=set(month_complete[month_complete<cutoff].index)
        paths=[Path(a['path']) for a in self.detail_manifest if a['month'] in months]
        if not paths:raise ValueError('No fully mature monthly O detail')
        source='read_parquet(['+','.join(map(lit,paths))+'])'
        detail=self.con.execute(f"SELECT o,kind,hour,vehicle,sum(n) AS n,sum(n_b) AS n_b FROM {source} WHERE level=0 AND data_window='DESCRIPTIVE_2024' GROUP BY ALL").df()
        hist=self.con.execute(f"SELECT o,travel_seconds,sum(n_b) AS n_b FROM {source} WHERE level=1 AND data_window='DESCRIPTIVE_2024' GROUP BY ALL").df()
        days=self.completion[self.completion.depart_day.dt.strftime('%Y%m').isin(months)].depart_day.map(self.calendar).value_counts().to_dict()
        original={'smooth_global_trips':30,'smooth_probability_trips':300,'low_n':300,'low_passing':30}
        funcs=legacy_functions(self.cfg,'o_features_grouping_20261111_v2/features.py',['build_table'],{'CFG':original})
        table=funcs['build_table'](detail,hist,days,3).sort_values('o').reset_index(drop=True)
        table['history_before']=cutoff;table['complete_months']=','.join(sorted(months))
        save_csv(table,self.cache/f'o_features_{fold}.csv')
        geom=legacy_functions(self.cfg,'o_features_grouping_20261111_v2/groups.py',['geometry'])
        x,transforms=geom['geometry'](table,'none')
        mappings={'ALL':dict.fromkeys(table.o,'ALL')}
        top=table.sort_values(['N_B','o'],ascending=[False,True]).o.head(20).tolist()
        mappings['TOP20']={o:o if o in top else 'OTHER' for o in table.o}
        for k in [5,10,20]:
            km=KMeans(n_clusters=k,n_init=self.cfg['kmeans']['n_init'],random_state=self.cfg['seed']).fit(x)
            mapping={o:f'G{c:02d}' for o,c in zip(table.o,km.labels_)};mappings[f'K{k:02d}']=mapping
            alt=KMeans(n_clusters=k,n_init=self.cfg['kmeans']['n_init'],random_state=self.cfg['seed']+1).fit_predict(x)
            self.stability.append({'fold':fold,'history_before':cutoff,'K':k,'comparison':'seed_only_geometry','ARI':adjusted_rand_score(km.labels_,alt),'predictive_evidence':False})
        for name,mp in mappings.items():
            block=table.assign(group=table.o.map(mp),fold=fold,grouping=name)
            self.assignments.append(block[['fold','grouping','o','group','N_all','N_B','history_before','complete_months']])
            summary=block.groupby('group').agg(O_count=('o','size'),N_all=('N_all','sum'),N_B=('N_B','sum'),lag_median_mean=('lag_median','mean'),p_mean=('p','mean'))
            summary['passing_share']=summary.N_B/table.N_B.sum();summary['departure_share']=summary.N_all/table.N_all.sum()
            for col in table.filter(regex='^pi_B_'):
                summary[col]=block.groupby('group').apply(lambda g:float((g[col].fillna(0)*g.N_B).sum()/g.N_B.sum()) if g.N_B.sum()>0 else np.nan,include_groups=False)
            self.profiles.append(summary.reset_index().assign(fold=fold,grouping=name,history_before=cutoff))
        save_json({'cutoff':cutoff,'mature_months':sorted(months),'transform':transforms,'groups':mappings,'new_O_fallback':'OTHER; prior training forecast has zero expected demand for unseen O'},self.cache/f'group_transform_{fold}.json')
        return mappings
    def block(self,profile,frame,version,mode,mapping):
        origins=profile['origins'].tolist();groups=sorted(set(mapping.values())|{'OTHER'})
        # K groups plus reserved OTHER. Since unseen-O demand is zero, OTHER is a
        # constant zero feature for Kmeans. Keep only columns with training mass.
        if 'OTHER' not in mapping.values():groups.remove('OTHER')
        if any(o not in mapping for o in origins) and 'OTHER' not in groups:groups.append('OTHER')
        matrix=np.zeros((len(origins),len(groups)))
        for i,o in enumerate(origins):matrix[i,groups.index(mapping.get(o,'ALL' if set(mapping.values())=={'ALL'} else 'OTHER'))]=1
        values=np.zeros((len(frame),len(groups)),dtype='float32')
        indices=pd.Series(np.arange(len(frame)),index=frame.index)
        for day,part in frame.groupby('day'):
            kind=self.calendar[pd.Timestamp(day)];source=profile[f'{version}_{mode}_{kind}'];arr=source[:SLOTS].copy()
            if mode=='PQ':
                max_days=max((len(profile[f'{version}_{mode}_{k}'])-1)//SLOTS for k in KINDS)
                for previous in range(1,max_days+1):
                    k=self.calendar.get(pd.Timestamp(day)-pd.Timedelta(days=previous))
                    if k:
                        carry=profile[f'{version}_{mode}_{k}'][previous*SLOTS:(previous+1)*SLOTS];arr[:len(carry)]+=carry
            selected=arr[part.slot.to_numpy(int)]
            aggregated=selected@matrix
            if not np.allclose(aggregated.sum(1),selected.sum(1),rtol=2e-6,atol=1e-5):raise ValueError('Group demand differs from ALL')
            values[indices.loc[part.index].to_numpy()]=aggregated
        return values,groups
    def make_fold(self,fold,h):
        key=f'{fold["id"]}_h{h}';dest=self.cache/f'features_{key}.parquet'
        if dest.exists() and dest.with_suffix('.json').exists():return
        src=self.legacy/f'horizon_15_30_20261111_v1/results/features_h{h}.csv'
        raw=pd.read_csv(src,parse_dates=['origin','target','day'])
        base=[c for c in raw if c.startswith(('cal_','vd_'))]
        raw=raw[['origin','target','day','slot','kind','y','n_valid_vd',*base]].rename(columns={'origin':'prediction_time','target':'target_time'})
        raw=raw.loc[raw.prediction_time.ge(self.cfg['warmup_before']) & raw.prediction_time.lt(fold['end'])].reset_index(drop=True)
        history_cut=pd.Timestamp(fold['start'])
        mapping=self.group_table(history_cut,fold['id'])
        raw['fold']=fold['id'];raw['split']=np.where(raw.prediction_time.ge(history_cut),'evaluation','training')
        raw['horizon']=h;raw['Y_version']=self.cfg['Y']['version']
        raw['vd_feature_available_at']=raw.prediction_time # lag10 + assumed10min latency
        raw['history_cutoff']=pd.NaT
        columnsets={'M0':base};specs=[('M1','B0','P','ALL'),('M2','B0','P','TOP20'),('M3','B0','P','K05'),('M4','B0','P','K10'),('M5','B0','P','K20'),('B0_PQ','B0','PQ','TOP20'),('B1_P','B1','P','TOP20'),('B1_PQ','B1','PQ','TOP20'),('B2_PQ','B2','PQ','TOP20')]
        # Training N/P/Q are out-of-date causal monthly estimates. Evaluation
        # remains frozen at the fold training cutoff. Group transform uses only
        # training-history months; no validation period enters it.
        for period,part in raw.groupby(raw.prediction_time.dt.to_period('M')):
            cutoff=min(pd.Timestamp(period.start_time),history_cut)
            profile=self.profile(str(cutoff.date()));raw.loc[part.index,'history_cutoff']=cutoff
            if not (cutoff<=part.prediction_time.min()):raise ValueError('Demand history from future')
            for name,version,mode,method in specs:
                values,groups=self.block(profile,part,version,mode,mapping[method])
                cols=[name+'__'+g for g in groups]
                for c in cols:
                    if c not in raw:raw[c]=np.float32(0)
                raw.loc[part.index,cols]=values
                columnsets[name]=base+sorted(set(columnsets.get(name,[]))-set(base)|set(cols))
        assert_available(raw.history_cutoff,raw.prediction_time)
        raw.to_parquet(dest,index=False)
        save_json({'columns':columnsets,'group_history_before':str(history_cut),'causal_NPQ':'monthly training histories; frozen evaluation cutoff','config_signature':signature(self.cfg),'horizon':h},dest.with_suffix('.json'))
        logging.info('Saved common feature frame %s rows=%s',key,len(raw));del raw;gc.collect()
    def run(self):
        folds=[*self.cfg['folds'],{'id':'EXP2025','start':'2025-01-01','end':'2026-01-01'}]
        for fold in folds:
            for h in [30,15]:self.make_fold(fold,h)
        if self.assignments:save_csv(pd.concat(self.assignments,ignore_index=True).drop_duplicates(['fold','grouping','o']),self.out/'kmeans_cluster_assignments.csv')
        if self.profiles:save_csv(pd.concat(self.profiles,ignore_index=True).drop_duplicates(['fold','grouping','group']),self.out/'cluster_profile.csv')
        if self.stability:save_csv(pd.DataFrame(self.stability).drop_duplicates(['fold','K']),self.out/'clustering_stability.csv')
        self.con.close()
