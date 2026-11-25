"""Binary congestion pilot: cached inputs only; python -m ... --smoke / --resume."""
from pathlib import Path
from types import SimpleNamespace
import argparse, ast, hashlib, json, zipfile, sys, shutil
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, precision_recall_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Load only the unchanged owning cache aggregation method. DuckDB is not
# needed for cached arrays and its native DLL is blocked by local application policy.
ENGINE_PATH=Path(__file__).resolve().parent/'features/engine.py'
_tree=ast.parse(ENGINE_PATH.read_text(encoding='utf-8-sig'))
_class=next(n for n in _tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
_block=next(n for n in _class.body if isinstance(n,ast.FunctionDef) and n.name=='block')
_ns={'np':np,'pd':pd,'KINDS':('WEEKEND','LONGHOLIDAY'),'SLOTS':288}
exec(compile(ast.Module(body=[_block],type_ignores=[]),str(ENGINE_PATH),'exec'),_ns)
cache_block=_ns['block']
from experiments.congestion_2026.data.contracts import train_mask, assert_train_validation, assert_probabilities

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/congestion_2026/outputs'
GROUPS=['ALL','TOP20','K05','GMM05','TS10']
SITES=['69.510','65.990','63.990','61.670']
GRID=np.round(np.arange(.2,.801,.05),2)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def js(x,p):
    t=Path(str(p)+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8');t.replace(p)
def csv(x,p):
    t=Path(str(p)+'.tmp');pd.DataFrame(x).to_csv(t,index=False,encoding='utf-8-sig');t.replace(p)
def parquet(x,p):
    t=Path(str(p)+'.tmp');x.to_parquet(t,index=False);t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def model(cfg,binary=False,weight=1):
    return LGBMClassifier(**({'objective':'binary'} if binary else {'objective':'multiclass','num_class':3}),
        learning_rate=.05,deterministic=True,force_col_wise=True,verbosity=-1,random_state=cfg['seed'],n_jobs=cfg['threads'],
        **cfg['models']['lightgbm'],**({'class_weight':{0:1,1:weight}} if binary else {}))
def demand(frame,cache,calendar,mapping):
    parts=[]
    for cutoff,part in frame.groupby('history_cutoff'):
        with np.load(cache/f'npq_{pd.Timestamp(cutoff):%Y%m%d}.npz',allow_pickle=False) as z:
            a,g=cache_block(SimpleNamespace(calendar=calendar),dict(z),part,'B0','PQ',mapping)
        parts.append(pd.DataFrame(a,index=part.index,columns=['demand__'+str(v) for v in g]))
    return pd.concat(parts).reindex(frame.index).fillna(0.).sort_index(axis=1)
def labels(grid,cut):
    s=grid[['speed_'+x for x in SITES]].to_numpy(float);o=grid[['occ_'+x for x in SITES]].to_numpy(float)
    thresholds=[]
    for i,site in enumerate(SITES):
        mask=(grid.time<pd.Timestamp(cut)).to_numpy()&np.isfinite(s[:,i])&(s[:,i]>=80)&np.isfinite(o[:,i])&(o[:,i]>=0)&(o[:,i]<=100)
        if not mask.any(): raise ValueError('No historical high-speed reference')
        thresholds.append(np.quantile(o[mask,i],.95))
    valid=np.isfinite(s)&(s>=0); vo=np.isfinite(o)&(o>=0)&(o<=100); low=valid&(s<40)
    known=valid&(~low|vo);positive=(low&vo&(o>=np.array(thresholds))).any(1)
    result={p:np.where(positive,1.,np.where(getattr(known,p)(1),0.,np.nan)) for p in ['all','any']}
    return {'Z0':grid.Z_original.to_numpy(), 'Z1_strict':result['all'],'Z1_available':result['any']},thresholds

def events(times,z):
    a=np.asarray(z);t=pd.DatetimeIndex(times).as_unit('ns')
    pos=a==1; consecutive=np.r_[False,np.diff(t.asi8)==pd.Timedelta(minutes=5).value]
    starts=np.flatnonzero(pos&~(np.r_[False,pos[:-1]]&consecutive));rows=[]
    for start in starts:
        end=start
        while end+1<len(a) and pos[end+1] and consecutive[end+1]:end+=1
        left=start==0 or not consecutive[start] or not np.isfinite(a[start-1]); right=end+1==len(a) or not consecutive[end+1] or not np.isfinite(a[end+1])
        clean=start>=6 and np.all(consecutive[start-5:start+1]) and np.all(a[start-6:start]==0)
        rows.append({'onset':t[start],'minutes_observed':(end-start+1)*5,'left_censored':left,'right_censored':right,'clean_30m':clean})
    return rows

def audit_labels(grid,folds,out):
    rows=[];ev=[];thresholds=[];aligned={}
    for f in folds:
        fid=f['id'];labs,th=labels(grid,f['start']); val=grid.time.ge(f['start'])&grid.time.lt(f['end'])
        aligned[fid]=pd.DataFrame({'target_time':grid.time,**labs})
        for site,t in zip(SITES,th):thresholds.append({'fold':fid,'history_before':f['start'],'site':site,'Q95':float(t)})
        for name,a in labs.items():
            a=a[val]; orig=grid.loc[val,'Y_original'].to_numpy(); z0=labs['Z0'][val]; common=np.isfinite(a)&np.isfinite(z0)
            e=events(grid.loc[val,'time'],a);ev.extend([{'fold':fid,'label':name,**r} for r in e])
            r={'fold':fid,'label':name,'n_total':len(a),'n_valid':int(np.isfinite(a).sum()),'missing':int(np.isnan(a).sum()),'missing_fraction':float(np.isnan(a).mean()),'positive':int((a==1).sum()),'common_valid':int(common.sum()),'changed_common':int(np.sum(a[common]!=z0[common])), 'original_Y0_to_positive':int(np.sum((orig==0)&(a==1))), 'event_count':len(e)}
            dur=[x['minutes_observed'] for x in e];r.update({'observed_event_median_min':float(np.median(dur)) if dur else np.nan,'observed_event_p90_min':float(np.quantile(dur,.9)) if dur else np.nan})
            for k in [1,2]:
                r.update({f'Y{k}_total':int((orig==k).sum()),f'Y{k}_retained_positive':int(((orig==k)&(a==1)).sum()),f'Y{k}_unknown':int(((orig==k)&np.isnan(a)).sum()),f'Y{k}_to_negative':int(((orig==k)&(a==0)).sum())})
            rows.append(r)
    csv(rows,out/'label_audit.csv');csv(ev,out/'label_events.csv');csv(thresholds,out/'occupancy_thresholds.csv')
    return aligned

def metrics(y,d,s,days):
    y=np.asarray(y,int);d=np.asarray(d,int);m=confusion_matrix(y,d,labels=[0,1]);tn,fp,fn,tp=m.ravel()
    p,r,f,n=precision_recall_fscore_support(y,d,labels=[0,1],zero_division=0)
    row={'n':len(y),'days':int(pd.Series(days).nunique()),'positive_n':int(y.sum()),'miss':int(fn),'false_alarm':int(fp),'tn':int(tn),'tp':int(tp),'false_alarm_bins_per_day':float(fp/pd.Series(days).nunique()),'AP':float(average_precision_score(y,s)) if y.sum() else np.nan}
    for k in [0,1]:row.update({f'precision_{k}':p[k],f'recall_{k}':r[k],f'f1_{k}':f[k],f'support_{k}':int(n[k])})
    return row,m

def evaluate(pred,fid,g,method,aligned):
    z=pred.merge(aligned,on='target_time',how='left',validate='one_to_one');rows=[];cms=[];sev=[]
    cases={'Z0':('Z0',z.Z0.notna())}
    for lab in ['Z1_strict','Z1_available']:
        common=z.Z0.notna()&z[lab].notna()
        cases[lab]=(lab,common);cases['Z0_common_'+lab]=('Z0',common)
    for name,(label,common) in cases.items():
        for threshold in ([None] if method=='B01' else GRID):
            d=(z.hard.to_numpy() if method=='B01' else (z.score>=threshold).to_numpy()).astype(int)
            peak=z.target_time.dt.hour.between(14,19)
            segments={'ALL':np.ones(len(z),bool),'14_20':peak,'OTHER_HOURS':~peak,'WEEKEND':z.kind.eq('WEEKEND'),'LONGHOLIDAY':z.kind.eq('LONGHOLIDAY')}
            for segment,mask in segments.items():
                keep=np.asarray(common&mask);a=z.loc[keep]
                if not len(a):continue
                row,mat=metrics(a[label],d[keep],a.score,a.day)
                key={'fold':fid,'grouping':g,'method':method,'label':name,'threshold':-1. if threshold is None else float(threshold),'segment':segment,'evaluation_type':'2025_exploratory' if fid=='EXP2025' else '2024_development_validation'}
                rows.append({**key,**row})
                for i in range(2):
                    for j in range(2):cms.append({**key,'actual':i,'predicted':j,'count':int(mat[i,j])})
                for k in [1,2]:
                    kk=a.y.eq(k).to_numpy();count=int(kk.sum());miss=int(np.sum((d[keep]==0)&kk))
                    sev.append({**key,'original_Y':k,'support':count,'miss':miss,'miss_fraction':miss/count if count else np.nan})
    return rows,cms,sev

def bootstrap(cells,out,reps):
    rng=np.random.default_rng(20261125);rows=[]
    for period,fids in [('2024',['F1','F2','F3']),('2025_exploratory',['EXP2025'])]:
        def joined(g,m):return pd.concat([cells[(f,g,m)] for f in fids],ignore_index=True).sort_values('target_time').reset_index(drop=True)
        reference=joined('ALL','B03')
        candidates=[(g,'B03','ALL','B03') for g in GROUPS if g!='ALL']+[(g,m,g,'B01') for g in GROUPS for m in ['B02','B03','B04_w2','B04_w4']]
        for g,m,rg,rm in candidates:
            a=joined(rg,rm);b=joined(g,m)
            if not a[['target_time','y']].equals(b[['target_time','y']]):raise ValueError('Bootstrap pairing mismatch')
            days=pd.DatetimeIndex(a.day).normalize();unique=days.unique().sort_values();ids=np.r_[0,np.cumsum(np.diff(unique.as_unit('ns').asi8)>pd.Timedelta(days=1).value)]
            blocks=[np.flatnonzero(days.isin(unique[ids==i])) for i in np.unique(ids)]
            day_sizes=a.groupby('day').day.transform('size').to_numpy();y=a.y.to_numpy()>0; da=a.hard.to_numpy() if rm=='B01' else (a.score.to_numpy()>=.5);db=b.hard.to_numpy() if m=='B01' else (b.score.to_numpy()>=.5)
            def stat(idx):
                yy=y[idx];aa=da[idx];bb=db[idx]
                def v(d):
                    tp=np.sum(yy&d);fp=np.sum(~yy&d);fn=np.sum(yy&~d)
                    return np.array([tp/(tp+fp) if tp+fp else np.nan,tp/(tp+fn) if tp+fn else np.nan,fp/np.sum(1. / day_sizes[idx]),fn])
                return v(bb)-v(aa)
            observed=stat(np.arange(len(y)));sim=np.array([stat(np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])) for _ in range(reps)])
            for j,metric in enumerate(['precision_1','recall_1','false_alarm_bins_per_day','miss']):
                good=sim[:,j][np.isfinite(sim[:,j])];lo,hi=np.quantile(good,[.025,.975]) if len(good) else [np.nan,np.nan]
                rows.append({'period':period,'grouping':g,'method':m,'reference_grouping':rg,'reference_method':rm,'metric':metric,'difference':observed[j],'lower_95':lo,'upper_95':hi,'repetitions':len(good),'date_blocks':len(blocks),'interpretation':'exploratory conditional on fitted models; fixed threshold 0.5; candidate minus reference'})
    csv(rows,out/'paired_effect.csv')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'binary_congestion_pilot');ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--reuse-models-from',type=Path);ap.add_argument('--bootstrap-reps',type=int,default=300);args=ap.parse_args()
    out=args.out.resolve();src=BASE/'workpackages_0ABCDE_v1';old=BASE/'grouping_pilot_v1';eda=BASE/'speed_occupancy_eda_v1';cache=src/'cache'
    if out in [src,old,eda] or any(x in out.parents for x in [src,old,eda]):raise ValueError('Separate output required')
    out.mkdir(parents=True,exist_ok=True)
    for d in ['cells','models','audit','figures']: (out/d).mkdir(exist_ok=True)
    cfg=read(src/'run_config.json')['config'];folds=cfg['folds']+[{'id':'EXP2025',**cfg['exploratory']}]
    if args.smoke:folds=folds[:1]
    groups=GROUPS[:1] if args.smoke else GROUPS
    inputs=[Path(__file__),src/'run_config.json',old/'cluster_assignments.csv',eda/'candidate_Y_grid.parquet',eda/'tables/04_candidate_thresholds_by_fold.csv',ROOT/'experiments/congestion_2026/features/engine.py',ROOT/'experiments/congestion_2026/data/contracts.py']
    for f in folds:
        fid=f['id'];inputs.extend([cache/f'features_{fid}_h30.parquet',cache/f'features_{fid}_h30.json',cache/f'group_transform_{fid}.json'])
        inputs.extend([old/'cells'/f'{fid}_h30_lightgbm_{g}_PQ_MULTI.parquet' for g in groups])
    inputs.extend(sorted(cache.glob('npq_*.npz')))
    manifest={'version':1,'smoke':args.smoke,'bootstrap_reps':args.bootstrap_reps,'threshold_grid':GRID.tolist(),'input_sha256':{str(p):sha(p) for p in inputs}}
    if args.reuse_models_from:
        prior=args.reuse_models_from.resolve(); previous_manifest=read(prior/'manifest.json')
        before={k:v for k,v in previous_manifest['input_sha256'].items() if k!=str(Path(__file__))}
        after={k:v for k,v in manifest['input_sha256'].items() if k!=str(Path(__file__))}
        if before!=after or previous_manifest['smoke']!=args.smoke:raise ValueError('Reused model inputs differ')
        if sha(prior/'source_script.py')!=previous_manifest['input_sha256'][str(Path(__file__))]:raise ValueError('Missing original source snapshot')
        artifacts=[p for folder in ['cells','models','audit'] for p in (prior/folder).iterdir() if p.suffix in ['.parquet','.joblib','.json']]
        manifest['reused_model_artifacts']={str(p):sha(p) for p in artifacts}
        manifest['reused_source_manifest_sha256']=sha(prior/'manifest.json')
    mp=out/'manifest.json'
    if mp.exists() and read(mp)!=manifest:raise ValueError('Source or settings changed: use new --out')
    if mp.exists() and not args.resume:raise ValueError('Use --resume for existing run')
    js(manifest,mp)
    if args.reuse_models_from:
        for p in artifacts:
            dest=out/p.parent.name/p.name
            if dest.exists():
                if sha(dest)!=sha(p):raise ValueError('Reused artifact differs')
            else:shutil.copy2(p,dest)
    grid=pd.read_parquet(eda/'candidate_Y_grid.parquet'); aligned=audit_labels(grid,folds,out)
    # Fold-specific Q95s must reproduce prior historical-reference EDA, never use full-2024 in 2024 folds.
    th=pd.read_csv(eda/'tables/04_candidate_thresholds_by_fold.csv');ours=pd.read_csv(out/'occupancy_thresholds.csv')
    for _,r in ours.iterrows():
        q=th[th.fold.eq(r.fold)&th.site.astype(str).eq(str(r.site))&th['quantile'].eq(.95)]
        assert len(q)==1 and np.isclose(q.threshold.iloc[0],r.Q95)
    cal=pd.read_csv(Path(cfg['data_root'])/cfg['inputs']['calendar'],dtype={'SourceDate':str});calendar=dict(zip(pd.to_datetime(cal.SourceDate,format='%Y%m%d'),cal.DayType))
    assignments=pd.read_csv(old/'cluster_assignments.csv',dtype={'o':str});prepared={};audits=[]
    # Audit every requested old model before any new binary fit.
    for f in folds:
        fid=f['id'];frame=pd.read_parquet(cache/f'features_{fid}_h30.parquet').reset_index(drop=True);meta=read(cache/f'features_{fid}_h30.json')
        fit=frame.loc[train_mask(frame,f['start'],30,cfg['warmup_before'])];val=frame.loc[frame.prediction_time.ge(f['start'])&frame.prediction_time.lt(f['end'])&frame.y.notna()]
        assert_train_validation(fit,val,30)
        assert (frame.target_time-frame.prediction_time).eq(pd.Timedelta(minutes=30)).all()
        assert frame.history_cutoff.le(frame.prediction_time).all() and frame.vd_feature_available_at.le(frame.prediction_time).all()
        expected=grid.set_index('time').Y_original.reindex(frame.target_time).to_numpy();assert np.array_equal(expected,frame.y.to_numpy(),equal_nan=True)
        base=[c for c in meta['columns']['M1'] if not c.startswith('M1__')];assert all(c.startswith(('cal_','vd_')) and c!='vd_feature_available_at' for c in base)
        for g in groups:
            tag=f'{fid}_{g}';z=assignments[assignments.fold.eq(fid)&assignments.grouping.eq(g)];assert not z.o.duplicated().any()
            x=pd.concat([frame[base],demand(frame,cache,calendar,dict(zip(z.o,z.group)))],axis=1)
            # Sum all group demand equals ALL at every target row.
            if g=='ALL':total=x.filter(like='demand__').sum(axis=1).to_numpy()
            else:assert np.allclose(x.filter(like='demand__').sum(axis=1),total,rtol=2e-5,atol=2e-4)
            if g=='TOP20':
                original=frame[[c for c in meta['columns']['B0_PQ'] if c.startswith('B0_PQ__')]].copy();original.columns=['demand__'+c.split('__',1)[1] for c in original]
                assert np.allclose(original.reindex(columns=x.filter(like='demand__').columns,fill_value=0),x.filter(like='demand__'),rtol=2e-5,atol=2e-4)
            previous=pd.read_parquet(old/'cells'/f'{fid}_h30_lightgbm_{g}_PQ_MULTI.parquet')
            for col in ['prediction_time','target_time','y']:assert np.array_equal(previous[col],val[col])
            assert_probabilities(previous[['p0','p1','p2']])
            check=out/'audit'/f'{tag}.json'
            if check.exists():record=read(check)
            else:
                mm=model(cfg);mm.fit(x.loc[fit.index],fit.y.astype(int));p=mm.predict_proba(x.loc[val.index]);err=float(np.max(np.abs(p-previous[['p0','p1','p2']].to_numpy())))
                if err>1e-10:raise ValueError(f'Old probabilities not reproduced {tag}: {err}')
                record={'fold':fid,'grouping':g,'train_n':len(fit),'validation_n':len(val),'features':list(x.columns),'max_probability_difference':err,'passed':True};js(record,check)
            audits.append(record);prepared[(fid,g)]=(x,fit,val,previous)
            print('AUDIT PASS',tag,len(fit),len(val),flush=True)
    csv(audits,out/'reproduction_audit.csv');print('ALL OLD MODELS REPRODUCED; binary training starts',flush=True)
    cells={};selection={}
    for f in folds:
        fid=f['id']
        # Freeze 2024 OOF exploratory thresholds before reading 2025 scores.
        if fid=='EXP2025':
            for g in groups:
                for method in ['B02','B03','B04_w2','B04_w4']:
                    a=pd.concat([cells[(ff['id'],g,method)] for ff in cfg['folds']]);zz=a.y.gt(0).to_numpy();candidates=[]
                    for t in GRID:
                        row,_=metrics(zz,a.score>=t,a.score,a.day)
                        if row['precision_1']>=.8:candidates.append((row['recall_1'],float(t)))
                    selection[g+'_'+method]=max(candidates)[1] if candidates else None
            js({'source':'2024 pooled OOF development exploration; not unbiased selected validation','uses_2025':False,'precision_min':.8,'thresholds':selection},out/'frozen_thresholds_2024.json')
        for g in groups:
            x,fit,val,previous=prepared[(fid,g)];a=previous[['prediction_time','target_time','day','kind','y']].copy();a['score']=previous.p1+previous.p2;a['hard']=(previous[['p0','p1','p2']].to_numpy().argmax(1)>0).astype(int)
            for method in ['B01','B02','B03','B04_w2','B04_w4']:
                path=out/'cells'/f'{fid}_{g}_{method}.parquet'
                if path.exists():pred=pd.read_parquet(path)
                elif method in ['B01','B02']:pred=a.copy();parquet(pred,path)
                else:
                    w={'B03':1,'B04_w2':2,'B04_w4':4}[method];mm=model(cfg,True,w);mm.fit(x.loc[fit.index],fit.y.gt(0).astype(int));pred=a.copy();pred['score']=mm.predict_proba(x.loc[val.index])[:,1];pred['hard']=(pred.score>=.5).astype(int)
                    joblib.dump(mm,out/'models'/f'{fid}_{g}_{method}.joblib',compress=3);parquet(pred,path)
                for col in ['prediction_time','target_time','y']:assert np.array_equal(pred[col],val[col])
                assert np.isfinite(pred.score).all() and pred.score.between(0,1).all()
                cells[(fid,g,method)]=pred
                print('CHECKPOINT',fid,g,method,flush=True)
    rows=[];cms=[];sev=[]
    for (fid,g,m),pred in cells.items():
        rr,cc,ss=evaluate(pred,fid,g,m,aligned[fid]);rows.extend(rr);cms.extend(cc);sev.extend(ss)
    metrics_df=pd.DataFrame(rows);csv(metrics_df,out/'threshold_sweep.csv');csv(cms,out/'confusion.csv');csv(sev,out/'original_severity_miss.csv')
    fixed=metrics_df[metrics_df.threshold.isin([-.1,-1.,.5])];csv(fixed,out/'metrics_by_fold.csv')
    keys=['grouping','method','label','threshold','segment'];dev=fixed[fixed.fold.ne('EXP2025')]
    summary=dev.groupby(keys,as_index=False).mean(numeric_only=True);summary['aggregation']='arithmetic mean of fold metrics; n is mean fold size';summary['fold_count']=dev.groupby(keys).size().to_numpy();summary['n_total']=dev.groupby(keys).n.sum().to_numpy();csv(summary,out/'comparison_2024.csv')
    csv(fixed[fixed.fold.eq('EXP2025')],out/'comparison_2025_exploratory.csv')
    feasible=[]
    for (g,m,fid),a in metrics_df[metrics_df.label.eq('Z0')&metrics_df.segment.eq('ALL')&metrics_df.method.ne('B01')&metrics_df.fold.ne('EXP2025')].groupby(['grouping','method','fold']):
        b=a[a.precision_1.ge(.8)].sort_values(['recall_1','threshold'],ascending=[False,False]);feasible.append({'grouping':g,'method':m,'fold':fid,'feasible':len(b)>0,'threshold':b.threshold.iloc[0] if len(b) else np.nan,'recall_1':b.recall_1.iloc[0] if len(b) else np.nan,'status':'2024 same-validation exploratory threshold comparison' if fid!='EXP2025' else '2025 prespecified-grid description only; never selection'})
    csv(feasible,out/'precision80_exploration.csv')
    frozen=[]
    for key,t in selection.items():
        g,m=key.split('_',1)
        if t is None:frozen.append({'grouping':g,'method':m,'feasible_2024':False});continue
        a=metrics_df[metrics_df.fold.eq('EXP2025')&metrics_df.grouping.eq(g)&metrics_df.method.eq(m)&metrics_df.label.eq('Z0')&metrics_df.threshold.eq(t)]
        frozen.extend(a.assign(feasible_2024=True).to_dict('records'))
    csv(frozen,out/'frozen_threshold_2025_results.csv')
    pr=[]
    for period,fids in [('2024',['F1','F2','F3']),('2025_exploratory',['EXP2025'])]:
        fids=[f for f in fids if any(k[0]==f for k in cells)]
        if not fids:continue
        fig,axes=plt.subplots(1,len(groups),figsize=(4*len(groups),4),squeeze=False)
        for ax,g in zip(axes[0],groups):
            for m in ['B02','B03','B04_w2','B04_w4']:
                a=pd.concat([cells[(fid,g,m)] for fid in fids]);p,r,t=precision_recall_curve(a.y>0,a.score);ax.plot(r,p,label=m)
                pr.extend({'period':period,'grouping':g,'method':m,'precision':pp,'recall':rr,'score_threshold':tt} for pp,rr,tt in zip(p[:-1],r[:-1],t))
            ax.axhline(.8,color='gray',ls=':');ax.set(xlabel='Recall',ylabel='Precision',title=g,xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
        fig.suptitle(period+' exact t+30 congestion');fig.tight_layout();fig.savefig(out/'figures'/f'PR_{period}.png',dpi=160);plt.close(fig)
    # Detailed PR coordinates stay local with row-level scores.
    csv(pr,out/'cells/pr_curves.csv')
    if not args.smoke:bootstrap(cells,out,args.bootstrap_reps)
    else:csv(pd.DataFrame(columns=['period','metric','difference','lower_95','upper_95']),out/'paired_effect.csv')
    # Descriptive current-state strata use observed truth only after prediction; never predictors or deployment gates.
    strata=[]
    for (fid,g,m),a in cells.items():
        now=grid.set_index('time').Z_original.reindex(a.prediction_time).to_numpy()
        for name,mask in [('ongoing_at_t',now==1),('not_congested_at_t',now==0),('unknown_at_t',~np.isfinite(now))]:
            if not mask.any():continue
            b=a.loc[mask];d=b.hard if m=='B01' else b.score.ge(.5);r,_=metrics(b.y>0,d,b.score,b.day);strata.append({'fold':fid,'grouping':g,'method':m,'stratum':name,**r})
    csv(strata,out/'current_state_strata.csv')
    passed=len(cells)==len(folds)*len(groups)*5 and len(audits)==len(folds)*len(groups)
    acceptance={'passed':passed,'scope':'smoke' if args.smoke else 'full_pilot','expected_prediction_cells':len(folds)*len(groups)*5,'completed_prediction_cells':len(cells),'new_binary_models':len(folds)*len(groups)*3,'reproduced_three_class_models':len(audits),'all_old_models_reproduced_before_binary_training':True,'fold_history_occupancy_threshold_verified':True,'unknown_never_imputed_negative':True,'fixed_threshold_validation':True,'selected_threshold_2024_is_exploratory':True,'2025_exploratory_only':True,'no_raw_trip_scan':True,'no_M0':True,'weighted_scores_not_calibrated':True,'Z1_sensitivity':'same fitted Z0 models; common valid samples; no Z1 retraining','event_warning_evaluation':'not performed; observed durations and t-state strata descriptive only','bootstrap':'conditional on trained models; date blocks; exploratory' if not args.smoke else 'not in smoke'}
    js(acceptance,out/'acceptance_report.json')
    print('DONE',out,flush=True)

if __name__=='__main__':main()
