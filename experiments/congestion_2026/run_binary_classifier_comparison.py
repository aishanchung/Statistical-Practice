"""Fixed Z0/PQ/h30 classifier comparison. Read existing caches; never raw trips."""
from pathlib import Path
import argparse, importlib.metadata, json, sys, time, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from experiments.congestion_2026.run_binary_congestion_pilot import (ROOT,BASE,GROUPS,GRID,read,js,csv,parquet,sha,demand,metrics,train_mask,assert_train_validation,assert_probabilities)

SOURCE=BASE/'binary_congestion_pilot'
CACHE=BASE/'workpackages_0ABCDE_v1/cache'
CONFIG=BASE/'workpackages_0ABCDE_v1/run_config.json'
LEGACY_XGB=ROOT.parent/'M1B_O_PILOT/pilot_2024_2025/.model_deps'
try:
    import xgboost
except ModuleNotFoundError:
    sys.path.append(str(LEGACY_XGB))
    import xgboost
from xgboost import XGBClassifier
ALGORITHMS=['logistic','xgboost','lightgbm']
METHODS=['B01','B02','B03','B04_w2','B04_w4']
# Fixed before fitting: no hyperparameter search, early stopping or external eval_set.
XGB_PARAMS=dict(n_estimators=160,max_depth=3,min_child_weight=8,learning_rate=.05,
                subsample=.9,colsample_bytree=1.,reg_lambda=10.,reg_alpha=0.,
                tree_method='hist',max_bin=256,device='cpu',verbosity=0)
META=['prediction_time','target_time','day','kind','y']

def classifier(algorithm,cfg,binary):
    if algorithm=='logistic':
        return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
            LogisticRegression(C=cfg['models']['logistic']['C'],max_iter=2000,random_state=cfg['seed']))
    return XGBClassifier(**XGB_PARAMS,random_state=cfg['seed'],n_jobs=cfg['threads'],
        **({'objective':'binary:logistic','eval_metric':'logloss'} if binary else {'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss'}))

def predict_path(out,fid,g,alg,method):
    return (SOURCE/'cells'/f'{fid}_{g}_{method}.parquet' if alg=='lightgbm' else out/'cells'/f'{fid}_{g}_{alg}_{"MULTI" if method in ["B01","B02"] else method}.parquet')

def load_prediction(out,fid,g,alg,method):return pd.read_parquet(predict_path(out,fid,g,alg,method))

def fit_one(alg,cfg,x,y,weight):
    binary=weight is not None;m=classifier(alg,cfg,binary)
    y=np.asarray(y>0 if binary else y,int);w=None if not binary else np.where(y==1,float(weight),1.)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        kw={} if w is None else {'logisticregression__sample_weight':w} if alg=='logistic' else {'sample_weight':w}
        m.fit(x,y,**kw)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Logistic did not converge; preserve checkpoint and diagnose training history')
    return m,time.perf_counter()-start

def coef_rows(m,x,fid,g,method):
    names=m[:-1].get_feature_names_out(list(x.columns));scaler=m.named_steps['standardscaler'];lr=m.named_steps['logisticregression']
    return [{'fold':fid,'grouping':g,'method':method,'feature':str(n),'coefficient_standardized':float(v),
        'training_mean_after_imputation':float(mu),'training_scale':float(sd),'intercept':float(lr.intercept_[0]),
        'direction':'positive' if v>0 else 'negative' if v<0 else 'zero','interpretation':'conditional log-odds association per training SD; not causal; groups fold-specific'}
        for n,v,mu,sd in zip(names,lr.coef_[0],scaler.mean_,scaler.scale_)]

def freeze_thresholds(out,folds,groups):
    result=[]
    for alg in ALGORITHMS:
        for g in groups:
            for method in METHODS[1:]:
                a=pd.concat([load_prediction(out,f['id'],g,alg,method) for f in folds]);choices=[]
                for t in GRID:
                    v,_=metrics(a.y>0,a.score>=t,a.score,a.day)
                    if v['precision_1']>=.8:choices.append((v['recall_1'],float(t),v['precision_1']))
                if choices:rr,t,pp=max(choices)
                else:rr,t,pp=None,None,None
                result.append({'algorithm':alg,'grouping':g,'method':method,'threshold':t,'development_precision':pp,'development_recall':rr,'feasible':bool(choices)})
    js({'selection':'2024 pooled OOF only; same development data, exploratory; no unbiased selected-validation claim','uses_2025':False,'frozen_before_new_2025_models':True,'grid':GRID.tolist(),'precision_min':.8,'candidates':result},out/'frozen_thresholds_2024.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'binary_classifier_comparison');ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    out=args.out.resolve()
    if out==SOURCE or SOURCE in out.parents or out in SOURCE.parents:raise ValueError('Separate output required')
    if out.exists() and any(out.iterdir()) and not args.resume:raise ValueError('Existing output: pass --resume')
    out.mkdir(parents=True,exist_ok=True)
    for sub in ['cells','models','checkpoints','figures']: (out/sub).mkdir(exist_ok=True)
    cfg=read(CONFIG)['config'];folds=cfg['folds']+[{'id':'EXP2025',**cfg['exploratory']}]
    if args.smoke:folds=folds[:1]
    groups=GROUPS[:1] if args.smoke else GROUPS
    assert read(SOURCE/'acceptance_report.json')['passed']
    assignments_path=BASE/'grouping_pilot_v1/cluster_assignments.csv'
    calendar_path=Path(cfg['data_root'])/cfg['inputs']['calendar']
    inputs=[Path(__file__),CONFIG,assignments_path,calendar_path,SOURCE/'manifest.json',SOURCE/'acceptance_report.json',ROOT/'experiments/congestion_2026/run_binary_congestion_pilot.py',ROOT/'experiments/congestion_2026/features/engine.py',ROOT/'experiments/congestion_2026/data/contracts.py',BASE/'speed_occupancy_eda_v1/candidate_Y_grid.parquet']
    existing_lr=[]
    for f in folds:
        fid=f['id'];inputs += [CACHE/f'features_{fid}_h30.parquet',CACHE/f'features_{fid}_h30.json',CACHE/f'group_transform_{fid}.json']
        for g in groups:
            inputs.append(SOURCE/'audit'/f'{fid}_{g}.json')
            inputs += [SOURCE/'cells'/f'{fid}_{g}_{m}.parquet' for m in METHODS]
            lr=BASE/'o_group_q_B0_v2/cells'/f'{fid}_h30_logistic_{g}_PQ.parquet'
            if lr.exists():inputs.append(lr);existing_lr.append(str(lr))
    inputs+=sorted(CACHE.glob('npq_*.npz'))
    # Seal the actual installed native XGBoost implementation, including legacy fallback.
    xgbroot=Path(xgboost.__file__).parent
    inputs += [Path(xgboost.__file__),xgbroot/'core.py',xgbroot/'sklearn.py',*xgbroot.glob('lib/*.dll')]
    manifest={'version':1,'source_commit':'12ef9da','smoke':args.smoke,'groups':groups,'methods':METHODS,'algorithms':ALGORITHMS,'horizon':30,'demand':'B0_PQ','label':'Z0','embargo':30,'xgboost_params':XGB_PARAMS,'logistic':{'C':1.,'max_iter':2000,'imputer':'median + indicator','scaler':'training mean/std','solver':'lbfgs','regularization':'L2'},'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','lightgbm','joblib']},'xgboost_version':xgboost.__version__,'input_sha256':{str(p.resolve()):sha(p) for p in inputs}}
    mp=out/'manifest.json'
    if mp.exists() and read(mp)!=manifest:raise ValueError('Source/settings/environment changed: use new output')
    js(manifest,mp)
    cal=pd.read_csv(calendar_path,dtype={'SourceDate':str});calendar=dict(zip(pd.to_datetime(cal.SourceDate,format='%Y%m%d'),cal.DayType));assignment=pd.read_csv(assignments_path,dtype={'o':str})
    audit=[];coefficients=[];statistics=[]
    for f in folds:
        fid=f['id']
        if fid=='EXP2025':freeze_thresholds(out,cfg['folds'],groups)
        frame=pd.read_parquet(CACHE/f'features_{fid}_h30.parquet').reset_index(drop=True);meta=read(CACHE/f'features_{fid}_h30.json')
        fit=frame.loc[train_mask(frame,f['start'],30,cfg['warmup_before'])];val=frame.loc[frame.prediction_time.ge(f['start'])&frame.prediction_time.lt(f['end'])&frame.y.notna()]
        assert_train_validation(fit,val,30)
        assert (frame.target_time-frame.prediction_time).eq(pd.Timedelta(minutes=30)).all()
        assert frame.history_cutoff.le(frame.prediction_time).all() and frame.vd_feature_available_at.le(frame.prediction_time).all()
        for g in groups:
            a=assignment[assignment.fold.eq(fid)&assignment.grouping.eq(g)];assert not a.o.duplicated().any()
            registered=read(SOURCE/'audit'/f'{fid}_{g}.json');base=[c for c in meta['columns']['M1'] if not c.startswith('M1__')]
            x=pd.concat([frame[base],demand(frame,CACHE,calendar,dict(zip(a.o,a.group)))],axis=1)
            assert list(x.columns)==registered['features'] and len(fit)==registered['train_n'] and len(val)==registered['validation_n']
            if g=='ALL':total=x.filter(like='demand__').sum(axis=1).to_numpy()
            else:assert np.allclose(x.filter(like='demand__').sum(axis=1),total,rtol=2e-5,atol=2e-4)
            for method in METHODS:
                prior=load_prediction(out,fid,g,'lightgbm',method)
                for c in META:assert np.array_equal(prior[c],val[c]),(fid,g,method,c)
            if g=='TOP20':
                old=frame[[c for c in meta['columns']['B0_PQ'] if c.startswith('B0_PQ__')]].copy();old.columns=['demand__'+c.split('__',1)[1] for c in old]
                assert np.allclose(old.reindex(columns=x.filter(like='demand__').columns,fill_value=0),x.filter(like='demand__'),rtol=2e-5,atol=2e-4)
            audit.append({'fold':fid,'grouping':g,'train_n':len(fit),'validation_n':len(val),'features':x.shape[1],'lightgbm_keys_verified':True,'feature_names_match_source':True,'train_target_max':fit.target_time.max(),'validation_prediction_min':val.prediction_time.min()})
            for alg in ALGORITHMS[:2]:
                for strategy,weight in [('MULTI',None),('B03',1),('B04_w2',2),('B04_w4',4)]:
                    tag=f'{fid}_{g}_{alg}_{strategy}';pp=out/'cells'/f'{tag}.parquet';jp=out/'models'/f'{tag}.joblib';cp=out/'checkpoints'/f'{tag}.json'
                    if cp.exists():
                        chk=read(cp)
                        if sha(pp)!=chk['prediction_sha256'] or sha(jp)!=chk['model_sha256']:raise ValueError('Checkpoint integrity failure '+tag)
                        pred=pd.read_parquet(pp);m=joblib.load(jp);print('REUSE',tag,flush=True)
                    else:
                        print('FIT',tag,'n=',len(fit),'p=',x.shape[1],flush=True)
                        m,seconds=fit_one(alg,cfg,x.loc[fit.index],fit.y,weight);p=m.predict_proba(x.loc[val.index]);assert list(m.classes_)==([0,1,2] if weight is None else [0,1])
                        pred=val[META].copy();pred['y']=pred.y.astype(int)
                        reused=None;err=None
                        if weight is None:
                            assert_probabilities(p);p=p/p.sum(1,keepdims=True)
                            lrpath=BASE/'o_group_q_B0_v2/cells'/f'{fid}_h30_logistic_{g}_PQ.parquet'
                            if alg=='logistic' and lrpath.exists():
                                previous=pd.read_parquet(lrpath)
                                for col in META:assert np.array_equal(previous[col],val[col])
                                oldp=previous[['p0','p1','p2']].to_numpy();err=float(np.max(np.abs(p-oldp)))
                                if err>1e-9:raise ValueError('Existing LR probability mismatch '+tag+str(err))
                                p=oldp;reused=str(lrpath)
                            pred[['p0','p1','p2']]=p;pred['score']=p[:,1]+p[:,2];pred['hard']=(p.argmax(1)>0).astype(int)
                        else:pred['score']=p[:,1];pred['hard']=(pred.score>=.5).astype(int)
                        assert np.isfinite(pred.score).all() and pred.score.between(0,1).all()
                        parquet(pred,pp);tmp=jp.with_suffix('.tmp');joblib.dump(m,tmp,compress=3);tmp.replace(jp)
                        chk={'tag':tag,'training_seconds':seconds,'train_n':len(fit),'validation_n':len(val),'input_columns':list(x.columns),'prediction_sha256':sha(pp),'model_sha256':sha(jp),'existing_three_class_predictions_reused':reused is not None,'legacy_probability_max_abs_difference':err,'weighted':weight,'classifier_params':m.get_params(deep=False) if alg=='xgboost' else m[-1].get_params(deep=False)}
                        chk['classifier_params']={k:('NaN (native missing marker)' if isinstance(v,float) and not np.isfinite(v) else v) for k,v in chk['classifier_params'].items()}
                        js(chk,cp)
                    for col in META:assert np.array_equal(pred[col],val[col])
                    statistics.append({k:v for k,v in chk.items() if k not in ['input_columns','classifier_params']})
                    if alg=='logistic' and weight is not None:coefficients.extend(coef_rows(m,x.loc[fit.index],fid,g,strategy))
            csv(audit,out/'sample_feature_audit.csv');csv(statistics,out/'training_statistics.csv');csv(coefficients,out/'logistic_standardized_coefficients.csv')
    js({'training_complete':True,'smoke':args.smoke,'new_or_reproduced_model_cells':len(statistics),'expected':len(folds)*len(groups)*8,'lightgbm_not_retrained':True,'all_checkpoint_hashes_valid':True,'scope':'training only; run analysis for final acceptance'},out/'training_acceptance.json')
    print('TRAINING COMPLETE',out,flush=True)

if __name__=='__main__':main()
