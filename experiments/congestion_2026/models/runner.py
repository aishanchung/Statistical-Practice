import json,logging,time,gc
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..common import *
from ..data.contracts import train_mask,assert_train_validation,assert_probabilities,assert_same_samples
from ..evaluation.metrics import evaluate

def factory(cfg):
    return legacy_functions(cfg,'pilot_2024_2025/model.py',['make_model','predict'],{'SimpleImputer':SimpleImputer,'StandardScaler':StandardScaler,'LogisticRegression':LogisticRegression,'make_pipeline':make_pipeline,'LABELS':[0,1,2]})
def specifications(h):
    specs=[(algorithm,name,name,1.0) for algorithm in ['lightgbm','logistic'] for name in ['M0','M1','M2','M3','M4','M5']]
    if h==30:
        specs += [('lightgbm',n,n,1.0) for n in ['B0_PQ','B1_P','B1_PQ','B2_PQ']]
        specs += [('lightgbm','C2_w15','M0',1.5),('lightgbm','C2_w20','M0',2.0)]
    return specs
def run(cfg):
    out=outdir(cfg);cells=out/'cells';cells.mkdir(exist_ok=True);models=out/'models';models.mkdir(exist_ok=True)
    legacy=factory(cfg);all_metrics=[];all_class=[];confusions=[]
    folds=[*cfg['folds'],{'id':'EXP2025','start':'2025-01-01','end':'2026-01-01'}]
    for fold in folds:
        if fold['id']=='EXP2025':
            validation=pd.DataFrame(all_metrics)
            rank=validation[validation.segment.eq('ALL')&validation.horizon.eq(30)&validation.configuration.isin(['M0','M1','M2','M3','M4','M5'])].groupby(['algorithm','configuration']).agg(average_precision_2=('average_precision_2','mean'),macro_f1=('macro_f1','mean'),log_loss=('log_loss','mean'),n_features=('n_features','max')).reset_index()
            best=[]
            for alg,g in rank.groupby('algorithm'):
                mx=g.average_precision_2.max();eligible=g[g.average_precision_2.ge(mx-cfg['selection']['practical_ap_tie'])]
                best.append(eligible.sort_values(['n_features','macro_f1','log_loss'],ascending=[True,False,True]).iloc[0].to_dict())
            save_csv(rank.sort_values(['algorithm','average_precision_2'],ascending=[True,False]),out/'selection_2024_only.csv')
            save_json({'selected':best,'selection_used':'2024 folds only; mean fold AP2 then <=.005 practical tie and fewer features','2025_used':False,'frozen_before_exploratory_training':True,'config_signature':signature(cfg)},out/'frozen_selection.json')
        for h in [30,15]:
            path=out/'cache'/f'features_{fold["id"]}_h{h}.parquet';frame=pd.read_parquet(path)
            meta=json.loads(path.with_suffix('.json').read_text(encoding='utf8'));columns=meta['columns']
            fit=frame.loc[train_mask(frame,fold['start'],cfg['availability']['embargo_minutes'],cfg['warmup_before'])].copy()
            val=frame.loc[frame.prediction_time.ge(fold['start'])&frame.prediction_time.lt(fold['end'])&frame.y.notna()].copy()
            assert_train_validation(fit,val,cfg['availability']['embargo_minutes'])
            if set(fit.y.unique())!={0,1,2}:raise ValueError('Training class absent')
            template=None
            for alg,name,feature_key,weight in specifications(h):
                tag=f'{fold["id"]}_h{h}_{alg}_{name}';predpath=cells/(tag+'.parquet');checkpoint=cells/(tag+'.json')
                cols=columns[feature_key];start=time.perf_counter()
                if predpath.exists() and checkpoint.exists():
                    chk=json.loads(checkpoint.read_text(encoding='utf8'))
                    if chk['config_signature']!=signature(cfg):raise ValueError('Checkpoint signature mismatch')
                    pred=pd.read_parquet(predpath);seconds=chk['training_seconds'];logging.info('Reuse %s',tag)
                else:
                    logging.info('Fit %s n_train=%s n_eval=%s features=%s',tag,len(fit),len(val),len(cols))
                    modelcfg={'seed':cfg['seed'],'model_threads':cfg['threads']}
                    model=legacy['make_model'](alg,cfg['models'][alg],modelcfg)
                    if weight!=1:model.set_params(class_weight={0:1.,1:weight,2:1.})
                    model.fit(fit[cols],fit.y.to_numpy(int));seconds=time.perf_counter()-start
                    p=legacy['predict'](model,val[cols]);assert_probabilities(p)
                    pred=val[['prediction_time','target_time','day','kind','y','fold','horizon','Y_version','history_cutoff','vd_feature_available_at']].copy()
                    pred['y']=pred.y.astype(int);pred['algorithm']=alg;pred['configuration']=name
                    pred['p0'],pred['p1'],pred['p2']=p.T;pred['predicted']=p.argmax(1)
                    pred['evaluation_type']='2025_exploratory' if fold['id']=='EXP2025' else '2024_temporal_validation'
                    pred.to_parquet(predpath,index=False);joblib.dump(model,models/(tag+'.joblib'),compress=3)
                    save_json({'config_signature':signature(cfg),'training_seconds':seconds,'train_n':len(fit),'eval_n':len(val),'columns':cols,'training_target_max':fit.target_time.max(),'evaluation_origin_min':val.prediction_time.min(),'embargo_minutes':cfg['availability']['embargo_minutes'],'model_params':model.get_params(deep=False)},checkpoint)
                if template is None:template=pred
                else:assert_same_samples(template,pred)
                for segment,keep in [('ALL',np.ones(len(pred),bool)),('WEEKEND',pred.kind.eq('WEEKEND')),('LONGHOLIDAY',pred.kind.eq('LONGHOLIDAY'))]:
                    z=pred.loc[keep]
                    if z.empty:continue
                    m,mat=evaluate(z.y.to_numpy(int),z[['p0','p1','p2']].to_numpy())
                    ident={'fold':fold['id'],'horizon':h,'algorithm':alg,'configuration':name,'segment':segment,'n_features':len(cols),'training_seconds':seconds,'evaluation_type':pred.evaluation_type.iloc[0]}
                    all_metrics.append({**ident,**m})
                    all_class.extend({**ident,'class':k,'precision':m[f'precision_{k}'],'recall':m[f'recall_{k}'],'f1':m[f'f1_{k}'],'support':m[f'support_{k}']} for k in [0,1,2])
                    confusions.extend({**ident,'actual':a,'predicted':b,'count':int(mat[a,b])} for a in [0,1,2] for b in [0,1,2])
                save_csv(pd.DataFrame(all_metrics),out/'metrics_by_fold.csv');save_csv(pd.DataFrame(all_class),out/'metrics_by_class.csv');save_csv(pd.DataFrame(confusions),out/'confusion.csv')
            del frame,fit,val;gc.collect()
    # Explicit local-only Parquet, never committed or uploaded.
    predictions=pd.concat([pd.read_parquet(p) for p in sorted(cells.glob('*.parquet'))],ignore_index=True)
    predictions.to_parquet(out/'predictions.parquet',index=False)
    logging.info('Saved %s real probability rows',len(predictions))
