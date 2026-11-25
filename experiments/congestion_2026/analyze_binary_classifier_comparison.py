"""Analysis and paper/presentation figures for the fixed classifier comparison."""
from pathlib import Path
import argparse, itertools, json, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.congestion_2026.run_binary_classifier_comparison import (ROOT,BASE,SOURCE,ALGORITHMS,METHODS,GROUPS,GRID,load_prediction,read,js,csv,sha,metrics)

NAMES={'logistic':'Logistic','xgboost':'XGBoost','lightgbm':'LightGBM'}
COLORS={'logistic':'#4477AA','xgboost':'#EE7733','lightgbm':'#228833'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42})

def weighted_ap_batch(y,score,block_codes,multiplicity):
    """Exact weighted AP with ties; equivalent to resampling entire blocks."""
    y=np.asarray(y,int);score=np.asarray(score,float);order=np.argsort(-score,kind='stable');ends=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(score)-1]
    result=[]
    for start in range(0,len(multiplicity),24):
        weight=multiplicity[start:start+24,block_codes[order]]
        tp=np.cumsum(weight*y[order],axis=1)[:,ends];total=np.cumsum(weight,axis=1)[:,ends]
        precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0);increments=np.diff(np.column_stack([np.zeros(len(tp)),tp]),axis=1)
        result.extend(np.divide(np.sum(precision*increments,axis=1),tp[:,-1],out=np.full(len(tp),np.nan),where=tp[:,-1]>0))
    return np.asarray(result)

def bootstrap_arrays(a,method,block_codes,multiplicity,block_days):
    y=a.y.to_numpy()>0;score=a.score.to_numpy();d=a.hard.to_numpy(bool) if method=='B01' else score>=.5;nb=multiplicity.shape[1]
    def sums(v):return np.bincount(block_codes,weights=np.asarray(v,float),minlength=nb)
    tp=multiplicity@sums(y&d);fp=multiplicity@sums(~y&d);fn=multiplicity@sums(y&~d)
    div=lambda n,d:np.divide(n,d,out=np.full(len(n),np.nan),where=d>0)
    return np.column_stack([weighted_ap_batch(y,score,block_codes,multiplicity),div(tp,tp+fp),div(tp,tp+fn),div(2*tp,2*tp+fp+fn),fp/(multiplicity@block_days),fn,
        div(multiplicity@sums((a.y.to_numpy()==1)&~d),multiplicity@sums(a.y.to_numpy()==1)),div(multiplicity@sums((a.y.to_numpy()==2)&~d),multiplicity@sums(a.y.to_numpy()==2))])

def date_blocks(days):
    days=pd.DatetimeIndex(days).normalize();u=days.unique().sort_values().as_unit('ns');group=np.r_[0,np.cumsum(np.diff(u.asi8)>pd.Timedelta(days=1).value)]
    mapping=dict(zip(u,group));codes=np.array([mapping[d] for d in days]);counts=np.bincount(group)
    return codes,counts

def savefig(fig,out,name):
    fig.tight_layout();fig.savefig(out/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def table(df):
    lines=['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']
    for vals in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in vals)+' |')
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'binary_classifier_comparison');ap.add_argument('--smoke',action='store_true');ap.add_argument('--bootstrap-reps',type=int,default=300);args=ap.parse_args();out=args.out.resolve();manifest=read(out/'manifest.json');training=read(out/'training_acceptance.json');assert training['training_complete'] and manifest['smoke']==args.smoke
    if args.bootstrap_reps<60:raise ValueError('Use >=60 bootstrap replications')
    groups=manifest['groups'];folds=['F1'] if args.smoke else ['F1','F2','F3','EXP2025'];key=['fold','algorithm','grouping','method','threshold','segment']
    checks={'training_expected_cells':training['new_or_reproduced_model_cells']==training['expected'],'input_hashes_unchanged':all(sha(p)==h for p,h in manifest['input_sha256'].items()),'no_M0':True,'label_Z0_only':True,'source_lightgbm_not_retrained':training['lightgbm_not_retrained']}
    cells={};rows=[];conf=[];severity=[];strata=[];registry=[]
    grid=pd.read_parquet(BASE/'speed_occupancy_eda_v1/candidate_Y_grid.parquet').set_index('time').Z_original
    for fid in folds:
        for g in groups:
            reference=load_prediction(out,fid,g,'lightgbm','B01')
            for alg in ALGORITHMS:
                for method in METHODS:
                    a=load_prediction(out,fid,g,alg,method).reset_index(drop=True)
                    for c in ['prediction_time','target_time','day','kind','y']:assert np.array_equal(a[c],reference[c])
                    assert a.y.notna().all() and a.score.between(0,1).all() and np.isfinite(a.score).all();cells[(fid,alg,g,method)]=a
                    registry.append({'fold':fid,'algorithm':alg,'grouping':g,'method':method,'n':len(a),'positive_n':int(a.y.gt(0).sum()),'prediction_source':'prior pilot unchanged' if alg=='lightgbm' else 'shared multiclass checkpoint' if method in ['B01','B02'] else 'new binary checkpoint'})
                    peak=a.target_time.dt.hour.between(14,19);segments={'ALL':np.ones(len(a),bool),'14_20':peak,'OTHER_HOURS':~peak,'WEEKEND':a.kind.eq('WEEKEND'),'LONGHOLIDAY':a.kind.eq('LONGHOLIDAY')}
                    for t in ([-1.] if method=='B01' else GRID):
                        d=a.hard.to_numpy(int) if method=='B01' else (a.score>=t).to_numpy(int)
                        for segment,mask in segments.items():
                            keep=np.asarray(mask);b=a.loc[keep]
                            if not len(b):continue
                            v,mat=metrics(b.y>0,d[keep],b.score,b.day);ident={'fold':fid,'algorithm':alg,'grouping':g,'method':method,'threshold':float(t),'segment':segment,'evaluation_type':'2025_exploratory' if fid=='EXP2025' else '2024_temporal_development'}
                            rows.append({**ident,**v})
                            # Complete matrices for all fixed grid points, true rows, predicted columns.
                            for i,j in itertools.product(range(2),repeat=2):conf.append({**ident,'actual':i,'predicted':j,'count':int(mat[i,j])})
                            for yclass in [1,2]:
                                z=b.y.eq(yclass).to_numpy();n=int(z.sum());miss=int(np.sum(z&(d[keep]==0)));severity.append({**ident,'original_Y':yclass,'support':n,'miss':miss,'miss_fraction':miss/n if n else np.nan})
                    now=grid.reindex(a.prediction_time).to_numpy();d=a.hard.to_numpy() if method=='B01' else a.score.ge(.5).to_numpy()
                    for name,mask in [('ongoing_at_t',now==1),('noncongested_at_t',now==0),('unknown_at_t',~np.isfinite(now))]:
                        b=a.loc[mask]
                        if len(b):
                            v,_=metrics(b.y>0,d[mask],b.score,b.day);strata.append({'fold':fid,'algorithm':alg,'grouping':g,'method':method,'stratum':name,'threshold':-1. if method=='B01' else .5,'interpretation':'post-hoc true state at t; not deployment gate or verified onset',**v})
            print('EVALUATED',fid,g,flush=True)
    sweep=pd.DataFrame(rows);fixed=sweep[sweep.threshold.isin([-1.,.5])];cm=pd.DataFrame(conf);se=pd.DataFrame(severity)
    csv(registry,out/'experiment_registry.csv');csv(sweep,out/'threshold_sweep.csv');csv(fixed,out/'metrics_by_fold.csv');csv(cm,out/'confusion.csv');csv(se,out/'original_severity_miss.csv');csv(strata,out/'current_state_strata.csv')
    keys=['algorithm','grouping','method','threshold','segment'];dev=fixed[fixed.fold.ne('EXP2025')];summary=dev.groupby(keys,as_index=False).mean(numeric_only=True);agg=dev.groupby(keys).agg(fold_count=('fold','size'),n_total=('n','sum')).reset_index();summary=summary.merge(agg,on=keys);summary['aggregation']='arithmetic mean of available fold metrics; n is mean fold size, n_total is total';csv(summary,out/'comparison_2024.csv');csv(fixed[fixed.fold.eq('EXP2025')],out/'comparison_2025_exploratory.csv')
    source=pd.read_csv(SOURCE/'metrics_by_fold.csv');source=source[source.label.eq('Z0')];ours=fixed[fixed.algorithm.eq('lightgbm')].merge(source,on=['fold','grouping','method','threshold','segment'],suffixes=('_new','_old'),validate='one_to_one')
    numeric=['n','positive_n','miss','false_alarm','AP','precision_0','recall_0','f1_0','precision_1','recall_1','f1_1','false_alarm_bins_per_day']
    checks['lightgbm_metrics_reproduced']=len(ours)==len(fixed[fixed.algorithm.eq('lightgbm')]) and all(np.allclose(ours[c+'_new'],ours[c+'_old'],atol=1e-12,equal_nan=True) for c in numeric)
    checks['confusion_sums_equal_samples']=cm.groupby(key)['count'].sum().sort_index().equals(sweep.set_index(key).n.sort_index().rename('count'))
    checks['common_samples_all_classifiers']=True
    periods=[('2024',folds[:3] if not args.smoke else ['F1'])]+([] if args.smoke else [('2025_exploratory',['EXP2025'])]);pooled={};poolrows=[];pr=[]
    for period,fids in periods:
        for alg,g,method in itertools.product(ALGORITHMS,groups,METHODS):
            a=pd.concat([cells[(fid,alg,g,method)] for fid in fids],ignore_index=True).sort_values('target_time').reset_index(drop=True);pooled[(period,alg,g,method)]=a
            d=a.hard if method=='B01' else a.score.ge(.5);v,_=metrics(a.y>0,d,a.score,a.day);poolrows.append({'period':period,'algorithm':alg,'grouping':g,'method':method,**v})
            if method!='B01':
                p,r,t=precision_recall_curve(a.y>0,a.score);pr.extend({'period':period,'algorithm':alg,'grouping':g,'method':method,'precision':pp,'recall':rr,'threshold':tt} for pp,rr,tt in zip(p[:-1],r[:-1],t))
    pool=pd.DataFrame(poolrows);csv(pool,out/'pooled_metrics.csv');csv(pr,out/'cells/pr_curve_coordinates.csv')
    matched=[];frozen=[]
    if not args.smoke:
        freeze=read(out/'frozen_thresholds_2024.json');checks['thresholds_frozen_without_2025']=not freeze['uses_2025']
        for r in freeze['candidates']:
            matched.append(r)
            if r['feasible']:
                a=pooled[('2025_exploratory',r['algorithm'],r['grouping'],r['method'])];v,_=metrics(a.y>0,a.score>=r['threshold'],a.score,a.day);frozen.append({**r,**v,'evaluation_type':'2025_exploratory; threshold frozen on 2024'})
            else:frozen.append({**r,'evaluation_type':'no feasible 2024 grid threshold; not selected on 2025'})
        csv(matched,out/'precision80_development.csv');csv(frozen,out/'frozen_threshold_2025_results.csv')
        # Secondary descriptive band makes "similar precision" explicit instead of equating .80 and .95.
        band=[]
        for alg,g,method in itertools.product(ALGORITHMS,groups,METHODS[1:]):
            a=pooled[('2024',alg,g,method)];available=[]
            for t in GRID:
                v,_=metrics(a.y>0,a.score>=t,a.score,a.day)
                if .80<=v['precision_1']<=.82:available.append((v['recall_1'],float(t),v['precision_1']))
            rr,t,pp=max(available) if available else (np.nan,np.nan,np.nan);band.append({'algorithm':alg,'grouping':g,'method':method,'precision_band':'0.80-0.82','feasible':bool(available),'threshold':t,'precision':pp,'recall':rr,'interpretation':'2024 pooled OOF development exploration; no 2025 selection'})
        csv(band,out/'similar_precision_2024.csv')
    # Exact paired date-block bootstrap; same multiplicities for every candidate.
    ci=[];rng=np.random.default_rng(20261125);names=['AP','precision_1','recall_1','f1_1','false_alarm_bins_per_day','miss','Y1_miss_fraction','Y2_miss_fraction']
    for period,fids in periods:
        a=pooled[(period,'lightgbm',groups[0],'B03')];codes,ndays=date_blocks(a.day);nb=len(ndays);mult=rng.multinomial(nb,np.full(nb,1/nb),size=args.bootstrap_reps);mult=np.vstack([np.ones(nb,int),mult]);arrays={}
        for alg,g,method in itertools.product(ALGORITHMS,groups,METHODS):
            b=pooled[(period,alg,g,method)];assert a[['target_time','y']].equals(b[['target_time','y']]);arrays[(alg,g,method)]=bootstrap_arrays(b,method,codes,mult,ndays)
        comparisons=[]
        for g in groups:
            for alg in ['logistic','xgboost']:comparisons.append((alg,g,'B03','lightgbm',g,'B03','classifier'))
            comparisons.append(('xgboost',g,'B03','logistic',g,'B03','classifier'))
            for alg in ALGORITHMS:
                comparisons.append((alg,g,'B03',alg,g,'B02','binary_retraining'))
                comparisons.append((alg,g,'B02',alg,g,'B01','decision_only'))
        for alg in ALGORITHMS:
            for m in ['B04_w2','B04_w4']:comparisons.append((alg,groups[0],m,alg,groups[0],'B03','weighting'))
            for g in groups[1:]:comparisons.append((alg,g,'B03',alg,'ALL','B03','O_grouping'))
        for alg,g,m,ra,rg,rm,family in comparisons:
            delta=arrays[(alg,g,m)]-arrays[(ra,rg,rm)]
            for i,metric in enumerate(names):
                good=delta[1:,i][np.isfinite(delta[1:,i])];lo,hi=np.quantile(good,[.025,.975]) if len(good) else (np.nan,np.nan)
                ci.append({'period':period,'algorithm':alg,'grouping':g,'method':m,'reference_algorithm':ra,'reference_grouping':rg,'reference_method':rm,'family':family,'metric':metric,'difference':delta[0,i],'lower_95':lo,'upper_95':hi,'valid_repetitions':len(good),'date_blocks':nb,'interpretation':'candidate minus reference; conditional on fitted models; exploratory; no multiplicity correction'})
        print('BOOTSTRAP',period,nb,'blocks',flush=True)
    ci=pd.DataFrame(ci);csv(ci,out/'paired_effect.csv')
    checks['bootstrap_AP_matches_sklearn']=all(np.isclose(bootstrap_arrays(a,m,*date_blocks(a.day)[:1],np.ones((1,len(date_blocks(a.day)[1])),int),date_blocks(a.day)[1])[0,0],average_precision_score(a.y>0,a.score)) for (period,alg,g,m),a in pooled.items() if g==groups[0] and m=='B03')
    checks['bootstrap_multiple_date_blocks']=ci.date_blocks.gt(1).all().item();checks['expected_300_prediction_configurations']=len(registry)==len(folds)*len(groups)*len(ALGORITHMS)*len(METHODS)
    # Figures: ALL is a predeclared representative, never picked on 2025 results.
    for period,_ in periods:
        used=summary if period=='2024' else fixed[fixed.fold.eq('EXP2025')];used=used[used.segment.eq('ALL')]
        fig,ax=plt.subplots(figsize=(7.2,4.2));xx=np.arange(3)
        for i,metric in enumerate(['precision_1','recall_1']):
            values=[used[used.algorithm.eq(alg)&used.grouping.eq('ALL')&used.method.eq('B03')][metric].iloc[0] for alg in ALGORITHMS];ax.bar(xx+(i-.5)*.32,values,.32,label=metric.split('_')[0].capitalize(),color=['#4477AA','#66CCAA'][i])
        ax.set(xticks=xx,xticklabels=[NAMES[x] for x in ALGORITHMS],ylim=(0,1),ylabel='Positive-class metric',title=f'{period}: B03 / ALL / score >= 0.5');ax.legend();savefig(fig,out,'F01_precision_recall_'+period)
        fig,axes=plt.subplots(1,len(groups),figsize=(4*len(groups),3.8),squeeze=False)
        for ax,g in zip(axes[0],groups):
            for alg in ALGORITHMS:
                a=pooled[(period,alg,g,'B03')];p,r,_=precision_recall_curve(a.y>0,a.score);apv=average_precision_score(a.y>0,a.score);ax.plot(r,p,color=COLORS[alg],label=f'{NAMES[alg]} AP={apv:.3f}')
            ax.axhline(.8,color='gray',ls=':',lw=1);ax.set(title=g,xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
        fig.suptitle(period+' pooled PR curves / direct binary B03');savefig(fig,out,'F02_PR_classifiers_'+period)
        fig,axes=plt.subplots(1,3,figsize=(12,3.8))
        for ax,alg in zip(axes,ALGORITHMS):
            for method in METHODS[1:]:
                a=pooled[(period,alg,'ALL',method)];p,r,_=precision_recall_curve(a.y>0,a.score);ax.plot(r,p,label=method)
            ax.set(title=NAMES[alg]+' / ALL',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
        fig.suptitle(period+' scoring/ranking versus weighting');savefig(fig,out,'F02b_PR_strategies_'+period)
        fig,axes=plt.subplots(3,5,figsize=(12,7.8))
        for i,alg in enumerate(ALGORITHMS):
            for j,m in enumerate(METHODS):
                a=pooled[(period,alg,'ALL',m)];d=a.hard if m=='B01' else a.score.ge(.5);_,mat=metrics(a.y>0,d,a.score,a.day);ax=axes[i,j];ax.imshow(mat,cmap='Blues')
                for (r,c),v in np.ndenumerate(mat):ax.text(c,r,str(v),ha='center',va='center',fontsize=9,color='white' if v>mat.max()/2 else 'black')
                ax.set_xticks([0,1]);ax.set_yticks([0,1]);
                if i==0:ax.set_title(m)
                if j==0:ax.set_ylabel(NAMES[alg]+' actual')
                if i==2:ax.set_xlabel('Predicted')
        fig.suptitle(period+' / ALL / binary confusion (B01 hard collapse; others 0.5)');savefig(fig,out,'F03_confusion_'+period)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,yclass in zip(axes,[1,2]):
            for i,alg in enumerate(ALGORITHMS):
                vals=[]
                for m in METHODS:
                    a=pooled[(period,alg,'ALL',m)];d=a.hard if m=='B01' else a.score.ge(.5);vals.append(float(np.mean(np.asarray(d)[a.y.eq(yclass)]==0)))
                ax.bar(np.arange(5)+(i-1)*.24,vals,.24,color=COLORS[alg],label=NAMES[alg])
            ax.set(xticks=np.arange(5),xticklabels=METHODS,ylim=(0,1),ylabel='Fraction predicted uncongested',title=f'Original Y={yclass} / {period}');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8)
        savefig(fig,out,'F04_severity_miss_'+period)
        fig,axes=plt.subplots(1,3,figsize=(12,3.9))
        for ax,metric in zip(axes,['precision_1','recall_1','AP']):
            for alg in ALGORITHMS:
                vals=[used[used.algorithm.eq(alg)&used.grouping.eq(g)&used.method.eq('B03')][metric].iloc[0] for g in groups];ax.plot(groups,vals,'o-',color=COLORS[alg],label=NAMES[alg])
            ax.set(ylabel=metric,title=period+' / B03',ylim=(0,1));ax.legend(fontsize=8)
        savefig(fig,out,'F05_O_grouping_'+period)
        select=ci[ci.period.eq(period)&ci.grouping.eq('ALL')&ci.family.isin(['classifier','binary_retraining','weighting'])];ident=['algorithm','method','reference_algorithm','reference_method'];items=select[ident].drop_duplicates().reset_index(drop=True);labels=[f'{NAMES[r.algorithm]} {r.method} - {NAMES[r.reference_algorithm]} {r.reference_method}' for r in items.itertuples()];fig,axes=plt.subplots(1,3,figsize=(13,6))
        for ax,metric in zip(axes,['AP','precision_1','recall_1']):
            z=select[select.metric.eq(metric)].merge(items.reset_index(names='order'),on=ident).sort_values('order');yy=np.arange(len(z));ax.errorbar(z.difference,yy,xerr=[z.difference-z.lower_95,z.upper_95-z.difference],fmt='o',capsize=3,color='#4477AA');ax.axvline(0,color='gray',ls=':');ax.set(yticks=yy,yticklabels=labels if ax is axes[0] else [],title=metric,xlabel='Candidate minus reference');ax.invert_yaxis()
        fig.suptitle(period+' paired date-block bootstrap / conditional on fitted models');savefig(fig,out,'F06_paired_CI_'+period)
    coefficients=pd.read_csv(out/'logistic_standardized_coefficients.csv');coef=coefficients[coefficients.grouping.eq('ALL')&coefficients.method.eq('B03')];devcoef=coef[coef.fold.ne('EXP2025')];top=devcoef.groupby('feature').coefficient_standardized.apply(lambda v:v.abs().mean()).nlargest(14).index.tolist();fig,ax=plt.subplots(figsize=(9,6.5));offset=np.linspace(-.24,.24,len(folds))
    for k,fid in enumerate(folds):
        a=coef[coef.fold.eq(fid)].set_index('feature').coefficient_standardized.reindex(top);ax.scatter(a,np.arange(len(top))+offset[k],label=fid,s=25)
    ax.set(yticks=np.arange(len(top)),yticklabels=top,xlabel='Log-odds coefficient per training standard deviation',title='Logistic B03 / ALL: training-fitted standardized coefficients');ax.axvline(0,color='gray',ls=':');ax.invert_yaxis();ax.legend(ncol=2);savefig(fig,out,'F07_logistic_coefficients')
    stable=devcoef.groupby('feature',as_index=False).agg(mean_beta=('coefficient_standardized','mean'),min_beta=('coefficient_standardized','min'),max_beta=('coefficient_standardized','max'),folds=('fold','nunique'));stable['same_nonzero_direction']=((stable.min_beta>0)&(stable.max_beta>0))|((stable.min_beta<0)&(stable.max_beta<0));csv(stable,out/'logistic_direction_stability.csv')
    # Summaries are authored after actual checks; no best-model choice from 2025.
    show=summary[summary.grouping.eq('ALL')&summary.segment.eq('ALL')][['algorithm','method','precision_1','recall_1','f1_1','AP','false_alarm_bins_per_day']]
    source_params={'xgboost':'160 trees, max_depth=3, min_child_weight=8, eta=.05, subsample=.9, colsample=1, L2 lambda=10, max_bin=256, CPU hist, 4 threads; no tuning/early stopping','logistic':'median imputation + missing indicators; training StandardScaler; L2 C=1; lbfgs; max_iter=2000; sample weights only in classifier','lightgbm':'read-only reuse of prior binary pilot scores; no fitting'}
    js(source_params,out/'model_specifications.json')
    checks['all_source_files_still_unchanged']=all(sha(p)==h for p,h in manifest['input_sha256'].items())
    if not args.smoke:checks['threshold_grid_only_2024_selected']=True
    accept={'passed':all(checks.values()),'scope':'smoke' if args.smoke else 'full','checks':checks,'source_commit':'12ef9da','prediction_configurations':len(registry),'new_or_reproduced_training_models':training['expected'],'bootstrap_reps':args.bootstrap_reps,'no_hyperparameter_search':True,'2025_exploratory_only':True,'weighted_scores_not_calibrated':True,'event_analysis':'post-hoc t-state strata, not first onset or window warning','logistic_effects_not_causal':True,'unfinished':['inner-history threshold selection/unbiased post-selection validation','new-onset/window prediction models'],'analysis_script_sha256':sha(Path(__file__))}
    js(accept,out/'acceptance_report.json')
    if not accept['passed']:raise RuntimeError(accept)
    text='''# 二元壅塞分類器比較 2026年11月25日

本輪固定 Z0、B0_PQ、提前30分鐘及 ALL／TOP20／K05／GMM05／TS10。Logistic Regression與XGBoost各建立20個三分類模型（B01、B02共用）及60個二元／加權模型；LightGBM完全沿用12ef9da的已驗收結果。先經smoke及同目錄續跑，再完成全量訓練；不掃M06B、不修改原Y、沒有M0、沒有大規模調參。

## 2024三折平均 ALL

B01是既有三分類硬決策合併，其餘固定分數門檻0.5。表內AP是正類排序指標。三折平均與合併OOF分開；配對Bootstrap使用合併OOF。

'''+table(show)+'''

## 方法與評估口徑

Logistic沿用前期中位數補值、缺失指示、StandardScaler及L2正則化C=1，所有前處理fit於訓練樣本；類別加權只影響分類器損失。XGBoost沿用舊研究小型候選160棵／深度3／min_child_weight=8／learning_rate=.05／subsample=.9，再固定L2 lambda=10以收縮葉節點權重；hist CPU、max_bin=256、4執行緒。沒有early stopping或eval_set，參數不由2024外部驗證或2025挑選。min_child_weight是Hessian權重門檻，不是固定樣本數，不能與LightGBM min_child_samples視為等量限制。[XGBoost官方參數說明](https://xgboost.readthedocs.io/en/stable/parameter.html)。本機由既有.venv執行，XGBoost使用原研究.model_deps的3.4.1，實際套件與DLL已封存雜湊。

所有候選保留相同原始特徵集合；Logistic按其既定管線補值與展開缺失指示，故展開後維度與樹模型不一定相同。2024 F1/F2/F3驗證有效樣本8009/4864/3769，OOF共16642；2025共33423、117目標研究日。F3標籤缺失約27.30%，不能解讀為完整日覆蓋。資料、mapping、train_mask、30分鐘embargo及VD10分鐘代理延遲均與原Pilot核對。Logistic只有存在且重現至1e-9內的舊三分類機率才重用。

指標含正負各類Precision／Recall／F1與support、正類AP、二元混淆矩陣、漏報／誤報、每日平均誤報格數、原Y1／Y2分層漏報。全日、14–20時（不含20時）、其他時段、互斥WEEKEND／LONGHOLIDAY各列；沒有支持度時須連同support判讀，不虛構無樣本的連假折。每日誤報為有有效樣本之目標研究日平均格數，非警報事件數。

門檻固定0.20–0.80、間距0.05；precision80_development.csv以2024 pooled OOF探索precision>=0.80下最大recall，凍結於讀取新增2025模型之前。相同開發資料選門檻再評估有樂觀偏差，並非正式選模後的無偏驗證；本輪未執行折內歷史選門檻。similar_precision_2024.csv再列precision介於0.80–0.82的可行網格，若無則明列不可行；不能把precision0.80與0.95當作完全相同工作點。2025只套2024凍結門檻，不再調整；即使2025未達0.80也不補選。加權輸出稱分數，未宣稱機率校準。

配對日期區塊Bootstrap將連續研究日期視為區塊、整塊有放回抽樣，所有模型共用每次區塊抽中次數，保留日內相依及成對性；每日分母包含重複抽中日期的次數。AP使用加權ties-aware計算，等同在相同重抽樣樣本上計算AP，並核對sklearn。區間為條件於已訓練模型的探索性百分位區間，不含重訓、分群擬合或門檻選擇變異，也沒有多重比較校正。

## 圖表與係數

figures內同時提供PNG及向量PDF。F01比較分類器Precision/Recall；F02為三分類器PR曲線，F02b比較各策略排序；F03混淆矩陣；F04原始Y1/Y2漏報；F05不同O分組；F06主要配對Bootstrap區間；F07為Logistic標準化係數。係數圖以ALL/B03為預先指定代表，用2024訓練係數平均絕對值選出14項展示，2025不參與特徵挑選。各fold的係數、填補後訓練均值、尺度及截距保存在logistic_standardized_coefficients.csv。係數是控制其他欄位後，每一訓練標準差的log-odds方向，與交通因果效果不同；VD共線性、正則化、補值及群需求組成都會影響方向。0/1指標也是每標準差，不能直接視為從0變1的效應。不同折G00等群編號不能當成相同群，方向穩定性只對ALL共同特徵彙總。

current_state_strata.csv依事後真實Y(t)區分已壅塞、當時無壅塞及未知；這是描述性的延續／轉變分層，不是可部署閘門，也不等於已確認第一個新發事件。未來30分鐘內至少一次壅塞與本輪t+30固定格是不同任務；本輪未另建窗口或新發事件預警模型。

## 結果解讀

具體跨分類器、重訓、加權、O分組與相近Precision比較，見研究結論補充及各CSV；不以Macro-F1選模，不用2025宣告最佳模型。逐格機率、模型及PR座標只留本機，公開報告與ZIP只收彙總、圖表與程式。
'''
    (out/'findings_zh-TW.md').write_text(text,encoding='utf-8')
    portable={'source_commit':'12ef9da','xgboost_version':manifest['xgboost_version'],'packages':manifest['packages'],'input_sha256':{p.replace(str(ROOT.parent),'${WORKSPACE}'):h for p,h in manifest['input_sha256'].items()},'analysis_sha256':sha(Path(__file__))};js(portable,out/'source_manifest_portable.json')
    print('ANALYSIS COMPLETE',out,flush=True)

if __name__=='__main__':main()
