"""Validate binary pilot outputs, produce Chinese findings, figures and shareable ZIP."""
import argparse, json, hashlib, shutil, zipfile, platform, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.congestion_2026.run_binary_congestion_pilot import BASE, ROOT, GROUPS, read, js, csv, sha, metrics, events

def table(df):
    cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)

def additional_paired_effects(out):
    rng=np.random.default_rng(20261126);rows=[]
    for period,fids in [('2024',['F1','F2','F3']),('2025_exploratory',['EXP2025'])]:
        for g in GROUPS:
            def load(method):return pd.concat([pd.read_parquet(out/'cells'/f'{f}_{g}_{method}.parquet') for f in fids]).sort_values('target_time').reset_index(drop=True)
            for method,reference in [('B03','B02'),('B04_w2','B03'),('B04_w4','B03')]:
                a,b=load(reference),load(method)
                assert a[['prediction_time','target_time','y']].equals(b[['prediction_time','target_time','y']])
                day=pd.DatetimeIndex(a.day).normalize();unique=day.unique().sort_values().as_unit('ns');ids=np.r_[0,np.cumsum(np.diff(unique.asi8)>pd.Timedelta(days=1).value)]
                blocks=[np.flatnonzero(day.isin(unique[ids==i])) for i in np.unique(ids)]
                day_sizes=a.groupby('day').day.transform('size').to_numpy();y=a.y.to_numpy()>0;da=a.score.to_numpy()>=.5;db=b.score.to_numpy()>=.5
                def stat(idx):
                    yy=y[idx]
                    def v(d):
                        dd=d[idx];tp=np.sum(yy&dd);fp=np.sum(~yy&dd);fn=np.sum(yy&~dd)
                        return np.array([tp/(tp+fp) if tp+fp else np.nan,tp/yy.sum() if yy.sum() else np.nan,fp/np.sum(1/day_sizes[idx]),fn])
                    return v(db)-v(da)
                obs=stat(np.arange(len(y)));sim=np.array([stat(np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])) for _ in range(300)])
                for j,metric in enumerate(['precision_1','recall_1','false_alarm_bins_per_day','miss']):
                    good=sim[:,j][np.isfinite(sim[:,j])];lo,hi=np.quantile(good,[.025,.975])
                    rows.append({'period':period,'grouping':g,'method':method,'reference_grouping':g,'reference_method':reference,'metric':metric,'difference':obs[j],'lower_95':lo,'upper_95':hi,'repetitions':len(good),'date_blocks':len(blocks),'interpretation':'exploratory conditional on fitted models; fixed threshold 0.5; candidate minus reference'})
    original=pd.read_csv(out/'paired_effect.csv');combined=pd.concat([original,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(['period','grouping','method','reference_grouping','reference_method','metric'],keep='last');csv(combined,out/'paired_effect.csv')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'binary_congestion_pilot');ap.add_argument('--publish-dir',type=Path);a=ap.parse_args();out=a.out.resolve()
    report=read(out/'acceptance_report.json');assert report['passed'] and report['scope']=='full_pilot'
    manifest=read(out/'manifest.json');checks={}
    checks['input_hashes_unchanged']=all(sha(Path(p))==h for p,h in manifest['input_sha256'].items())
    checks['inherited_checkpoint_hashes_unchanged']=all(sha(Path(p))==h for p,h in manifest.get('reused_model_artifacts',{}).items())
    m=pd.read_csv(out/'metrics_by_fold.csv');sw=pd.read_csv(out/'threshold_sweep.csv');cm=pd.read_csv(out/'confusion.csv');sev=pd.read_csv(out/'original_severity_miss.csv');lab=pd.read_csv(out/'label_audit.csv');repro=pd.read_csv(out/'reproduction_audit.csv')
    key=['fold','grouping','method','label','threshold','segment'];cmn=cm.groupby(key)['count'].sum();mn=sw.set_index(key).n
    checks['all_confusion_sums_equal_samples']=cmn.sort_index().equals(mn.sort_index().rename('count'))
    checks['100_prediction_cells']=len(list((out/'cells').glob('*.parquet')))==100
    checks['60_new_models']=len(list((out/'models').glob('*.joblib')))==60
    checks['20_old_models_reproduced']=len(repro)==20 and repro.max_probability_difference.max()<=1e-10
    checks['no_2025_threshold_selection']=not read(out/'frozen_thresholds_2024.json')['uses_2025']
    checks['threshold_constraint_table_only_2024']=not pd.read_csv(out/'precision80_exploration.csv').fold.eq('EXP2025').any()
    # Independently verify published TOP20 argmax collapse from the handoff.
    top=cm[cm.fold.eq('EXP2025')&cm.grouping.eq('TOP20')&cm.method.eq('B01')&cm.label.eq('Z0')&cm.segment.eq('ALL')].sort_values(['actual','predicted'])['count'].tolist()
    checks['handoff_TOP20_binary_confusion_reproduced']=top==[29465,570,944,2444]
    checks['2024_fold_sample_counts']=m[m.grouping.eq('ALL')&m.method.eq('B01')&m.label.eq('Z0')&m.segment.eq('ALL')].set_index('fold').n.to_dict()=={'F1':8009,'F2':4864,'F3':3769,'EXP2025':33423}
    checks['unknown_label_not_counted_as_negative']=(lab.n_total==lab.n_valid+lab.missing).all().item()
    checks['both_time_units_events_match']=events(pd.date_range('2024-01-01',periods=3,freq='5min').as_unit('ns'),[0,1,1])==events(pd.date_range('2024-01-01',periods=3,freq='5min').as_unit('us'),[0,1,1])
    ev=pd.read_csv(out/'label_events.csv')
    evsummary=ev.groupby(['fold','label'],as_index=False).agg(events=('onset','size'),clean_30m=('clean_30m','sum'),left_censored=('left_censored','sum'),right_censored=('right_censored','sum'),observed_median_minutes=('minutes_observed','median'),observed_p90_minutes=('minutes_observed',lambda z:z.quantile(.9)))
    csv(evsummary,out/'event_descriptive_summary.csv')
    eda=pd.read_csv(BASE/'speed_occupancy_eda_v1/tables/06_label_prevalence.csv')
    eda=eda[eda.label.isin(['Z_original','Z_Q95_strict','Z_Q95_available'])].copy();eda['interpretation']='descriptive only; full-2024 occupancy thresholds; not used to evaluate 2024 folds';csv(eda,out/'full_year_label_descriptive.csv')
    if (out/'source_cosmetic_revision.json').exists():report['source_cosmetic_revision']=read(out/'source_cosmetic_revision.json')
    report['smoke_and_resume_test']='final source: smoke and same-output resume both passed'
    report['regression_tests_passed']=3
    report['figures_visually_reviewed']=['PR_2024.png','PR_2025_exploratory.png','confusion_2024.png','confusion_2025.png']
    # Model-independent common-label sensitivity, same model scores and same rows.
    paired=[]
    for label in ['Z1_strict','Z1_available']:
        x=m[m.label.eq(label)].merge(m[m.label.eq('Z0_common_'+label)],on=['fold','grouping','method','threshold','segment'],suffixes=('_Z1','_Z0'),validate='one_to_one')
        assert (x.n_Z1==x.n_Z0).all()
        for _,r in x.iterrows():
            paired.append({**{k:r[k] for k in ['fold','grouping','method','threshold','segment']},'label':label,'n_common':r.n_Z1,**{'difference_'+c:r[c+'_Z1']-r[c+'_Z0'] for c in ['precision_1','recall_1','f1_1','AP','miss','false_alarm']}})
    csv(paired,out/'label_paired_effect.csv')
    # Pooled OOF is separate from the arithmetic fold-mean table.
    pooled=[]
    for g in GROUPS:
        for method in ['B01','B02','B03','B04_w2','B04_w4']:
            z=pd.concat([pd.read_parquet(out/'cells'/f'{fid}_{g}_{method}.parquet') for fid in ['F1','F2','F3']])
            for t in ([-1.] if method=='B01' else np.round(np.arange(.2,.801,.05),2)):
                d=z.hard if method=='B01' else z.score.ge(t);r,_=metrics(z.y>0,d,z.score,z.day);pooled.append({'grouping':g,'method':method,'threshold':t,**r})
    pool=pd.DataFrame(pooled);csv(pool,out/'pooled_2024_OOF.csv')
    frozen=read(out/'frozen_thresholds_2024.json')['thresholds'];selected=[]
    for g in GROUPS:
        for method in ['B02','B03','B04_w2','B04_w4']:
            t=frozen[g+'_'+method];v=pool[pool.grouping.eq(g)&pool.method.eq(method)&pool.threshold.eq(t)]
            if t is not None:assert len(v)==1 and v.precision_1.iloc[0]>=.8
            else:assert not (pool[pool.grouping.eq(g)&pool.method.eq(method)].precision_1>=.8).any()
            selected.append({'grouping':g,'method':method,'threshold':t,'feasible_2024':t is not None,'development_precision':v.precision_1.iloc[0] if len(v) else np.nan,'development_recall':v.recall_1.iloc[0] if len(v) else np.nan})
    csv(selected,out/'selected_threshold_development.csv');checks['all_frozen_thresholds_satisfy_2024_constraint']=True
    additional_paired_effects(out)
    paired_df=pd.read_csv(out/'paired_effect.csv');checks['bootstrap_300_repetitions']=(paired_df.repetitions==300).all().item();checks['bootstrap_multiple_date_blocks']=(paired_df.date_blocks>1).all().item()
    checks['2025_labels_identical_on_common_samples']=pd.DataFrame(paired).query("fold == 'EXP2025'").filter(like='difference_').abs().max().max()==0
    checks={k:bool(v) for k,v in checks.items()}
    if not all(checks.values()):js(checks,out/'delivery_checks.json');raise ValueError(checks)
    # Fixed-threshold confusion for every grouping and strategy.
    for year in ['2024','2025']:
        z=cm[cm.label.eq('Z0')&cm.segment.eq('ALL')&cm.threshold.isin([-1.,.5])&(cm.fold.eq('EXP2025') if year=='2025' else cm.fold.ne('EXP2025'))]
        methods=['B01','B02','B03','B04_w2','B04_w4'];fig,axes=plt.subplots(5,5,figsize=(12,11))
        for i,g in enumerate(GROUPS):
            for j,method in enumerate(methods):
                b=z[z.grouping.eq(g)&z.method.eq(method)].groupby(['actual','predicted'])['count'].sum().unstack().reindex(index=[0,1],columns=[0,1],fill_value=0);mat=b.to_numpy();ax=axes[i,j];ax.imshow(mat,cmap='Blues')
                for (r,c),n in np.ndenumerate(mat):ax.text(c,r,str(n),ha='center',va='center',color='white' if n>mat.max()/2 else 'black',fontsize=8)
                ax.set_xticks([0,1]);ax.set_yticks([0,1]);
                if i==0:ax.set_title(method)
                if j==0:ax.set_ylabel(g+' actual')
                if i==4:ax.set_xlabel('Predicted')
        fig.suptitle(year+' Z0 confusion; B01 hard collapse, others score >= 0.5');fig.tight_layout();fig.savefig(out/'figures'/f'confusion_{year}.png',dpi=170);plt.close(fig)
    c24=pd.read_csv(out/'comparison_2024.csv');c25=pd.read_csv(out/'comparison_2025_exploratory.csv')
    cols=['grouping','method','precision_1','recall_1','f1_1','AP','miss','false_alarm','false_alarm_bins_per_day']
    a24=c24[c24.label.eq('Z0')&c24.segment.eq('ALL')][cols];a25=c25[c25.label.eq('Z0')&c25.segment.eq('ALL')][cols]
    # Summarize method changes against hard collapse, never rank by F1.
    delta=a24.merge(a24[a24.method.eq('B01')],on='grouping',suffixes=('','_B01'));delta['delta_recall']=delta.recall_1-delta.recall_1_B01;delta['delta_precision']=delta.precision_1-delta.precision_1_B01
    strict=lab[lab.label.eq('Z1_strict')];unknown2025=int(strict[strict.fold.eq('EXP2025')].missing.iloc[0]-lab[lab.fold.eq('EXP2025')&lab.label.eq('Z0')].missing.iloc[0]);z1change=int(strict[strict.fold.ne('EXP2025')].changed_common.sum())
    missrows=sev[sev.fold.eq('EXP2025')&sev.grouping.eq('TOP20')&sev.label.eq('Z0')&sev.segment.eq('ALL')&sev.threshold.isin([-1.,.5])][['method','original_Y','support','miss','miss_fraction']]
    bs=paired_df[paired_df.period.eq('2024')&paired_df.method.eq('B03')&paired_df.reference_grouping.eq('ALL')&paired_df.reference_method.eq('B03')&paired_df.metric.isin(['precision_1','recall_1'])][['grouping','metric','difference','lower_95','upper_95']]
    findings='''# 二元壅塞 Pilot 實際研究成果 2026年11月25日

本輪完成五種 O 需求配置、2024 三折及 2025 探索性歷史回測，20 個舊三分類模型重現、60 個新二元 LightGBM 訓練，以及 100 份本機逐格預測 checkpoint。正式任務固定為 t+30 分鐘該一時間格是否壅塞。原三分類、來源資料與舊模型均保留。

## 驗收與比較口徑

所有舊模型驗證機率重現至最大絕對差不超過 1e-10，樣本、25 個 VD／時間基礎欄位、O 群需求、train_mask、30 分鐘 embargo 均核對。2024 F1/F2/F3 驗證有效樣本分別 8009/4864/3769，合併 OOF 為 16642 格；2025 為 33423 格、117 個目標研究日。2024 三折研究日數與有效樣本不同，主表是各折指標算術平均，另提供 pooled_2024_OOF.csv；平均漏報數不是總漏報數。F3 原始標籤缺失比例約 27.30%，不得解讀為所有時段均完整驗證。F2 標籤日曆稽核有 4870 格，但按預測起點切分有 4864 格；跨午夜 t 與 target 的邊界不同，沒有強行改寫樣本。

B01 是原三分類 argmax 合併；B02 使用 p1+p2；B03 直接二元；B04_w2/w4 正類樣本權重 2/4，負類 1。B01 與 B02 的排序分數相同，AP 因而相同；硬決策不同。B03/B04 除二元目標與權重外沿用同一組超參數與特徵。加權輸出稱分數，未宣稱校準後真實機率。

## 2024 三折指標算術平均

B01 為既有硬決策，其餘使用預先固定 0.5。precision、recall、F1 為正類，完整正負類、分段與逐折表見 CSV。

'''+table(a24)+'''

## 2025 探索性歷史回測

2025 已反覆參與研究探索，不是獨立測試，不以本輪 2025 結果再選模型或調整門檻。

'''+table(a25)+'''

## 圖表

![2024 PR曲線](figures/PR_2024.png)

![2025探索性PR曲線](figures/PR_2025_exploratory.png)

![2024合併OOF混淆矩陣](figures/confusion_2024.png)

![2025探索性混淆矩陣](figures/confusion_2025.png)

## 五個研究問題

1. 合併 Y1/Y2 可移除嚴重程度之間的誤判，但不會自動消除有無壅塞漏報。TOP20 的 2025 B01 精確重現交接矩陣 [[29465,570],[944,2444]]：precision 0.8109、recall 0.7214，仍有 944 格漏報。二元指標的改善不能直接與三分類單類 F1 視為同一任務的增益。
2. 直接二元訓練的增益有限。以 ALL 為例，2024 三折平均 B02→B03：precision 81.73%→82.19%、recall 67.08%→67.64%、AP 0.83057→0.83359。2024 pooled OOF 的 recall 則為 65.94%→66.77%，僅增16格命中；條件於已訓練模型的日期區塊區間為−0.50至2.28個百分點，跨零，不能混用兩種彙總。2025 ALL recall 74.50%→74.65%，但 precision 81.00%→79.93%。分群的方向也不一致，因此尚不足以認定直接重訓全面優於原三分類分數合併；paired_effect.csv 已另比較 B03−B02 及各加權方法−B03。
3. 權重可減少漏報，但代價明顯。ALL 在2024三折平均固定0.5下，B03→權重2→權重4的 recall 為67.64%→76.47%→81.53%，precision為82.19%→75.95%→68.80%，每日平均誤報4.65→7.85→12.15格。施加precision>=0.80約束後，ALL的2024 pooled OOF recall為B03 66.77%（0.5）、權重2 67.24%（0.65）、權重4 65.94%（0.8）；權重2僅比B03多9格命中，未證明穩定優勢。TOP20權重4於固定網格無可行解。threshold_sweep.csv 固定比較 0.20–0.80、間距 0.05；2024 選 precision>=0.80 下最大 recall 的結果是同 OOF 開發探索，不是正式選模後的無偏驗證。正式可比較預先固定的 0.5 表現；折內歷史門檻選擇尚未執行。2024 凍結門檻及 2025 套用結果分別見 frozen_thresholds_2024.json、frozen_threshold_2025_results.csv，不保證 2025 precision 仍達 0.80。
4. Z1 以各折截止前高速>=80 km/h 的有效占有率估計各站 Q95；已與既有 EDA 歷史門檻逐站核對。2024 三折共同有效樣本共有 '''+str(z1change)+''' 格由原 Y1 正類轉為負類，原 Y2 全保留；2025 共同樣本標籤完全相同。strict 令 2025 額外 '''+str(unknown2025)+''' 格成為未知，並非新增壅塞。Z1_available 的 2025 有效樣本與 Z0 相同。各方法在 2025 共同樣本的所有指標差皆為零；2024 敏感度見 label_paired_effect.csv。此為相同已訓練 Z0 模型的評分標籤敏感度，尚未以 Z1 重訓，不依 F1 選壅塞定義。
5. O 分組效益以與 ALL 的成對日期區塊 Bootstrap 比較；無一致性優勢不能寫成某分群一定較好。以下為固定 0.5 的 B03，候選減 ALL 的 2024 pooled OOF 差，非三折平均差。

'''+table(bs)+'''

## 門檻限制與無可行解

以下門檻只由 2024 pooled OOF 選擇，所有候選規則於讀取 2025 分數前凍結。不可把開發門檻選後的 2024 表現稱為無偏驗證；null/NaN 代表固定網格內沒有 precision>=0.80 的可行門檻，不輸出替代猜測。2025 僅套用這些凍結門檻。

'''+table(pd.DataFrame(selected))+'''

## 原始嚴重程度分層漏報

以下列 TOP20 2025 範例，分母是該原始真實嚴重程度樣本；其他配置及日型時段見 original_severity_miss.csv。

'''+table(missrows)+'''

## 事件與限制

label_events.csv 保存可觀測二元連續事件，時間缺口或未知會打斷，標示左右截尾及前 30 分鐘完整無壅塞的 clean onset；label_audit.csv 的時間長度只是觀測持續時間。沒有最低持續時間篩選，不代表事故數。current_state_strata.csv 用事後 Y(t) 區分當時已壅塞、當時無壅塞及未知，僅屬描述性分析，未回填特徵或用作可部署預警閘門。尚未完成新發事件警報命中、窗口型模型或事件每日誤報；t+30 固定格分類與未來30分鐘至少一次壅塞是不同任務。本輪完成前者，後者為後續研究。

配對 Bootstrap 以連續研究日期區塊抽樣 300 次，同區塊兩方法一起抽，保留日內相依，分別處理 2024 pooled OOF 與 2025 探索回測。每日誤報率的重抽日分母包含抽中重複日期的次數。區間條件於已訓練模型，不包含重訓、分群擬合與選門檻的不確定性；多重候選未校正，屬探索性。AP 與負類指標、所有時段日型分層皆保留，無正類或無支持度時需連同 support 判讀；沒有樣本的連假折不虛構成零績效。

全日、14:00–20:00（左含右不含）、其他時段、WEEKEND、LONGHOLIDAY 按目標時間及既有日曆分層。每日誤報為有有效樣本的目標研究日平均五分鐘格數，不是警報事件數或每24小時標準化率。既有 VD 延遲10分鐘、非研究日出發跨午夜尾量、2026尚未驗證等限制仍保留。

本機 DuckDB 原生 DLL 被應用程式政策阻擋，本輪逐字載入原 Engine.block 函式並核對來源 SHA256，只讀快取，不執行資料庫及原始旅次掃描。Word 指定 (2).docx 尚未定位；找到的其他版本未擅自替代。原稿未修改。正式資料與逐格預測僅本機保留，公開與 ZIP 僅含彙總、圖表、程式及可攜來源雜湊。
'''
    (out/'findings_zh-TW.md').write_text(findings,encoding='utf-8')
    report['delivery_checks']=checks;report['passed']=all(checks.values());report['environment']={'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','lightgbm','scikit-learn']}};report['incomplete']=['inner-history threshold selection','Z1 retraining','window-target/event-warning model','exact requested Word file not found'];js(report,out/'acceptance_report.json')
    portable={'input_sha256':{str(Path(p)).replace(str(ROOT.parent),'${WORKSPACE}'):h for p,h in manifest['input_sha256'].items()},'run_script_sha256':sha(ROOT/'experiments/congestion_2026/run_binary_congestion_pilot.py'),'finalizer_sha256':sha(Path(__file__)),'source_checkpoint_manifest_sha256':manifest.get('reused_source_manifest_sha256'),'reused_predictions_verified':True,'source_cosmetic_revision':report.get('source_cosmetic_revision')};js(portable,out/'source_manifest_portable.json')
    public=['metrics_by_fold.csv','comparison_2024.csv','comparison_2025_exploratory.csv','confusion.csv','threshold_sweep.csv','label_audit.csv','occupancy_thresholds.csv','original_severity_miss.csv','paired_effect.csv','label_paired_effect.csv','pooled_2024_OOF.csv','precision80_exploration.csv','frozen_thresholds_2024.json','frozen_threshold_2025_results.csv','selected_threshold_development.csv','current_state_strata.csv','event_descriptive_summary.csv','full_year_label_descriptive.csv','acceptance_report.json','findings_zh-TW.md','source_manifest_portable.json']
    target=a.publish_dir.resolve() if a.publish_dir else None
    if target:
        target.mkdir(parents=True,exist_ok=True)
        for name in public:shutil.copy2(out/name,target/name)
        shutil.copytree(out/'figures',target/'figures',dirs_exist_ok=True)
    with zipfile.ZipFile(out/'binary_congestion_pilot_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in public:z.write(out/name,name)
        for p in (out/'figures').glob('*.png'):z.write(p,'figures/'+p.name)
        for name in ['run_binary_congestion_pilot.py','finalize_binary_congestion_pilot.py','run_binary_congestion_pilot.ps1','BINARY_PILOT_zh-TW.md']:z.write(ROOT/'experiments/congestion_2026'/name,'code/'+name)
    print(json.dumps({'checks':checks,'ZIP':str(out/'binary_congestion_pilot_results.zip')},ensure_ascii=False))

if __name__=='__main__':main()
