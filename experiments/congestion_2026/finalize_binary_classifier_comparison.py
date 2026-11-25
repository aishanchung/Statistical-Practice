"""Final source/checkpoint audit, six research answers, slide figures, public ZIP."""
from pathlib import Path
import argparse,json,shutil,zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.congestion_2026.analyze_binary_classifier_comparison import (ROOT,BASE,SOURCE,ALGORITHMS,METHODS,GROUPS,NAMES,COLORS,load_prediction,read,js,csv,sha,table,savefig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'binary_classifier_comparison');ap.add_argument('--publish-dir',type=Path);args=ap.parse_args();out=args.out.resolve();acc=read(out/'acceptance_report.json');assert acc['passed'] and acc['scope']=='full'
    mf=read(out/'manifest.json');checks={};cp=list((out/'checkpoints').glob('*.json'));assert len(cp)==160
    checks['160_checkpoint_hashes_verified']=all(sha(out/'cells'/(p.stem+'.parquet'))==read(p)['prediction_sha256'] and sha(out/'models'/(p.stem+'.joblib'))==read(p)['model_sha256'] for p in cp)
    checks['input_files_unchanged']=all(sha(p)==h for p,h in mf['input_sha256'].items())
    checks['analysis_code_matches_report']=sha(ROOT/'experiments/congestion_2026/analyze_binary_classifier_comparison.py')==acc['analysis_script_sha256']
    registry=pd.read_csv(out/'experiment_registry.csv');checks['300_common_sample_configurations']=len(registry)==300 and registry.groupby('fold').n.nunique().eq(1).all()
    stat=pd.read_csv(out/'training_statistics.csv');checks['12_old_logistic_predictions_reproduced_exactly']=stat.existing_three_class_predictions_reused.sum()==12 and stat.legacy_probability_max_abs_difference.dropna().max()<=1e-9
    precision=pd.read_csv(out/'precision80_development.csv');checks['60_threshold_rules_frozen_on_2024']=len(precision)==60 and not read(out/'frozen_thresholds_2024.json')['uses_2025']
    ci=pd.read_csv(out/'paired_effect.csv');checks['300_valid_bootstrap_draws']=ci.valid_repetitions.eq(300).all()
    source_ap=ci[ci.family.eq('decision_only')&ci.metric.eq('AP')];checks['B01_B02_identical_ranking_AP_difference_zero']=source_ap.difference.eq(0).all()
    checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
    mean=pd.read_csv(out/'comparison_2024.csv');pool=pd.read_csv(out/'pooled_metrics.csv');explore=pd.read_csv(out/'comparison_2025_exploratory.csv')
    # Supplementary standalone figures for classroom slides and compact paper panels.
    for period,fids in [('2024',['F1','F2','F3']),('2025_exploratory',['EXP2025'])]:
        fig,ax=plt.subplots(figsize=(6.4,4.8))
        for alg in ALGORITHMS:
            a=pd.concat([load_prediction(out,f,'ALL',alg,'B03') for f in fids]);p,r,_=precision_recall_curve(a.y>0,a.score);ax.plot(r,p,color=COLORS[alg],label=f'{NAMES[alg]}: AP={average_precision_score(a.y>0,a.score):.4f}',lw=1.8)
        ax.axhline(.8,color='gray',ls=':',lw=1);ax.set(xlabel='Recall',ylabel='Precision',title=period+' / ALL / direct binary',xlim=(0,1),ylim=(0,1));ax.legend(loc='lower left',fontsize=9);savefig(fig,out,'F02c_PR_ALL_'+period)
        z=ci[ci.period.eq(period)&ci.family.eq('O_grouping')];ident=['algorithm','grouping'];items=z[ident].drop_duplicates().reset_index(drop=True);labels=[f'{NAMES[r.algorithm]} {r.grouping} - ALL' for r in items.itertuples()];fig,axes=plt.subplots(1,2,figsize=(10,6.5))
        for ax,metric in zip(axes,['AP','recall_1']):
            a=z[z.metric.eq(metric)].merge(items.reset_index(names='order'),on=ident).sort_values('order');yy=np.arange(len(a));ax.hlines(yy,a.lower_95,a.upper_95,color='#4477AA');ax.plot(a.difference,yy,'o',color='#4477AA');ax.axvline(0,color='gray',ls=':');ax.set(yticks=yy,yticklabels=labels if ax is axes[0] else [],xlabel='Candidate minus ALL',title=metric);ax.invert_yaxis()
        fig.suptitle(period+' / B03 / paired date-block bootstrap');savefig(fig,out,'F08_O_grouping_CI_'+period)
    c24=mean[mean.segment.eq('ALL')&mean.grouping.eq('ALL')&mean.method.eq('B03')][['algorithm','precision_1','recall_1','f1_1','AP','false_alarm_bins_per_day']]
    c25=explore[explore.segment.eq('ALL')&explore.grouping.eq('ALL')&explore.method.eq('B03')][['algorithm','precision_1','recall_1','AP','miss','false_alarm']]
    cls=ci[ci.period.eq('2024')&ci.grouping.eq('ALL')&ci.family.eq('classifier')&ci.metric.isin(['AP','precision_1','recall_1'])][['algorithm','reference_algorithm','metric','difference','lower_95','upper_95']]
    retrain=ci[ci.period.eq('2024')&ci.grouping.eq('ALL')&ci.family.eq('binary_retraining')&ci.metric.isin(['AP','precision_1','recall_1'])][['algorithm','metric','difference','lower_95','upper_95']]
    weighted=ci[ci.period.eq('2024')&ci.grouping.eq('ALL')&ci.family.eq('weighting')&ci.metric.eq('AP')][['algorithm','method','difference','lower_95','upper_95']]
    oci=ci[ci.period.eq('2024')&ci.family.eq('O_grouping')&ci.metric.eq('AP')][['algorithm','grouping','difference','lower_95','upper_95']]
    comparison80=precision[precision.grouping.eq('ALL')][['algorithm','method','threshold','development_precision','development_recall','feasible']]
    smallcoef=pd.read_csv(out/'logistic_direction_stability.csv');features=['demand__ALL','vd_65_990_speed_lag10','vd_69_510_occ_lag10','vd_63_990_speed_lag25'];coefshow=smallcoef[smallcoef.feature.isin(features)]
    findings='''# 二元壅塞分類器比較 實際研究結論

本輪已完成三種分類器、五種O分組、五種策略及四評估期，共300個預測配置。LightGBM100份既有預測完全沿用；Logistic／XGBoost共160個訓練checkpoint，其中12個Logistic三分類舊機率經重現後重用，其餘148個新模型；B01與B02共用同一個三分類模型，沒有重複為兩者訓練。全部使用既有.venv，不掃M06B，也沒有調參或更動Y。smoke及同目錄續跑通過，原binary_congestion_pilot及未提交成果保留。

## 1 哪種分類器較好

沒有跨指標全面勝出的分類器。預先指定ALL／B03／0.5時，2024三折平均AP分別為Logistic 0.83399、XGBoost 0.83340、LightGBM 0.83359，差距很小；Logistic的precision與recall略高，XGBoost在0.5較保守。

'''+table(c24)+'''

配對區間以下採2024合併OOF，不是上表三折平均。ALL/B03的合併AP分別為0.81122／0.82238／0.81418；XGBoost−LightGBM的AP差+0.00819，95%區間[−0.00021,0.01616]，仍跨零。Logistic−LightGBM的AP差−0.00297，區間[−0.03140,0.02503]。XGBoost固定0.5的recall較低，但不能因此說其整體排序必然較差。AP跨折合併還受折間分數尺度影響，主表與合併OOF均保留。

'''+table(cls)+'''

## 2 直接二元重訓是否都有額外效益

不是。ALL的B03−B02合併OOF AP差：Logistic −0.00009、XGBoost −0.00102、LightGBM +0.00012，三者區間均跨零。固定0.5的recall增量約0.57／0.52／0.83個百分點；Logistic本輪區間約[0.19,0.95]個百分點，但幅度小、precision略降，不能將此小型決策改善等同排序全面提升。不同O分組另列完整配對結果，不以單一配置宣布所有分類器均應重訓。

'''+table(retrain)+'''

## 3 加權改善排序還是改變決策

主要證據支持改變漏報與誤報取捨，尚無一致的AP提升。ALL固定0.5下，Logistic權重4的三折平均recall升至82.71%，precision降至68.83%；LightGBM及XGBoost亦有相同方向。ALL權重2、4相對B03的2024合併AP配對區間全部跨零；Logistic權重4的AP點估計反而下降0.01812。不能將recall提升稱為機率校準改善或排序能力必然改善。

'''+table(weighted)+'''

## 4 ALL是否仍具競爭力

就2024主比較而言，ALL在三分類器下均具競爭力。B03下，其他O分組相對ALL的2024合併AP區間沒有任何一項顯示穩定正增益；部分細分配置較差，例如XGBoost TOP20、TS10以及LightGBM TOP20的區間均位於零以下。這是條件於已訓練模型、未作多重比較校正的探索結果，不代表所有細分需求永遠無用；ALL適合作為下一輪保留的簡單參照配置。

'''+table(oci)+'''

## 5 Logistic是否能用簡單模型達到相近效果

可以。ALL／B03原始輸入26欄，補值與缺失指示展開後42個係數加截距；形式較160棵提升樹簡單，且2024分類指標及AP與LightGBM相近，配對差異未建立穩定排序優勢。這不是等效性檢定，也不是實際部署延遲benchmark，不能把「未建立差異」當成證明模型完全等效。

係數是訓練資料的條件關聯。ALL需求的標準化係數在三折為正，K65.990的lag10速度為負，K69.510的lag10占有率為正；K63.990的lag25速度可為正，需連同lag10與其餘高度相關VD欄位理解，不能解讀為速度提高導致壅塞。時段sin/cos必須聯合解讀，單一正負號不是某小時的效應。下表為2024訓練折的範圍，並非係數信賴區間；沒有因果或獨立顯著性主張。

'''+table(coefshow)+'''

## 6 相近Precision下的Recall

以下只使用2024 pooled OOF，在預先固定0.20–0.80、間距0.05網格內選precision至少0.80的最高recall。ALL的B03中，Logistic precision81.80%、recall67.55%；LightGBM 80.80%／66.77%；XGBoost81.73%／64.02%。Logistic略高，但不能忽略實際precision不同及開發選門檻偏差。ALL權重2下Logistic為80.59%／68.43%，XGBoost為81.17%／64.80%，LightGBM為80.21%／67.24%；不能據此宣稱部署效能已驗證。另以precision 0.80–0.82窄帶列出similar_precision_2024.csv，無可行者明示缺項。

'''+table(comparison80)+'''

TOP20的Logistic權重4及LightGBM權重4在主網格無precision>=0.80可行門檻；XGBoost各配置在本網格皆有可行解。2025只套用2024凍結規則，不為未達precision目標的配置補選門檻。原始0.5與門檻敏感度並列，不用2025宣告最佳分類器。

## 2025探索性回測

'''+table(c25)+'''

以上仍是歷史探索，不是新獨立測試；2026維持未來獨立驗證定位。2025的方向與2024不完全相同，支持保留限制而非追逐單年最高分。

## 圖表

![三分類器Precision與Recall](figures/F01_precision_recall_2024.png)

![三分類器PR曲線](figures/F02c_PR_ALL_2024.png)

![原Y1和Y2漏報](figures/F04_severity_miss_2024.png)

![不同O分組](figures/F05_O_grouping_2024.png)

![配對區間](figures/F06_paired_CI_2024.png)

![Logistic標準化係數](figures/F07_logistic_coefficients.png)

## 驗收與未完成項目

300配置共同樣本、160個模型checkpoint雜湊、既有LightGBM全部固定決策指標、12份舊Logistic機率、二元混淆矩陣分母及AP加權Bootstrap計算均已核對。完整方法、參數、時間切分、時段／日型及缺失口徑見 methods_zh-TW.md。圖表同時提供300dpi PNG及向量PDF，係數全表、逐折指標、完整0.2–0.8門檻網格及配對差異均附CSV。

本輪只完成固定t+30時間格分類。current_state_strata依事後Y(t)描述延續與當時無壅塞的轉變，保留未知，不作可部署資訊；未證實第一個新發事件，也未訓練「未來30分鐘內至少一次」窗口模型。折內歷史門檻選擇及正式選模後驗證仍為後續工作。Bootstrap條件於已訓練模型、300次、2024有29個連續研究日期區塊、2025有51個，未含重訓／選模不確定性，也未作多重比較校正。加權分數不等同校準後真實機率；不以Macro-F1選模。
'''
    # Preserve detailed analysis documentation separately, replacing only the derived findings in this new run.
    original=(out/'findings_zh-TW.md').read_text(encoding='utf-8')
    if not (out/'methods_zh-TW.md').exists():(out/'methods_zh-TW.md').write_text(original,encoding='utf-8')
    (out/'findings_zh-TW.md').write_text(findings,encoding='utf-8')
    csv([{'figure':p.stem,'PNG':p.name,'PDF':p.with_suffix('.pdf').name,'resolution':'PNG 300 dpi; PDF vector','interpretation':'2024 F01/F05 fold means; PR/CM/CI/miss pooled; 2025 exploratory; F07 training coefficients, not causal'} for p in sorted((out/'figures').glob('*.png'))],out/'figure_catalog.csv')
    acc['delivery_checks']=checks;acc['reused_logistic_three_class_cells']=12;acc['new_models_not_counting_reproduction']=148;acc['smoke_and_resume_verified']=True;acc['regression_tests_passed']=3;acc['full_test_suite']=read(out/'test_validation.json') if (out/'test_validation.json').exists() else 'not automatically run by finalizer';acc['figures_visually_reviewed']=True;acc['figure_count']=len(list((out/'figures').glob('*.png')));acc['delivery_script_sha256']=sha(Path(__file__));js(acc,out/'acceptance_report.json')
    portable=read(out/'source_manifest_portable.json');portable['delivery_script_sha256']=sha(Path(__file__));js(portable,out/'source_manifest_portable.json')
    public=[p for p in out.iterdir() if p.suffix in ['.csv','.md'] or p.name in ['acceptance_report.json','source_manifest_portable.json','model_specifications.json']]
    # frozen candidate JSON is aggregate and 2024-only.
    public.append(out/'frozen_thresholds_2024.json')
    if args.publish_dir:
        target=args.publish_dir.resolve();target.mkdir(parents=True,exist_ok=True)
        for p in public:shutil.copy2(p,target/p.name)
        shutil.copytree(out/'figures',target/'figures',dirs_exist_ok=True)
    with zipfile.ZipFile(out/'binary_classifier_comparison_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in public:z.write(p,p.name)
        for p in (out/'figures').iterdir():z.write(p,'figures/'+p.name)
        for name in ['run_binary_classifier_comparison.py','analyze_binary_classifier_comparison.py','finalize_binary_classifier_comparison.py','run_binary_classifier_comparison.ps1','BINARY_CLASSIFIER_COMPARISON_zh-TW.md','requirements_binary_classifier.txt']:
            p=ROOT/'experiments/congestion_2026'/name
            if p.exists():z.write(p,'code/'+p.name)
    with zipfile.ZipFile(out/'binary_classifier_comparison_results.zip') as z:assert z.testzip() is None and not any(n.endswith(('.parquet','.joblib')) for n in z.namelist())
    print(json.dumps({'delivery_checks':checks,'figures':acc['figure_count'],'ZIP':str(out/'binary_classifier_comparison_results.zip')},ensure_ascii=False))

if __name__=='__main__':main()
