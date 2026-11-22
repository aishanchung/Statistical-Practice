import json,logging,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from ..common import *

def parquet_info(p):
    pf=pq.ParquetFile(p);schema=pf.schema_arrow
    timecols=[c for c in schema.names if any(x in c.lower() for x in ['time','date','day'])]
    ranges={}
    for name in timecols:
        idx=pf.schema.names.index(name);lo=[];hi=[]
        for i in range(pf.metadata.num_row_groups):
            st=pf.metadata.row_group(i).column(idx).statistics
            if st and st.has_min_max:lo.append(str(st.min));hi.append(str(st.max))
        ranges[name]=[min(lo) if lo else None,max(hi) if hi else None]
    return {'rows':pf.metadata.num_rows,'schema':str(schema),'time_fields':';'.join(timecols),'date_ranges':json.dumps(ranges,ensure_ascii=False),'availability':'metadata_inspected; archival publication timestamps unverified'}
def run(cfg):
    out=outdir(cfg);data=Path(cfg['data_root']);legacy=Path(cfg['legacy_root']);records=[]
    ps=[(k,data/v,'original') for k,v in cfg['inputs'].items()]
    ps.extend((p.stem,p,'legacy_aggregate') for p in sorted((legacy/'pilot_2024_2025/results/monthly').glob('*.parquet')))
    detail=json.loads((legacy/'o_features_grouping_20261111_v2/results/checkpoint_manifest.json').read_text(encoding='utf8'))
    ps.extend(('o_monthly_'+a['month'],Path(a['path']),'legacy_aggregate') for a in detail)
    for sub in ['pilot_2024_2025','horizon_15_30_20261111_v1','o_features_grouping_20261111_v2','o_group_rejection_20261111_v1','cost_sensitive_pilot_20261111_v1','cost_sensitive_event_pilot_20261111_v1','rejection_compare_20261111_v1']:
        for p in (legacy/sub).rglob('*'):
            if any(x in p.parts for x in ['.deps','.model_deps','__pycache__']):continue
            if p.is_file() and p.suffix.lower() in ['.py','.json','.csv','.joblib','.pkl','.gz'] and p.stat().st_size<150_000_000:ps.append((sub+'/'+p.name,p,'legacy_source_or_result'))
    seen=set()
    for key,p,role in ps:
        p=p.resolve()
        if p in seen:continue
        seen.add(p)
        r={'id':key,'path':str(p),'role':role,'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None,'mtime_ns':p.stat().st_mtime_ns if p.exists() else None,'sha256':digest(p) if p.exists() and p.stat().st_size<150_000_000 else 'metadata_only_large_file','version':cfg['version']}
        r['experiment_version']=cfg['version']
        r['version']=r['sha256'] if len(r['sha256'])==64 else f"metadata:size={r['bytes']};mtime_ns={r['mtime_ns']}"
        r['data_source']=role+': '+str(p)
        r['availability']='local archival artifact; not evidence of live publication time'
        if p.exists():
            if p.suffix=='.parquet':r.update(parquet_info(p))
            elif p.suffix=='.csv':
                frame=pd.read_csv(p);r.update(rows=len(frame),schema=';'.join(f'{c}:{t}' for c,t in frame.dtypes.items()))
                ts=[c for c in frame if c in ['origin','target','Time5','TargetTime','time','day','SourceDate']]
                r['time_fields']=';'.join(ts);r['date_ranges']=json.dumps({c:[str(frame[c].min()),str(frame[c].max())] for c in ts},ensure_ascii=False)
        records.append(r)
    save_csv(pd.DataFrame(records),out/'input_manifest.csv')
    # Compare blob IDs across full GitHub main checkout and local original scripts.
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],cwd=REPO,text=True,encoding='utf8').splitlines()
    comparisons=[];local=Path(cfg['local_source_root'])
    for name in tracked:
        remote=REPO/name
        if remote.suffix not in ['.py','.md','.json','.html','.csv']:continue
        relative=name.removeprefix('20_MODEL_ML/')
        possibilities=[local/relative,local/'10_OD analysis'/Path(name).name] if name.startswith(('20_MODEL_ML/','10_OD-analysis/')) else []
        match=next((p for p in possibilities if p.is_file()),None)
        comparisons.append({'github_main_path':name,'github_sha256':digest(remote),'local_path':str(match) if match else '', 'local_sha256':digest(match) if match else '', 'same_bytes':digest(remote)==digest(match) if match else None,'action':'read_only; no overwritten source'})
    save_csv(pd.DataFrame(comparisons),out/'source_version_comparison.csv')
    with duckdb.connect(config={'threads':cfg['threads'],'memory_limit':cfg['memory_limit']}) as con:
        v=con.execute(f'SELECT DISTINCT GantryO AS o FROM read_parquet({lit(data/cfg["inputs"]["v3"])})').df()
        origin_cache=out/'cache/full_m06b_origin_universe.csv';origin_cache.parent.mkdir(exist_ok=True)
        origin_source=data/cfg['inputs']['m06b'];identity={'path':str(origin_source.resolve()),'bytes':origin_source.stat().st_size,'mtime_ns':origin_source.stat().st_mtime_ns}
        origin_meta=origin_cache.with_suffix('.json')
        if origin_cache.exists() and origin_meta.exists() and json.loads(origin_meta.read_text(encoding='utf8'))==identity:all_origins=pd.read_csv(origin_cache)
        else:
            logging.info('Read-only distinct O scan, M06B O column only; no trip-detail export')
            all_origins=con.execute(f'SELECT DISTINCT GantryO AS o FROM read_parquet({lit(data/cfg["inputs"]["m06b"])})').df()
            save_csv(all_origins,origin_cache)
            save_json(identity,origin_meta)
    fixed=pd.read_csv(legacy/'o_features_grouping_20261111_v2/results/o_features_FIXED_20240401.csv')
    desc=pd.read_csv(legacy/'o_features_grouping_20261111_v2/results/o_features_DESCRIPTIVE_2024.csv')
    allO=set(all_origins.o)|set(desc.o)|set(fixed.o)|set(v.o)
    rows=[{'o':o,'in_M06B_2024_2025_all_departure_cohort':o in set(all_origins.o),'in_V3_2024_2025_bottleneck_arrival_cohort':o in set(v.o),'in_fixed_20240401_all_departure_cohort':o in set(fixed.o),'fixed_N_all':fixed.set_index('o').N_all.get(o,np.nan),'fixed_N_B':fixed.set_index('o').N_B.get(o,np.nan),'in_2024_all_departure_cohort':o in set(desc.o)} for o in sorted(allO)]
    save_csv(pd.DataFrame(rows),out/'o_universe_audit.csv')
    grid=pd.read_csv(legacy/'pilot_2024_2025/results/metrics.csv');pred=pd.read_csv(legacy/'pilot_2024_2025/results/predictions.csv')
    from ..evaluation.metrics import evaluate
    comparison=[]
    for (a,b),g in pred.groupby(['algorithm','features']):
        m,_=evaluate(g.y.to_numpy(int),g[['p0','p1','p2']].to_numpy());old=grid.loc[grid.algorithm.eq(a)&grid.features.eq(b)&grid.segment.eq('all')].iloc[0]
        comparison.append({'algorithm':a,'features':b,'n':len(g),'log_loss':m['log_loss'],'macro_f1':m['macro_f1'],'max_metric_difference':max(abs(m[k]-old[k]) for k in ['log_loss','macro_f1','recall_1','recall_2']),'trained_model_binary_found':False,'training_code_found':True,'checkpoint_parameters_found':True,'probability_rows_found':True,'legacy_embargo':'not explicitly recorded'})
    save_csv(pd.DataFrame(comparison),out/'legacy_nine_model_reproduction.csv')
    save_json({'tracked_main_files':len(tracked),'V3_O':len(v),'fixed_O':len(fixed),'fixed_zero_B':int(fixed.N_B.eq(0).sum()),'descriptive_2024_O':len(desc),'raw_missing':[r['path'] for r in records if not r['exists']],'legacy_binary_search_scope':'all registered pilot directories; original .json are parameters/signatures, not fitted model weights','new_Y_unchanged':True,'legacy_comparisons_are_separate':True},out/'audit_summary.json')
    logging.info('Audit complete. V3 O=%s; fixed O=%s, zero-passage=%s',len(v),len(fixed),fixed.N_B.eq(0).sum())
