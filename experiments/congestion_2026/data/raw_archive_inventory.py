"""Metadata-only inventory of optional raw CSV archives, never a model input."""
import ast,os,json,logging
from pathlib import Path
import pandas as pd
from ..common import *

def run(cfg):
    out=outdir(cfg);parent=Path(cfg['data_root']).parent
    explicit=os.environ.get('CONGESTION_RAW_M06B_ROOT')
    candidates=[Path(explicit)] if explicit else [p for p in parent.iterdir() if p.is_dir() and 'M06B' in p.name and '原始' in p.name]
    records=[];summary=[]
    source=Path(cfg['local_source_root'])/'02_資料分割/M06B_2024_2025_假日原始資料合併.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'));names=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_COLUMNS' for t in node.targets):names=ast.literal_eval(node.value)
    if not names:raise ValueError('Cannot verify headerless raw schema')
    for root in candidates:
        dates=sorted(p for p in root.iterdir() if p.is_dir() and len(p.name)==8 and p.name.isdigit())
        for date in dates:
            for p in date.glob('*/*.csv'):
                records.append({'path':str(p.resolve()),'source_date':date.name,'source_hour':p.parent.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'schema':','.join(names),'row_count':None,'row_count_status':'not scanned; raw candidate not used in this model','availability':'not yet complete-cohort/publication audited','role':'raw_archive_candidate_only'})
        summary.append({'root':str(root.resolve()),'date_directories':len(dates),'first_date_directory':dates[0].name if dates else None,'last_date_directory':dates[-1].name if dates else None,'csv_files':sum(Path(r['path']).is_relative_to(root.resolve()) for r in records),'complete_rows_verified':False,'current_model_uses_raw_candidate':False})
    save_csv(pd.DataFrame(records),out/'raw_archive_inventory.csv');save_json({'archives':summary,'schema_owner':str(source),'no_trip_details_exported':True,'next_step':'Audit only required neighboring nonholiday cohorts before expanding N/P/Q; do not merge unverified raw files into B0'},out/'raw_archive_summary.json')
    # Link the optional raw candidate inventory from the common manifest. The
    # primary model-input rows retain their separately audited exact counts.
    manifest=pd.read_csv(out/'input_manifest.csv');manifest=manifest[manifest.id.ne('optional_raw_archive_inventory')]
    p=out/'raw_archive_inventory.csv'
    row={'id':'optional_raw_archive_inventory','path':str(p.resolve()),'role':'metadata_index_of_raw_candidates_not_model_input','exists':True,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':digest(p),'version':digest(p),'experiment_version':cfg['version'],'data_source':';'.join(s['root'] for s in summary),'availability':'raw candidate only; not used for current N/P/Q','rows':len(records),'schema':'hourly CSV paths, source date/hour, file metadata, verified nine-column schema; raw row counts unscanned','time_fields':'SourceDate,SourceHour','date_ranges':json.dumps(summary,ensure_ascii=False)}
    save_csv(pd.concat([manifest,pd.DataFrame([row])],ignore_index=True),out/'input_manifest.csv')
    # Existing completed A-class diagnostics use V3 arrivals only, not an invalid
    # V3/M06B passage probability. Preserve this denominator explicitly.
    old=pd.read_csv(Path(cfg['legacy_root'])/'pilot_2024_2025/results/a_arrival_hour.csv')
    a=old.groupby('bhour')[['n_v3','n_a']].sum().reset_index();a['A_share_among_V3_arrivals']=a.n_a/a.n_v3
    a['denominator']='V3 bottleneck arrivals, not M06B all departures';save_csv(a,out/'A_crossday_by_arrival_hour.csv')
    logging.info('Optional raw archive metadata: %s date directories, %s CSV candidates; zero new trip processing',sum(s['date_directories'] for s in summary),len(records))
if __name__=='__main__':
    cfg=load_config();setup(cfg);run(cfg)
