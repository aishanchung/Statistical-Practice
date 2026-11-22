from __future__ import annotations
import ast,hashlib,json,logging,os,platform,subprocess,sys,time
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
def load_config(file=None):
    cfg=json.loads(Path(file or EXP/'config/default.json').read_text(encoding='utf-8-sig'))
    for key in ('data_root','legacy_root','local_source_root','output_root'):
        override=os.environ.get('CONGESTION_'+key.upper())
        cfg[key]=str((REPO/Path(os.path.expandvars(override or cfg[key]))).resolve())
    if cfg['primary_horizon_minutes']!=30:raise ValueError('Primary protocol is h=30')
    if cfg['availability']['embargo_minutes']<max([cfg['primary_horizon_minutes'],*cfg['sensitivity_horizons_minutes']]):raise ValueError('Embargo shorter than horizon')
    read_only_guard([cfg['data_root'],cfg['legacy_root'],cfg['local_source_root']],[Path(cfg['output_root'])/cfg['run_id']])
    return cfg
def outdir(cfg):
    p=Path(cfg['output_root'])/cfg['run_id'];p.mkdir(parents=True,exist_ok=True);return p
def save_json(v,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8');q.replace(p)
def save_csv(v,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');v.to_csv(q,index=False,encoding='utf-8-sig');q.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def signature(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
def setup(cfg):
    out=outdir(cfg)
    existing=out/'run_config.json'
    previous=json.loads(existing.read_text(encoding='utf8')) if existing.exists() else {}
    if previous and previous['config_signature']!=signature(cfg):raise ValueError('Config changed. Use a new run_id; do not mix checkpoints.')
    sealed=out/'sealed_provenance.json'
    if sealed.exists():
        approved=json.loads(sealed.read_text(encoding='utf8'))
        if provenance(cfg)!=approved:raise ValueError('Sealed source/data/dependencies changed. Use a new run_id instead of reusing incompatible checkpoints.')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log',encoding='utf8')],force=True)
    save_json({'config':cfg,'config_signature':signature(cfg),'git_main_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','duckdb','scikit-learn','lightgbm','streamlit','plotly']},'platform':platform.platform()},existing)
    return out
def provenance(cfg):
    """Fingerprint primary data, reused owning code, aggregates and new source."""
    legacy=Path(cfg['legacy_root']);files=[Path(cfg['data_root'])/v for v in cfg['inputs'].values()]
    files += sorted((legacy/'pilot_2024_2025/results/monthly').glob('*.parquet'))
    manifest=legacy/'o_features_grouping_20261111_v2/results/checkpoint_manifest.json'
    files += [Path(r['path']) for r in json.loads(manifest.read_text(encoding='utf8'))]
    files += [legacy/p for p in ['pilot_2024_2025/features.py','pilot_2024_2025/model.py','o_features_grouping_20261111_v2/features.py','o_features_grouping_20261111_v2/groups.py','pilot_2024_2025/results/cohort_completion.csv','horizon_15_30_20261111_v1/results/features_h15.csv','horizon_15_30_20261111_v1/results/features_h30.csv']]
    files += [p for p in EXP.rglob('*.py') if not any(part in ['outputs','tests'] for part in p.relative_to(EXP).parts)]
    return {'config_signature':signature(cfg),'files':{str(p.resolve()):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':digest(p) if p.stat().st_size<5_000_000 else 'metadata_fingerprint_large_source'} for p in sorted(set(files))},'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','duckdb','scikit-learn','lightgbm']}}
def seal(cfg):save_json(provenance(cfg),outdir(cfg)/'sealed_provenance.json')
def registry(cfg,stage,status,seconds,detail=''):
    p=outdir(cfg)/'experiment_registry.csv'
    row={'stage':stage,'status':status,'seconds':seconds,'detail':detail,'timestamp':pd.Timestamp.now().isoformat(),'config_signature':signature(cfg)}
    df=pd.read_csv(p) if p.exists() else pd.DataFrame();save_csv(pd.concat([df,pd.DataFrame([row])],ignore_index=True),p)
def lit(v):return "'"+str(v).replace("'","''")+"'"
def legacy_functions(cfg,relative,names,environment=None):
    """Compile named owning functions verbatim without running legacy module globals.

    This preserves legacy math and prevents imports from creating/writing old OUT.
    Source hashes are recorded in the input manifest.
    """
    path=Path(cfg['legacy_root'])/relative
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError(f'Missing legacy functions in {path}')
    ns={'np':np,'pd':pd,'json':json,'Path':Path,'lit':lit,'literal':lit,'KINDS':('WEEKEND','LONGHOLIDAY'),'SLOTS':288,'SITES':('69.510','65.990','63.990','61.670')}
    ns.update(environment or {});exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns
def read_only_guard(inputs,outputs):
    ins=[Path(p).resolve() for p in inputs]
    for p in outputs:
        p=Path(p).resolve()
        if any(p==q or q in p.parents for q in ins):raise ValueError('Output would overwrite an input')
    return True
