"""Windows: python -m experiments.congestion_2026.run --stage all"""
import argparse,logging,time
from .common import load_config,setup,registry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');ap.add_argument('--stage',choices=['audit','features','train','analyze','export','all'],default='all');args=ap.parse_args()
    cfg=load_config(args.config);setup(cfg)
    for stage in (['audit','features','train','analyze','export'] if args.stage=='all' else [args.stage]):
        start=time.perf_counter();logging.info('START %s',stage)
        try:
            if stage=='audit':
                from .data.audit import run;run(cfg)
                from .data.raw_archive_inventory import run as raw_inventory;raw_inventory(cfg)
            elif stage=='features':
                from .features.engine import Engine;Engine(cfg).run()
            elif stage=='train':
                from .models.runner import run;run(cfg)
            elif stage=='analyze':
                from .evaluation.analysis import run;run(cfg)
            elif stage=='export':
                from .data.export_dashboard import run;run(cfg)
            registry(cfg,stage,'complete',time.perf_counter()-start)
        except Exception as error:
            registry(cfg,stage,'failed',time.perf_counter()-start,repr(error));logging.exception('Stage failed');raise
if __name__=='__main__':main()
