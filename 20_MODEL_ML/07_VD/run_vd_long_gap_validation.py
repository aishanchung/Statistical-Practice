#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VD long-gap retrospective imputation benchmark (2026-10-29).

Run from the directory containing the four project scripts:
  vd_spatial_methods_benchmark.py
  vd_gp_kriging_experiment.py
  vd_robust_mask_report.py
  vd_imputation_experiment.py

Examples (PowerShell):
  py run_vd_long_gap_validation.py "VD_N3_N_2024_2025_LITE.parquet"
  py run_vd_long_gap_validation.py "VD_N3_N_2024_2025_LITE.parquet" --only-hours 6 24
  py run_vd_long_gap_validation.py "VD_N3_N_2024_2025_LITE.parquet" --dry-run

Uses only VD speeds, never OD. Always holds out 2025 known truth for scoring.
The old parquet file is never changed. Method counts per time horizon can differ
when there are not enough usable, non-overlapping 2025 intervals.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# (Duration in hours, desired artificial-missing blocks PER sampling stratum)
# 12+12 for six-hour gaps, 8+8 for 24-hour, 6+6 for 72-hour, 4+4 for 151.5-hour.
# Long durations naturally offer fewer nearly independent candidate windows.
PLAN = [(6.0, 12), (24.0, 8), (72.0, 6), (151.5, 4)]
SCRIPT_FILES = (
    'vd_imputation_experiment.py',
    'vd_spatial_methods_benchmark.py',
    'vd_gp_kriging_experiment.py',
    'vd_robust_mask_report.py',
)


def run_command(args: list[str], *, dry_run: bool) -> None:
    printable = subprocess.list2cmdline(args)
    print('RUN:', printable, flush=True)
    if not dry_run:
        subprocess.run(args, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('parquet', type=Path, help='Complete five-station VD .parquet')
    ap.add_argument('--output', type=Path, default=Path('outputs/vd_long_gap_validation'))
    ap.add_argument('--only-hours', type=float, nargs='+', default=None)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--context-hours', type=float, default=24.0,
                    help='GP/Kriging smoothing context on EACH side of gap')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-gp', action='store_true',
                    help='First pass: nine methods only, then separately add GP.')
    ns = ap.parse_args()

    # Follow the research project's convention: run from the script folder.
    home = Path(__file__).resolve().parent
    os.chdir(home)
    parquet = ns.parquet.expanduser().resolve()
    out = ns.output.expanduser().resolve()
    if not ns.dry_run and not parquet.is_file():
        ap.error(f'VD source file not found: {parquet}')
    missing = [f for f in SCRIPT_FILES if not (home / f).is_file()]
    if missing and not ns.dry_run:
        ap.error('Place the scripts in the same directory: ' + ', '.join(missing))
    if ns.bootstrap < 100:
        ap.error('--bootstrap must be at least 100')

    requested = {round(h, 3) for h in ns.only_hours} if ns.only_hours else None
    plan = [(hours, blocks) for hours, blocks in PLAN
            if requested is None or hours in requested]
    if not plan:
        ap.error('No valid durations: choose from 6, 24, 72, 151.5')

    all_reports = []
    for hours, nblocks in plan:
        label = str(hours).replace('.', 'p')
        root = out / f'gap_{label}h'
        nine, gp, combined = root / 'nine', root / 'gp', root / 'eleven'
        for dest in (nine, gp, combined):
            if not ns.dry_run:
                dest.mkdir(parents=True, exist_ok=True)

        # The 9-method benchmark selects test blocks from 2025.
        # For each block it masks 1, 2, 3, 4, or 5 adjacent stations.
        cmd1 = [sys.executable, str(home / 'vd_spatial_methods_benchmark.py'),
                str(parquet), '--output', str(nine),
                '--hours', str(hours), '--trials-per-stratum', str(nblocks),
                '--rf-rows', '30000', '--rf-trees', '50',
                '--knn-rows', '30000', '--threads', str(ns.threads),
                '--seed', str(ns.seed), '--skip-actual']
        run_command(cmd1, dry_run=ns.dry_run)

        details = [str(nine / 'benchmark_by_block_station.csv')]
        if not ns.skip_gp:
            # The two GP-family models reuse the EXACT SAME held-out blocks.
            cmd2 = [sys.executable, str(home / 'vd_gp_kriging_experiment.py'),
                    str(parquet), '--previous', str(nine), '--output', str(gp),
                    '--hours', str(hours), '--blocks-per-stratum', str(nblocks),
                    '--context-hours', str(ns.context_hours), '--skip-actual']
            run_command(cmd2, dry_run=ns.dry_run)

        # Bootstrap resamples complete masked blocks, not single 5-min rows.
        cmd3 = [sys.executable, str(home / 'vd_robust_mask_report.py'),
                details[0], '--output', str(combined),
                '--bootstrap', str(ns.bootstrap), '--seed', str(ns.seed)]
        if not ns.skip_gp:
            cmd3 += ['--extra-details', str(gp / 'gp_kriging_by_block_station.csv')]
        run_command(cmd3, dry_run=ns.dry_run)
        if not ns.dry_run:
            import pandas as pd
            df = pd.read_csv(combined / 'robust_metrics_with_ci.csv')
            # Keep all 11 methods and 1..5 stations in the combined numeric CSV.
            all_reports.append(df)

    if ns.dry_run:
        print('Dry-run completed: NO experiments were executed.')
        return

    import pandas as pd
    result = pd.concat(all_reports, ignore_index=True)
    out.mkdir(exist_ok=True, parents=True)
    csvpath = out / 'long_gap_all_methods.csv'
    result.to_csv(csvpath, index=False, encoding='utf-8-sig')

    # For presentation, prioritize congestion-enriched samples with coverage >=95%.
    frame = result.loc[(result['stratum'] == 'congestion_enriched') &
                       (result['n_stations_missing'].isin([1, 5]))].copy()
    frame = frame[frame['coverage'] >= 0.95]
    frame = frame.sort_values(['duration_hours', 'n_stations_missing',
                               'congestion_f1'], ascending=[True, True, False])
    summary = frame.groupby(['duration_hours', 'n_stations_missing'],
                            as_index=False, group_keys=False).head(3)
    summary_csv = out / 'long_gap_top3_per_scenario.csv'
    summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    fields = ['duration_hours', 'n_stations_missing', 'method', 'n_blocks',
              'n_test_truth', 'n_true_congestion_scored', 'coverage',
              'mae_kmh', 'congestion_recall', 'congestion_f1',
              'congestion_f1_ci_lower', 'congestion_f1_ci_upper']
    print('\nTOP 3 METHODS PER SCENARIO (congestion-enriched):')
    print(summary[fields].round(3).to_string(index=False))
    print('\nSaved:', csvpath)
    print('Saved:', summary_csv)
    print('Caution: block bootstrap intervals for four or fewer blocks are exploratory.')
    print('Do not use 2025 test-period rankings as final locked validation.')


if __name__ == '__main__':
    main()
