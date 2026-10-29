#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize VD artificial-mask experiments with block-level uncertainty.

Usage (after vd_spatial_methods_benchmark.py has produced its CSV):
    python vd_robust_mask_report.py \
        outputs/vd_robust_short_20261029/benchmark_by_block_station.csv \
        --output outputs/vd_robust_short_20261029

Each mask block (a separated time interval) is the bootstrap sampling unit.
Do NOT bootstrap individual 5-minute VD measurements as independent rows:
neighboring times and multiple stations within an event are correlated.

The 'congestion_enriched' and 'general' strata are always reported separately.
Their simple combined average would NOT estimate real-world traffic prevalence.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


CM_COLS = [f"cm_{i}{j}" for i in range(3) for j in range(3)]
SUM_COLS = ["n_truth", "n_pred", "ae_sum", *CM_COLS]


def metrics(sums: np.ndarray) -> dict:
    """Calculate speed/congestion metrics from confusion-matrix totals."""
    n_truth, n_pred, ae_sum = sums[:3]
    cm = sums[3:].reshape(3, 3)
    tp = cm[0, 0]
    true_jam = cm[0, :].sum()
    predicted_jam = cm[:, 0].sum()
    recall = tp / true_jam if true_jam else np.nan
    precision = tp / predicted_jam if predicted_jam else np.nan
    f1 = 2 * tp / (true_jam + predicted_jam) if true_jam + predicted_jam else np.nan
    return {
        "n_test_truth": int(round(n_truth)),
        "n_scored": int(round(n_pred)),
        "n_true_congestion_scored": int(round(true_jam)),
        "n_congestion_recovered": int(round(tp)),
        "coverage": n_pred / n_truth if n_truth else np.nan,
        "mae_kmh": ae_sum / n_pred if n_pred else np.nan,
        "congestion_recall": recall,
        "congestion_precision": precision,
        "congestion_f1": f1,
    }


def boot_metrics(block_values: np.ndarray, rng: np.random.Generator,
                 resamples: int) -> dict:
    """Percentile CIs by resampling entire time blocks, with replacement."""
    n_blocks = len(block_values)
    sample_idx = rng.integers(0, n_blocks, size=(resamples, n_blocks))
    sm = block_values[sample_idx].sum(axis=1)
    n_pred, ae_sum = sm[:, 1], sm[:, 2]
    cm = sm[:, 3:].reshape(resamples, 3, 3)
    tp = cm[:, 0, 0]
    tru = cm[:, 0, :].sum(axis=1)
    predicted = cm[:, :, 0].sum(axis=1)
    ma = np.divide(ae_sum, n_pred, out=np.full(resamples, np.nan), where=n_pred > 0)
    rec = np.divide(tp, tru, out=np.full(resamples, np.nan), where=tru > 0)
    f1 = np.divide(2*tp, tru+predicted, out=np.full(resamples, np.nan),
                   where=tru+predicted > 0)
    out = {}
    for name, vals in [("mae_kmh", ma), ("congestion_recall", rec),
                       ("congestion_f1", f1)]:
        finite = vals[np.isfinite(vals)]
        lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        out[name + "_ci_lower"] = float(lo)
        out[name + "_ci_upper"] = float(hi)
    return out


def paired_bootstrap(blocks_a: np.ndarray, blocks_b: np.ndarray,
                     rng: np.random.Generator, resamples: int) -> dict:
    """Paired F1/MAE difference: KNN minus linear, same masked blocks."""
    n = len(blocks_a)
    idx = rng.integers(0, n, size=(resamples, n))
    def scores(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = x[idx].sum(axis=1)
        cm = s[:, 3:].reshape(resamples, 3, 3)
        tp, tru, pred = cm[:, 0, 0], cm[:, 0, :].sum(axis=1), cm[:, :, 0].sum(axis=1)
        f1 = np.divide(2*tp, tru+pred, out=np.full(resamples,np.nan),
                       where=tru+pred > 0)
        mae = np.divide(s[:, 2],s[:, 1],out=np.full(resamples,np.nan),
                        where=s[:, 1] > 0)
        return f1,mae
    af,am = scores(blocks_a)
    bf,bm = scores(blocks_b)
    results = {}
    for label, delta in [("f1",af-bf),("mae_kmh",am-bm)]:
        val = delta[np.isfinite(delta)]
        results[label + "_delta_ci_low"] = float(np.quantile(val, .025)) if len(val) else np.nan
        results[label + "_delta_ci_high"] = float(np.quantile(val, .975)) if len(val) else np.nan
    return results


def main():
    # Consistent with the user's Windows workflow: relative paths are resolved
    # from this script's directory even if launched from a different cwd.
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("details_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 resamples for confidence intervals")
    data = pd.read_csv(args.details_csv)
    required = {"block_id","stratum","duration_hours","n_stations_missing","method",*SUM_COLS}
    if not required.issubset(data.columns):
        parser.error(f"Missing columns: {sorted(required - set(data.columns))}")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    block_groups = {}
    for key, subset in data.groupby(["duration_hours","stratum","n_stations_missing","method"],sort=True):
        blocks = subset.groupby("block_id",sort=True)[SUM_COLS].sum()
        block_groups[key] = blocks
        rec = {**dict(zip(["duration_hours","stratum","n_stations_missing","method"],key)),
               "n_blocks":len(blocks),
               **metrics(blocks.to_numpy(dtype=float).sum(axis=0)),
               **boot_metrics(blocks.to_numpy(dtype=float),rng,args.bootstrap)}
        rec["eligible_95pct_coverage"] = bool(rec["coverage"] >= .95)
        records.append(rec)
    all_metrics = pd.DataFrame(records).sort_values(
        ["duration_hours","stratum","n_stations_missing","congestion_f1"],
        ascending=[True,True,True,False])
    all_metrics.to_csv(args.output/"robust_metrics_with_ci.csv",index=False,encoding="utf-8-sig")

    paired = []
    for duration,stratum,missing in sorted({k[:3] for k in block_groups}):
        a = block_groups.get((duration,stratum,missing,"historical_knn"))
        b = block_groups.get((duration,stratum,missing,"linear_offline"))
        if a is None or b is None: continue
        shared = a.index.intersection(b.index)
        if len(shared) < 2: continue
        aa,bb = a.loc[shared].to_numpy(dtype=float),b.loc[shared].to_numpy(dtype=float)
        ma,mb = metrics(aa.sum(axis=0)),metrics(bb.sum(axis=0))
        paired.append({"duration_hours":duration,"stratum":stratum,
                       "n_stations_missing":missing,"n_blocks":len(shared),
                       "knn_f1":ma["congestion_f1"],"linear_f1":mb["congestion_f1"],
                       "knn_minus_linear_f1":ma["congestion_f1"]-mb["congestion_f1"],
                       "knn_minus_linear_mae_kmh":ma["mae_kmh"]-mb["mae_kmh"],
                       **paired_bootstrap(aa,bb,rng,args.bootstrap)})
    pd.DataFrame(paired).to_csv(args.output/"paired_knn_vs_linear_block_ci.csv",index=False,encoding="utf-8-sig")
    print("Saved metrics and block-bootstrap 95% confidence intervals to",args.output)
    sel=all_metrics.query('stratum=="congestion_enriched" and n_stations_missing in [1,5] and eligible_95pct_coverage')
    best=sel.loc[sel.groupby(["duration_hours","n_stations_missing"])["congestion_f1"].idxmax()]
    print(best[["duration_hours","n_stations_missing","method","n_blocks","n_test_truth","n_true_congestion_scored","n_congestion_recovered","mae_kmh","congestion_recall","congestion_f1","congestion_f1_ci_lower","congestion_f1_ci_upper"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
