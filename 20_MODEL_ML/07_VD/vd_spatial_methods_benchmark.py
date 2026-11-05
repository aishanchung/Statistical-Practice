#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N3 northbound VD-only spatial/nonlinear imputation benchmark.

Run:
  py -m pip install duckdb pandas numpy scikit-learn matplotlib
  py vd_spatial_methods_benchmark.py VD_N3_N_2024_2025_LITE.parquet \
      --output vd_spatial_methods_results

No OD, weather, or trip features are used. Original source is NEVER altered.
2024 observed data train models; 2025 observed data are artificially masked.
All time labels are the original Taiwan-local five-minute slot-end timestamps.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from vd_imputation_experiment import (
    VDExperiment, LAGS, SPEED_MAX, SPEED_MIN, genuine_missing_runs,
)

METHODS = (
    "seasonal_2024", "linear_offline", "spatial_ridge", "spacetime_ridge",
    "idw_spatial", "historical_knn", "random_forest", "asm_observed",
    "asm_seasonal_fallback",
)
LABELS = ("壅塞(<40)", "車多(40–<80)", "暢通(≥80)")


def classes(speed: np.ndarray) -> np.ndarray:
    """Exactly the user's 3 mutually exclusive congestion categories."""
    return np.where(speed < 40, 0, np.where(speed < 80, 1, 2))


def class_confusion(truth: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    y = classes(truth)
    p = classes(estimate)
    return np.bincount(3 * y + p, minlength=9).reshape(3, 3)


def f_scores(confusion: np.ndarray) -> dict:
    if confusion.sum() == 0:
        return {
            "macro_f1_3class": np.nan,
            "congestion_f1": np.nan,
            "congestion_recall": np.nan,
            "congestion_precision": np.nan,
            "true_congestion_n": 0, "true_heavy_n": 0, "true_clear_n": 0,
            "f1_congestion": np.nan, "f1_heavy": np.nan, "f1_clear": np.nan,
        }
    diag = np.diag(confusion).astype(float)
    true = confusion.sum(axis=1).astype(float)
    pred = confusion.sum(axis=0).astype(float)
    f1 = np.divide(2 * diag, true + pred,
                   out=np.zeros(3), where=(true + pred) > 0)
    recall = np.divide(diag, true, out=np.full(3, np.nan), where=true > 0)
    precision = np.divide(diag, pred, out=np.full(3, np.nan), where=pred > 0)
    return {
        "macro_f1_3class": float(np.mean(f1)),
        "congestion_f1": float(f1[0]) if true[0] + pred[0] > 0 else np.nan,
        "congestion_recall": float(recall[0]),
        "congestion_precision": float(precision[0]),
        "true_congestion_n": int(true[0]),
        "true_heavy_n": int(true[1]),
        "true_clear_n": int(true[2]),
        "f1_congestion": float(f1[0]),
        "f1_heavy": float(f1[1]),
        "f1_clear": float(f1[2]),
    }


class SpatialBench:
    def __init__(self, exp: VDExperiment, seed: int, rf_rows: int, rf_trees: int,
                 knn_rows: int, threads: int):
        self.e = exp
        self.rng = np.random.default_rng(seed)
        self.miles = np.array([float(z.split("-")[3]) for z in exp.ids])
        self.x = -self.miles  # increasing in northbound driving direction
        self.rf_rows = rf_rows
        self.knn_rows = knn_rows
        self.rf_trees = rf_trees
        self.threads = threads
        self.models = []
        self.knn = {}
        self.lags = np.stack(
            [np.column_stack(exp._lag_arrays(j)) for j in range(exp.k)], axis=1
        )
        self.knn_calendar = self._knn_calendar()
        self._fit_forests()

    def _knn_calendar(self):
        e = self.e
        tod = 2 * np.pi * e.slot / 288
        week = 2 * np.pi * e.dow / 7
        year = 2 * np.pi * e.times.dayofyear.to_numpy() / 365.25
        return np.column_stack([
            np.sin(tod) * 1.5, np.cos(tod) * 1.5,
            np.sin(2 * tod) * .65, np.cos(2 * tod) * .65,
            np.sin(week) * .70, np.cos(week) * .70,
            (e.dow >= 5).astype(float) * .65,
            np.sin(year) * .35, np.cos(year) * .35,
        ]).astype(np.float32)

    def _rf_features(self, j: int, t: np.ndarray, observed: np.ndarray,
                     own_lag: np.ndarray) -> np.ndarray:
        """Observed is shape [time, 5]; lag is [time,4].

        Missing inputs use *2024 seasonal* proxies and explicit missing flags.
        No use of y(t) or labels, and no recursive synthetic model outputs.
        """
        e = self.e
        donor = [k for k in range(e.k) if k != j]
        cur = observed[:, donor]
        donor_flag = np.isfinite(cur)
        donor_filled = np.where(donor_flag, cur, e.seasonal[t][:, donor])
        lag_flag = np.isfinite(own_lag)
        lag_index = t[:, None] - np.asarray(LAGS)[None, :]
        lag_seasonal = np.zeros_like(own_lag)
        for h in range(len(LAGS)):
            prev = lag_index[:, h]
            lag_seasonal[:, h] = e.seasonal[np.maximum(0, prev), j]
        lag_filled = np.where(lag_flag, own_lag, lag_seasonal)
        return np.column_stack([
            donor_filled, donor_flag.astype(float), lag_filled,
            lag_flag.astype(float), e.cal[t],
        ]).astype(np.float32)

    def _fit_forests(self):
        e = self.e
        for j in range(e.k):
            training = np.flatnonzero(e.train & np.isfinite(e.v[:, j]))
            if len(training) > self.rf_rows:
                training = np.sort(self.rng.choice(training, self.rf_rows,
                                                   replace=False))
            originals = e.v[training].copy()
            lags = self.lags[training, j, :].copy()
            # Several synthetic missingness regimes in the 2024 training set.
            xs, ys = [], []
            for variant in range(4):
                donor_values = originals.copy()
                hist = lags.copy()
                if variant == 1:
                    donor_values[self.rng.random(donor_values.shape) < .27] = np.nan
                    hist[self.rng.random(hist.shape) < .25] = np.nan
                elif variant == 2:
                    donor_values[self.rng.random(donor_values.shape) < .75] = np.nan
                    hist[self.rng.random(hist.shape) < .72] = np.nan
                elif variant == 3:
                    donor_values[:] = np.nan
                    hist[:] = np.nan
                # Target sensor is ALWAYS missing at prediction time; no y(t) feature.
                donor_values[:, j] = np.nan
                xs.append(self._rf_features(j, training, donor_values, hist))
                ys.append(e.v[training, j])
            model = RandomForestRegressor(
                n_estimators=self.rf_trees, max_depth=16, min_samples_leaf=12,
                max_features=.80, n_jobs=self.threads,
                random_state=290 + j,
            )
            model.fit(np.vstack(xs), np.concatenate(ys))
            self.models.append(model)
            print(f"Random forest {j+1}/5 trained: {e.ids[j]}, "
                  f"{len(training):,} observed 2024 dates × 4 masking regimes", flush=True)

    def _knn_data(self, j: int, donor_tuple: tuple[int, ...]):
        """Fit historical analog search once for each donor-availability pattern."""
        key = (j, donor_tuple)
        if key not in self.knn:
            e = self.e
            eligible = e.train & np.isfinite(e.v[:, j])
            if donor_tuple:
                eligible &= np.isfinite(e.v[:, list(donor_tuple)]).all(axis=1)
            ids = np.flatnonzero(eligible)
            if len(ids) > self.knn_rows:
                # Deterministic, evenly spread through the year (not one period).
                ids = ids[np.linspace(0, len(ids)-1, self.knn_rows,
                                     dtype=np.int64)]
            x = self._knn_features(ids, donor_tuple, e.v[ids])
            n_neighbors = min(50 if not donor_tuple else 20, len(ids))
            model = NearestNeighbors(n_neighbors=n_neighbors,
                                     algorithm="ball_tree", n_jobs=self.threads)
            model.fit(x)
            self.knn[key] = (model, e.v[ids, j], n_neighbors)
        return self.knn[key]

    def _knn_features(self, t: np.ndarray, donor_tuple: tuple[int, ...],
                      observed: np.ndarray) -> np.ndarray:
        features = [self.knn_calendar[t]]
        if donor_tuple:
            features.append((observed[:, list(donor_tuple)] / 15).astype(np.float32))
        return np.column_stack(features).astype(np.float32)

    def _knn_predict(self, j: int, t: np.ndarray, observed: np.ndarray):
        e = self.e
        p = np.full(len(t), np.nan)
        donors = [k for k in range(e.k) if k != j]
        # Group by the *actual* donor availability at each target time.
        flags = np.isfinite(observed[:, donors])
        code = np.sum(flags.astype(np.int8) * (1 << np.arange(4)), axis=1)
        for pattern in np.unique(code):
            select = code == pattern
            donor_tuple = tuple(donors[i] for i in range(4) if pattern & (1 << i))
            model, ys, count = self._knn_data(j, donor_tuple)
            x = self._knn_features(t[select], donor_tuple, observed[select])
            distances, idx = model.kneighbors(x, n_neighbors=count)
            weights = 1 / np.maximum(distances, .05)
            p[select] = np.sum(weights * ys[idx], axis=1) / weights.sum(axis=1)
        return np.clip(p, SPEED_MIN, SPEED_MAX)

    def _idw(self, j: int, observed: np.ndarray) -> np.ndarray:
        e = self.e
        distances = np.abs(self.miles[j] - self.miles)
        weights = np.where(distances > 0, np.divide(1.0, distances**2,
                                  out=np.zeros(e.k), where=distances > 0), 0.)
        valid = np.isfinite(observed)
        weight_mat = valid * weights[None, :]
        total = weight_mat.sum(axis=1)
        estimate = np.sum(np.nan_to_num(observed, nan=0.) * weight_mat,
                          axis=1)
        return np.divide(estimate, total, out=np.full(len(total), np.nan),
                         where=total > 0)

    def _spatial_ridge(self, j: int, t: np.ndarray, observed: np.ndarray):
        e = self.e
        a, b = e.donors[j]
        donors = observed[:, [a, b]]
        valid = np.isfinite(donors).all(axis=1)
        p = np.full(len(t), np.nan)
        if valid.any():
            x = np.column_stack([donors[valid], e.cal[t[valid]]])
            p[valid] = e.spatial_models[j].predict_many(x)
        return np.clip(p, SPEED_MIN, SPEED_MAX)

    def _spacetime_ridge(self, j: int, t: np.ndarray, observed: np.ndarray,
                         start: int, end: int, masked: set[int]):
        """Existing 2024 trained Ridge, recursive *target* lag handling."""
        e = self.e
        a, b = e.donors[j]
        p = np.full(len(t), np.nan)
        for h, idx in enumerate(t):
            history = []
            for lag in LAGS:
                prev = int(idx - lag)
                if start <= prev < end and j in masked:
                    value = p[prev-start]
                else:
                    value = e.v[prev, j] if prev >= 0 else np.nan
                if not np.isfinite(value):
                    value = e.seasonal[max(prev, 0), j]
                history.append(value)
            two = observed[h, [a, b]]
            available = np.isfinite(two).astype(float)
            two = np.where(np.isfinite(two), two, e.seasonal[idx, [a, b]])
            features = np.concatenate([history, two, available, e.cal[idx]])
            p[h] = e.st_models[j].predict_one(features)
        return np.clip(p, SPEED_MIN, SPEED_MAX)

    def _asm(self, j: int, t: np.ndarray, source: np.ndarray,
             source_first_idx: int) -> np.ndarray:
        """Simple offline adaptive smoothing on observed VD values only.

        Traffic characteristic velocities, in driving (northbound) direction:
        free: +90 km/h, congested: -15 km/h. Two shifted exponential
        filters are blended according to estimated traffic state. This is an
        ASM-inspired *fixed-parameter prototype*, NOT a calibrated deployment.
        It may look before AND after t within +/-60 minutes (retrospective).
        """
        result = []
        rel = t - source_first_idx
        # Around ±60 minutes, spatial bandwidth 3 km, temporal bandwidth 10m.
        for char_speed in (90.0, -15.0):
            num = np.zeros(len(t)); den = np.zeros(len(t))
            for k in range(self.e.k):
                dist = abs(self.x[j] - self.x[k])
                dx = self.x[j] - self.x[k]
                for lag in range(-12, 13):
                    pos = rel + lag
                    safe = np.clip(pos, 0, len(source)-1)
                    values = source[safe, k]
                    valid = (pos >= 0) & (pos < len(source)) & np.isfinite(values)
                    if not valid.any():
                        continue
                    weight = np.exp(-dist/3.0 - abs(lag * 5 + dx * 60/char_speed)/10.)
                    num[valid] += weight * values[valid]
                    den[valid] += weight
            filtered = np.divide(num, den, out=np.full(len(t), np.nan),
                                 where=den > 1e-8)
            result.append(filtered)
        free, congested = result
        both = np.isfinite(free) & np.isfinite(congested)
        out = np.where(np.isfinite(free), free, congested)
        if both.any():
            # Congestion wave mixing based on preliminary speed estimate.
            base = np.minimum(free[both], congested[both])
            w = 1. / (1. + np.exp(np.clip((base-60.) / 10., -30, 30)))
            out[both] = (1 - w) * free[both] + w * congested[both]
        return np.clip(out, SPEED_MIN, SPEED_MAX)

    def _observed_and_lags(self, start: int, end: int, masked: set[int]):
        e = self.e
        t = np.arange(start, end)
        observed = e.v[start:end].copy()
        observed[:, list(masked)] = np.nan
        # Build a local source window so ASM sees genuine 5m readings beyond
        # the masked block but NEVER true (held-out) measurements within it.
        lo, hi = max(0, start-12), min(e.n, end+12)
        source = e.v[lo:hi].copy()
        source[start-lo:end-lo, list(masked)] = np.nan
        return t, observed, source, lo

    def predict(self, start: int, end: int, masked: tuple[int, ...],
                j: int, precomputed=None) -> dict[str, np.ndarray]:
        e = self.e
        assert j in masked
        subset = set(masked)
        if precomputed is None:
            t, observed, source, lo = self._observed_and_lags(start, end, subset)
        else:
            t, observed, source, lo = precomputed
        lagged = self.lags[t, j].copy()
        for h, lag in enumerate(LAGS):
            lagged[t-lag >= start, h] = np.nan
        rf_x = self._rf_features(j, t, observed, lagged)
        asm = self._asm(j, t, source, lo)
        return {
            "seasonal_2024": e.seasonal[t, j].copy(),
            "linear_offline": e.predict_linear(j, start, end),
            "spatial_ridge": self._spatial_ridge(j, t, observed),
            "spacetime_ridge": self._spacetime_ridge(j, t, observed, start, end, subset),
            "idw_spatial": self._idw(j, observed),
            "historical_knn": self._knn_predict(j, t, observed),
            "random_forest": np.clip(self.models[j].predict(rf_x), SPEED_MIN, SPEED_MAX),
            "asm_observed": asm,
            "asm_seasonal_fallback": np.where(np.isfinite(asm), asm,
                                               e.seasonal[t, j]),
        }

    def predict_actual(self, j: int, start: int, end: int) -> dict:
        """All original missing sensors stay hidden (no imputed donor use)."""
        e = self.e
        t = np.arange(start, end)
        observed = e.v[start:end].copy()
        lagged = self.lags[t, j]
        lo, hi = max(0, start-12), min(e.n, end+12)
        asm = self._asm(j, t, e.v[lo:hi], lo)
        return {
            "seasonal_2024": e.seasonal[t, j].copy(),
            "linear_offline": e.predict_linear(j, start, end),
            "spatial_ridge": self._spatial_ridge(j, t, observed),
            "spacetime_ridge": e.predict_recursive(j, start, end,
                                                    spatiotemporal=True,
                                                    mask_all=False),
            "idw_spatial": self._idw(j, observed),
            "historical_knn": self._knn_predict(j, t, observed),
            "random_forest": np.clip(self.models[j].predict(
                self._rf_features(j, t, observed, lagged)), SPEED_MIN, SPEED_MAX),
            "asm_observed": asm,
            "asm_seasonal_fallback": np.where(np.isfinite(asm), asm,
                                               e.seasonal[t, j]),
        }


def select_mask_blocks(e: VDExperiment, hours: list[float],
                       trials_each_stratum: int, seed: int):
    """Half general periods, half periods with known true <40 observations.

    Selection uses *truth only for scoring design*, never as a model input.
    Report each stratum separately to avoid misleading population estimates.
    """
    rng = np.random.default_rng(seed)
    good = np.isfinite(e.v)
    cum = np.vstack([np.zeros((1, e.k), dtype=int),
                     np.cumsum(good, axis=0, dtype=int)])
    jam_any = np.any(good & (e.v < 40), axis=1).astype(int)
    jam_cum = np.concatenate([[0], np.cumsum(jam_any)])
    test_idx = np.flatnonzero(e.test)
    first, last = int(test_idx[0]), int(test_idx[-1] + 1)
    out = []
    for duration in hours:
        length = int(round(duration*12))
        candidates = np.arange(first+24, last-length-24, 12)
        valid = (cum[candidates+length] - cum[candidates]).min(axis=1) >= .88 * length
        candidates = candidates[valid]
        jam_presence = (jam_cum[candidates+length] - jam_cum[candidates]) > 0
        selected = []
        for stratum, sample in [
            ("general", candidates),
            ("congestion_enriched", candidates[jam_presence]),
        ]:
            sample = sample.copy(); rng.shuffle(sample)
            count = 0
            for start in sample:
                if any(abs(start - previous) < length+288 for previous in selected):
                    continue
                selected.append(int(start))
                out.append({
                    "block_id": len(out)+1,
                    "duration_hours": duration,
                    "start_idx": int(start), "end_idx": int(start+length),
                    "start_time": str(e.times[start]),
                    "last_hidden_time": str(e.times[start+length-1]),
                    "stratum": stratum,
                    "congested_5min_slots_any_station": int(
                        jam_cum[start+length] - jam_cum[start]),
                })
                count += 1
                if count >= trials_each_stratum:
                    break
            if count < trials_each_stratum:
                print(f"NOTE: {duration}h {stratum} selected {count}/"
                      f"{trials_each_stratum} suitable non-overlapping blocks", flush=True)
    return out


def masked_patterns() -> list[tuple[int, ...]]:
    """Contiguous station groups by order on the highway: 1,2,3,4,5 stations."""
    return [tuple(range(start, start+size))
            for size in range(1, 6) for start in range(6-size)]


def run_masked(e: VDExperiment, bench: SpatialBench, blocks: list,
               output: Path, save_example=True):
    rows = []
    example = None
    for n, block in enumerate(blocks, 1):
        start, end = block["start_idx"], block["end_idx"]
        for mask in masked_patterns():
            pre = bench._observed_and_lags(start, end, set(mask))
            for j in mask:
                truth = e.v[start:end, j]
                for method, predicted in bench.predict(start, end, mask, j, pre).items():
                    valid_truth = np.isfinite(truth)
                    scored = valid_truth & np.isfinite(predicted)
                    t = truth[scored]
                    p = predicted[scored]
                    cm = class_confusion(t, p) if len(t) else np.zeros((3, 3), int)
                    err = p-t
                    row = {
                        "block_id": n, "start_time": block["start_time"],
                        "stratum": block["stratum"],
                        "duration_hours": block["duration_hours"],
                        "n_stations_missing": len(mask),
                        "masked_indices": ",".join(map(str, mask)),
                        "VDID": e.ids[j], "method": method,
                        "n_truth": int(valid_truth.sum()),
                        "n_pred": len(t),
                        "ae_sum": float(np.abs(err).sum()),
                        "se_sum": float((err*err).sum()),
                        "bias_sum": float(err.sum()),
                    }
                    for a, b in itertools.product(range(3), repeat=2):
                        row[f"cm_{a}{b}"] = int(cm[a, b])
                    rows.append(row)
                    if (save_example and example is None and
                        block["duration_hours"] == 72 and
                        len(mask) == 5 and j == 2 and method == "seasonal_2024"):
                        example = {"times": e.times[start:end], "truth": truth,
                                   "pred": {}, "start": block["start_time"]}
                    if (example is not None and example["start"] == block["start_time"]
                            and len(mask) == 5 and j == 2):
                        example["pred"][method] = predicted
        print(f"Masked validation {n}/{len(blocks)}: "
              f"{block['duration_hours']}h {block['stratum']} {block['start_time']}",
              flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(output/"benchmark_by_block_station.csv", index=False,
                  encoding="utf-8-sig")
    for cols, filename in [
        (["stratum", "duration_hours", "n_stations_missing", "method"],
         "benchmark_by_missing_pattern.csv"),
        (["stratum", "duration_hours", "n_stations_missing", "VDID", "method"],
         "benchmark_by_station.csv"),
    ]:
        summary = aggregate(detail, cols)
        summary.to_csv(output/filename, index=False, encoding="utf-8-sig")
    if example is not None:
        plot_example(example, output/"example_72h_all_five.png")
    return detail


def aggregate(detail: pd.DataFrame, keys: list[str]):
    cm_names = [f"cm_{a}{b}" for a in range(3) for b in range(3)]
    records=[]
    for k, group in detail.groupby(keys, sort=True, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        t, n = int(group.n_truth.sum()), int(group.n_pred.sum())
        cm = group[cm_names].sum().to_numpy().reshape(3,3)
        row = {
            **dict(zip(keys, k)),
            "blocks": group.block_id.nunique(),
            "n_truth": t, "n_pred": n,
            "coverage": n/t if t else np.nan,
            "mae_kmh": group.ae_sum.sum()/n if n else np.nan,
            "rmse_kmh": np.sqrt(group.se_sum.sum()/n) if n else np.nan,
            "bias_kmh": group.bias_sum.sum()/n if n else np.nan,
            **f_scores(cm),
        }
        for a, b in itertools.product(range(3), repeat=2):
            row[f"cm_{a}{b}"] = int(cm[a,b])
        records.append(row)
    return pd.DataFrame(records)


def plot_example(example: dict, target: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(example["times"], example["truth"], c="#1e3349", lw=1.4,
            label="Observed speed (artificially hidden)")
    colors = {"seasonal_2024":"#3b77ba", "historical_knn":"#e39931",
              "random_forest":"#9e4799", "spacetime_ridge":"#2b9785",
              "asm_observed":"#c14b49", "asm_seasonal_fallback":"#9babb8"}
    for method, color in colors.items():
        if method in example["pred"]:
            ax.plot(example["times"], example["pred"][method], color=color,
                    linewidth=.9, alpha=.83, label=method)
    for speed, label in [(40, "Congestion <40"), (80, "Heavy traffic <80")]:
        ax.axhline(speed, ls="--", lw=.9, c="#6c7782",
                   label=label)
    ax.set(title=f"63.990K: five-station missing 72h, {example['start']}",
           ylabel="Speed (km/h)", xlabel="Taiwan time")
    ax.grid(alpha=.15); ax.legend(fontsize=8, ncol=4)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(target, dpi=165)
    plt.close(fig)


def run_actual_missing(e: VDExperiment, bench: SpatialBench, output: Path):
    rows=[]
    for j in range(e.k):
        for start, end in genuine_missing_runs(e.missing[:, j]):
            p = bench.predict_actual(j, start, end)
            for h, idx in enumerate(range(start, end)):
                row = {
                    "TargetTime": str(e.times[idx]), "VDID": e.ids[j],
                    "VDType_original": str(e.types[idx, j]),
                    "Was_missing_original": True,
                    "Speed_original_code": float(e.raw[idx,j]),
                    "gap_length_bins": end-start,
                    "gap_length_hours": (end-start)/12,
                    "other_VD_valid_speed_count": int(np.isfinite(e.v[idx]).sum()),
                }
                for method, result in p.items():
                    row[method] = float(result[h]) if np.isfinite(result[h]) else np.nan
                row["asm_used_seasonal_fallback"] = not np.isfinite(p["asm_observed"][h])
                rows.append(row)
        print(f"Real missing candidates {j+1}/5, {e.ids[j]}", flush=True)
    df = pd.DataFrame(rows).sort_values(["TargetTime", "VDID"])
    if len(df) != int(e.missing.sum()):
        raise AssertionError("Missing cells changed unexpectedly")
    df.to_csv(output/"real_missing_candidate_estimates.csv", index=False,
              encoding="utf-8-sig", na_rep="")
    return df


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("parquet", type=Path)
    pa.add_argument("--output", type=Path, default=Path("vd_spatial_methods_results"))
    pa.add_argument("--hours", nargs="+", type=float,
                    default=[.25, 1, 6, 24, 72, 151.5])
    pa.add_argument("--trials-per-stratum", type=int, default=4,
                    help="Per duration: N general + up to N congestion-enriched blocks")
    pa.add_argument("--rf-rows", type=int, default=30000)
    pa.add_argument("--rf-trees", type=int, default=50)
    pa.add_argument("--knn-rows", type=int, default=30000)
    pa.add_argument("--threads", type=int, default=4)
    pa.add_argument("--seed", type=int, default=20261029)
    pa.add_argument("--skip-actual", action="store_true")
    args = pa.parse_args()
    if any(abs(h*12-round(h*12))>1e-8 for h in args.hours):
        pa.error("--hours values must be multiples of 5 minutes")
    if any(h<=0 for h in args.hours):
        pa.error("--hours must be >0")
    if not args.parquet.exists():
        pa.error(f"File does not exist: {args.parquet}")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output/"mplcache"))
    e = VDExperiment(args.parquet)
    bench = SpatialBench(e, args.seed, args.rf_rows, args.rf_trees,
                         args.knn_rows, args.threads)
    blocks = select_mask_blocks(e, args.hours, args.trials_per_stratum, args.seed)
    pd.DataFrame(blocks).to_csv(args.output/"masked_blocks_manifest.csv",
                                index=False, encoding="utf-8-sig")
    details = run_masked(e, bench, blocks, args.output)
    if not args.skip_actual:
        actual = run_actual_missing(e, bench, args.output)
        print("Actual missing cells, candidate only:", len(actual), flush=True)
    conf = {
        "source": str(args.parquet), "seed": args.seed,
        "train_year": 2024, "heldout_year": 2025,
        "hours": args.hours, "trials_per_stratum": args.trials_per_stratum,
        "rf_rows": args.rf_rows, "rf_trees": args.rf_trees,
        "knn_rows": args.knn_rows,
        "missing_patterns": "contiguous groups of 1, 2, 3, 4, 5 stations",
        "period_selection": "Two strata: general periods and known-congested periods; do not mix as an estimate of real-world performance",
        "class_definition": {"0":"speed<40", "1":"40<=speed<80", "2":"speed>=80"},
        "rf_features": "four donor sensor speeds, own 2024/2025 historical lagged speeds, missing indicators, calendar; 2024 seasonal substitutes",
        "rf_training_missingness": "original, 27%, 75%, 100% random donor/lag masking regimes (2024)",
        "historical_knn": "2024 analog periods with donor speeds if available, calendar-only if all five missing",
        "idw": "inverse squared distance of any same-time observed other VD",
        "asm": "offline (past AND future +/-60m); fixed characteristic velocities +90 / -15km/h, space scale 3km, time scale 10min; NOT CALIBRATED",
        "asm_seasonal_fallback": "seasonal 2024 value only when no observed sensor signal within +/-60min",
        "speed_minus_one": "no passing vehicles: not observed speed for model, not an imputation target or congestion evidence",
        "provenance": "Outputs are candidate predictions; original data never changed",
        "no_od_or_weather": True,
        "warning": "When 5 stations simultaneously missing for long periods, no method has contemporary spatial evidence; macro-F1 uses all 3 classes and coverage must be read beside accuracy",
    }
    (args.output/"experiment_config.json").write_text(
        json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    bypattern = aggregate(details, ["stratum", "duration_hours", "n_stations_missing", "method"])
    chosen = bypattern[(bypattern.duration_hours.isin([6, 72, 151.5]))
                       & bypattern.n_stations_missing.isin([1, 5])]
    print("\nSelected results (speed MAE, three-class Macro-F1):")
    print(chosen[["stratum","duration_hours","n_stations_missing","method",
                  "coverage","mae_kmh","macro_f1_3class","congestion_recall"]]
          .round(3).to_string(index=False))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
