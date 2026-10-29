#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VD 5-minute speed imputation benchmark (N3 northbound, five stations).

Purpose
-------
1. Fit candidate models using only 2024 observed speeds.
2. In 2025, deliberately hide continuous observed blocks; compare reconstructed
   speeds with the original *hidden* speeds (single station vs all five).
3. Generate *candidate estimates*, NOT verified observations, for actual
   OFFICIAL_NA/VD_ABSENT/FILE_UNAVAILABLE rows.

Run (Windows PowerShell or Google Colab):
  pip install numpy pandas duckdb scikit-learn matplotlib
  python vd_imputation_experiment.py VD_N3_N_2024_2025_LITE.parquet \
      --output vd_imputation_results

The original Parquet file is read-only. This experiment does not impute
Speed=-1 (a no-vehicle observation). Units: speed = km/h; time = 5 minutes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MISSING_TYPES = {"VD_ABSENT", "OFFICIAL_NA", "FILE_UNAVAILABLE"}
METHODS = [
    "linear_offline",
    "seasonal_2024",
    "previous_week",
    "recursive_ar",
    "spatial_ridge",
    "spacetime_ridge",
]
SPEED_MIN, SPEED_MAX = 0.0, 150.0
LAGS = (1, 12, 288, 2016)  # 5 min, 1 hour, 1 day, 1 week
THRESHOLDS = (40, 60, 80)  # provisional speed thresholds, km/h


def read_parquet(path: Path) -> pd.DataFrame:
    """DuckDB avoids needing pyarrow in the user's restricted Windows setup."""
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit(
            "Need DuckDB. Install: py -m pip install duckdb numpy pandas "
            "scikit-learn matplotlib (or run in Google Colab)."
        ) from exc
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(
            "SELECT TargetTime, VDID, Speed, VDType FROM read_parquet(?)",
            [str(path.resolve())],
        ).df()
    finally:
        con.close()


def calendar_matrix(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slot = (times.hour * 12 + times.minute // 5).to_numpy(dtype=int)
    dow = times.dayofweek.to_numpy(dtype=int)
    year_day = times.dayofyear.to_numpy(dtype=float)
    day_angle = 2 * np.pi * slot / 288
    week_angle = 2 * np.pi * dow / 7
    year_angle = 2 * np.pi * year_day / 365.25
    weekend = (dow >= 5).astype(float)
    # Weekday/weekend interaction captures different peak-hour patterns.
    cal = np.column_stack(
        [
            np.sin(day_angle), np.cos(day_angle),
            np.sin(2 * day_angle), np.cos(2 * day_angle),
            np.sin(3 * day_angle), np.cos(3 * day_angle),
            np.sin(week_angle), np.cos(week_angle),
            np.sin(year_angle), np.cos(year_angle),
            weekend,
            weekend * np.sin(day_angle), weekend * np.cos(day_angle),
            weekend * np.sin(2 * day_angle),
            weekend * np.cos(2 * day_angle),
        ]
    )
    return cal.astype(float), dow, slot


def shifted(arr: np.ndarray, lag: int) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    out[:lag] = np.nan
    out[lag:] = arr[:-lag]
    return out


@dataclass
class LinearModel:
    coef: np.ndarray
    intercept: float

    def predict_one(self, features: np.ndarray) -> float:
        return float(self.intercept + self.coef @ features)

    def predict_many(self, features: np.ndarray) -> np.ndarray:
        return self.intercept + features @ self.coef


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 35.0) -> LinearModel:
    scale = StandardScaler()
    z = scale.fit_transform(x)
    model = Ridge(alpha=alpha)
    model.fit(z, y)
    coef = model.coef_ / scale.scale_
    intercept = model.intercept_ - float(coef @ scale.mean_)
    return LinearModel(np.asarray(coef, float), float(intercept))


class VDExperiment:
    def __init__(self, source: Path):
        df = read_parquet(source)
        df["TargetTime"] = pd.to_datetime(df["TargetTime"])
        self.ids = sorted(df["VDID"].unique(), key=lambda x: -float(x.split("-")[3]))
        if len(self.ids) != 5:
            raise ValueError(f"Expected five N3 stations; found {self.ids}")
        if df.duplicated(["TargetTime", "VDID"]).any():
            raise ValueError("Duplicate TargetTime x VDID; input is not a valid panel.")
        speed = df.pivot(index="TargetTime", columns="VDID", values="Speed")[self.ids]
        vtype = df.pivot(index="TargetTime", columns="VDID", values="VDType")[self.ids]
        self.times = pd.DatetimeIndex(speed.index)
        if len(self.times) != 210528:
            raise ValueError(f"Expected 210,528 time slots; got {len(self.times)}")
        if not (self.times.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all():
            raise ValueError("Time grid has discontinuities.")
        self.raw = speed.to_numpy(dtype=float)
        self.types = vtype.to_numpy(dtype=object)
        self.missing = np.isin(self.types, list(MISSING_TYPES))
        valid = np.isfinite(self.raw) & (self.raw >= SPEED_MIN) & ~self.missing
        self.v = np.where(valid, self.raw, np.nan)
        self.n, self.k = self.v.shape
        self.cal, self.dow, self.slot = calendar_matrix(self.times)
        self.train = (self.times.year == 2024)
        self.test = (self.times.year == 2025)
        self.profile = self._seasonal_profile()
        self.seasonal = self.profile[:, self.dow, self.slot].T.copy()
        self.donors = self._closest_two()
        self.ar_models: list[LinearModel] = []
        self.spatial_models: list[LinearModel] = []
        self.st_models: list[LinearModel] = []
        self._fit_models()

    def _closest_two(self) -> list[list[int]]:
        mile = [float(v.split("-")[3]) for v in self.ids]
        return [
            sorted([j for j in range(self.k) if j != i],
                   key=lambda j: abs(mile[i] - mile[j]))[:2]
            for i in range(self.k)
        ]

    def _seasonal_profile(self) -> np.ndarray:
        profile = np.empty((self.k, 7, 288))
        for j in range(self.k):
            d = pd.DataFrame({
                "weekday": self.dow[self.train],
                "slot": self.slot[self.train],
                "speed": self.v[self.train, j],
            })
            med = d.pivot_table(
                index="weekday", columns="slot", values="speed",
                aggfunc="median", observed=False,
            ).reindex(index=range(7), columns=range(288))
            global_med = float(np.nanmedian(d["speed"]))
            # Fill rare empty slot groups by the same weekday median.
            med = med.T.fillna(med.median(axis=1)).T.fillna(global_med)
            profile[j] = med.to_numpy(dtype=float)
        return profile

    def _lag_arrays(self, j: int) -> list[np.ndarray]:
        return [shifted(self.v[:, j], lag) for lag in LAGS]

    def _fit_models(self) -> None:
        for j in range(self.k):
            y = self.v[:, j]
            lagged = self._lag_arrays(j)
            lags = np.column_stack(lagged)
            ar_x = np.column_stack([lags, self.cal])
            ar_ok = self.train & np.isfinite(y) & np.isfinite(lags).all(axis=1)
            if ar_ok.sum() < 5000:
                raise RuntimeError("Not enough observed samples to fit AR model.")
            self.ar_models.append(fit_ridge(ar_x[ar_ok], y[ar_ok]))

            a, b = self.donors[j]
            sp_x = np.column_stack([self.v[:, [a, b]], self.cal])
            sp_ok = self.train & np.isfinite(y) & np.isfinite(sp_x).all(axis=1)
            self.spatial_models.append(fit_ridge(sp_x[sp_ok], y[sp_ok]))

            donor_values = self.v[:, [a, b]].copy()
            observed = np.isfinite(donor_values).astype(float)
            donor_values = np.where(
                np.isfinite(donor_values),
                donor_values, self.seasonal[:, [a, b]]
            )
            st_x = np.column_stack([lags, donor_values, observed, self.cal])
            st_ok = self.train & np.isfinite(y) & np.isfinite(lags).all(axis=1)
            self.st_models.append(fit_ridge(st_x[st_ok], y[st_ok]))
            print(f"Trained {self.ids[j]}: AR={ar_ok.sum():,}, "
                  f"spatial={sp_ok.sum():,}, space-time={st_ok.sum():,}", flush=True)

    def select_blocks(self, lengths: list[int], trials: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        valid = np.isfinite(self.v).astype(np.int32)
        cum = np.vstack([np.zeros((1, self.k), dtype=np.int64),
                         np.cumsum(valid, axis=0, dtype=np.int64)])
        first = int(np.flatnonzero(self.test)[0]) + 24
        end_year = int(np.flatnonzero(self.test)[-1]) + 1
        out = []
        for length in lengths:
            # Random starts at every 1-hour boundary, but never span years.
            candidates = np.arange(first, end_year - length - 24, 12)
            rng.shuffle(candidates)
            chosen = []
            for start in candidates:
                stop = int(start) + length
                counts = cum[stop] - cum[start]
                if counts.min() / length < 0.88:
                    continue
                if any(abs(start - other) < length + 288 for other in chosen):
                    continue
                chosen.append(int(start))
                out.append({
                    "length_bins": length,
                    "start_idx": int(start),
                    "end_idx": stop,
                    "start_time": str(self.times[start]),
                    "last_hidden_time": str(self.times[stop - 1]),
                    "min_observed_fraction": float(counts.min() / length),
                })
                if len(chosen) == trials:
                    break
            if len(chosen) < trials:
                raise RuntimeError(
                    f"Only {len(chosen)} eligible blocks for length {length}; "
                    "reduce --trials-per-length or observed-fraction requirement."
                )
        return out

    def _outside_value(self, j: int, idx: int, direction: int,
                       start: int, end: int, search: int = 24):
        """Find a valid endpoint *outside* the artificially missing block."""
        for offset in range(search):
            pos = idx + direction * offset
            if pos < 0 or pos >= self.n or (start <= pos < end):
                break
            if np.isfinite(self.v[pos, j]):
                return pos, float(self.v[pos, j])
        return None

    def predict_linear(self, j: int, start: int, end: int) -> np.ndarray:
        left = self._outside_value(j, start - 1, -1, start, end)
        right = self._outside_value(j, end, 1, start, end)
        if left is None or right is None:
            return np.full(end - start, np.nan)
        return np.interp(
            np.arange(start, end),
            [left[0], right[0]], [left[1], right[1]]
        )

    def predict_week(self, j: int, start: int, end: int) -> np.ndarray:
        result = self.seasonal[start:end, j].copy()
        for h, t in enumerate(range(start, end)):
            prev = t - 2016
            if prev >= 0 and not (start <= prev < end):
                if np.isfinite(self.v[prev, j]):
                    result[h] = self.v[prev, j]
        return result

    def _lag_value(self, j: int, t: int, lag: int,
                   start: int, predicted: np.ndarray) -> float:
        prev = t - lag
        if prev >= start:
            value = predicted[prev - start]
        else:
            value = self.v[prev, j] if prev >= 0 else np.nan
        if not np.isfinite(value):
            value = self.seasonal[prev, j] if prev >= 0 else self.seasonal[t, j]
        return float(value)

    def predict_recursive(self, j: int, start: int, end: int,
                          spatiotemporal: bool, mask_all: bool) -> np.ndarray:
        predicted = np.full(end - start, np.nan)
        model = self.st_models[j] if spatiotemporal else self.ar_models[j]
        a, b = self.donors[j]
        for h, t in enumerate(range(start, end)):
            history = [self._lag_value(j, t, lag, start, predicted) for lag in LAGS]
            if spatiotemporal:
                donors = [np.nan, np.nan] if mask_all else [self.v[t, a], self.v[t, b]]
                available = [float(np.isfinite(z)) for z in donors]
                donors = [
                    float(z) if np.isfinite(z) else float(self.seasonal[t, donor])
                    for z, donor in zip(donors, (a, b))
                ]
                x = np.concatenate([history, donors, available, self.cal[t]])
            else:
                x = np.concatenate([history, self.cal[t]])
            predicted[h] = np.clip(
                model.predict_one(x), SPEED_MIN, SPEED_MAX
            )
        return predicted

    def predict_spatial(self, j: int, start: int, end: int,
                        mask_all: bool) -> np.ndarray:
        if mask_all:
            return np.full(end - start, np.nan)
        a, b = self.donors[j]
        donors = self.v[start:end, [a, b]].T
        pair = np.column_stack([donors[0], donors[1]])
        x = np.column_stack([pair, self.cal[start:end]])
        okay = np.isfinite(pair).all(axis=1)
        p = np.full(end - start, np.nan)
        if okay.any():
            p[okay] = np.clip(
                self.spatial_models[j].predict_many(x[okay]),
                SPEED_MIN, SPEED_MAX
            )
        return p

    def predictions(self, j: int, start: int, end: int, mask_all: bool):
        return {
            "linear_offline": self.predict_linear(j, start, end),
            "seasonal_2024": self.seasonal[start:end, j].copy(),
            "previous_week": self.predict_week(j, start, end),
            "recursive_ar": self.predict_recursive(j, start, end, False, mask_all),
            "spatial_ridge": self.predict_spatial(j, start, end, mask_all),
            "spacetime_ridge": self.predict_recursive(j, start, end, True, mask_all),
        }


def score_block(truth: np.ndarray, predicted: np.ndarray,
                threshold: float) -> dict:
    valid_truth = np.isfinite(truth)
    eligible = valid_truth & np.isfinite(predicted)
    n_true = int(valid_truth.sum())
    n = int(eligible.sum())
    if n == 0:
        return dict(n_true=n_true, n_pred=0, abs_sum=0.0, sq_sum=0.0,
                    p90_ae=np.nan, bias_sum=0.0, tp=0, fp=0, fn=0)
    err = predicted[eligible] - truth[eligible]
    congested = truth[eligible] < threshold
    guessed = predicted[eligible] < threshold
    return {
        "n_true": n_true, "n_pred": n,
        "abs_sum": float(np.abs(err).sum()),
        "sq_sum": float((err ** 2).sum()),
        "p90_ae": float(np.quantile(np.abs(err), 0.90)),
        "bias_sum": float(err.sum()),
        "tp": int((congested & guessed).sum()),
        "fp": int((~congested & guessed).sum()),
        "fn": int((congested & ~guessed).sum()),
    }


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["scenario", "missing_hours", "method", "congestion_threshold"]
    for name, grp in detail.groupby(keys, sort=True):
        total = grp[["n_true", "n_pred", "abs_sum", "sq_sum",
                     "bias_sum", "tp", "fp", "fn"]].sum()
        n = int(total["n_pred"])
        tp, fp, fn = [int(total[x]) for x in ("tp", "fp", "fn")]
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        records.append({
            **dict(zip(keys, name)),
            "blocks": int(grp["block_id"].nunique()),
            "n_true": int(total["n_true"]),
            "n_pred": n,
            "coverage": n / total["n_true"] if total["n_true"] else np.nan,
            "mae_kmh": total["abs_sum"] / n if n else np.nan,
            "rmse_kmh": math.sqrt(total["sq_sum"] / n) if n else np.nan,
            "bias_kmh": total["bias_sum"] / n if n else np.nan,
            "precision_congestion": precision,
            "recall_congestion": recall,
            "f1_congestion": f1,
            "n_congestion_truth_scored": tp + fn,
        })
    return pd.DataFrame.from_records(records)


def artificial_block_experiment(
    exp: VDExperiment, lengths: list[int], trials: int, seed: int,
    threshold: float,
):
    blocks = exp.select_blocks(lengths, trials, seed)
    results = []
    example = None
    scored_thresholds = tuple(sorted(set((*THRESHOLDS, threshold))))
    for bidx, block in enumerate(blocks, 1):
        start, end, length = (
            block["start_idx"], block["end_idx"], block["length_bins"]
        )
        for j in range(exp.k):
            truth = exp.v[start:end, j]
            for scenario in ("single", "all_five"):
                pred = exp.predictions(j, start, end, mask_all=(scenario == "all_five"))
                for method, values in pred.items():
                    for th in scored_thresholds:
                        metrics = score_block(truth, values, th)
                        results.append({
                            "block_id": bidx,
                            "start_time": block["start_time"],
                            "last_hidden_time": block["last_hidden_time"],
                            "scenario": scenario,
                            "VDID": exp.ids[j],
                            "missing_hours": length / 12,
                            "method": method,
                            "congestion_threshold": th,
                            **metrics,
                        })
                # Show one held-out real-world 72-hour trace around the bottleneck.
                if (
                    scenario == "all_five"
                    and j == 2 and length == 72 * 12 and example is None
                ):
                    example = {
                        "times": exp.times[start:end],
                        "truth": truth,
                        "predictions": pred,
                        "start": block["start_time"],
                    }
        print(f"Validated artificial block {bidx:02d}/{len(blocks)}: "
              f"{length/12:.1f}h, {block['start_time']}", flush=True)
    details = pd.DataFrame(results)
    summary = summarise(details)
    return blocks, details, summary, example


def genuine_missing_runs(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    for part in np.split(idx, breaks):
        yield int(part[0]), int(part[-1]) + 1


def actual_missing_candidates(exp: VDExperiment) -> pd.DataFrame:
    rows = []
    for j in range(exp.k):
        for start, end in genuine_missing_runs(exp.missing[:, j]):
            p = exp.predictions(j, start, end, mask_all=False)
            for h, t in enumerate(range(start, end)):
                row = {
                    "TargetTime": str(exp.times[t]),
                    "VDID": exp.ids[j],
                    "VDType_original": str(exp.types[t, j]),
                    "Speed_original_code": float(exp.raw[t, j]),
                    "gap_length_bins": end - start,
                    "gap_length_hours": (end - start) / 12,
                    "other_VD_valid_speed_count": int(
                        np.isfinite(exp.v[t]).sum()
                    ),
                }
                row.update({
                    name: float(series[h]) if np.isfinite(series[h]) else np.nan
                    for name, series in p.items()
                })
                rows.append(row)
    out = pd.DataFrame(rows).sort_values(["TargetTime", "VDID"]).reset_index(drop=True)
    if len(out) != 29592:
        raise AssertionError(f"Expected 29,592 original missing cells, got {len(out)}")
    return out


def plot_example(example: dict, destination: Path, threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4.7))
    ax.plot(example["times"], example["truth"], color="#1A344A",
            linewidth=1.4, label="True speed (artificially hidden)")
    colors = {
        "linear_offline": "#EA8A32",
        "seasonal_2024": "#368CCB",
        "recursive_ar": "#B13C8C",
        "spacetime_ridge": "#38866B",
    }
    for name, color in colors.items():
        ax.plot(example["times"], example["predictions"][name],
                linewidth=1.05, alpha=0.84, color=color, label=name)
    ax.axhline(threshold, color="#AD4E4E", linestyle="--", linewidth=1,
               label=f"Congestion threshold ({threshold:g} km/h)")
    ax.set(title=f"63.990K: 72-hour held-out validation block starting {example['start']}",
           ylabel="Speed (km/h)", xlabel="Time (Taiwan local)")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.grid(alpha=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vd_imputation_results"))
    parser.add_argument("--trials-per-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--congestion-threshold", type=float, default=60.0)
    parser.add_argument(
        "--hours", type=float, nargs="+", default=[0.25, 1, 6, 24, 72, 151.5],
        help="Artificially hidden lengths in hours (must be multiples of 5 min)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "mplcache"))
    lengths = [int(round(h * 12)) for h in args.hours]
    if any(abs(h * 12 - n) > 1e-6 for h, n in zip(args.hours, lengths)):
        parser.error("Each --hours value must be a multiple of five minutes.")
    if not args.parquet.exists():
        parser.error(f"Cannot find Parquet: {args.parquet}")
    exp = VDExperiment(args.parquet)
    print("Loaded", exp.n, "five-minute slots x", exp.k,
          "VDs; original true missing", int(exp.missing.sum()), flush=True)
    blocks, details, summary, example = artificial_block_experiment(
        exp, lengths, args.trials_per_length, args.seed, args.congestion_threshold
    )
    real = actual_missing_candidates(exp)
    pd.DataFrame(blocks).to_csv(args.output / "masked_blocks_manifest.csv", index=False,
                                encoding="utf-8-sig")
    details.to_csv(args.output / "benchmark_by_block.csv", index=False,
                   encoding="utf-8-sig")
    summary.to_csv(args.output / "benchmark_summary.csv", index=False,
                   encoding="utf-8-sig")
    real.to_csv(args.output / "real_missing_candidate_estimates.csv", index=False,
                encoding="utf-8-sig", na_rep="")
    if example is not None:
        plot_example(example, args.output / "example_72h_all_five.png",
                     args.congestion_threshold)
    conf = {
        "source_file": args.parquet.name,
        "train_year": 2024, "heldout_year": 2025,
        "seed": args.seed, "trials_per_length": args.trials_per_length,
        "length_hours": args.hours, "congestion_threshold_kmh": args.congestion_threshold,
        "scored_thresholds_kmh": sorted(set((*THRESHOLDS, args.congestion_threshold))),
        "speed_prediction_range_kmh": [SPEED_MIN, SPEED_MAX],
        "speed_minus_one": "Observed no vehicle: excluded from speed targets, not true missing",
        "genuine_missing_types": sorted(MISSING_TYPES),
        "linear_offline": "Uses observed valid endpoints before and after the hidden block",
        "recursive_ar": "Uses past speeds (including recursively forecast values), time features",
        "spatial_ridge": "Requires both nearby VD speed measurements at the same time",
        "spacetime_ridge": (
            "Own recursive lags, calendar, two nearby speeds; missing neighbors "
            "replaced by 2024 seasonal medians with availability flags. "
            "All-five-missing extrapolation is high-risk."
        ),
        "status": "Candidate estimates only; no original speed was overwritten",
        "weather_and_OD": "Not used in this pilot: future independent covariates to be tested",
        "real_2024_caution": "2024 candidate models were trained on other 2024 observations, including later dates (retrospective only)",
    }
    (args.output / "experiment_config.json").write_text(
        json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selected = summary[
        np.isclose(summary["congestion_threshold"], args.congestion_threshold)
    ]
    print(f"\nBenchmark summary ({args.congestion_threshold:g} km/h threshold):")
    print(selected[["scenario", "missing_hours", "method", "coverage",
                  "mae_kmh", "rmse_kmh", "f1_congestion"]].round(3).to_string(index=False))
    print("\nOutput folder:", args.output.resolve())


if __name__ == "__main__":
    main()
