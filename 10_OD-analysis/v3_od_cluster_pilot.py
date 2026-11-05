#!/usr/bin/env python3
r"""V3 OD-pair clustering pilot from the completed quick-EDA CSV exports.

Unit: ordered (GantryO, GantryD). Fit on 2024 ONLY, with 2024 >= 300 trips.
2025 outcomes, weather, VD speed and Y are never used for training or ranking.

Example (PowerShell):
  .\.venv\Scripts\python.exe .\10_OD-analysis\v3_od_cluster_pilot.py `
    --eda-dir "D:\\statistical_practice\\v3_od_eda" `
    --output "D:\\statistical_practice\\v3_od_cluster_pilot" `
    --methods kmeans gmm kmedoids sparse --restarts 3

This is an exploratory pilot. The best predictive clustering requires 2024
time-ordered CV and downstream B0/B1/B2 evaluation; do NOT select using 2025.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

KEYS = ["GantryO", "GantryD"]
SETS = {
    "F1_weekend_time": ["weekend"],
    "F2_time_holiday": ["weekend", "holiday"],
    "F3_plus_distance": ["weekend", "holiday", "distance"],
    "F4_plus_vehicle": ["weekend", "holiday", "distance", "vehicle"],
}


def kmedoids_alternate(x: np.ndarray, k: int, seed: int, max_iter: int = 15) -> np.ndarray:
    """Euclidean alternating K-medoids; intentionally not full PAM-swap."""
    init = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(x)
    centers = np.stack([x[init == j].mean(axis=0) for j in range(k)])
    medoids = [int(np.argmin(cdist(centers[j:j + 1], x))) for j in range(k)]
    distance = cdist(x, x)
    for _ in range(max_iter):
        labels = np.argmin(distance[:, medoids], axis=1)
        next_medoids = []
        for j in range(k):
            members = np.flatnonzero(labels == j)
            if not len(members):
                members = np.array([int(np.argmax(distance[:, medoids].min(axis=1)))])
            within = distance[np.ix_(members, members)].sum(axis=1)
            next_medoids.append(int(members[np.argmin(within)]))
        if next_medoids == medoids:
            break
        medoids = next_medoids
    return np.argmin(distance[:, medoids], axis=1)


def sparse_weights(between: np.ndarray, l1: float) -> np.ndarray:
    between = np.maximum(between, 0)
    if not np.any(between):
        return np.ones_like(between) / math.sqrt(len(between))
    full = between / np.linalg.norm(between)
    if full.sum() <= l1:
        return full
    left, right = 0., float(between.max())
    for _ in range(45):
        middle = (left + right) / 2
        weights = np.maximum(between - middle, 0)
        if not np.any(weights):
            right = middle
            continue
        if (weights / np.linalg.norm(weights)).sum() > l1:
            left = middle
        else:
            right = middle
    weights = np.maximum(between - right, 0)
    return weights / np.linalg.norm(weights) if np.any(weights) else full


def sparse_kmeans(x: np.ndarray, k: int, seed: int, l1: float, max_iter: int = 25):
    weights = np.ones(x.shape[1]) / math.sqrt(x.shape[1])
    old_labels = None
    for iteration in range(max_iter):
        labels = KMeans(n_clusters=k, n_init=3 if iteration == 0 else 1,
                        random_state=seed + iteration).fit_predict(x * np.sqrt(weights))
        mu = x.mean(axis=0)
        between = np.zeros(x.shape[1])
        for j in range(k):
            members = x[labels == j]
            if len(members):
                between += len(members) * (members.mean(axis=0) - mu) ** 2
        updated = sparse_weights(between, min(l1, math.sqrt(x.shape[1])))
        if old_labels is not None and np.array_equal(old_labels, labels) and np.max(np.abs(updated - weights)) < 1e-5:
            weights = updated
            break
        weights, old_labels = updated, labels
    labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(x * np.sqrt(weights))
    return labels, weights


def fit_one(x: np.ndarray, method: str, k: int, seed: int, args: argparse.Namespace):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x), None, np.nan
    if method == "kmedoids":
        return kmedoids_alternate(x, k, seed), None, np.nan
    if method == "gmm":
        gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-5,
                              n_init=2, random_state=seed, max_iter=200)
        return gmm.fit_predict(x), None, float(gmm.bic(x))
    labels, weights = sparse_kmeans(x, k, seed, args.sparse_l1)
    return labels, weights, np.nan


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eda-dir", type=Path, required=True,
                   help="Directory with 00_calendar_days, 04_od_hourly_profiles and 08_od_feature_screen_2024 CSV files")
    p.add_argument("--output", type=Path, default=Path("outputs/v3_od_cluster_pilot"))
    p.add_argument("--min-trips", type=int, default=300)
    p.add_argument("--methods", nargs="+", choices=["kmeans", "gmm", "kmedoids", "sparse"],
                   default=["kmeans", "gmm"])
    p.add_argument("--feature-sets", nargs="+", choices=list(SETS), default=list(SETS))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--sparse-l1", type=float, default=2.5)
    p.add_argument("--smoothing-trips", type=float, default=10.0)
    p.add_argument("--silhouette-sample", type=int, default=1000)
    return p.parse_args()


def read(name: str, directory: Path) -> pd.DataFrame:
    return pd.read_csv(directory / name, encoding="utf-8-sig", dtype={"GantryO": "string", "GantryD": "string"})


def prepare(eda_dir: Path, min_trips: int, alpha: float) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    screen = read("08_od_feature_screen_2024.csv", eda_dir)
    hourly = read("04_od_hourly_profiles.csv", eda_dir)
    days = read("00_calendar_days.csv", eda_dir)
    if screen.duplicated(KEYS).any():
        raise ValueError("Duplicate OD identifiers in feature screen")
    if not {"Year", "DayType", "hour", "trips"}.issubset(hourly.columns):
        raise ValueError("Hourly source is missing required columns")
    if not {"n_2024", "triplen_median_raw", "n_2024_weekend", "n_2024_longholiday"}.issubset(screen.columns):
        raise ValueError("Feature screen is missing required training columns")
    screen = screen.loc[screen.n_2024 >= min_trips].copy().sort_values(KEYS).reset_index(drop=True)
    hourly = hourly.loc[hourly.Year == 2024].copy()
    hourly["bin"] = (hourly.hour // 4).astype(int)
    blocks: dict[str, list[str]] = {"weekend": [], "holiday": [], "distance": [], "vehicle": []}

    # Build each OD's 6-bin distribution from actual 2024 counts; use global
    # 2024 distribution as a Dirichlet prior for small/absent holiday samples.
    distributions: dict[str, np.ndarray] = {}
    for daytype in ["WEEKEND", "LONGHOLIDAY"]:
        data = hourly.loc[hourly.DayType == daytype]
        pivot = data.pivot_table(index=KEYS, columns="bin", values="trips", aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(columns=range(6), fill_value=0)
        grid = screen[KEYS].merge(pivot.reset_index(), on=KEYS, how="left", validate="one_to_one")
        counts = grid[list(range(6))].fillna(0).to_numpy(dtype=float)
        totals = counts.sum(axis=1)
        prior = counts.sum(axis=0)
        prior = prior / prior.sum() if prior.sum() else np.repeat(1 / 6, 6)
        distributions[daytype] = (counts + alpha * prior) / (totals[:, None] + alpha)
        expected = screen["n_2024_" + daytype.lower()].to_numpy(dtype=float)
        if not np.array_equal(totals.astype(int), expected.astype(int)):
            raise ValueError(f"2024 {daytype} hourly counts differ from feature-screen totals")

    for b in range(6):
        name = f"weekend_b{b}_share"
        screen[name] = distributions["WEEKEND"][:, b]
        blocks["weekend"].append(name)
    # Six shifts add to zero: omit last bin to avoid a mechanically redundant column.
    for b in range(5):
        name = f"holiday_minus_weekend_b{b}_share"
        screen[name] = distributions["LONGHOLIDAY"][:, b] - distributions["WEEKEND"][:, b]
        blocks["holiday"].append(name)

    # Avoid comparing raw totals from very different numbers of observation days.
    # Exposures come from the EDA calendar, rather than the count of OD-present days.
    day_counts = {(int(r.Year), str(r.DayType)): int(r.days) for r in days.itertuples()}
    nw = day_counts.get((2024, "WEEKEND"), 0)
    nh = day_counts.get((2024, "LONGHOLIDAY"), 0)
    if nw <= 0 or nh <= 0:
        raise ValueError("2024 exposure-day counts are missing")
    name = "log_holiday_weekend_daily_ratio"
    screen[name] = np.log((screen.n_2024_longholiday + 0.5) / nh) - np.log((screen.n_2024_weekend + 0.5) / nw)
    blocks["holiday"].append(name)

    screen["log_distance"] = np.log1p(screen.triplen_median_raw.clip(lower=0))
    blocks["distance"].append("log_distance")
    # Four independent vehicle shares; the fifth/OTHER values are almost fully
    # determined by the rest and VehicleType codes are NOT decoded here.
    for code in ["31", "32", "41", "42"]:
        source = f"vehicle_share_pct_{code}"
        if source in screen:
            dest = f"vehicle_{code}_share"
            screen[dest] = screen[source] / 100
            blocks["vehicle"].append(dest)
    if not blocks["vehicle"]:
        raise ValueError("No vehicle mix columns are available")
    if screen[sum(blocks.values(), [])].isna().any().any():
        raise ValueError("Missing training feature after 2024 feature engineering")
    # Keep held-out-year aggregates completely out of the clustering export.
    screen = screen.drop(columns=[col for col in screen if "2025" in col])
    return screen, blocks, {"n_OD": len(screen), "2024_weekend_days": nw, "2024_holiday_days": nh,
                            "n_train_trips": int(screen.n_2024.sum())}


def design_matrix(frame: pd.DataFrame, selected: list[str], blocks: dict[str, list[str]]) -> np.ndarray:
    xx = frame[selected].astype(float)
    low, high = xx.quantile(.01), xx.quantile(.99)
    xx = xx.clip(lower=low, upper=high, axis=1)
    x = xx.to_numpy(dtype=float)
    med = np.median(x, axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1
    x = (x - x.mean(axis=0)) / sd
    # Give each conceptual feature block equal total squared weight.
    for group in blocks.values():
        ix = [selected.index(c) for c in group if c in selected]
        if ix:
            x[:, ix] /= math.sqrt(len(ix))
    return x


def run(args: argparse.Namespace) -> None:
    if args.k_min < 2 or args.k_max < args.k_min or args.restarts < 1:
        raise ValueError("Require K >= 2 and at least one restart")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame, blocks, facts = prepare(args.eda_dir, args.min_trips, args.smoothing_trips)
    frame.to_csv(out / "v3_od_features_2024.csv", index=False, encoding="utf-8-sig")
    metrics: list[dict] = []
    assignment: list[pd.DataFrame] = []
    profiles: list[dict] = []
    weight_rows: list[dict] = []
    for fs in args.feature_sets:
        selected = [col for group in SETS[fs] for col in blocks[group]]
        x = design_matrix(frame, selected, blocks)
        for method in args.methods:
            for k in range(args.k_min, args.k_max + 1):
                trials = []
                for r in range(args.restarts):
                    seed = args.seed + r * 73
                    labels, weights, bic = fit_one(x, method, k, seed, args)
                    if len(np.unique(labels)) != k:
                        continue
                    sil = float(silhouette_score(x, labels, sample_size=min(len(x), args.silhouette_sample),
                                                 random_state=args.seed))
                    trials.append((labels, weights, bic, sil))
                if not trials:
                    continue
                best = max(trials, key=lambda t: t[3])
                labels = best[0]
                odcounts = np.bincount(labels, minlength=k)
                tripcounts = np.bincount(labels, weights=frame.n_2024.to_numpy(), minlength=k)
                stability = (float(np.mean([adjusted_rand_score(trials[0][0], a[0]) for a in trials[1:]]))
                             if len(trials) > 1 else np.nan)
                metric = {"feature_set": fs, "method": method, "K": k, "n_OD": len(frame),
                          "n_features": len(selected), "silhouette": best[3],
                          "mean_silhouette": float(np.mean([a[3] for a in trials])),
                          "restart_ARI": stability, "min_OD_share": float(odcounts.min() / len(frame)),
                          "min_trip_share": float(tripcounts.min() / tripcounts.sum()),
                          "gmm_bic": (float(min(a[2] for a in trials)) if method == "gmm" else np.nan),
                          "n_restarts": len(trials)}
                metrics.append(metric)
                each = frame[KEYS + ["n_2024"]].copy()
                each.insert(0, "feature_set", fs)
                each.insert(1, "method", method)
                each.insert(2, "K", k)
                each["cluster"] = labels + 1
                assignment.append(each)
                for j in range(k):
                    subset = frame.iloc[np.flatnonzero(labels == j)]
                    profiles.append({
                        "feature_set": fs, "method": method, "K": k, "cluster": j + 1,
                        "n_OD": len(subset), "n_trips_2024": int(subset.n_2024.sum()),
                        "share_trips_2024": float(subset.n_2024.sum() / frame.n_2024.sum()),
                        "median_distance_raw": float(subset.triplen_median_raw.median()),
                        "median_holiday_weekend_daily_ratio_2024": float(subset.holiday_weekend_daily_ratio_2024.median()),
                        "mean_weekend_00_03_pct": float(subset.weekend_b0_share.mean() * 100),
                        "mean_weekend_12_15_pct": float(subset.weekend_b3_share.mean() * 100),
                        "mean_weekend_16_19_pct": float(subset.weekend_b4_share.mean() * 100),
                        "mean_holiday_shift_00_03_pp": float(subset.holiday_minus_weekend_b0_share.mean() * 100),
                        "mean_holiday_shift_16_19_pp": float(subset.holiday_minus_weekend_b4_share.mean() * 100),
                    })
                if best[1] is not None:
                    for feature, w in zip(selected, best[1]):
                        weight_rows.append({"feature_set": fs, "method": method, "K": k,
                                            "feature": feature, "sparse_weight": float(w)})
                print(f"{fs:22} {method:9} K={k} silhouette={best[3]:.3f} "
                      f"ARI={stability:.3f} smallest_OD={odcounts.min()}", flush=True)

    score = pd.DataFrame(metrics).sort_values(["feature_set", "method", "K"])
    score.to_csv(out / "v3_method_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profiles).to_csv(out / "v3_cluster_profiles_all.csv", index=False, encoding="utf-8-sig")
    label_long = pd.concat(assignment, ignore_index=True)
    label_long.to_csv(out / "v3_od_labels_all_candidates.csv", index=False, encoding="utf-8-sig")
    # Compact, one-row-per-OD matrix for downstream B2 comparisons.
    label_long["candidate"] = (label_long.feature_set + "__" + label_long.method + "__K" +
                               label_long.K.astype(str))
    label_wide = label_long.pivot(index=KEYS, columns="candidate", values="cluster").reset_index()
    label_wide.to_csv(out / "v3_od_labels_wide.csv", index=False, encoding="utf-8-sig")
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(out / "v3_sparse_weights.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 5))
    for (fs, method), part in score.groupby(["feature_set", "method"], sort=True):
        plt.plot(part.K, part.silhouette, marker="o", label=f"{fs} / {method}")
    plt.xticks(range(args.k_min, args.k_max + 1))
    plt.xlabel("Number of OD clusters (K)")
    plt.ylabel("Silhouette (exploratory only)")
    plt.title("V3 OD clustering: 2024 training data")
    plt.grid(alpha=.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out / "v3_pilot_silhouette.png", dpi=160)
    plt.close()
    (out / "v3_run_summary.json").write_text(json.dumps({"facts": facts, "methods": args.methods,
        "feature_sets": {fs: [col for g in SETS[fs] for col in blocks[g]] for fs in args.feature_sets},
        "K_range": [args.k_min, args.k_max], "restarts": args.restarts,
        "warning": "Exploratory fit only. 2024 blocked-CV downstream evaluation is required; 2025 untouched."},
        indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run(parse())
