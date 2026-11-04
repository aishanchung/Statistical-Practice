#!/usr/bin/env python3
"""OD-level demand clustering for the N3 northbound M06B/V2_LITE Parquet parts.

The raw 45M trip rows are scanned by DuckDB, never loaded into pandas. The
clustering unit is an ordered (GantryO, GantryD) pair, not an individual trip.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DAYTYPES = ("WEEKDAY", "WEEKEND", "LONGHOLIDAY", "STANDALONEHOLIDAY")
REQUIRED = ("GantryO", "GantryD", "BottleneckTime", "DayType", "VehicleType", "TripLength", "DetectionTimeD")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="Folder of part_*.parquet files, a glob, or one Parquet file")
    p.add_argument("--output", default="outputs/od_clustering")
    p.add_argument("--train-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--min-trips", type=int, default=100)
    p.add_argument("--thresholds", type=int, nargs="+", default=[30, 100, 300, 500, 1000])
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--methods", nargs="+", default=["kmeans", "kmedoids", "gmm", "sparse"],
                   choices=["kmeans", "kmedoids", "gmm", "sparse"])
    p.add_argument("--feature-sets", nargs="+", default=["time", "time_vehicle", "time_vehicle_distance", "time_vehicle_distance_volume", "all_spatial"])
    p.add_argument("--vehicle-types", default="31,32,41,42,5", help="Observed VehicleType codes; other codes become OTHER")
    p.add_argument("--smooth", type=float, default=10.0, help="Dirichlet shrinkage strength for low-count profiles")
    p.add_argument("--sparse-l1", type=float, default=2.5, help="Sparse k-means L1 bound; must be >= 1")
    p.add_argument("--gantry-map", help="Optional CSV with GantryID, latitude, longitude for spatial features")
    p.add_argument("--exclude-date", action="append", default=[], help="Explicit QC date YYYY-MM-DD to omit; repeatable")
    p.add_argument("--five-minute", choices=["none", "bottleneck", "completed", "both"], default="none",
                   help="Optional extra full-data scan; completed uses DetectionTimeD")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-limit", default="4GB", help="DuckDB memory cap, e.g. 4GB")
    p.add_argument("--seed", type=int, default=20261104)
    return p.parse_args()


def discover_files(spec: str) -> list[str]:
    path = Path(spec).expanduser()
    if path.is_dir():
        files = sorted(str(f.resolve()) for f in path.glob("part_*.parquet"))
        if not files:
            files = sorted(str(f.resolve()) for f in path.glob("*.parquet"))
    elif path.is_file():
        files = [str(path.resolve())]
    else:
        files = sorted(str(Path(f).resolve()) for f in glob.glob(spec))
    if not files:
        raise FileNotFoundError(f"No Parquet files found for --input {spec!r}")
    if len({Path(f).name for f in files}) != len(files):
        raise ValueError("Duplicate part filenames found; use one input directory")
    part_numbers = sorted(int(m.group(1)) for f in files if (m := re.fullmatch(r"part_(\d+)\.parquet", Path(f).name)))
    if part_numbers and len(part_numbers) == len(files):
        missing = sorted(set(range(part_numbers[0], part_numbers[-1] + 1)) - set(part_numbers))
        if missing:
            raise ValueError(f"Missing Parquet part numbers: {missing}; do not run on incomplete data")
    return files


def sql_source() -> str:
    return "read_parquet(?, union_by_name = true)"


def quote_code(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def date_filter(args: argparse.Namespace, year: int) -> str:
    clauses = [f"BottleneckTime >= TIMESTAMP '{year}-01-01 00:00:00'",
               f"BottleneckTime < TIMESTAMP '{year + 1}-01-01 00:00:00'",
               "GantryO IS NOT NULL", "GantryD IS NOT NULL"]
    for value in args.exclude_date:
        pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
        clauses.append(f"CAST(BottleneckTime AS DATE) <> DATE {quote_code(value)}")
    return " AND ".join(clauses)


def prepare_aggregates(conn: duckdb.DuckDBPyConnection, files: list[str], args: argparse.Namespace,
                       out: Path, vehicle_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    names = set(conn.execute(f"DESCRIBE SELECT * FROM {sql_source()}", [files]).df()["column_name"])
    absent = set(REQUIRED) - names
    if absent:
        raise ValueError(f"Missing required columns: {sorted(absent)}")

    # The same raw input may be scanned several times, but only aggregated tables enter pandas.
    start = ["GantryO", "GantryD", "COUNT(*) AS n_train",
             "approx_quantile(TripLength, 0.25) AS dist_q25",
             "approx_quantile(TripLength, 0.50) AS dist_median",
             "approx_quantile(TripLength, 0.75) AS dist_q75"]
    for dt in DAYTYPES:
        start.append(f"COUNT(*) FILTER (WHERE DayType = {quote_code(dt)}) AS n_{dt}")
    for dt, prefix in (("WEEKDAY", "wd"), ("WEEKEND", "we")):
        for block in range(6):
            start.append(f"COUNT(*) FILTER (WHERE DayType = {quote_code(dt)} AND "
                         f"FLOOR(EXTRACT(HOUR FROM BottleneckTime) / 4) = {block}) AS n_{prefix}_b{block}")
    for vt in vehicle_types:
        start.append(f"COUNT(*) FILTER (WHERE VehicleType = {quote_code(vt)}) AS n_vehicle_{vt}")
    start.append("COUNT(*) FILTER (WHERE VehicleType NOT IN (" + ",".join(map(quote_code, vehicle_types)) +
                 ") OR VehicleType IS NULL) AS n_vehicle_OTHER")
    query = ("SELECT " + ",\n".join(start) + f"\nFROM {sql_source()}\nWHERE {date_filter(args, args.train_year)}\n"
             "GROUP BY GantryO, GantryD")
    print("[1/5] Aggregating OD features for training year...", flush=True)
    train = conn.execute(query, [files]).df()
    train.to_csv(out / "od_aggregates_train.csv", index=False, encoding="utf-8-sig")

    print("[2/5] Counting OD demand in held-out year...", flush=True)
    test = conn.execute(f"SELECT GantryO, GantryD, COUNT(*) AS n_test FROM {sql_source()} "
                        f"WHERE {date_filter(args, args.test_year)} GROUP BY 1, 2", [files]).df()
    test.to_csv(out / "od_counts_test.csv", index=False, encoding="utf-8-sig")

    # Unique observed days avoid comparing raw weekday/weekend totals with unequal exposures.
    exposure = conn.execute(f"SELECT DayType, COUNT(DISTINCT CAST(BottleneckTime AS DATE)) AS n_days "
                            f"FROM {sql_source()} WHERE {date_filter(args, args.train_year)} "
                            "GROUP BY DayType", [files]).fetchall()
    days = {str(key): int(value) for key, value in exposure if key is not None}
    (out / "observed_daytype_days.json").write_text(json.dumps(days, ensure_ascii=False, indent=2), encoding="utf-8")

    merged = train[["GantryO", "GantryD", "n_train"]].merge(test, on=["GantryO", "GantryD"], how="outer")
    merged[["n_train", "n_test"]] = merged[["n_train", "n_test"]].fillna(0).astype(int)
    merged.to_csv(out / "od_yearly_counts.csv", index=False, encoding="utf-8-sig")
    denom = int(test.n_test.sum())
    cover = []
    for threshold in sorted(set(args.thresholds + [args.min_trips])):
        selected = merged.n_train >= threshold
        cover.append({"min_train_trips": threshold, "eligible_od_count": int(selected.sum()),
                      "test_trip_coverage": float(merged.loc[selected, "n_test"].sum() / denom) if denom else np.nan,
                      "eligible_test_trips": int(merged.loc[selected, "n_test"].sum()),
                      "all_test_trips": denom})
    pd.DataFrame(cover).to_csv(out / "threshold_coverage.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([v["min_train_trips"] for v in cover], [100 * v["test_trip_coverage"] for v in cover], "o-")
    ax.set(xlabel=f"Minimum {args.train_year} trips per OD", ylabel=f"{args.test_year} demand coverage (%)",
           title="OD threshold sensitivity")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "threshold_coverage.png", dpi=150)
    plt.close(fig)
    print(f"  {len(train):,} training OD pairs, {denom:,} held-out trips.", flush=True)
    return train, test, days


def add_features(agg: pd.DataFrame, days: dict, vehicle_types: list[str], alpha: float,
                 gantry_map: str | None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    data = agg.copy()
    blocks: dict[str, list[str]] = {key: [] for key in ["time", "date", "vehicle", "distance", "volume", "spatial"]}
    global_days = max(sum(days.values()), 1)

    for dt, prefix in (("WEEKDAY", "wd"), ("WEEKEND", "we")):
        cols = [f"n_{prefix}_b{i}" for i in range(6)]
        hist = np.array([data[c].sum() for c in cols], dtype=float)
        prior = hist / hist.sum() if hist.sum() else np.full(6, 1 / 6)
        for i, col in enumerate(cols):
            name = f"p_{prefix}_b{i}"
            data[name] = (data[col] + alpha * prior[i]) / (data[f"n_{dt}"] + alpha)
            blocks["time"].append(name)

    def daily(dt: str) -> pd.Series:
        return data[f"n_{dt}"] / max(days.get(dt, 0), 1)

    weekday = daily("WEEKDAY")
    weekend = daily("WEEKEND")
    data["log_weekend_weekday"] = np.log1p(weekend) - np.log1p(weekday)
    blocks["date"].append("log_weekend_weekday")
    for dt in ("LONGHOLIDAY", "STANDALONEHOLIDAY"):
        if days.get(dt, 0):
            name = f"log_{dt.lower()}_weekday"
            data[name] = np.log1p(daily(dt)) - np.log1p(weekday)
            blocks["date"].append(name)

    types = vehicle_types + ["OTHER"]
    n_by_type = np.array([data[f"n_vehicle_{t}"].sum() for t in types], dtype=float)
    prior = n_by_type / max(n_by_type.sum(), 1)
    for vt, p in zip(types, prior):
        name = f"p_vehicle_{vt}"
        data[name] = (data[f"n_vehicle_{vt}"] + alpha * p) / (data.n_train + alpha)
        blocks["vehicle"].append(name)

    data["log_dist_median"] = np.log1p(data.dist_median.clip(lower=0))
    data["log_dist_iqr"] = np.log1p((data.dist_q75 - data.dist_q25).clip(lower=0))
    blocks["distance"] = ["log_dist_median", "log_dist_iqr"]
    data["log_daily_volume"] = np.log1p(data.n_train / global_days)
    blocks["volume"] = ["log_daily_volume"]

    if gantry_map:
        gm = pd.read_csv(gantry_map, dtype={"GantryID": "string"})
        required = {"GantryID", "latitude", "longitude"}
        if not required.issubset(gm.columns):
            raise ValueError("--gantry-map needs columns: GantryID, latitude, longitude")
        if gm.GantryID.duplicated().any():
            raise ValueError("Gantry map contains duplicate GantryID values")
        gm = gm[["GantryID", "latitude", "longitude"]]
        for side in "OD":
            origin = gm.rename(columns={"GantryID": f"Gantry{side}", "latitude": f"lat_{side}", "longitude": f"lon_{side}"})
            data = data.merge(origin, on=f"Gantry{side}", how="left", validate="many_to_one")
        blocks["spatial"] = ["lat_O", "lon_O", "lat_D", "lon_D"]
        missing = data[blocks["spatial"]].isna().any(axis=1).sum()
        print(f"  Spatial mapping missing for {missing:,} OD pairs; median-imputed for clustering.")
    return data, blocks


def feature_groups(blocks: dict[str, list[str]], requested: list[str]) -> dict[str, list[str]]:
    specifications = {
        "time": ["time", "date"],
        "time_vehicle": ["time", "date", "vehicle"],
        "time_vehicle_distance": ["time", "date", "vehicle", "distance"],
        "time_vehicle_distance_volume": ["time", "date", "vehicle", "distance", "volume"],
        "all_spatial": ["time", "date", "vehicle", "distance", "spatial"],
    }
    invalid = set(requested) - set(specifications)
    if invalid:
        raise ValueError(f"Unknown feature sets: {invalid}")
    return {name: [c for b in specifications[name] for c in blocks[b]] for name in requested
            if all(blocks[b] for b in specifications[name])}


def matrix(data: pd.DataFrame, columns: list[str], blocks: dict[str, list[str]]) -> np.ndarray:
    x = data[columns].replace([np.inf, -np.inf], np.nan).astype(float)
    x = x.fillna(x.median()).fillna(0)
    low, high = x.quantile(0.01), x.quantile(0.99)
    x = x.clip(lower=low, upper=high, axis=1)
    transformed = StandardScaler().fit_transform(x)
    # Equalize feature BLOCKS; 12 time bins should not automatically swamp distance.
    for group, members in blocks.items():
        indices = [columns.index(c) for c in members if c in columns]
        if indices:
            transformed[:, indices] /= math.sqrt(len(indices))
    return transformed


def kmedoids_alternate(x: np.ndarray, k: int, seed: int, max_iter: int = 15) -> np.ndarray:
    labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(x)
    centers = np.vstack([x[labels == j].mean(axis=0) for j in range(k)])
    initial = cdist(centers, x)
    medoids = [int(np.argmin(initial[j])) for j in range(k)]
    distance = cdist(x, x)
    for _ in range(max_iter):
        labels = np.argmin(distance[:, medoids], axis=1)
        updated = []
        for j in range(k):
            members = np.where(labels == j)[0]
            if not len(members):
                members = np.array([int(np.argmax(distance[:, medoids].min(axis=1)))])
            within = distance[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[np.argmin(within)]))
        if updated == medoids:
            break
        medoids = updated
    return np.argmin(distance[:, medoids], axis=1)


def sparse_weights(between: np.ndarray, bound: float) -> np.ndarray:
    b = np.maximum(between, 0)
    if not np.any(b):
        return np.ones_like(b) / math.sqrt(len(b))
    full = b / np.linalg.norm(b)
    if full.sum() <= bound:
        return full
    left, right = 0.0, float(b.max())
    for _ in range(45):
        middle = (left + right) / 2
        sparse = np.maximum(b - middle, 0)
        if not np.any(sparse):
            right = middle
            continue
        w = sparse / np.linalg.norm(sparse)
        if w.sum() > bound:
            left = middle
        else:
            right = middle
    sparse = np.maximum(b - right, 0)
    return sparse / np.linalg.norm(sparse) if np.any(sparse) else full


def sparse_kmeans(x: np.ndarray, k: int, seed: int, l1: float, max_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
    if l1 < 1:
        raise ValueError("--sparse-l1 must be >= 1")
    w = np.ones(x.shape[1]) / math.sqrt(x.shape[1])
    old_labels = None
    for iteration in range(max_iter):
        labels = KMeans(n_clusters=k, n_init=3 if iteration == 0 else 1,
                        random_state=seed + iteration).fit_predict(x * np.sqrt(w))
        mu = x.mean(axis=0)
        between = np.zeros(x.shape[1])
        for group in range(k):
            members = x[labels == group]
            if len(members):
                between += len(members) * (members.mean(axis=0) - mu) ** 2
        new_w = sparse_weights(between, min(l1, math.sqrt(x.shape[1])))
        if old_labels is not None and np.array_equal(labels, old_labels) and np.max(np.abs(w - new_w)) < 1e-5:
            w = new_w
            break
        w, old_labels = new_w, labels
    final = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(x * np.sqrt(w))
    return final, w


def fit_one(x: np.ndarray, method: str, k: int, seed: int, args: argparse.Namespace):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x), None, np.nan
    if method == "kmedoids":
        return kmedoids_alternate(x, k, seed), None, np.nan
    if method == "gmm":
        gm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-5,
                             n_init=2, random_state=seed, max_iter=200)
        return gm.fit_predict(x), None, float(gm.bic(x))
    labels, weights = sparse_kmeans(x, k, seed, args.sparse_l1)
    return labels, weights, np.nan


def compare_models(data: pd.DataFrame, blocks: dict[str, list[str]], args: argparse.Namespace,
                   out: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray | None, dict]:
    features = feature_groups(blocks, args.feature_sets)
    if not features:
        raise ValueError("No usable feature set. Check gantry map and --feature-sets.")
    n = len(data)
    if n < max(args.k_max + 1, 12):
        raise ValueError(f"Only {n} eligible OD pairs; lower --min-trips or --k-max")
    results = []
    weight_rows = []
    saved = {}
    print("[3/5] Comparing methods and feature sets...", flush=True)
    for set_name, columns in features.items():
        x = matrix(data, columns, blocks)
        print(f"  {set_name}: {len(columns)} features", flush=True)
        for method in args.methods:
            for k in range(args.k_min, min(args.k_max, n - 1) + 1):
                trials = []
                for j in range(args.restarts):
                    try:
                        labels, weights, bic = fit_one(x, method, k, args.seed + j * 73, args)
                        if len(np.unique(labels)) < 2:
                            continue
                        sil = float(silhouette_score(x, labels, sample_size=min(n, 1200),
                                                     random_state=args.seed))
                        trials.append((labels, weights, bic, sil))
                    except (ValueError, np.linalg.LinAlgError) as error:
                        print(f"    SKIP {set_name}/{method}/{k}: {error}")
                if not trials:
                    continue
                stability = float(np.mean([adjusted_rand_score(trials[0][0], t[0]) for t in trials[1:]])) if len(trials) > 1 else np.nan
                best = max(trials, key=lambda t: t[3])
                proportions = np.bincount(best[0], minlength=k) / n
                row = {"feature_set": set_name, "method": method, "k": k, "silhouette": best[3],
                       "mean_silhouette": float(np.mean([t[3] for t in trials])), "stability_ari": stability,
                       "min_cluster_od_share": float(proportions.min()), "n_features": len(columns),
                       "bic": float(min(t[2] for t in trials)) if method == "gmm" else np.nan,
                       "n_successful_restarts": len(trials)}
                # Heuristic for the output candidate; final scientific choice needs interpretation.
                row["exploratory_score"] = row["mean_silhouette"] + 0.15 * (stability if np.isfinite(stability) else 0)
                results.append(row)
                saved[(set_name, method, k)] = (best[0], best[1], columns)
                if method == "sparse" and best[1] is not None:
                    for name, weight in zip(columns, best[1]):
                        weight_rows.append({"feature_set": set_name, "k": k, "feature": name,
                                            "sparse_weight": float(weight), "silhouette": best[3]})
    metrics = pd.DataFrame(results)
    if metrics.empty:
        raise RuntimeError("Every clustering candidate failed")
    metrics.sort_values("exploratory_score", ascending=False).to_csv(out / "method_scores.csv", index=False, encoding="utf-8-sig")
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(out / "sparse_feature_weights_all.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 5))
    for (fs, meth), part in metrics.groupby(["feature_set", "method"]):
        ax.plot(part.k, part.mean_silhouette, marker=".", label=f"{meth} / {fs}")
    ax.set(xlabel="Number of clusters (K)", ylabel="Mean silhouette", title="OD clustering comparison")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "method_comparison.png", dpi=150)
    plt.close(fig)
    admissible = metrics[metrics.min_cluster_od_share >= 0.01]
    if admissible.empty:
        admissible = metrics
    winner = admissible.sort_values("exploratory_score", ascending=False).iloc[0]
    key = (winner.feature_set, winner.method, int(winner.k))
    labels, weights, cols = saved[key]
    print(f"  Exploratory candidate: {key}; silhouette={winner.silhouette:.3f}, stability={winner.stability_ari:.3f}")
    return metrics, labels, cols, weights, winner.to_dict()


def write_profiles(data: pd.DataFrame, labels: np.ndarray, test: pd.DataFrame, cols: list[str],
                   weights: np.ndarray | None, args: argparse.Namespace, out: Path) -> pd.DataFrame:
    od = data[["GantryO", "GantryD", "n_train"] + cols].copy()
    od["cluster"] = [f"C{int(v) + 1}" for v in labels]
    od = od.merge(test, on=["GantryO", "GantryD"], how="left")
    od["n_test"] = od.n_test.fillna(0).astype(int)
    od.to_csv(out / "od_cluster_labels.csv", index=False, encoding="utf-8-sig")
    summary = od.groupby("cluster").agg(n_od=("GantryO", "size"), train_trips=("n_train", "sum"),
                                         test_trips=("n_test", "sum")).reset_index()
    for name in cols:
        summary[name] = od.groupby("cluster")[name].mean().reindex(summary.cluster).values
    summary.to_csv(out / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    if weights is not None:
        pd.DataFrame({"feature": cols, "sparse_weight": weights}).sort_values("sparse_weight", ascending=False).to_csv(
            out / "selected_feature_weights.csv", index=False, encoding="utf-8-sig" )
    # A heatmap of the actual standardized cluster means, not invented cluster names.
    means = od.groupby("cluster")[cols].mean()
    values = (means - od[cols].mean()) / od[cols].std().replace(0, np.nan)
    values = values.fillna(0).clip(-3, 3)
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.35), max(4, len(means) * 0.65)))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cols)), cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(means)), means.index)
    ax.set_title("OD cluster profiles (standardized mean difference)")
    fig.colorbar(im, ax=ax, label="Standard deviations")
    fig.tight_layout()
    fig.savefig(out / "cluster_profiles.png", dpi=150)
    plt.close(fig)
    return od


def five_min_counts(conn: duckdb.DuckDBPyConnection, files: list[str], od: pd.DataFrame,
                    args: argparse.Namespace, out: Path) -> None:
    conn.register("selected_od", od[["GantryO", "GantryD", "cluster"]])
    kinds = ["bottleneck", "completed"] if args.five_minute == "both" else [args.five_minute]
    for kind in kinds:
        time_col = "BottleneckTime" if kind == "bottleneck" else "DetectionTimeD"
        print(f"  Generating 5-minute {kind} historical counts...", flush=True)
        sql = (f"SELECT time_bucket(INTERVAL '5 minutes', raw.{time_col}) AS Time5, "
               "COALESCE(map.cluster, 'OTHER') AS cluster, COUNT(*) AS trip_count "
               f"FROM {sql_source()} AS raw LEFT JOIN selected_od AS map "
               "ON raw.GantryO = map.GantryO AND raw.GantryD = map.GantryD "
               f"WHERE raw.{time_col} >= TIMESTAMP '{args.train_year}-01-01' AND "
               f"raw.{time_col} < TIMESTAMP '{args.test_year + 1}-01-01' "
               "GROUP BY 1, 2 ORDER BY 1, 2")
        counts = conn.execute(sql, [files]).df()
        wide = counts.pivot_table(index="Time5", columns="cluster", values="trip_count", aggfunc="sum", fill_value=0)
        if "OTHER" not in wide:
            wide["OTHER"] = 0
        wide = wide.add_prefix("trips_")
        wide["trips_total"] = wide.sum(axis=1)
        for col in [c for c in wide if c.startswith("trips_") and c != "trips_total"]:
            wide[col.replace("trips_", "share_", 1)] = wide[col] / wide.trips_total.replace(0, np.nan)
        wide.reset_index().to_csv(out / f"five_min_{kind}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = arguments()
    if args.train_year >= args.test_year:
        raise ValueError("train-year must precede test-year")
    if args.k_min < 2 or args.restarts < 1 or args.min_trips < 1:
        raise ValueError("k-min >= 2, restarts >= 1, min-trips >= 1 are required")
    files = discover_files(args.input)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps({**vars(args), "n_files": len(files)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Discovered {len(files)} Parquet parts; training={args.train_year}, held-out={args.test_year}")
    conn = duckdb.connect()
    conn.execute(f"SET threads = {int(args.threads)}")
    conn.execute("SET memory_limit = ?", [args.memory_limit])
    conn.execute("SET temp_directory = ?", [str(output / "duckdb_tmp")])
    vtypes = [t.strip() for t in args.vehicle_types.split(",") if t.strip()]
    if not vtypes or len(vtypes) != len(set(vtypes)):
        raise ValueError("--vehicle-types must contain unique non-empty codes")
    agg, test, days = prepare_aggregates(conn, files, args, output, vtypes)
    eligible = agg[agg.n_train >= args.min_trips].copy().reset_index(drop=True)
    print(f"Eligible OD pairs: {len(eligible):,}; observed day counts={days}")
    data, blocks = add_features(eligible, days, vtypes, args.smooth, args.gantry_map)
    data.to_csv(output / "od_features_train.csv", index=False, encoding="utf-8-sig")
    metrics, labels, cols, weights, chosen = compare_models(data, blocks, args, output)
    print("[4/5] Writing labels, profiles, and plots...", flush=True)
    selected = write_profiles(data, labels, test, cols, weights, args, output)
    summary = {"candidate": {key: chosen[key] for key in ("method", "feature_set", "k", "silhouette", "stability_ari")},
               "eligible_od": len(eligible), "selected_feature_names": cols,
               "warning": "Candidate is selected by an exploratory heuristic, not by held-out congestion prediction.",
               "future_information_warning": "BottleneckTime uses retrospective trip information; for forecasting, audit when GantryD becomes known."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("[5/5] Optional five-minute aggregation...", flush=True)
    if args.five_minute != "none":
        five_min_counts(conn, files, selected, args, output)
    conn.close()
    print(f"Done. Outputs: {output}")


if __name__ == "__main__":
    main()
