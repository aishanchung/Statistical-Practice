from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

# ============================================================
# 0. 基本設定
# ============================================================

OD_SUMMARY = Path(
    r"D:\統計實務\Data\OD_ANALYSIS_N3_N_2024_2025_V2"
    r"\01_od_summary_all.parquet"
)

OUT_ROOT = Path(
    r"D:\統計實務\Data\OD_K_SELECTION_N3_N"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 固定同一批 OD，先只研究「24 小時通過瓶頸 profile」的 K。
#
# threshold 先固定 100，
# 避免把 threshold effect 和 K effect 混在一起。
# ------------------------------------------------------------

MIN_TRIPS = 100

# K = 2 ~ 10
K_VALUES = list(range(2, 11))

# ------------------------------------------------------------
# Stability：
# 每次抽 80% OD，重新分群，
# 再用 ARI 比較完整資料分群 vs 抽樣分群。
#
# 10 次通常已足夠先看趨勢。
# 想做正式 sensitivity 可以改 20 或 30。
# ------------------------------------------------------------

STABILITY_SAMPLE_FRAC = 0.80
STABILITY_REPEATS = 10

RANDOM_STATE = 20261021

# silhouette 最多抽 5000 OD
SILHOUETTE_SAMPLE = 5000

# ------------------------------------------------------------
# 第一輪只跑相對快的方法。
#
# KMeans             ：Euclidean baseline
# CosineKMeans       ：更強調 profile shape
# Agglomerative Ward ：hierarchical structure
# GMM                ：probabilistic clustering
#
# KMedoids / KShape / DTW 等 K 範圍縮小後再跑。
# ------------------------------------------------------------

METHODS = [
    "KMeans",
    "CosineKMeans",
    "AgglomerativeWard",
    "GMM",
]

HOUR_COLS = [
    f"HourShare_{h:02d}"
    for h in range(24)
]


# ============================================================
# 1. Helper
# ============================================================

def print_title(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def load_data():

    if not OD_SUMMARY.exists():
        raise FileNotFoundError(
            f"找不到 OD summary：\n{OD_SUMMARY}"
        )

    df = pd.read_parquet(
        OD_SUMMARY
    )

    required = [
        "GantryO",
        "GantryD",
        "Trips",
    ] + HOUR_COLS

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"OD summary 缺少必要欄位：{missing}"
        )

    df = (
        df.loc[
            df["Trips"] >= MIN_TRIPS
        ]
        .dropna(
            subset=HOUR_COLS
        )
        .copy()
        .reset_index(drop=True)
    )

    X = (
        df[HOUR_COLS]
        .to_numpy(
            dtype=float
        )
    )

    # 每組 OD 的 24 小時 share 理論上約加總為 1
    share_sum = X.sum(axis=1)

    print(
        f"MIN_TRIPS = {MIN_TRIPS}"
    )

    print(
        f"OD N = {len(df):,}"
    )

    print(
        f"Trips = {df['Trips'].sum():,}"
    )

    print(
        "24h share sum:"
        f" min={share_sum.min():.6f},"
        f" median={np.median(share_sum):.6f},"
        f" max={share_sum.max():.6f}"
    )

    return df, X


def get_method_space(method, X):

    if method == "CosineKMeans":
        return normalize(
            X,
            norm="l2"
        )

    return X


def fit_predict_method(
    method,
    X,
    k,
    seed
):

    if method == "KMeans":

        model = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=seed
        )

        labels = model.fit_predict(
            X
        )

        extra = {
            "NativeObjective":
                float(model.inertia_),
            "BIC":
                np.nan,
            "AIC":
                np.nan,
        }

        return labels, model, extra


    if method == "CosineKMeans":

        # X 傳進來時已經 L2 normalize
        model = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=seed
        )

        labels = model.fit_predict(
            X
        )

        extra = {
            "NativeObjective":
                float(model.inertia_),
            "BIC":
                np.nan,
            "AIC":
                np.nan,
        }

        return labels, model, extra


    if method == "AgglomerativeWard":

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )

        labels = model.fit_predict(
            X
        )

        extra = {
            "NativeObjective":
                np.nan,
            "BIC":
                np.nan,
            "AIC":
                np.nan,
        }

        return labels, model, extra


    if method == "GMM":

        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=10,
            random_state=seed
        )

        model.fit(
            X
        )

        labels = model.predict(
            X
        )

        extra = {
            "NativeObjective":
                float(
                    model.lower_bound_
                ),
            "BIC":
                float(
                    model.bic(X)
                ),
            "AIC":
                float(
                    model.aic(X)
                ),
        }

        return labels, model, extra


    raise ValueError(
        f"未知 method：{method}"
    )


def within_cluster_sse(
    X,
    labels
):

    labels = np.asarray(
        labels
    )

    total = 0.0

    for c in np.unique(labels):

        idx = (
            labels == c
        )

        Xc = X[idx]

        center = Xc.mean(
            axis=0
        )

        total += (
            (
                Xc
                -
                center
            )
            ** 2
        ).sum()

    return float(
        total
    )


def cluster_size_stats(
    df,
    labels
):

    tmp = pd.DataFrame({
        "Cluster": labels,
        "Trips": df["Trips"].to_numpy(),
    })

    od_n = (
        tmp.groupby(
            "Cluster"
        )
        .size()
    )

    trip_n = (
        tmp.groupby(
            "Cluster"
        )["Trips"]
        .sum()
    )

    return {
        "ObservedClusters":
            int(len(od_n)),

        "SmallestClusterOD_N":
            int(od_n.min()),

        "SmallestClusterODShare":
            float(
                od_n.min()
                /
                len(tmp)
            ),

        "LargestClusterODShare":
            float(
                od_n.max()
                /
                len(tmp)
            ),

        "SmallestClusterTripShare":
            float(
                trip_n.min()
                /
                trip_n.sum()
            ),

        "LargestClusterTripShare":
            float(
                trip_n.max()
                /
                trip_n.sum()
            ),
    }


def calculate_metrics(
    X_metric,
    labels,
    silhouette_metric
):

    labels = np.asarray(
        labels
    )

    if len(
        np.unique(labels)
    ) < 2:

        return {
            "Silhouette": np.nan,
            "DaviesBouldin": np.nan,
            "CalinskiHarabasz": np.nan,
        }

    sil = silhouette_score(
        X_metric,
        labels,
        metric=silhouette_metric,
        sample_size=min(
            SILHOUETTE_SAMPLE,
            len(X_metric)
        ),
        random_state=RANDOM_STATE
    )

    # DB / CH 本身是基於 Euclidean geometry。
    db = davies_bouldin_score(
        X_metric,
        labels
    )

    ch = calinski_harabasz_score(
        X_metric,
        labels
    )

    return {
        "Silhouette":
            float(sil),

        "DaviesBouldin":
            float(db),

        "CalinskiHarabasz":
            float(ch),
    }


# ============================================================
# 2. Elbow detector
# ============================================================

def detect_elbow(
    ks,
    ys
):
    """
    用「點到首尾連線的最大距離」找 elbow。

    僅作為參考，不可單獨決定 K。
    """

    ks = np.asarray(
        ks,
        dtype=float
    )

    ys = np.asarray(
        ys,
        dtype=float
    )

    valid = (
        np.isfinite(ks)
        &
        np.isfinite(ys)
    )

    ks = ks[valid]
    ys = ys[valid]

    if len(ks) < 3:
        return np.nan

    x_range = (
        ks.max()
        -
        ks.min()
    )

    y_range = (
        ys.max()
        -
        ys.min()
    )

    if x_range == 0 or y_range == 0:
        return np.nan

    x = (
        ks
        -
        ks.min()
    ) / x_range

    y = (
        ys
        -
        ys.min()
    ) / y_range

    # 首點 A、末點 B
    A = np.array([
        x[0],
        y[0]
    ])

    B = np.array([
        x[-1],
        y[-1]
    ])

    AB = (
        B
        -
        A
    )

    denom = np.linalg.norm(
        AB
    )

    if denom == 0:
        return np.nan

    distances = []

    for xi, yi in zip(
        x,
        y
    ):

        P = np.array([
            xi,
            yi
        ])

        # 2D cross product magnitude / |AB|
        cross = abs(
            AB[0]
            *
            (A[1] - P[1])
            -
            (A[0] - P[0])
            *
            AB[1]
        )

        distances.append(
            cross
            /
            denom
        )

    idx = int(
        np.argmax(
            distances
        )
    )

    return int(
        ks[idx]
    )


# ============================================================
# 3. Full-data K scan
# ============================================================

def scan_all_k(
    df,
    X
):

    print_title(
        "STEP 1：K = 2 ~ 10"
    )

    metric_rows = []
    assignments_long = []
    profile_rows = []

    full_labels = {}

    total_trip = (
        df["Trips"]
        .sum()
    )

    for method in METHODS:

        print()
        print(
            "-" * 100
        )

        print(
            method
        )

        print(
            "-" * 100
        )

        X_method = get_method_space(
            method,
            X
        )

        silhouette_metric = (
            "cosine"
            if method == "CosineKMeans"
            else "euclidean"
        )

        for k in K_VALUES:

            print(
                f"{method}: K={k}"
            )

            t0 = (
                time.perf_counter()
            )

            labels, model, extra = (
                fit_predict_method(
                    method,
                    X_method,
                    k,
                    RANDOM_STATE
                )
            )

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            full_labels[
                (method, k)
            ] = labels

            metric = (
                calculate_metrics(
                    X_method,
                    labels,
                    silhouette_metric
                )
            )

            size_stat = (
                cluster_size_stats(
                    df,
                    labels
                )
            )

            # 為所有方法都計算一個
            # centroid-based WCSS，
            # 讓 elbow 圖有共同的描述性指標。
            wcss = within_cluster_sse(
                X_method,
                labels
            )

            metric_rows.append({
                "Method":
                    method,

                "K":
                    k,

                "OD_N":
                    len(df),

                "Trip_N":
                    int(total_trip),

                "RuntimeSec":
                    float(elapsed),

                "WithinClusterSSE":
                    wcss,

                **metric,

                **size_stat,

                **extra,
            })

            # assignments
            tmp_assign = pd.DataFrame({
                "GantryO":
                    df["GantryO"],

                "GantryD":
                    df["GantryD"],

                "Trips":
                    df["Trips"],

                "Method":
                    method,

                "K":
                    k,

                "Cluster":
                    labels,
            })

            assignments_long.append(
                tmp_assign
            )

            # cluster profiles
            temp = df.copy()
            temp["Cluster"] = labels

            for cluster_id, g in temp.groupby(
                "Cluster"
            ):

                equal_profile = (
                    g[HOUR_COLS]
                    .mean()
                    .to_numpy(
                        dtype=float
                    )
                )

                weighted_profile = np.average(
                    g[HOUR_COLS].to_numpy(
                        dtype=float
                    ),
                    axis=0,
                    weights=g["Trips"].to_numpy(
                        dtype=float
                    )
                )

                row = {
                    "Method":
                        method,

                    "K":
                        k,

                    "Cluster":
                        cluster_id,

                    "OD_N":
                        len(g),

                    "Trip_N":
                        int(
                            g["Trips"].sum()
                        ),

                    "TripShare":
                        float(
                            g["Trips"].sum()
                            /
                            total_trip
                        ),

                    "EqualPeakHour":
                        int(
                            np.argmax(
                                equal_profile
                            )
                        ),

                    "WeightedPeakHour":
                        int(
                            np.argmax(
                                weighted_profile
                            )
                        ),

                    "EqualNightShare_00_05":
                        float(
                            equal_profile[
                                0:6
                            ].sum()
                        ),
                }

                for h in range(24):

                    row[
                        f"Equal_Hour_{h:02d}"
                    ] = (
                        equal_profile[h]
                    )

                    row[
                        f"Weighted_Hour_{h:02d}"
                    ] = (
                        weighted_profile[h]
                    )

                profile_rows.append(
                    row
                )

    metrics = pd.DataFrame(
        metric_rows
    )

    assignments = pd.concat(
        assignments_long,
        ignore_index=True
    )

    profiles = pd.DataFrame(
        profile_rows
    )

    metrics.to_csv(
        OUT_ROOT
        / "01_k_scan_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    assignments.to_parquet(
        OUT_ROOT
        / "02_assignments_all_k.parquet",
        index=False
    )

    profiles.to_csv(
        OUT_ROOT
        / "03_cluster_profiles_all_k.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return (
        metrics,
        assignments,
        profiles,
        full_labels
    )


# ============================================================
# 4. Resampling stability
# ============================================================

def run_stability(
    df,
    X,
    full_labels
):

    print_title(
        "STEP 2：Resampling stability"
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n = len(df)

    sample_n = int(
        round(
            n
            *
            STABILITY_SAMPLE_FRAC
        )
    )

    rows = []

    for method in METHODS:

        print()
        print(
            "-" * 100
        )

        print(
            f"Stability: {method}"
        )

        print(
            "-" * 100
        )

        X_method = get_method_space(
            method,
            X
        )

        for k in K_VALUES:

            reference = (
                full_labels[
                    (method, k)
                ]
            )

            print(
                f"{method}: K={k}"
            )

            for r in range(
                STABILITY_REPEATS
            ):

                idx = rng.choice(
                    n,
                    size=sample_n,
                    replace=False
                )

                X_sub = (
                    X_method[
                        idx
                    ]
                )

                labels_sub, model, extra = (
                    fit_predict_method(
                        method,
                        X_sub,
                        k,
                        RANDOM_STATE
                        +
                        1000 * k
                        +
                        r
                    )
                )

                ari = adjusted_rand_score(
                    reference[
                        idx
                    ],
                    labels_sub
                )

                rows.append({
                    "Method":
                        method,

                    "K":
                        k,

                    "Repeat":
                        r + 1,

                    "SampleFrac":
                        STABILITY_SAMPLE_FRAC,

                    "SampleOD_N":
                        sample_n,

                    "ARI":
                        float(ari),
                })

    detail = pd.DataFrame(
        rows
    )

    summary = (
        detail
        .groupby(
            [
                "Method",
                "K"
            ]
        )["ARI"]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
        .rename(
            columns={
                "mean":
                    "StabilityARI_Mean",

                "median":
                    "StabilityARI_Median",

                "std":
                    "StabilityARI_SD",

                "min":
                    "StabilityARI_Min",

                "max":
                    "StabilityARI_Max",
            }
        )
    )

    # P10 / P25
    quant = (
        detail
        .groupby(
            [
                "Method",
                "K"
            ]
        )["ARI"]
        .quantile(
            [
                0.10,
                0.25
            ]
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.10:
                    "StabilityARI_P10",

                0.25:
                    "StabilityARI_P25",
            }
        )
    )

    summary = summary.merge(
        quant,
        on=[
            "Method",
            "K"
        ],
        how="left"
    )

    detail.to_csv(
        OUT_ROOT
        / "04_stability_resample_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUT_ROOT
        / "05_stability_by_k.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return (
        detail,
        summary
    )


# ============================================================
# 5. K recommendation table
# ============================================================

def build_k_recommendation(
    metrics,
    stability
):

    print_title(
        "STEP 3：K recommendation summary"
    )

    merged = metrics.merge(
        stability,
        on=[
            "Method",
            "K"
        ],
        how="left"
    )

    merged.to_csv(
        OUT_ROOT
        / "06_k_selection_full_table.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rows = []

    for method in METHODS:

        x = (
            merged.loc[
                merged[
                    "Method"
                ]
                ==
                method
            ]
            .sort_values(
                "K"
            )
            .copy()
        )

        elbow_k = detect_elbow(
            x["K"],
            x["WithinClusterSSE"]
        )

        silhouette_k = int(
            x.loc[
                x[
                    "Silhouette"
                ].idxmax(),
                "K"
            ]
        )

        db_k = int(
            x.loc[
                x[
                    "DaviesBouldin"
                ].idxmin(),
                "K"
            ]
        )

        ch_k = int(
            x.loc[
                x[
                    "CalinskiHarabasz"
                ].idxmax(),
                "K"
            ]
        )

        stability_k = int(
            x.loc[
                x[
                    "StabilityARI_Median"
                ].idxmax(),
                "K"
            ]
        )

        row = {
            "Method":
                method,

            "ElbowK":
                elbow_k,

            "BestSilhouetteK":
                silhouette_k,

            "BestDaviesBouldinK":
                db_k,

            "BestCalinskiHarabaszK":
                ch_k,

            "BestStabilityK":
                stability_k,

            "BestBICK":
                np.nan,

            "BestAICK":
                np.nan,
        }

        if method == "GMM":

            valid_bic = (
                x["BIC"]
                .dropna()
            )

            valid_aic = (
                x["AIC"]
                .dropna()
            )

            if len(
                valid_bic
            ) > 0:

                row[
                    "BestBICK"
                ] = int(
                    x.loc[
                        x[
                            "BIC"
                        ].idxmin(),
                        "K"
                    ]
                )

            if len(
                valid_aic
            ) > 0:

                row[
                    "BestAICK"
                ] = int(
                    x.loc[
                        x[
                            "AIC"
                        ].idxmin(),
                        "K"
                    ]
                )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUT_ROOT
        / "07_k_recommendation_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        out.to_string(
            index=False
        )
    )

    return (
        merged,
        out
    )


# ============================================================
# 6. Plots
# ============================================================

def plot_metric_by_method(
    df,
    y_col,
    title,
    ylabel,
    filename
):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for method in METHODS:

        x = (
            df.loc[
                df["Method"]
                ==
                method
            ]
            .sort_values(
                "K"
            )
        )

        ax.plot(
            x["K"],
            x[y_col],
            marker="o",
            linewidth=1.5,
            label=method
        )

    ax.set_xticks(
        K_VALUES
    )

    ax.set_xlabel(
        "Number of clusters K"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / filename,
        dpi=180
    )

    plt.close(fig)


def make_all_plots(
    merged
):

    print_title(
        "STEP 4：Plot K-selection dashboard"
    )

    # 1. Elbow
    plot_metric_by_method(
        merged,
        "WithinClusterSSE",
        "Elbow / within-cluster SSE",
        "Within-cluster SSE",
        "01_elbow_within_cluster_sse.png"
    )

    # 2. Silhouette
    plot_metric_by_method(
        merged,
        "Silhouette",
        "Silhouette score by K",
        "Silhouette (higher is better)",
        "02_silhouette_by_k.png"
    )

    # 3. DB
    plot_metric_by_method(
        merged,
        "DaviesBouldin",
        "Davies-Bouldin index by K",
        "Davies-Bouldin (lower is better)",
        "03_davies_bouldin_by_k.png"
    )

    # 4. CH
    plot_metric_by_method(
        merged,
        "CalinskiHarabasz",
        "Calinski-Harabasz index by K",
        "Calinski-Harabasz (higher is better)",
        "04_calinski_harabasz_by_k.png"
    )

    # 5. Stability median
    plot_metric_by_method(
        merged,
        "StabilityARI_Median",
        (
            "Resampling cluster stability "
            f"({int(STABILITY_SAMPLE_FRAC * 100)}% OD samples)"
        ),
        "Median ARI (higher is more stable)",
        "05_stability_ari_by_k.png"
    )

    # 6. Smallest cluster OD share
    plot_metric_by_method(
        merged,
        "SmallestClusterODShare",
        "Smallest cluster OD share by K",
        "Smallest cluster OD share",
        "06_smallest_cluster_od_share.png"
    )

    # 7. Smallest cluster trip share
    plot_metric_by_method(
        merged,
        "SmallestClusterTripShare",
        "Smallest cluster trip share by K",
        "Smallest cluster trip share",
        "07_smallest_cluster_trip_share.png"
    )

    # 8. Runtime
    plot_metric_by_method(
        merged,
        "RuntimeSec",
        "Runtime by K",
        "Seconds",
        "08_runtime_by_k.png"
    )

    # 9. GMM BIC / AIC
    gmm = (
        merged.loc[
            merged["Method"]
            ==
            "GMM"
        ]
        .sort_values(
            "K"
        )
    )

    if len(
        gmm
    ) > 0:

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.plot(
            gmm["K"],
            gmm["BIC"],
            marker="o",
            label="BIC"
        )

        ax.plot(
            gmm["K"],
            gmm["AIC"],
            marker="o",
            label="AIC"
        )

        ax.set_xticks(
            K_VALUES
        )

        ax.set_xlabel(
            "Number of components K"
        )

        ax.set_ylabel(
            "Information criterion"
        )

        ax.set_title(
            "GMM BIC / AIC"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            PLOT_DIR
            / "09_gmm_bic_aic.png",
            dpi=180
        )

        plt.close(fig)


# ============================================================
# 7. Cluster-size detail
# ============================================================

def build_cluster_size_detail(
    assignments
):

    detail = (
        assignments
        .groupby(
            [
                "Method",
                "K",
                "Cluster"
            ]
        )
        .agg(
            OD_N=(
                "Trips",
                "size"
            ),
            Trip_N=(
                "Trips",
                "sum"
            ),
        )
        .reset_index()
    )

    totals = (
        detail
        .groupby(
            [
                "Method",
                "K"
            ]
        )
        .agg(
            TotalOD_N=(
                "OD_N",
                "sum"
            ),
            TotalTrip_N=(
                "Trip_N",
                "sum"
            ),
        )
        .reset_index()
    )

    detail = detail.merge(
        totals,
        on=[
            "Method",
            "K"
        ],
        how="left"
    )

    detail[
        "ODShare"
    ] = (
        detail[
            "OD_N"
        ]
        /
        detail[
            "TotalOD_N"
        ]
    )

    detail[
        "TripShare"
    ] = (
        detail[
            "Trip_N"
        ]
        /
        detail[
            "TotalTrip_N"
        ]
    )

    detail.to_csv(
        OUT_ROOT
        / "08_cluster_size_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return detail


# ============================================================
# 8. Main
# ============================================================

def main():

    print_title(
        "N3 北向 OD：K-selection dashboard"
    )

    print(
        "固定 threshold，掃描 K=2~10。"
    )

    print(
        "先用快速方法縮小合理 K 範圍，"
        "再把 KMedoids / KShape / DTW 加回來。"
    )

    print()
    print(
        f"OD summary:\n{OD_SUMMARY}"
    )

    print()
    print(
        f"Output:\n{OUT_ROOT}"
    )

    # --------------------------------------------------------
    # data
    # --------------------------------------------------------

    df, X = load_data()

    # --------------------------------------------------------
    # settings
    # --------------------------------------------------------

    settings = pd.DataFrame([{
        "MIN_TRIPS":
            MIN_TRIPS,

        "K_MIN":
            min(
                K_VALUES
            ),

        "K_MAX":
            max(
                K_VALUES
            ),

        "STABILITY_SAMPLE_FRAC":
            STABILITY_SAMPLE_FRAC,

        "STABILITY_REPEATS":
            STABILITY_REPEATS,

        "OD_N":
            len(df),

        "Trip_N":
            int(
                df[
                    "Trips"
                ].sum()
            ),

        "Methods":
            ",".join(
                METHODS
            ),
    }])

    settings.to_csv(
        OUT_ROOT
        / "00_run_settings.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # K scan
    # --------------------------------------------------------

    (
        metrics,
        assignments,
        profiles,
        full_labels,
    ) = scan_all_k(
        df,
        X
    )

    # --------------------------------------------------------
    # stability
    # --------------------------------------------------------

    (
        stability_detail,
        stability_summary,
    ) = run_stability(
        df,
        X,
        full_labels
    )

    # --------------------------------------------------------
    # recommendation
    # --------------------------------------------------------

    merged, recommendation = (
        build_k_recommendation(
            metrics,
            stability_summary
        )
    )

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------

    make_all_plots(
        merged
    )

    # --------------------------------------------------------
    # cluster sizes
    # --------------------------------------------------------

    build_cluster_size_detail(
        assignments
    )

    print_title(
        "ALL DONE"
    )

    print(
        "最重要先看："
    )

    print(
        "07_k_recommendation_summary.csv"
    )

    print(
        "06_k_selection_full_table.csv"
    )

    print(
        "08_cluster_size_detail.csv"
    )

    print()
    print(
        "plots 裡優先看："
    )

    print(
        "01_elbow_within_cluster_sse.png"
    )

    print(
        "02_silhouette_by_k.png"
    )

    print(
        "03_davies_bouldin_by_k.png"
    )

    print(
        "05_stability_ari_by_k.png"
    )

    print(
        "09_gmm_bic_aic.png"
    )


if __name__ == "__main__":
    main()
