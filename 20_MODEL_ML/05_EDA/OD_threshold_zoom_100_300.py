from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

OD_SUMMARY = Path(
    r"D:\統計實務\Data\OD_ANALYSIS_N3_N_2024_2025_V2"
    r"\01_od_summary_all.parquet"
)

STABILITY_PARQUET = Path(
    r"D:\統計實務\Data\OD_THRESHOLD_STABILITY_N3_N_2024_2025_EXTENDED"
    r"\03_od_profile_stability_2024_2025.parquet"
)

STABILITY_CSV = Path(
    r"D:\統計實務\Data\OD_THRESHOLD_STABILITY_N3_N_2024_2025_EXTENDED"
    r"\03_od_profile_stability_2024_2025.csv"
)

OUT_ROOT = Path(
    r"D:\統計實務\Data\OD_THRESHOLD_ZOOM_100_300"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [100, 150, 200, 250, 300]
K_VALUES = [2, 3]

RANDOM_STATE = 20261021
SILHOUETTE_SAMPLE = 5000

HOUR_COLS = [
    f"HourShare_{h:02d}"
    for h in range(24)
]


def print_title(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def load_inputs():
    if not OD_SUMMARY.exists():
        raise FileNotFoundError(
            f"找不到 OD summary：\n{OD_SUMMARY}"
        )

    od = pd.read_parquet(OD_SUMMARY)

    if STABILITY_PARQUET.exists():
        stability = pd.read_parquet(STABILITY_PARQUET)
    elif STABILITY_CSV.exists():
        stability = pd.read_csv(STABILITY_CSV)
    else:
        raise FileNotFoundError(
            "找不到前一輪的 profile stability 檔案：\n"
            f"{STABILITY_PARQUET}\n"
            f"{STABILITY_CSV}"
        )

    return od, stability


def get_total_trips_for_stability(df):
    candidates = [
        "TotalTrips",
        "TripsTotal",
        "Trips_2024_2025",
        "Trips",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "stability 檔找不到總 Trips 欄位。\n"
        f"目前欄位：{df.columns.tolist()}"
    )


def cosine_col_name(df):
    candidates = [
        "CosineSimilarity",
        "Cosine",
        "ProfileCosine",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "找不到 Cosine similarity 欄位。\n"
        f"目前欄位：{df.columns.tolist()}"
    )


def l1_col_name(df):
    candidates = [
        "L1Distance",
        "L1",
        "ProfileL1Distance",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "找不到 L1 distance 欄位。\n"
        f"目前欄位：{df.columns.tolist()}"
    )


def analyze_stability(stability):
    print_title(
        "STEP 1：100 / 150 / 200 / 250 / 300 profile stability"
    )

    trip_col = get_total_trips_for_stability(stability)
    cos_col = cosine_col_name(stability)
    l1_col = l1_col_name(stability)

    rows = []

    for t in THRESHOLDS:
        x = stability.loc[
            stability[trip_col] >= t
        ].copy()

        cos = x[cos_col].dropna()
        l1 = x[l1_col].dropna()

        rows.append({
            "MinTrips": t,
            "OD_N": len(x),
            "MedianCosine": float(cos.median()),
            "MeanCosine": float(cos.mean()),
            "P10Cosine": float(cos.quantile(0.10)),
            "P25Cosine": float(cos.quantile(0.25)),
            "PropCosine_GE_090": float((cos >= 0.90).mean()),
            "PropCosine_GE_095": float((cos >= 0.95).mean()),
            "MedianL1Distance": float(l1.median()),
            "P90L1Distance": float(l1.quantile(0.90)),
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUT_ROOT / "01_stability_100_300.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(out.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        out["MinTrips"],
        out["MedianCosine"],
        marker="o",
        label="Median cosine"
    )

    ax.plot(
        out["MinTrips"],
        out["P10Cosine"],
        marker="o",
        label="P10 cosine"
    )

    ax.set_xlabel("Minimum trips per OD")
    ax.set_ylabel("2024 vs 2025 cosine similarity")
    ax.set_title("OD profile stability: threshold 100–300")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT_ROOT / "02_stability_100_300.png",
        dpi=180
    )

    plt.close(fig)

    return out


def run_time_clustering(od):
    print_title(
        "STEP 2：Time-profile KMeans 100 / 150 / 200 / 250 / 300"
    )

    all_score_rows = []
    all_profile_rows = []
    assignments = {}

    total_trip_all = od["Trips"].sum()

    for t in THRESHOLDS:
        print()
        print("-" * 100)
        print(f"Threshold >= {t}")
        print("-" * 100)

        df = (
            od.loc[
                od["Trips"] >= t
            ]
            .dropna(subset=HOUR_COLS)
            .copy()
        )

        X = df[HOUR_COLS].to_numpy(dtype=float)

        trip_coverage = (
            df["Trips"].sum()
            /
            total_trip_all
        )

        print(f"OD N = {len(df):,}")
        print(f"Trip coverage = {trip_coverage:.4%}")

        for k in K_VALUES:
            model = KMeans(
                n_clusters=k,
                n_init=50,
                random_state=RANDOM_STATE
            )

            labels = model.fit_predict(X)

            df_k = df.copy()
            df_k["Cluster"] = labels

            assignments[(t, k)] = (
                df_k[
                    ["GantryO", "GantryD", "Cluster"]
                ].copy()
            )

            score_row = {
                "MinTrips": t,
                "K": k,
                "OD_N": len(df_k),
                "Trip_N": int(df_k["Trips"].sum()),
                "TripCoverage": float(trip_coverage),
                "Inertia": float(model.inertia_),
                "Silhouette": float(
                    silhouette_score(
                        X,
                        labels,
                        sample_size=min(
                            SILHOUETTE_SAMPLE,
                            len(X)
                        ),
                        random_state=RANDOM_STATE
                    )
                ),
                "DaviesBouldin": float(
                    davies_bouldin_score(
                        X,
                        labels
                    )
                ),
                "CalinskiHarabasz": float(
                    calinski_harabasz_score(
                        X,
                        labels
                    )
                ),
            }

            all_score_rows.append(score_row)

            print(
                f"K={k}: "
                f"Sil={score_row['Silhouette']:.4f}, "
                f"DB={score_row['DaviesBouldin']:.4f}"
            )

            cluster_rows = []

            for cluster_id, g in df_k.groupby("Cluster"):
                equal_profile = (
                    g[HOUR_COLS]
                    .mean()
                    .to_numpy(dtype=float)
                )

                weighted_profile = np.average(
                    g[HOUR_COLS].to_numpy(dtype=float),
                    axis=0,
                    weights=g["Trips"].to_numpy(dtype=float)
                )

                equal_peak = int(np.argmax(equal_profile))
                weighted_peak = int(np.argmax(weighted_profile))

                trip_n = int(g["Trips"].sum())

                row = {
                    "MinTrips": t,
                    "K": k,
                    "Cluster": cluster_id,
                    "OD_N": len(g),
                    "Trip_N": trip_n,
                    "TripShareWithinThreshold": float(
                        trip_n / df_k["Trips"].sum()
                    ),
                    "EqualProfilePeakHour": equal_peak,
                    "WeightedProfilePeakHour": weighted_peak,
                    "EqualProfileNightShare_00_05": float(
                        equal_profile[0:6].sum()
                    ),
                    "TripLengthMedian_MedianAcrossOD": float(
                        g["TripLengthMedian"].median()
                    )
                    if "TripLengthMedian" in g.columns
                    else np.nan,
                    "WeekendShare_Mean": float(
                        g["WeekendShare"].mean()
                    )
                    if "WeekendShare" in g.columns
                    else np.nan,
                    "LongHolidayShare_Mean": float(
                        g["LongHolidayShare"].mean()
                    )
                    if "LongHolidayShare" in g.columns
                    else np.nan,
                }

                for h in range(24):
                    row[f"Equal_Hour_{h:02d}"] = equal_profile[h]
                    row[f"Weighted_Hour_{h:02d}"] = weighted_profile[h]

                all_profile_rows.append(row)
                cluster_rows.append(row)

            fig, ax = plt.subplots(figsize=(11, 7))

            for row in cluster_rows:
                y = [
                    row[f"Equal_Hour_{h:02d}"]
                    for h in range(24)
                ]

                ax.plot(
                    range(24),
                    y,
                    marker="o",
                    linewidth=1.5,
                    label=(
                        f"Cluster {row['Cluster']} "
                        f"(OD={row['OD_N']})"
                    )
                )

            ax.set_xticks(range(24))
            ax.set_xlabel("Hour")
            ax.set_ylabel("Mean trip share within OD")
            ax.set_title(
                f"Time profile — T>={t}, K={k} — Equal OD weight"
            )
            ax.legend()

            fig.tight_layout()

            fig.savefig(
                OUT_ROOT
                / f"time_profile_T{t}_K{k}_equal.png",
                dpi=180
            )

            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 7))

            for row in cluster_rows:
                y = [
                    row[f"Weighted_Hour_{h:02d}"]
                    for h in range(24)
                ]

                ax.plot(
                    range(24),
                    y,
                    marker="o",
                    linewidth=1.5,
                    label=(
                        f"Cluster {row['Cluster']} "
                        f"(Trips={row['Trip_N']:,})"
                    )
                )

            ax.set_xticks(range(24))
            ax.set_xlabel("Hour")
            ax.set_ylabel("Trip-weighted share")
            ax.set_title(
                f"Time profile — T>={t}, K={k} — Trip weighted"
            )
            ax.legend()

            fig.tight_layout()

            fig.savefig(
                OUT_ROOT
                / f"time_profile_T{t}_K{k}_weighted.png",
                dpi=180
            )

            plt.close(fig)

            pca = PCA(n_components=2)
            Z = pca.fit_transform(X)

            scatter_df = pd.DataFrame({
                "GantryO": df_k["GantryO"].to_numpy(),
                "GantryD": df_k["GantryD"].to_numpy(),
                "Trips": df_k["Trips"].to_numpy(),
                "Cluster": labels,
                "PC1": Z[:, 0],
                "PC2": Z[:, 1],
            })

            scatter_df.to_csv(
                OUT_ROOT
                / f"time_scatter_T{t}_K{k}_points.csv",
                index=False,
                encoding="utf-8-sig"
            )

            fig, ax = plt.subplots(figsize=(9, 7))

            sc = ax.scatter(
                Z[:, 0],
                Z[:, 1],
                c=labels,
                s=12,
                alpha=0.55
            )

            centers_pca = pca.transform(
                model.cluster_centers_
            )

            ax.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                marker="X",
                s=150,
                linewidths=1.5,
                label="KMeans centers"
            )

            ax.set_xlabel(
                "PC1 "
                f"({pca.explained_variance_ratio_[0]:.1%})"
            )

            ax.set_ylabel(
                "PC2 "
                f"({pca.explained_variance_ratio_[1]:.1%})"
            )

            ax.set_title(
                f"OD time-profile PCA scatter — T>={t}, K={k}"
            )

            fig.colorbar(
                sc,
                ax=ax,
                label="Cluster"
            )

            ax.legend()

            fig.tight_layout()

            fig.savefig(
                OUT_ROOT
                / f"time_scatter_T{t}_K{k}_PCA.png",
                dpi=180
            )

            plt.close(fig)

    scores = pd.DataFrame(all_score_rows)

    scores.to_csv(
        OUT_ROOT / "03_time_kmeans_scores_100_300.csv",
        index=False,
        encoding="utf-8-sig"
    )

    profiles = pd.DataFrame(all_profile_rows)

    profiles.to_csv(
        OUT_ROOT / "04_time_cluster_profiles_100_300.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return scores, profiles, assignments


def compare_ari(assignments):
    print_title(
        "STEP 3：Threshold-to-threshold ARI"
    )

    rows = []

    for k in K_VALUES:
        for ta, tb in combinations(THRESHOLDS, 2):
            a = assignments[(ta, k)].rename(
                columns={"Cluster": "ClusterA"}
            )

            b = assignments[(tb, k)].rename(
                columns={"Cluster": "ClusterB"}
            )

            m = a.merge(
                b,
                on=["GantryO", "GantryD"],
                how="inner"
            )

            ari = adjusted_rand_score(
                m["ClusterA"],
                m["ClusterB"]
            )

            rows.append({
                "K": k,
                "ThresholdA": ta,
                "ThresholdB": tb,
                "CommonOD_N": len(m),
                "ARI": float(ari),
            })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUT_ROOT / "05_threshold_ari_100_300.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(out.to_string(index=False))

    return out


def build_transition_table(
    scores,
    ari,
    stability_summary
):
    print_title(
        "STEP 4：100 -> 150 -> 200 -> 250 -> 300 transition"
    )

    rows = []

    for t in THRESHOLDS:
        k2 = scores.loc[
            (scores["MinTrips"] == t)
            &
            (scores["K"] == 2)
        ].iloc[0]

        k3 = scores.loc[
            (scores["MinTrips"] == t)
            &
            (scores["K"] == 3)
        ].iloc[0]

        st = stability_summary.loc[
            stability_summary["MinTrips"] == t
        ].iloc[0]

        rows.append({
            "MinTrips": t,
            "OD_N": int(k2["OD_N"]),
            "TripCoverage": float(k2["TripCoverage"]),
            "MedianCosine": float(st["MedianCosine"]),
            "P10Cosine": float(st["P10Cosine"]),
            "K2_Silhouette": float(k2["Silhouette"]),
            "K3_Silhouette": float(k3["Silhouette"]),
            "K3_minus_K2_Silhouette": float(
                k3["Silhouette"] - k2["Silhouette"]
            ),
            "K2_DaviesBouldin": float(
                k2["DaviesBouldin"]
            ),
            "K3_DaviesBouldin": float(
                k3["DaviesBouldin"]
            ),
        })

    out = pd.DataFrame(rows)

    out["Adjacent_K2_ARI"] = np.nan
    out["Adjacent_K3_ARI"] = np.nan

    for i in range(1, len(out)):
        ta = int(out.loc[i - 1, "MinTrips"])
        tb = int(out.loc[i, "MinTrips"])

        for k, col in [
            (2, "Adjacent_K2_ARI"),
            (3, "Adjacent_K3_ARI"),
        ]:
            x = ari.loc[
                (ari["K"] == k)
                &
                (ari["ThresholdA"] == ta)
                &
                (ari["ThresholdB"] == tb)
            ]

            if len(x) == 1:
                out.loc[i, col] = float(
                    x["ARI"].iloc[0]
                )

    out.to_csv(
        OUT_ROOT / "06_transition_summary_100_300.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(out.to_string(index=False))

    return out


def main():
    print_title(
        "N3 北向 OD threshold zoom：100 / 150 / 200 / 250 / 300"
    )

    print(
        "這支程式不會再重讀 45M trips。"
    )

    print(
        "它直接使用前一輪已建立好的 OD summary "
        "與 2024/2025 stability 結果。"
    )

    print()
    print(f"Output:\n{OUT_ROOT}")

    od, stability = load_inputs()

    stability_summary = analyze_stability(
        stability
    )

    scores, profiles, assignments = run_time_clustering(
        od
    )

    ari = compare_ari(
        assignments
    )

    build_transition_table(
        scores,
        ari,
        stability_summary
    )

    print_title("ALL DONE")

    print("最重要先看：")
    print("06_transition_summary_100_300.csv")

    print()
    print("以及 K=3 的：")

    for t in THRESHOLDS:
        print(f"time_profile_T{t}_K3_equal.png")
        print(f"time_scatter_T{t}_K3_PCA.png")


if __name__ == "__main__":
    main()
