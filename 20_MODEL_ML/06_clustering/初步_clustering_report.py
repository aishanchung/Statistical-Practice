from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse
from matplotlib import font_manager

from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import multivariate_normal


OUT_DIR = Path("animations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261022
FPS = 1.2


# ============================================================
# Font
# ============================================================

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    r"C:\Windows\Fonts\msjh.ttc",
]

for p in FONT_CANDIDATES:
    if Path(p).exists():
        try:
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
        except Exception:
            pass

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Helpers
# ============================================================

def decorate(ax, title, subtitle, step=None):
    ax.set_title(title, fontsize=18, fontweight="bold", pad=34)
    ax.text(
        0.5, 1.008, subtitle,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=11
    )

    if step:
        ax.text(
            0.02, 0.96, step,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.12)
        )

    ax.grid(alpha=0.13, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def save(anim, fig, name, fps=FPS):
    out = OUT_DIR / name
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


# ============================================================
# 1. K-means
# ============================================================

def make_kmeans_data():
    X, _ = make_blobs(
        n_samples=33,
        centers=[(-3.1, -1.4), (0.2, 2.4), (3.0, -0.8)],
        cluster_std=[0.72, 0.72, 0.68],
        random_state=SEED,
    )
    return X


def kmeans_frames(X, k=3, max_iter=8):
    # 故意用較差初始中心，讓點換群的過程看得見
    init_idx = np.array([1, 8, 15])
    centers = X[init_idx].copy()

    frames = [{
        "kind": "init",
        "centers": centers.copy(),
        "labels": None,
        "frame_no": 1
    }]

    for it in range(1, max_iter + 1):
        D = cdist(X, centers)
        labels = D.argmin(axis=1)

        frames.append({
            "kind": "assign",
            "centers": centers.copy(),
            "labels": labels.copy(),
            "iteration": it
        })

        new_centers = centers.copy()

        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                new_centers[c] = members.mean(axis=0)

        frames.append({
            "kind": "move",
            "old_centers": centers.copy(),
            "centers": new_centers.copy(),
            "labels": labels.copy(),
            "iteration": it
        })

        if np.allclose(new_centers, centers, atol=1e-4):
            frames.append({
                "kind": "done",
                "centers": new_centers.copy(),
                "labels": labels.copy(),
                "iteration": it
            })
            break

        centers = new_centers

    for i, f in enumerate(frames, start=1):
        f["frame_no"] = i

    return frames


def animate_kmeans():
    X = make_kmeans_data()
    frames = kmeans_frames(X)

    fig, ax = plt.subplots(figsize=(9, 6.4))

    def draw(frame):
        ax.clear()

        if frame["kind"] == "init":
            ax.scatter(X[:, 0], X[:, 1], s=68, alpha=0.65)
            ax.scatter(
                frame["centers"][:, 0],
                frame["centers"][:, 1],
                marker="X", s=300, linewidths=2
            )
            subtitle = "先放入 K 個暫時中心"

        else:
            labels = frame["labels"]
            centers = frame["centers"]

            ax.scatter(
                X[:, 0], X[:, 1],
                c=labels, s=72, alpha=0.84
            )

            if frame["kind"] == "assign":
                for i, p in enumerate(X):
                    c = centers[labels[i]]
                    ax.plot(
                        [p[0], c[0]],
                        [p[1], c[1]],
                        linewidth=0.8, alpha=0.18
                    )

                ax.scatter(
                    centers[:, 0], centers[:, 1],
                    marker="X", s=300, linewidths=2
                )
                subtitle = "分配：每個點先歸到目前最近的中心"

            elif frame["kind"] == "move":
                old = frame["old_centers"]

                ax.scatter(
                    old[:, 0], old[:, 1],
                    marker="x", s=160, linewidths=2, alpha=0.5
                )
                ax.scatter(
                    centers[:, 0], centers[:, 1],
                    marker="X", s=300, linewidths=2
                )

                for a, b in zip(old, centers):
                    ax.annotate(
                        "",
                        xy=b, xytext=a,
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=2.3,
                            shrinkA=6,
                            shrinkB=8
                        )
                    )

                subtitle = "更新：重新算群平均，中心移動"

            else:
                ax.scatter(
                    centers[:, 0], centers[:, 1],
                    marker="X", s=300, linewidths=2
                )
                subtitle = "收斂：中心與群組不再改變"

        decorate(
            ax,
            "K-means",
            subtitle,
            f"STEP {frame['frame_no']} / {len(frames)}"
        )
        ax.set_xlim(-5.2, 5.0)
        ax.set_ylim(-4.0, 4.5)

    anim = FuncAnimation(
        fig, draw, frames=frames,
        interval=1100, repeat=True
    )

    return save(anim, fig, "01_kmeans.gif")


# ============================================================
# 2. Hierarchical + dendrogram + cut into 3 clusters
# ============================================================

def hierarchical_data():
    X, _ = make_blobs(
        n_samples=15,
        centers=[(-3.0, -1.5), (0.2, 2.4), (3.1, -0.8)],
        cluster_std=[0.58, 0.62, 0.56],
        random_state=SEED + 2,
    )
    return X


def build_hierarchical_frames(X):
    # 動畫用 single linkage，因為能直觀顯示「最近兩群」
    Z = linkage(X, method="single")

    clusters = {i: [i] for i in range(len(X))}
    edges = []
    frames = []

    frames.append({
        "kind": "merge",
        "clusters": [v.copy() for v in clusters.values()],
        "edges": [],
        "new_edge": None,
        "merge_count": 0
    })

    for step, row in enumerate(Z):
        a = int(row[0])
        b = int(row[1])

        members_a = clusters[a]
        members_b = clusters[b]

        D = cdist(X[members_a], X[members_b])
        pos = np.unravel_index(np.argmin(D), D.shape)

        pa = members_a[pos[0]]
        pb = members_b[pos[1]]

        edges.append((pa, pb))

        new_id = len(X) + step
        clusters[new_id] = members_a + members_b

        del clusters[a]
        del clusters[b]

        frames.append({
            "kind": "merge",
            "clusters": [v.copy() for v in clusters.values()],
            "edges": edges.copy(),
            "new_edge": (pa, pb),
            "merge_count": step + 1
        })

        # 在剩 3 群時停一格
        if len(clusters) == 3:
            frames.append({
                "kind": "cut3",
                "clusters": [v.copy() for v in clusters.values()],
                "edges": edges.copy(),
                "new_edge": None,
                "merge_count": step + 1
            })

    for i, f in enumerate(frames, start=1):
        f["frame_no"] = i

    return Z, frames


def animate_hierarchical():
    X = hierarchical_data()
    Z, frames = build_hierarchical_frames(X)

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[0, 1])

    def draw(frame):
        ax_scatter.clear()
        ax_tree.clear()

        labels = np.zeros(len(X), dtype=int)
        for c_id, members in enumerate(frame["clusters"]):
            labels[members] = c_id

        ax_scatter.scatter(
            X[:, 0], X[:, 1],
            c=labels, s=88, alpha=0.88
        )

        for e in frame["edges"]:
            a = X[e[0]]
            b = X[e[1]]
            lw = 3.0 if e == frame["new_edge"] else 1.5
            alpha = 0.9 if e == frame["new_edge"] else 0.28
            ax_scatter.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                linewidth=lw,
                alpha=alpha
            )

        if frame["kind"] == "cut3":
            subtitle = "在這個層級切開 → 得到 3 群"
        else:
            subtitle = "逐步合併目前最接近的兩群；舊連線持續保留"

        decorate(
            ax_scatter,
            "Hierarchical clustering",
            subtitle,
            f"STEP {frame['frame_no']} / {len(frames)}"
        )

        ax_scatter.text(
            0.98, 0.04,
            f"目前群數：{len(frame['clusters'])}",
            transform=ax_scatter.transAxes,
            ha="right", va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", alpha=0.12)
        )

        ax_scatter.set_xlim(-5.2, 5.0)
        ax_scatter.set_ylim(-4.0, 4.5)

        # dendrogram：以 merge_count 決定畫到哪個高度
        dendrogram(
            Z,
            ax=ax_tree,
            no_labels=True,
            color_threshold=None,
        )

        ax_tree.set_title("Dendrogram", fontsize=15, fontweight="bold")
        ax_tree.set_xlabel("Observations")
        ax_tree.set_ylabel("Merge distance")
        ax_tree.grid(alpha=0.10)

        if frame["kind"] == "cut3":
            # 3 群時的切割高度：介於造成 4->3 與 3->2 的距離之間
            n = len(X)
            # 3 群需要完成 n-3 次 merge
            idx_done = n - 3 - 1
            if 0 <= idx_done < len(Z)-1:
                h1 = Z[idx_done, 2]
                h2 = Z[idx_done + 1, 2]
                cut_h = (h1 + h2) / 2
                ax_tree.axhline(cut_h, linestyle="--", linewidth=2)

                ax_tree.text(
                    0.98, 0.94,
                    "在此高度切成 3 群",
                    transform=ax_tree.transAxes,
                    ha="right", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.35", alpha=0.12)
                )

    anim = FuncAnimation(
        fig, draw, frames=frames,
        interval=1000, repeat=True
    )

    return save(anim, fig, "02_hierarchical.gif", fps=1.0)


# ============================================================
# 3. DBSCAN
# ============================================================

def dbscan_data():
    X, _ = make_moons(
        n_samples=58,
        noise=0.055,
        random_state=SEED + 4
    )

    # 額外放幾個明顯孤立點
    extra = np.array([
        [-1.15, 1.15],
        [2.10, 1.05],
        [1.65, -0.62],
        [-0.90, -0.48],
    ])

    return np.vstack([X, extra])


def dbscan_frames(X, eps=0.22, min_samples=4):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    core_idx = set(model.core_sample_indices_.tolist())

    # 選一個 core point 作教學展示
    example_core = sorted(core_idx)[len(core_idx)//3]

    frames = [
        {"kind": "raw"},
        {"kind": "eps", "idx": example_core},
        {"kind": "core"},
        {"kind": "expand", "cluster_id": 0},
        {"kind": "expand", "cluster_id": 1},
        {"kind": "final"},
    ]

    for i, f in enumerate(frames, start=1):
        f["frame_no"] = i

    return labels, core_idx, frames


def animate_dbscan():
    X = dbscan_data()
    eps = 0.22
    min_samples = 4

    labels, core_idx, frames = dbscan_frames(
        X,
        eps=eps,
        min_samples=min_samples
    )

    fig, ax = plt.subplots(figsize=(9, 6.4))

    def draw(frame):
        ax.clear()
        kind = frame["kind"]

        if kind == "raw":
            ax.scatter(X[:, 0], X[:, 1], s=72, alpha=0.70)
            subtitle = "先看資料分布，不需要事先指定 K"

        elif kind == "eps":
            ax.scatter(X[:, 0], X[:, 1], s=72, alpha=0.68)

            idx = frame["idx"]
            p = X[idx]

            circ = Circle(
                p, eps,
                fill=False,
                linewidth=2.2
            )
            ax.add_patch(circ)

            ax.scatter(
                [p[0]], [p[1]],
                s=210, facecolors="none",
                linewidths=2.3
            )

            subtitle = f"ε-neighborhood：半徑 ε={eps} 內有足夠鄰居嗎？"

        elif kind == "core":
            core_mask = np.array(
                [i in core_idx for i in range(len(X))]
            )
            ax.scatter(
                X[~core_mask, 0], X[~core_mask, 1],
                s=62, alpha=0.35
            )
            ax.scatter(
                X[core_mask, 0], X[core_mask, 1],
                s=90, alpha=0.88
            )

            subtitle = f"Core points：鄰域內至少有 {min_samples} 個點"

        elif kind == "expand":
            cid = frame["cluster_id"]

            base_mask = labels != cid
            cluster_mask = labels == cid
            noise_mask = labels == -1

            ax.scatter(
                X[base_mask & ~noise_mask, 0],
                X[base_mask & ~noise_mask, 1],
                s=62, alpha=0.25
            )

            ax.scatter(
                X[cluster_mask, 0],
                X[cluster_mask, 1],
                c=np.full(cluster_mask.sum(), cid),
                s=82, alpha=0.90
            )

            ax.scatter(
                X[noise_mask, 0],
                X[noise_mask, 1],
                marker="x", s=74, linewidths=2, alpha=0.50
            )

            subtitle = f"Density expansion：由 core point 往外擴張 Cluster {cid + 1}"

        else:
            noise_mask = labels == -1
            non_noise = ~noise_mask

            ax.scatter(
                X[non_noise, 0],
                X[non_noise, 1],
                c=labels[non_noise],
                s=82, alpha=0.90
            )

            ax.scatter(
                X[noise_mask, 0],
                X[noise_mask, 1],
                marker="x", s=82, linewidths=2
            )

            subtitle = "完成：密集區形成群；孤立點可以保留為 noise"

        decorate(
            ax,
            "DBSCAN",
            subtitle,
            f"STEP {frame['frame_no']} / {len(frames)}"
        )

        ax.set_xlim(-1.6, 2.5)
        ax.set_ylim(-0.9, 1.6)

    anim = FuncAnimation(
        fig, draw, frames=frames,
        interval=1200, repeat=True
    )

    return save(anim, fig, "03_dbscan.gif", fps=1.0)


# ============================================================
# 4. GMM
# ============================================================

def gmm_data():
    X, _ = make_blobs(
        n_samples=39,
        centers=[(-2.7, -1.4), (0.2, 2.3), (2.9, -0.6)],
        cluster_std=[0.75, 0.82, 0.73],
        random_state=SEED + 6
    )
    return X


def manual_em_frames(X, k=3, rounds=5):
    means = np.array([
        [-1.4, -0.2],
        [1.4, 1.1],
        [1.5, -1.4],
    ], dtype=float)

    covs = np.array([
        np.eye(2) * 2.0,
        np.eye(2) * 1.9,
        np.eye(2) * 1.7,
    ])

    weights = np.ones(k) / k
    frames = []

    for r in range(rounds):
        dens = np.zeros((len(X), k))

        for j in range(k):
            dens[:, j] = (
                weights[j]
                * multivariate_normal.pdf(
                    X,
                    mean=means[j],
                    cov=covs[j]
                )
            )

        resp = dens / dens.sum(axis=1, keepdims=True)

        frames.append({
            "kind": "E",
            "round": r + 1,
            "means": means.copy(),
            "covs": covs.copy(),
            "resp": resp.copy(),
        })

        Nk = resp.sum(axis=0)
        new_weights = Nk / len(X)
        new_means = (resp.T @ X) / Nk[:, None]

        new_covs = []

        for j in range(k):
            diff = X - new_means[j]

            cov = (
                (resp[:, j][:, None] * diff).T @ diff
                / Nk[j]
            )

            cov += np.eye(2) * 1e-4
            new_covs.append(cov)

        new_covs = np.array(new_covs)

        frames.append({
            "kind": "M",
            "round": r + 1,
            "old_means": means.copy(),
            "means": new_means.copy(),
            "covs": new_covs.copy(),
            "resp": resp.copy(),
        })

        means = new_means
        covs = new_covs
        weights = new_weights

    for i, f in enumerate(frames, start=1):
        f["frame_no"] = i

    return frames


def add_ellipse(ax, mean, cov):
    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(
        np.arctan2(
            vecs[1, 0],
            vecs[0, 0]
        )
    )

    width, height = 4.0 * np.sqrt(vals)

    e = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2.2,
        alpha=0.70
    )

    ax.add_patch(e)


def animate_gmm():
    X = gmm_data()
    frames = manual_em_frames(X)

    fig, ax = plt.subplots(figsize=(9, 6.4))

    def draw(frame):
        ax.clear()

        resp = frame["resp"]
        hard = resp.argmax(axis=1)
        certainty = resp.max(axis=1)

        # 讓不確定點更透明，直觀看 soft membership
        for cid in range(resp.shape[1]):
            idx = hard == cid

            ax.scatter(
                X[idx, 0], X[idx, 1],
                c=np.full(idx.sum(), cid),
                s=72,
                alpha=float(np.clip(np.median(certainty[idx]), 0.35, 0.9))
            )

        for m, cov in zip(frame["means"], frame["covs"]):
            add_ellipse(ax, m, cov)

        ax.scatter(
            frame["means"][:, 0],
            frame["means"][:, 1],
            marker="X", s=260, linewidths=2
        )

        if frame["kind"] == "E":
            idx = int(np.argmin(certainty))
            p = X[idx]
            probs = resp[idx]

            ax.scatter(
                [p[0]], [p[1]],
                s=230,
                facecolors="none",
                linewidths=2.2
            )

            for prob, mean in zip(probs, frame["means"]):
                ax.plot(
                    [p[0], mean[0]],
                    [p[1], mean[1]],
                    linewidth=1.2 + 2.4 * prob,
                    alpha=0.15 + 0.65 * prob
                )

            ax.text(
                0.98, 0.05,
                "邊界點 membership\n"
                + "   ".join(
                    f"C{j+1}: {prob:.0%}"
                    for j, prob in enumerate(probs)
                ),
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.45", alpha=0.12)
            )

            subtitle = "E-step：不是硬切；每個點對各群都有機率"

        else:
            old = frame["old_means"]

            for a, b in zip(old, frame["means"]):
                ax.annotate(
                    "",
                    xy=b, xytext=a,
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2.1,
                        shrinkA=6,
                        shrinkB=8
                    )
                )

            subtitle = "M-step：根據 membership 機率更新 mean 與 covariance"

        decorate(
            ax,
            "Gaussian Mixture Model (GMM)",
            subtitle,
            f"EM ROUND {frame['round']} — {frame['kind']}-STEP"
        )

        ax.text(
            0.02, 0.04,
            "橢圓只是 2D Gaussian 的等密度輪廓，重點是 soft membership",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", alpha=0.10)
        )

        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-4.0, 4.5)

    anim = FuncAnimation(
        fig, draw, frames=frames,
        interval=1200, repeat=True
    )

    return save(anim, fig, "04_gmm.gif")


# ============================================================
# Main
# ============================================================

def main():
    tasks = [
        ("K-means", animate_kmeans),
        ("Hierarchical", animate_hierarchical),
        ("DBSCAN", animate_dbscan),
        ("GMM", animate_gmm),
    ]

    outputs = []

    for name, func in tasks:
        print(f"Generating {name}...")
        try:
            out = func()
            outputs.append(out)
            print(" ->", out)
        except Exception as e:
            print(" !! failed:", repr(e))

    print("\nDONE")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
