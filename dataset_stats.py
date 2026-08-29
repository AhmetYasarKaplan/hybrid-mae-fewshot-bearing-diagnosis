from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_pipeline as tp
from feature_extraction import extract_features_batch, get_feature_names

LOAD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e3e3e0"
SURFACE = "#fcfcfb"

SEQ_BLUE = ["#eaf1fb", "#c3d9f4", "#8ab6e9", "#5295dd", "#2a78d6", "#1d5aa3", "#143d70"]


def _style_axes(ax, ylabel=None, xlabel=None, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def window_stats(X: np.ndarray) -> dict[str, np.ndarray]:
    x = X.reshape(X.shape[0], -1) if X.ndim > 2 else X
    mean = x.mean(1)
    std = x.std(1)
    rms = np.sqrt((x ** 2).mean(1))
    peak = np.abs(x).max(1)
    xc = x - mean[:, None]
    sd = xc.std(1) + 1e-12
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": x.max(1) - x.min(1),
        "crest_factor": peak / (rms + 1e-12),
        "kurtosis": ((xc / sd[:, None]) ** 4).mean(1) - 3.0,
        "skewness": ((xc / sd[:, None]) ** 3).mean(1),
        "energy": (x ** 2).sum(1),
    }


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "var": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def fig_amplitude_by_load(per_load_windows, loads, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, key, label in zip(axes, ("rms", "kurtosis"),
                              ("RMS amplitude", "Kurtosis (impulsiveness)")):
        data = [per_load_windows[l][key] for l in loads]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color=INK, linewidth=1.4),
                        whiskerprops=dict(color=INK_MUTED, linewidth=1.0),
                        capprops=dict(color=INK_MUTED, linewidth=1.0),
                        flierprops=dict(marker="o", markersize=2.5,
                                        markerfacecolor=INK_MUTED,
                                        markeredgecolor="none", alpha=0.35))
        for patch, colour in zip(bp["boxes"], LOAD_COLORS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(2.0)
        ax.set_xticklabels([f"L{l}" for l in loads])
        _style_axes(ax, ylabel=label, xlabel="Operating condition", title=label)
    fig.suptitle("Signal statistics shift with operating condition",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_load_distance(dist, loads, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    im = ax.imshow(dist, cmap=cmap, vmin=0)
    ax.set_xticks(range(len(loads)), [f"L{l}" for l in loads])
    ax.set_yticks(range(len(loads)), [f"L{l}" for l in loads])
    hi = dist.max() if dist.max() > 0 else 1.0
    for i in range(len(loads)):
        for j in range(len(loads)):
            ax.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if dist[i, j] > 0.6 * hi else INK)
    ax.set_title("Distance between load feature distributions\n"
                 "(standardised feature-space centroids)",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_pca_loads(feats, load_of_row, loads, out_path):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = PCA(n_components=2, random_state=0).fit_transform(
        StandardScaler().fit_transform(feats))
    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor=SURFACE)
    for colour, load in zip(LOAD_COLORS, loads):
        m = load_of_row == load
        ax.scatter(Z[m, 0], Z[m, 1], s=9, c=colour, alpha=0.55,
                   linewidths=0, label=f"L{load}", zorder=3)
    # clip axes to robust percentiles so a few outlier windows don't compress the rest
    lo_x, hi_x = np.percentile(Z[:, 0], [0.5, 99.5])
    lo_y, hi_y = np.percentile(Z[:, 1], [0.5, 99.5])
    pad_x, pad_y = 0.08 * (hi_x - lo_x), 0.08 * (hi_y - lo_y)
    ax.set_xlim(lo_x - pad_x, hi_x + pad_x)
    ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
    n_clipped = int(((Z[:, 0] < lo_x) | (Z[:, 0] > hi_x) |
                     (Z[:, 1] < lo_y) | (Z[:, 1] > hi_y)).sum())

    leg = ax.legend(frameon=False, fontsize=9, markerscale=1.6,
                    title="Operating condition", loc="best")
    leg.get_title().set_color(INK)
    leg.get_title().set_fontsize(9)
    for t in leg.get_texts():
        t.set_color(INK)
    _style_axes(ax, ylabel="PC 2", xlabel="PC 1",
                title="Feature space separates by operating condition")
    ax.grid(color=GRID, linewidth=0.8)
    if n_clipped:
        ax.text(0.995, -0.13, f"axes clipped to the 0.5-99.5 percentile "
                              f"({n_clipped} of {len(Z)} windows outside)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="CWRU statistics across loads")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--loads", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--channels", choices=["de", "fe", "both"], default="both")
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--window-cache-dir", type=str, default=None)
    ap.add_argument("--max-windows-per-load", type=int, default=1500,
                    help="Subsample for the feature/PCA stage (keeps it fast).")
    ap.add_argument("--output-dir", type=str, default="results_stats")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    per_load_windows, per_load_feats, per_load_labels = {}, {}, {}
    summary: dict = {"loads": args.loads, "channels": args.channels, "per_load": {}}

    print("Loading CWRU ...")
    for load in args.loads:
        ld = tp.load_windows(
            args.data_dir, load,
            window_size=args.window_size, stride=args.stride,
            channels=args.channels,
            window_cache_dir=Path(args.window_cache_dir) if args.window_cache_dir else None,
        )
        X, y = ld.X, ld.y
        if len(X) > args.max_windows_per_load:
            keep = rng.choice(len(X), args.max_windows_per_load, replace=False)
            X, y = X[keep], y[keep]

        per_load_windows[load] = window_stats(X)
        per_load_feats[load] = extract_features_batch(X, fs=ld.fs, feature_set="all")
        per_load_labels[load] = y
        summary["per_load"][str(load)] = {
            "n_windows": int(len(X)),
            "n_classes": int(len(np.unique(y))),
            "fs": int(ld.fs),
            "stats": {k: describe(v) for k, v in per_load_windows[load].items()},
        }
        print(f"  L{load}: {len(X)} windows, {len(np.unique(y))} classes")

    metrics = list(per_load_windows[args.loads[0]].keys())
    lines = ["load," + ",".join(f"{m}_{s}" for m in metrics
                                for s in ("mean", "std", "median"))]
    for load in args.loads:
        row = [f"L{load}"]
        for m in metrics:
            d = describe(per_load_windows[load][m])
            row += [f"{d['mean']:.6g}", f"{d['std']:.6g}", f"{d['median']:.6g}"]
        lines.append(",".join(row))
    (out / "stats_per_load.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["load,class,n," + ",".join(f"{m}_{s}" for m in metrics
                                        for s in ("mean", "std"))]
    per_class: dict = {}
    for load in args.loads:
        y = per_load_labels[load]
        for cls in sorted(np.unique(y)):
            m = y == cls
            row = [f"L{load}", str(cls), str(int(m.sum()))]
            entry = {}
            for metric in metrics:
                d = describe(per_load_windows[load][metric][m])
                row += [f"{d['mean']:.6g}", f"{d['std']:.6g}"]
                entry[metric] = {"mean": d["mean"], "std": d["std"]}
            per_class[f"L{load}|{cls}"] = entry
            lines.append(",".join(row))
    (out / "stats_per_class.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["per_class"] = per_class

    feats = np.vstack([per_load_feats[l] for l in args.loads])
    load_of_row = np.concatenate([np.full(len(per_load_feats[l]), l) for l in args.loads])
    mu, sigma = feats.mean(0), feats.std(0) + 1e-12
    Zf = (feats - mu) / sigma
    centroids = np.stack([Zf[load_of_row == l].mean(0) for l in args.loads])
    n = len(args.loads)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))

    lines = ["," + ",".join(f"L{l}" for l in args.loads)]
    for i, l in enumerate(args.loads):
        lines.append(f"L{l}," + ",".join(f"{dist[i, j]:.4f}" for j in range(n)))
    (out / "load_distance.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mean_dist = {f"L{l}": float(dist[i].sum() / max(1, n - 1))
                 for i, l in enumerate(args.loads)}
    summary["load_distance_matrix"] = dist.tolist()
    summary["mean_distance_to_other_loads"] = mean_dist
    summary["most_isolated_load"] = max(mean_dist, key=mean_dist.get)
    summary["feature_names"] = get_feature_names("all")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_figures:
        print("Rendering figures ...")
        fig_amplitude_by_load(per_load_windows, args.loads,
                              out / "fig_amplitude_by_load.png")
        fig_load_distance(dist, args.loads, out / "fig_load_distance.png")
        fig_pca_loads(feats, load_of_row, args.loads, out / "fig_pca_loads.png")

    print("\nMean distance to the other loads (higher = more isolated):")
    for k, v in sorted(mean_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v:.3f}")
    print(f"\nMost isolated operating condition: {summary['most_isolated_load']}")
    print(f"Outputs -> {out}")


if __name__ == "__main__":
    main()
