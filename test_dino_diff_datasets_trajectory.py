"""
Figure 4-style trajectory visualization on top of ReStraV curvature t-SNE.

Why this script exists:
- The paper/repo describe the feature extraction and curvature pipeline, but do not
  publish a dedicated Figure 4 plotting script.
- The published figure shows connected line-like structures; this script preserves
  the same curvature t-SNE method and adds a kNN edge overlay to reproduce that
  visual style.

What is reused from test_dino_diff_datasets.py:
- Sampling: centered 2-second window, T=24
- Preprocessing: resize to 224, scale to [0,1]
- DINO representation: cls + patch tokens
- Curvature: angle between consecutive displacement vectors
- t-SNE input: one curvature vector per video

This script keeps one global hyperparameter for all sources:
- N_VIDEOS_PER_SOURCE

And supports extra datasets (UCF101, DAVIS) through the same toggles as base script.
"""

import importlib.util
from pathlib import Path
import random
from typing import Any, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch


_SCRIPT_DIR = Path(__file__).parent.resolve()
_BASE_SCRIPT = _SCRIPT_DIR / "test_dino_diff_datasets.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_fig4", _BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script at {_BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = cast(Any, load_base_module())


# -----------------------------------------------------------------------------
# User-facing config
# -----------------------------------------------------------------------------

# Keep a single N hyperparameter for every source.
N_VIDEOS_PER_SOURCE = base.N_VIDEOS_PER_SOURCE

# Reuse base dataset toggles/paths by default.
USE_EXTRA_NATURAL_DATASETS = base.USE_EXTRA_NATURAL_DATASETS
USE_UCF101_NATURAL = base.USE_UCF101_NATURAL
USE_DAVIS_NATURAL = base.USE_DAVIS_NATURAL

# kNN edge count for line overlay (visual trajectory style).
KNN_EDGES_PER_POINT = 4

OUTPUT_FILE = str(_SCRIPT_DIR / "dino_viz" / "figure4_curvature_tsne_with_trajectories.png")


# Push local config into base module so extraction uses the same N and toggles.
setattr(base, "N_VIDEOS_PER_SOURCE", N_VIDEOS_PER_SOURCE)
setattr(base, "USE_EXTRA_NATURAL_DATASETS", USE_EXTRA_NATURAL_DATASETS)
setattr(base, "USE_UCF101_NATURAL", USE_UCF101_NATURAL)
setattr(base, "USE_DAVIS_NATURAL", USE_DAVIS_NATURAL)


def build_curvature_sources(model):
    """Load datasets and compute curvature vectors exactly like the base script."""
    dino_curves_per_source = {}
    pixel_curves_per_source = {}

    # VidProM natural
    nat_files = base.collect_video_files(base.VIDPROM_NAT_DIR, max_n=base.N_VIDEOS_PER_SOURCE * 4)
    if nat_files:
        dino_curves, pixel_curves = base.extract_curvature_vectors_from_files(
            nat_files,
            base.N_VIDEOS_PER_SOURCE,
            model,
            "Natural",
        )
        if dino_curves:
            dino_curves_per_source["Natural"] = dino_curves
            pixel_curves_per_source["Natural"] = pixel_curves
    else:
        print(f"  Natural videos not found at: {base.VIDPROM_NAT_DIR}")

    # Extra natural sets
    if base.USE_EXTRA_NATURAL_DATASETS:
        if base.USE_UCF101_NATURAL:
            ucf_files = base.collect_video_files(
                base.UCF101_NAT_DIR,
                exts={".avi", ".mp4"},
                max_n=base.N_VIDEOS_PER_SOURCE * 4,
            )
            if ucf_files:
                dino_curves, pixel_curves = base.extract_curvature_vectors_from_files(
                    ucf_files,
                    base.N_VIDEOS_PER_SOURCE,
                    model,
                    "UCF101",
                )
                if dino_curves:
                    dino_curves_per_source["UCF101"] = dino_curves
                    pixel_curves_per_source["UCF101"] = pixel_curves
            else:
                print(f"  UCF101 videos not found at: {base.UCF101_NAT_DIR}")

        if base.USE_DAVIS_NATURAL:
            davis_folders = base.collect_image_sequence_folders(
                base.DAVIS_NAT_DIR,
                max_n=base.N_VIDEOS_PER_SOURCE * 4,
            )
            if davis_folders:
                dino_curves, pixel_curves = base.extract_curvature_vectors_from_image_folders(
                    davis_folders,
                    base.N_VIDEOS_PER_SOURCE,
                    model,
                    "DAVIS",
                )
                if dino_curves:
                    dino_curves_per_source["DAVIS"] = dino_curves
                    pixel_curves_per_source["DAVIS"] = pixel_curves
            else:
                print(f"  DAVIS sequences not found at: {base.DAVIS_NAT_DIR}")

    # AI sets
    for source_name, source_dir in base.VIDPROM_AI_DIRS.items():
        video_files = base.collect_video_files(source_dir, max_n=base.N_VIDEOS_PER_SOURCE * 4)
        if not video_files:
            print(f"  {source_name} videos not found at: {source_dir}")
            continue

        dino_curves, pixel_curves = base.extract_curvature_vectors_from_files(
            video_files,
            base.N_VIDEOS_PER_SOURCE,
            model,
            source_name,
        )
        if dino_curves:
            dino_curves_per_source[source_name] = dino_curves
            pixel_curves_per_source[source_name] = pixel_curves

    return pixel_curves_per_source, dino_curves_per_source


def draw_knn_edges(ax, xy: np.ndarray, labels: list[str], k_edges: int = 4):
    """Draw source-wise kNN edges to mimic the trajectory-like line texture in Figure 4."""
    segments = []
    colors = []

    for source, style in base.SOURCE_STYLE.items():
        idx = np.array([i for i, lab in enumerate(labels) if lab == source], dtype=np.int64)
        if len(idx) < 2:
            continue

        pts = xy[idx]
        k = min(k_edges, len(pts) - 1)
        if k <= 0:
            continue

        graph = kneighbors_graph(pts, k, mode="connectivity", include_self=False)
        rows, cols = cast(Any, graph).nonzero()

        for r, c in zip(rows.tolist(), cols.tolist()):
            if r < c:
                segments.append([pts[r], pts[c]])
                colors.append(str(style["color"]))

    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=0.35, alpha=0.30, zorder=1)
        ax.add_collection(lc)


def plot_tsne_panel_with_trajectories(ax, xy: np.ndarray, labels: list[str], title: str):
    """Scatter points + kNN line overlay (trajectory-like appearance)."""
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    draw_knn_edges(ax, xy, labels, k_edges=KNN_EDGES_PER_POINT)

    for source in base.SOURCE_STYLE:
        idx = [i for i, lab in enumerate(labels) if lab == source]
        if not idx:
            continue
        style = base.SOURCE_STYLE[source]
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            c=str(style["color"]),
            s=int(style["size"]),
            marker=str(style["marker"]),
            alpha=0.80,
            linewidths=0,
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(pixel_curves_per_source: dict, dino_curves_per_source: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    print("\n=== t-SNE: Pixel Domain ===")
    xy_pix, labels_pix = base.run_tsne_on_curvature(pixel_curves_per_source)
    plot_tsne_panel_with_trajectories(axes[0], xy_pix, labels_pix, "Pixel Domain")

    print("\n=== t-SNE: Representation Domain ===")
    xy_dino, labels_dino = base.run_tsne_on_curvature(dino_curves_per_source)
    plot_tsne_panel_with_trajectories(axes[1], xy_dino, labels_dino, "Representation Domain (DINOv2 ViT-S/14)")

    legend_items = []
    present_sources = set(labels_pix) | set(labels_dino)
    for src, style in base.SOURCE_STYLE.items():
        if src in present_sources:
            legend_items.append(mpatches.Patch(color=str(style["color"]), label=str(style["label"])))

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=min(6, max(1, len(legend_items))),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "t-SNE of Curvature Trajectories with Line Overlay (Figure 4-style)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved figure -> {out_path}")
    plt.show()


def print_curvature_stats(dino_curves_per_source: dict):
    print("\nCurvature Statistics (DINO domain)")
    print(f"{'Source':<12} {'Mean(deg)':>12} {'Std(deg)':>12} {'N videos':>10}")
    print("-" * 50)
    for source, curves in dino_curves_per_source.items():
        arr = np.concatenate(curves)
        print(f"{source:<12} {arr.mean():>12.2f} {arr.std():>12.2f} {len(curves):>10}")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Using base extraction pipeline from test_dino_diff_datasets.py")
    print(f"N_VIDEOS_PER_SOURCE = {base.N_VIDEOS_PER_SOURCE}")

    model = base.load_dinov2()
    pixel_curves_per_source, dino_curves_per_source = build_curvature_sources(model)

    if not dino_curves_per_source:
        raise RuntimeError("No valid videos were processed. Check dataset paths in base script.")

    print_curvature_stats(dino_curves_per_source)
    make_figure(pixel_curves_per_source, dino_curves_per_source)


if __name__ == "__main__":
    main()
