"""
ReStraV Figure 4 replication (corrected)
========================================

This script follows the Figure 4 protocol from the ReStraV paper/repo as
closely as possible, with one requested change:
  - N_VIDEOS_PER_SOURCE is kept as a user hyperparameter.

What this script does:
  1. Sample T=24 frames from a centered 2-second window per video.
  2. Resize frames to 224x224 and scale to [0, 1].
  3. Extract DINOv2 features from cls + patch tokens (flattened per frame).
  4. Compute per-video curvature trajectories (length T-2 = 22).
  5. Run t-SNE on video-level curvature vectors (one point per video).
  6. Plot two panels: Pixel Domain vs Representation Domain.

Datasets used for Figure 4 replication:
  - VidProM natural
  - VidProM AI: Pika, VideoCrafter2, Text2Video-Zero, ModelScope

Run:
  python test_dino_diff_datasets.py
"""

from pathlib import Path
import random
import warnings
from typing import cast

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()


def _p(rel_path: str) -> str:
    return str(_SCRIPT_DIR / rel_path)


VIDPROM_NAT_DIR = _p("Datasets/VidProM/natural")
UCF101_NAT_DIR = _p("Datasets/UCF-101")
DAVIS_NAT_DIR = _p("Datasets/DAVIS/JPEGImages/480p")

# Simple switch: set True to include UCF101 and DAVIS as additional natural sources.
USE_EXTRA_NATURAL_DATASETS = True
USE_UCF101_NATURAL = True
USE_DAVIS_NATURAL = True

VIDPROM_AI_DIRS = {
    "Pika": _p("Datasets/VidProM/ai/pika"),
    "VC2": _p("Datasets/VidProM/ai/vc2_videos"),
    "T2VZ": _p("Datasets/VidProM/ai/text2video_zero"),
    "ModelScope": _p("Datasets/VidProM/ai/ms_videos"),
}

# User-requested: keep this as a tunable hyperparameter.
N_VIDEOS_PER_SOURCE = 200

T_FRAMES = 24
VIDEO_DURATION_S = 2.0
FRAME_SIZE = 224
MODEL_BATCH_SIZE = 8

TSNE_PERPLEXITY = 30
TSNE_ITER = 2000
TSNE_RANDOM_STATE = 42

OUTPUT_FILE = str(_SCRIPT_DIR / "dino_viz" / "figure4_curvature_tsne.png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


SOURCE_STYLE = {
    "Natural": dict(color="#1f77b4", marker="o", size=20, label="Natural (VidProM)"),
    "UCF101": dict(color="#17becf", marker="o", size=20, label="Natural (UCF101)"),
    "DAVIS": dict(color="#aec7e8", marker="o", size=20, label="Natural (DAVIS)"),
    "Pika": dict(color="#d62728", marker="^", size=18, label="Pika (AI)"),
    "VC2": dict(color="#8c1a11", marker="v", size=18, label="VideoCrafter2 (AI)"),
    "T2VZ": dict(color="#ff7f0e", marker="<", size=18, label="Text2Video-Zero (AI)"),
    "ModelScope": dict(color="#ffbb78", marker="s", size=18, label="ModelScope (AI)"),
}


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def load_dinov2():
    """Load DINOv2 ViT-S/14."""
    print("Loading DINOv2 ViT-S/14...")
    model = cast(torch.nn.Module, torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"))
    model.eval().to(DEVICE)
    return model


# -----------------------------------------------------------------------------
# Video utilities
# -----------------------------------------------------------------------------

def collect_video_files(root: str, exts: set[str] | None = None, max_n: int | None = None) -> list[Path]:
    """Recursively collect video files from a root folder."""
    if exts is None:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    root_path = Path(root)
    if not root_path.exists():
        return []

    files = [p for p in root_path.rglob("*") if p.suffix.lower() in exts]
    random.shuffle(files)
    if max_n is not None:
        files = files[:max_n]
    return files


def collect_image_sequence_folders(root: str, max_n: int | None = None) -> list[Path]:
    """Collect folders containing image sequences (e.g., DAVIS)."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    folders = [d for d in root_path.iterdir() if d.is_dir()]
    random.shuffle(folders)
    if max_n is not None:
        folders = folders[:max_n]
    return folders


def sample_frames_from_video(
    video_path: str,
    n_frames: int = T_FRAMES,
    duration_s: float = VIDEO_DURATION_S,
) -> np.ndarray | None:
    """
    Sample n_frames over a centered duration window.
    Uses repeat-last behavior when a read fails after at least one valid frame,
    matching the intent of ReStraV's decoding policy.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    duration_total = total_frames / fps
    center_time = duration_total / 2.0
    half = duration_s / 2.0
    start_time = max(0.0, center_time - half)
    end_time = min(duration_total, center_time + half)

    if end_time <= start_time:
        start_time = 0.0
        end_time = max(duration_total, 1.0 / fps)

    start_idx = int(round(start_time * fps))
    end_idx = int(round(end_time * fps)) - 1
    end_idx = max(start_idx, end_idx)

    indices = np.linspace(start_idx, end_idx, n_frames, dtype=int)

    frames = []
    last_good = None
    for idx in indices:
        idx = int(np.clip(idx, 0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            if last_good is None:
                break
            frame = last_good.copy()
        else:
            last_good = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC)
        frames.append(frame.astype(np.float32) / 255.0)

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames[:n_frames], axis=0)


def sample_frames_from_image_folder(folder: str, n_frames: int = T_FRAMES) -> np.ndarray | None:
    """Sample n_frames uniformly from an image-sequence folder."""
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])
    if len(imgs) == 0:
        return None

    indices = np.linspace(0, len(imgs) - 1, n_frames, dtype=int)
    frames = []
    last_good = None

    for idx in indices:
        img = cv2.imread(str(imgs[idx]))
        if img is None:
            if last_good is None:
                continue
            img = last_good.copy()
        else:
            last_good = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC)
        frames.append(img.astype(np.float32) / 255.0)

    if len(frames) == 0:
        return None

    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames[:n_frames], axis=0)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def compute_curvature(z: np.ndarray) -> np.ndarray:
    """
    Compute curvature trajectory in degrees.
    z: (T, D)
    returns: (T-2,)
    """
    delta = np.diff(z, axis=0)
    norms = np.linalg.norm(delta, axis=1)

    curves = []
    for i in range(len(delta) - 1):
        denom = norms[i] * norms[i + 1]
        if denom < 1e-8:
            curves.append(90.0)
            continue
        cos_val = np.dot(delta[i], delta[i + 1]) / denom
        cos_val = np.clip(cos_val, -1.0, 1.0)
        curves.append(np.degrees(np.arccos(cos_val)))

    return np.array(curves, dtype=np.float32)


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_dino_token_trajectory(frames: np.ndarray, model) -> np.ndarray:
    """
    Extract per-frame flattened (cls + patch) token vectors.
    Input frames are already [0,1] and resized to 224x224.
    """
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(DEVICE)

    chunks = []
    for i in range(0, len(t), MODEL_BATCH_SIZE):
        batch = t[i:i + MODEL_BATCH_SIZE]
        feats = model.forward_features(batch)
        cls = feats["x_norm_clstoken"].unsqueeze(1)
        patches = feats["x_norm_patchtokens"]
        tokens = torch.cat([cls, patches], dim=1)
        chunks.append(tokens.flatten(1).cpu().numpy())

    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def extract_curvature_vectors_from_files(
    video_files: list[Path],
    n_videos: int,
    model,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns:
      dino_curves: list of (T-2,) curvature vectors in DINO representation space
      pixel_curves: list of (T-2,) curvature vectors in pixel space
    """
    dino_curves = []
    pixel_curves = []

    for vf in tqdm(video_files, desc=f"Processing [{label}]", leave=False):
        if len(dino_curves) >= n_videos:
            break

        frames = sample_frames_from_video(str(vf))
        if frames is None:
            continue

        z_dino = extract_dino_token_trajectory(frames, model)
        dino_curves.append(compute_curvature(z_dino))

        z_pixel = frames.reshape(frames.shape[0], -1).astype(np.float32)
        pixel_curves.append(compute_curvature(z_pixel))

    print(f"  {label}: loaded {len(dino_curves)} videos")
    return dino_curves, pixel_curves


@torch.no_grad()
def extract_curvature_vectors_from_image_folders(
    folders: list[Path],
    n_videos: int,
    model,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Same as video extraction, but for DAVIS-style image sequence folders.
    """
    dino_curves = []
    pixel_curves = []

    for folder in tqdm(folders, desc=f"Processing [{label}]", leave=False):
        if len(dino_curves) >= n_videos:
            break

        frames = sample_frames_from_image_folder(str(folder))
        if frames is None:
            continue

        z_dino = extract_dino_token_trajectory(frames, model)
        dino_curves.append(compute_curvature(z_dino))

        z_pixel = frames.reshape(frames.shape[0], -1).astype(np.float32)
        pixel_curves.append(compute_curvature(z_pixel))

    print(f"  {label}: loaded {len(dino_curves)} videos")
    return dino_curves, pixel_curves


# -----------------------------------------------------------------------------
# t-SNE and plotting
# -----------------------------------------------------------------------------

def run_tsne_on_curvature(curves_per_source: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    """
    t-SNE on curvature vectors.
    One point per video.
    """
    all_vecs = []
    labels = []
    for source, curves in curves_per_source.items():
        for c in curves:
            all_vecs.append(c)
            labels.append(source)

    if not all_vecs:
        raise RuntimeError("No curvature vectors available for t-SNE.")

    X = np.stack(all_vecs, axis=0)
    print(f"Running t-SNE on {X.shape[0]} videos x {X.shape[1]} curvature steps...")

    # Guard against invalid perplexity when the user sets a very small N.
    perplexity = min(TSNE_PERPLEXITY, max(5, X.shape[0] - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=TSNE_ITER,
        random_state=TSNE_RANDOM_STATE,
        metric="euclidean",
        init="pca",
        learning_rate="auto",
    )
    xy = tsne.fit_transform(X)
    return xy, labels


def plot_tsne_panel(ax, xy: np.ndarray, labels: list[str], title: str):
    """Plot a single t-SNE scatter panel."""
    ax.set_facecolor("#f8f8f8")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    for source in SOURCE_STYLE:
        idx = [i for i, lab in enumerate(labels) if lab == source]
        if not idx:
            continue
        style = SOURCE_STYLE[source]
        color = str(style["color"])
        marker = str(style["marker"])
        size = int(style["size"])
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            c=color,
            s=size,
            marker=marker,
            alpha=0.8,
            linewidths=0,
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_full_figure(pixel_curves_per_source: dict, dino_curves_per_source: dict):
    """Create the Figure 4 style two-panel plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    print("\n=== t-SNE: Pixel Domain ===")
    xy_pix, labels_pix = run_tsne_on_curvature(pixel_curves_per_source)
    plot_tsne_panel(axes[0], xy_pix, labels_pix, "Pixel Domain")

    print("\n=== t-SNE: Representation Domain ===")
    xy_dino, labels_dino = run_tsne_on_curvature(dino_curves_per_source)
    plot_tsne_panel(axes[1], xy_dino, labels_dino, "Representation Domain (DINOv2 ViT-S/14)")

    legend_items = []
    present_sources = set(labels_pix) | set(labels_dino)
    for src, style in SOURCE_STYLE.items():
        if src in present_sources:
            legend_items.append(mpatches.Patch(color=str(style["color"]), label=str(style["label"])))

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=min(5, max(1, len(legend_items))),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "t-SNE of Curvature Trajectories: Natural vs AI-Generated",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
    print(f"\nSaved figure -> {OUTPUT_FILE}")
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = load_dinov2()

    dino_curves_per_source = {}
    pixel_curves_per_source = {}

    # Natural source
    nat_files = collect_video_files(VIDPROM_NAT_DIR, max_n=N_VIDEOS_PER_SOURCE * 4)
    if nat_files:
        dino_curves, pixel_curves = extract_curvature_vectors_from_files(
            nat_files,
            N_VIDEOS_PER_SOURCE,
            model,
            "Natural",
        )
        if dino_curves:
            dino_curves_per_source["Natural"] = dino_curves
            pixel_curves_per_source["Natural"] = pixel_curves
    else:
        print(f"  Natural videos not found at: {VIDPROM_NAT_DIR}")

    if USE_EXTRA_NATURAL_DATASETS:
        # UCF101 natural videos
        if USE_UCF101_NATURAL:
            ucf_files = collect_video_files(UCF101_NAT_DIR, exts={".avi", ".mp4"}, max_n=N_VIDEOS_PER_SOURCE * 4)
            if ucf_files:
                dino_curves, pixel_curves = extract_curvature_vectors_from_files(
                    ucf_files,
                    N_VIDEOS_PER_SOURCE,
                    model,
                    "UCF101",
                )
                if dino_curves:
                    dino_curves_per_source["UCF101"] = dino_curves
                    pixel_curves_per_source["UCF101"] = pixel_curves
            else:
                print(f"  UCF101 videos not found at: {UCF101_NAT_DIR}")

        # DAVIS image-sequence folders
        if USE_DAVIS_NATURAL:
            davis_folders = collect_image_sequence_folders(DAVIS_NAT_DIR, max_n=N_VIDEOS_PER_SOURCE * 4)
            if davis_folders:
                dino_curves, pixel_curves = extract_curvature_vectors_from_image_folders(
                    davis_folders,
                    N_VIDEOS_PER_SOURCE,
                    model,
                    "DAVIS",
                )
                if dino_curves:
                    dino_curves_per_source["DAVIS"] = dino_curves
                    pixel_curves_per_source["DAVIS"] = pixel_curves
            else:
                print(f"  DAVIS sequences not found at: {DAVIS_NAT_DIR}")

    # AI sources
    for source_name, source_dir in VIDPROM_AI_DIRS.items():
        video_files = collect_video_files(source_dir, max_n=N_VIDEOS_PER_SOURCE * 4)
        if not video_files:
            print(f"  {source_name} videos not found at: {source_dir}")
            continue

        dino_curves, pixel_curves = extract_curvature_vectors_from_files(
            video_files,
            N_VIDEOS_PER_SOURCE,
            model,
            source_name,
        )
        if dino_curves:
            dino_curves_per_source[source_name] = dino_curves
            pixel_curves_per_source[source_name] = pixel_curves

    if not dino_curves_per_source:
        raise RuntimeError(
            "No valid videos were processed. Check VidProM paths in the config section."
        )

    print("\nCurvature Statistics (DINO domain)")
    print(f"{'Source':<12} {'Mean(deg)':>12} {'Std(deg)':>12} {'N videos':>10}")
    print("-" * 50)
    for source, curves in dino_curves_per_source.items():
        arr = np.concatenate(curves)
        print(f"{source:<12} {arr.mean():>12.2f} {arr.std():>12.2f} {len(curves):>10}")

    make_full_figure(pixel_curves_per_source, dino_curves_per_source)


if __name__ == "__main__":
    main()
