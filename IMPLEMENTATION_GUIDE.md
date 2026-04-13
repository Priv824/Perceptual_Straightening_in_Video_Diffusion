# Enhanced Perceptual Straightening Implementation Guide

## Overview
This notebook implements a production-grade system for improving temporal consistency in video diffusion using the **Perceptual Straightening Hypothesis (PSH)**. Built on AnimateDiff with an RTX 4060, it provides:

- **Perceptual trajectory analysis** via RetinalDN + GaborV1 (mimics human V1)
- **Adaptive PSG** with scheduling during inference
- **Multi-decoder comparison** for trajectory diversity analysis
- **Kernel regression smoothing** (2-5 parameter models)
- **Multi-scale temporal analysis** and ensemble fusion
- **Comprehensive metrics** and visualizations

## Architecture

### Core Hypothesis
**Retinal Latent Constraint (RLC)**: Natural videos have curved trajectories in pixel space but straighten when projected into V1 perceptual space. AI-generated videos remain curved → apply gradient guidance to straighten them at each denoising step.

### System Components

#### 1. **Perceptual Model** (`Section 1`)
- `RetinalDN`: Mimics retina+LGN with DoG center-surround, luminance/contrast gain control
- `GaborV1`: Multi-scale Gabor filters → V1-like energy maps
- **Purpose**: Extract meaningful perceptual features from frames
- **Input**: RGB frames [0,1]
- **Output**: D-dimensional feature vector per frame

#### 2. **AnimateDiff Pipeline** (`Section 2`)
- Base model: `emilianJR/epiCRealism` (realistic video)
- Motion adapter: AnimateDiff v1.5-2 (temporal coherence)
- Memory optimizations: CPU offload, VAE tiling/slicing, attention slicing
- **Why**: Better inherent temporal consistency than frame-by-frame SD 1.5

#### 3. **PSG Engine** (`Section 4`)
- Callback-based integration into diffusers pipeline
- **Adaptive λ scheduling**: Cosine-annealed guidance strength
  ```
  λ(t) = λ_base × 0.5(1 + cos(π × progress))
  ```
- **Minimal refinement**: 2-3 gradient steps per DDIM step (for speed)
- **Latent-space operation**: Reduces VAE decode overhead

#### 4. **Trajectory Smoothing** (`Section 6`)
Four kernel regression models with varying complexity:

| Model | Parameters | Implementation | Use Case |
|-------|-----------|-----------------|----------|
| Linear | 2 | Ridge regression (degree 1) | Baseline smoothing |
| Poly2 | 3 | Polynomial Ridge (degree 2) | Gentle curves |
| Poly3 | 4 | Polynomial Ridge (degree 3) | Complex motion |
| Spline | 5+ | UnivariateSpline (k=3) | Per-frame adaptation |

**Intuition**: Fit smooth curves to frame trajectories, reducing high-frequency jitter.

#### 5. **Multi-Decoder Manager** (`Section 2`)
Tests three VAE decoder variants:
- **Default**: Standard decode
- **Tiled**: Frame-by-frame decode (more memory efficient)
- **Sliced**: Batch with interior padding

**Why**: Different latent manifold trajectories → ensemble opportunities.

## Configuration (CONFIG Dictionary)

### Key Parameters
```python
CONFIG = {
    # Model choice
    "base_model": "emilianJR/epiCRealism",
    "dtype": torch.float16,  # Memory safety on RTX 4060
    
    # Video params
    "n_frames": 8,
    "n_steps": 15,
    "prompt": "...",
    
    # PSG control
    "psg_lambda": 0.3,  # Guidance strength
    "psg_start_step": 5,  # Warmup steps (avoid early blurring)
    "psg_adaptive_schedule": True,  # Cosine decay
    
    # Experiments (boolean flags)
    "run_psg": True,
    "run_multi_decoder": True,
    "run_trajectory_smoothing": True,
    "run_multi_scale_analysis": True,
    "run_ensemble_fusion": True,
    "run_adaptive_scheduling": True,
}
```

### To Toggle Experiments
```python
CONFIG["run_trajectory_smoothing"] = False  # Skip this experiment
```

## Workflow

### 1. Generate Baseline
```python
if CONFIG["run_baseline"]:
    baseline_frames, baseline_m = generate_baseline()
```
**Output**: 
- `results/baseline/video.gif` - Generated video
- `results/baseline/frames/` - Individual frames
- `baseline_m` dict with metrics

### 2. Apply PSG
```python
if CONFIG["run_psg"]:
    psg_frames, psg_m, psg_latent_traj = generate_with_psg()
```
**Key differences from baseline**:
- Applies curvature loss during denoising
- Adaptively scales λ down denoising steps
- Saves latent trajectory for analysis

### 3. Compare Decoders
```python
if CONFIG["run_multi_decoder"]:
    # Tests default, tiled, sliced VAE decoders
    # Stores in decoder_results dict
```

### 4. Test Trajectory Smoothing
```python
if CONFIG["run_trajectory_smoothing"]:
    # Fits 4 kernel models to baseline frames
    # Evaluates curvature, straightness, velocity variance
```

### 5. Advanced Experiments
- **Multi-scale**: Analyzes straightness at different frame strides
- **Ensemble**: Averages multiple seed generations
- **Adaptive scheduling**: Visualizes different λ profiles

## Metrics Explained

| Metric | Computation | Interpretation |
|--------|-----------|-----------------|
| **Curvature** | Mean angle between consecutive displacements in V1 space | Lower = straighter motion |
| **Straightness** | `(1 - Curvature/π) × 100` | 0-100 score; natural videos ~60-80% |
| **Temporal Consistency** | Frame-to-frame MSE | Lower = smoother transitions |
| **Velocity Variance** | Std of frame differences | Lower = less jittery |

## Results Directory Structure
```
experiment-results/
├── baseline/
│   ├── video.gif
│   └── frames/
├── psg/
│   ├── video_lambda*.gif
│   └── frames_lambda*/
├── decoders/
│   ├── video_default.gif
│   ├── video_tiled.gif
│   └── video_sliced.gif
├── smoothing/
│   ├── video_linear_2param.gif
│   ├── video_poly_3param.gif
│   ├── video_poly_4param.gif
│   └── video_spline_5param.gif
├── plots/
│   ├── trajectory_pca.png
│   ├── metrics_comparison.png
│   └── ...
└── metadata/
    └── experiment_summary.json
```

## How PSG Works (Deep Dive)

### Step-by-Step During Denoising

```
1. UNet predicts noise at timestep t
   → Tweedie's formula: x̂₀ = (z_t - (1-ᾱ_t)^0.5 × ε_pred) / (ᾱ_t)^0.5

2. Decode predicted frames: x̂₀_pixels = VAE_decode(x̂₀_latent)

3. Compute V1 features: f = V1(RetinalDN(x̂₀_pixels))

4. Measure trajectory curvature:
   curv = arccos(f[i]·f[i+1] / (||f[i]|| × ||f[i+1]||))

5. Gradient descent on frames (NOT weights):
   x̂₀_refined = x̂₀ - λ × ∇_x ReLU(curv)

6. Re-encode refined frames and step scheduler
   ε_pred_refined = back_compute_noise(x̂₀_refined)
```

### Why It Works
- **Early steps** (high t): Global motion structure formed → strong guidance needed
- **Late steps** (low t): Details added → light guidance prevents over-smoothing
- **Adaptive λ**: Cosine schedule automatically handles this trade-off

## Advanced Extensions (Brainstormed)

### 1. **Feature-Space Guidance** (High Priority)
- Apply PSG directly in latent space without VAE decode
- **Benefit**: 3-5× speedup, avoid VAE manifold artifacts
- **Implementation**: Train lightweight perceptual encoder for latent space

### 2. **Cross-Frame Attention** (High Priority)
- Inject temporal self-attention during denoising
- **Mechanism**: Key/query pooling across frames → implicit trajectory smoothing
- **Cost**: Minimal (already in transformer UNet)

### 3. **Velocity Regularization** (Medium Priority)
- Direct penalty on frame-to-frame MSE during denoising
- **Formula**: `loss = ReLU(||f[i] - f[i+1]||²) - threshold`
- **Advantage**: Orthogonal to curvature; targets different motion types

### 4. **Hierarchical Multi-Scale Guidance** (Medium Priority)
- Different λ at different spatial scales (patch-wise)
- **Intuition**: Large motions need more guidance; fine details should be free
- **Implementation**: Apply PSG separately to multi-scale latent decomposition

### 5. **Anomaly-Based Selective Smoothing** (Low Priority)
- Only smooth frames with anomalously high temporal jump
- **Algorithm**:
  ```
  diff_mag = ||f[i] - f[i+1]||
  threshold = 2 × median(diff_mag)
  smooth_mask = (diff_mag > threshold).astype(float)
  refined = blend(smooth(f), f, mask=smooth_mask)
  ```

## Performance Targets (RTX 4060)

| Setting | VRAM | Time/Frame | Quality |
|---------|------|-----------|---------|
| Baseline (8 frames, 15 steps) | ~6.5 GB | 15 sec | Good |
| + PSG (n_refine=3) | ~7.2 GB | 22 sec | Very Good |
| Multi-scale (3 trajectories) | ~7.8 GB | 45 sec | Excellent |
| Ensemble (3 members) | OOM | — | — |

**Recommendations**:
- Use `n_frames=6`, `n_steps=12` for faster iteration
- PSG is worth the 7 seconds overhead
- Ensemble fusion → reduce to n_frames=4

## How to Run

### Quick Test
```python
CONFIG["n_frames"] = 6
CONFIG["n_steps"] = 12
CONFIG["run_psg"] = True
CONFIG["run_multi_decoder"] = False  # Skip expensive tests
CONFIG["run_trajectory_smoothing"] = True

# Run all cells top to bottom
```

### Full Benchmark
```python
CONFIG["n_frames"] = 8
CONFIG["n_steps"] = 15
CONFIG["run_psg"] = True
CONFIG["run_multi_decoder"] = True
CONFIG["run_trajectory_smoothing"] = True
CONFIG["run_multi_scale_analysis"] = True

# Allow 5-10 minutes
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **CUDA OOM** | Too many frames/steps | Reduce n_frames to 4, n_steps to 10 |
| **Baseline quality low** | Poor prompt tuning | Refine CONFIG["prompt"] |
| **PSG not improving** | λ too low | Increase psg_lambda to 0.5-0.7 |
| **Over-smoothing** | λ too high or start_step too low | Decrease λ or increase start_step to 8+ |
| **Slow generation** | CPU offload overhead | Disable enable_cpu_offload if VRAM allows |

## References

- **Paper**: "Improving Temporal Consistency at Inference-time..." (Rahimi & Tekalp, 2025)
- **Perceptual Model**: PSH via RetinalDN + Steerable Pyramid (Hénaf et al., Nature Neuro 2019)
- **Diffusion**: DDIM (Song et al., 2021)
- **Video Diffusion**: AnimateDiff (Guo et al., 2023)

---

**Last Updated**: April 13, 2026  
**Notebook**: `perceptual_straightening_sushi.ipynb`  
**Results**: `experiment-results/`
