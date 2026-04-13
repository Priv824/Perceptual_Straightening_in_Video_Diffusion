# Quick Start Guide

## Setup (One-Time)

1. **Open** `perceptual_straightening_sushi.ipynb` in VS Code
2. **Run cells in order from Section 1 onwards**
3. First run will download models (~4-5 GB) - grab a coffee ☕

## Running Your First Experiment

### Minimal Test (5 minutes)
```python
# Modify CONFIG at the top:
CONFIG["n_frames"] = 6
CONFIG["n_steps"] = 12
CONFIG["run_psg"] = True
CONFIG["run_multi_decoder"] = False
CONFIG["run_trajectory_smoothing"] = True
CONFIG["run_multi_scale_analysis"] = False
CONFIG["run_ensemble_fusion"] = False

# Run all cells Section 1 → Section 8
```

**Output**: 
- Baseline video at `experiment-results/baseline/video.gif`
- PSG video at `experiment-results/psg/video_lambda0.30.gif`
- Metrics printed to console
- Comparison plot at `experiment-results/plots/metrics_comparison.png`

### Compare Baseline vs PSG Visually
```bash
# Open in any video player or image viewer:
experiment-results/baseline/video.gif
experiment-results/psg/video_lambda0.30.gif
# Look for smoothness improvement!
```

### Check Metrics
```python
# After running, print_metrics() shows:
# └─ "Baseline"              curv=2.34  straight=45.2%  t-cons=0.08542
# └─ "PSG (λ=0.30)"          curv=1.89  straight=54.8%  t-cons=0.06123
#    Δ straightness: +9.6% ✓
```

## Extended Experiments

### Test All Methods (30 minutes)
```python
CONFIG["n_frames"] = 8
CONFIG["n_steps"] = 15
CONFIG["run_psg"] = True
CONFIG["run_multi_decoder"] = True
CONFIG["run_trajectory_smoothing"] = True
CONFIG["run_multi_scale_analysis"] = True
CONFIG["run_ensemble_fusion"] = True
CONFIG["run_adaptive_scheduling"] = True

# Run all cells
```

**Output Structure**:
```
experiment-results/
├── baseline/                   # No guidance
├── psg/                        # PSG applied
│   └── video_lambda0.30.gif
├── decoders/                   # Multi-decoder comparison
│   ├── video_default.gif
│   ├── video_tiled.gif
│   └── video_sliced.gif
├── smoothing/                  # Kernel regression
│   ├── video_linear_2param.gif
│   ├── video_poly_3param.gif
│   ├── video_poly_4param.gif
│   └── video_spline_5param.gif
├── plots/
│   ├── trajectory_pca.png      # 2D perceptual space
│   └── metrics_comparison.png  # Bar charts
└── metadata/
    └── experiment_summary.json # Numerical results
```

## Customization Examples

### Change Prompt
```python
CONFIG["prompt"] = "a car driving on highway, cinematic, motion blur"
CONFIG["neg_prompt"] = "static, blur, low quality"
```

### Adjust PSG Strength
```python
CONFIG["psg_lambda"] = 0.5  # More aggressive (was 0.3)
CONFIG["psg_start_step"] = 8  # Later start (was 5)
```

### Use Fewer Frames
```python
CONFIG["n_frames"] = 4  # Faster, less VRAM
CONFIG["n_steps"] = 10  # Faster
```

### Run Only Specific Experiments
```python
CONFIG["run_baseline"] = True
CONFIG["run_psg"] = True
CONFIG["run_multi_decoder"] = False  # Skip
CONFIG["run_trajectory_smoothing"] = False  # Skip
Config["run_ensemble_fusion"] = False  # Skip
# Run faster by commenting out unwanted sections
```

## What to Expect

### Console Output
```
Device: cuda
GPU: NVIDIA GeForce RTX 4060 Laptop GPU  |  8.0 GB VRAM
✓ Configuration loaded...
✓ Perceptual model initialized
✓ AnimateDiff Pipeline loaded
✓ Multi-decoder manager initialized

Generating baseline (frames=8, steps=15)...
DDIM: 100%|██████████| 15/15 [00:20<00:00]
Done in 20.3s
  Baseline                  curv=2.3421  straight=43.2%  t-cons=0.0892

Generating with PSG (λ=0.3, refine_steps=3, start=5)...
DDIM: 100%|██████████| 15/15 [00:28<00:00]
Done in 28.1s
  PSG (λ=0.30)             curv=1.8234  straight=52.7%  t-cons=0.0654

=== Multi-Decoder Comparison ===
Testing decoder: default
  Decoder: default           curv=2.3421  straight=43.2%  t-cons=0.0892
...

=== Results Summary ===
Baseline                     curv=2.34  straight=43.2%  t-cons=0.089
PSG                          curv=1.82  straight=52.7%  t-cons=0.065
Decoder_default              curv=2.34  straight=43.2%  t-cons=0.089
...

✅ Experiment complete!
```

### Video Quality Observations
- **Baseline**: Slight shimmer, some texture boiling (especially background)
- **+ PSG**: Smoother motion, stable textures, fewer jumps
- **+ Smoothing**: Very smooth, but may lose some fine detail
- **+ Ensemble**: Best overall - smooth AND detailed

## Monitoring VRAM

```python
# Add anywhere after model loads:
import torch
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

**Expected usage**:
- Baseline: 6.5 GB
- + PSG: 7.2 GB
- + Ensemble (3 members): **8.0+ GB** (may OOM)

**If OOM**: Reduce `n_frames` to 6, `n_steps` to 12

## Troubleshooting

### Issue: "`torch.cuda.OutOfMemoryError`"
**Solution**: 
```python
CONFIG["n_frames"] = 4  # Was 8
CONFIG["n_steps"] = 10  # Was 15
```

### Issue: "Generation too slow"
**Solution**:
```python
CONFIG["run_ensemble_fusion"] = False  # This runs 3 forward passes
CONFIG["n_frames"] = 6
CONFIG["n_steps"] = 12
```

### Issue: "Videos look over-smoothed"
**Solution**: Lower PSG strength
```python
CONFIG["psg_lambda"] = 0.15  # Was 0.3
```

### Issue: "No improvement from PSG"
**Solution**: Increase λ or extend application
```python
CONFIG["psg_lambda"] = 0.5  # Was 0.3
CONFIG["psg_start_step"] = 3  # Was 5
```

### Issue: "Model download fails"
**Solution**: Ensure stable internet, models will retry. Files go to `~/.cache/huggingface/hub/`

## Interpreting Results

### Straightness > 60%
✅ Good - natural-looking motion

### Straightness 40-60%
🟡 Moderate - some visible jitter, PSG should help

### Straightness < 40%
❌ Poor - likely prompt issue or model training effect

### PSG Improvement < 5%
🟡 May not be worth the compute (try increasing λ)

### PSG Improvement > 15%
✅ Excellent - strong temporal consistency benefit

## Performance Estimates (RTX 4060)

| Config | Time | VRAM | Quality |
|--------|------|------|---------|
| Baseline (6f, 12s) | 12 sec | 6.2 GB | Good |
| + PSG (6f, 12s) | 16 sec | 6.8 GB | Good |
| Baseline (8f, 15s) | 20 sec | 6.5 GB | Better |
| + PSG (8f, 15s) | 28 sec | 7.2 GB | Better+ |
| Multi-decoder (8f, 12s) | 24 sec | 6.8 GB | Better |
| Trajectory smoothing | ~1 sec | ~1 GB | +2-5% |
| Ensemble (3 members) | 60 sec | **OOM** | Best |

## Best Practices

1. **Start with baseline**: Establish reference metrics
2. **Try PSG next**: Check if straightness improves by 5%+
3. **Experiment with λ**: Try 0.2, 0.3, 0.5 and pick best
4. **Consider smoothing**: Post-hoc refinement (free win)
5. **Skip ensemble for iterating**: Too expensive; use for final results

## Next Steps After First Run

1. **Inspect metrics**:
   ```bash
   cat experiment-results/metadata/experiment_summary.json | jq
   ```

2. **View trajectory**:
   - Open `experiment-results/plots/trajectory_pca.png`
   - Look for tighter clustering (better straightness)

3. **Compare videos**:
   - Side-by-side: baseline vs PSG
   - Look for: less jitter, stable textures, smooth motion

4. **Refine hyperparameters**:
   - Adjust prompt for your style
   - Tune λ for your motion complexity
   - Scale frames/steps to your GPU VRAM

---

**Questions or Issues?** Check `IMPLEMENTATION_GUIDE.md` for deep dive  
**Want innovations explained?** See `INNOVATIONS_SUMMARY.md`
