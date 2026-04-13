## Plan: New Notebook Perceptual Polynomial Experiment (3 Encoder-Decoder Pairs)

Goal: run the same experiment structure as sushi-does-it-again.ipynb, but move polynomial fitting from diffusion latent space to encoder perceptual space, and execute it across 3 paper-mentioned encoder-decoder pairs in a new notebook.

## Scope Lock

1. New notebook only. Do not modify sushi-does-it-again.ipynb.
2. Keep experiment flow identical to sushi-does-it-again.ipynb:
- baseline generation
- guided generation
- kernel degree sweep
- temporal curvature reporting and plots
3. Change only one core mechanism:
- from latent polynomial fitting
- to perceptual encoder-space polynomial fitting

## Selected Encoder-Decoder Pairs (3)

1. Pair A: SDXL-style pair
- encoder/decoder: AutoencoderKL pair used with SDXL-style path
2. Pair B: Stable Diffusion 1.5 pair
- encoder/decoder: AutoencoderKL pair for SD 1.5
3. Pair C: PixArt-style pair
- encoder/decoder: compatible AutoencoderKL pair for PixArt-style path

Note: If Pair C cannot be loaded in the runtime environment, replace Pair C with a Consistency-Model-compatible encoder-decoder path and keep the same experiment protocol.

## New Notebook Design

Target notebook: sushi-perceptual-poly-encdec.ipynb

### Cell 1 - Config and imports

Add config keys for:
1. enc_dec_pairs_to_test (exactly 3 active pairs)
2. perceptual_poly_degrees = [2, 3, 4, 5]
3. experiment_seed = 42
4. run_baseline, run_guided, run_degree_sweep, run_mpes
5. perceptual_fit_dtype = float32
6. save_dir for per-pair outputs

### Cell 2 - Perceptual encoder model

Define perceptual feature extraction used for fitting:
1. perceptual_encode(frames_tensor) -> features [T, D]
2. same preprocessing pipeline for all 3 pairs
3. curvature function unchanged from sushi style

### Cell 3 - Polynomial fitting in perceptual space

Implement:
1. fit_polynomial_features(features, degree)
- build Vandermonde matrix in float32
- solve with lstsq/pinv for stability
2. apply_perceptual_poly_guidance(latents)
- decode current latents to frames
- encode to perceptual features
- fit polynomial in feature space
- optimize latents for 1-2 refinement steps so decoded-frame perceptual features match fitted features

### Cell 4 - Guided callback (same interface as sushi)

Keep callback structure identical, but replace latent smoothing call with perceptual-space guidance call.

### Cell 5 - Same generation protocol per pair

For each of the 3 pairs:
1. run baseline video with fixed seed and same params
2. run guided video with callback
3. export videos to pair-specific output folders

### Cell 6 - Same degree experiment per pair

Replicate sushi degree sweep exactly:
1. degrees = [2, 3, 4, 5]
2. same seed, same motion/noise settings, same scheduling cadence
3. one output gif per degree per pair

### Cell 7 - Same evaluation protocol

Replicate sushi metrics and plotting style:
1. base curvature vs guided curvature
2. degree-wise curvature trend
3. one comparison plot per pair
4. one combined summary plot across the 3 pairs

## Output Layout

experiment-results/
1. perceptual_poly_pair_A/
2. perceptual_poly_pair_B/
3. perceptual_poly_pair_C/
4. plots/
5. metadata/

Each pair folder stores:
1. baseline gif
2. guided gif
3. degree sweep gifs
4. per-run metrics json

## Verification Checklist

1. Exact protocol parity with sushi confirmed:
- same seeds
- same degree list
- same callback cadence
- same metric definitions
2. Polynomial fit confirmed in perceptual feature space (not latent tensor space).
3. All 3 pairs produce full baseline/guided/degree outputs.
4. Curvature metrics are finite for all runs.
5. Notebook runs end-to-end on RTX 4060 with memory-safe settings.

## Decisions

1. Included:
- new notebook implementation only
- 3 encoder-decoder pairs
- sushi-equivalent experiment protocol
- perceptual-space polynomial fitting
2. Excluded:
- editing sushi-does-it-again.ipynb
- changing experiment definition beyond latent->perceptual fitting substitution
