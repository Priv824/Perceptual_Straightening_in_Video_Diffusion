# EXPERIMENT REPORT: Implementing Retinal Latent Constraint (RLC) in Video Diffusion

---

## 1. The Initial Hypothesis: "Retinal Persistence"

**The Goal:** Video diffusion models (like AnimateDiff) often suffer from temporal jitter (flickering details) because each frame's noise is predicted somewhat independently. We hypothesized that we could enforce temporal stability mathematically by mimicking the human retina's persistence of vision.

**The Strategy:** We injected a custom mathematical constraint into the generation loop between the UNet's noise prediction and the Scheduler's step.

We built `apply_retinal_constraint` based on two pillars:

- **Trajectory Smoothing (Inertia):** We took the noisy image representations (latents) and forced them onto a smoothed timeline. If Frame 1 is at Point A, and Frame 3 is at Point C, Frame 2 must be pushed toward Point B.

- **Orthogonal Projection:** We split the model's noise prediction into "Motion" (parallel to the trajectory) and "Jitter" (perpendicular). We wanted to keep the motion and delete the jitter.

---

## 2. Timeline of Execution and Failures

### Phase 1: The "Deep Fried" Era

**The Code:** We wrote a function that calculated `trajectory_correction = latents_smooth - latents`. We directly subtracted this correction from the model's `noise_pred`.

**The Result:** The videos exploded into high-contrast colors, random static, and "deep-fried" artifacts.

**The Diagnosis (Unit Mismatch):** We made a physics error. `latents` represent **Position** (where the pixels are). `noise_pred` represents **Velocity** (where the pixels should go). By subtracting a position error directly from a velocity vector, the numbers became astronomically large. It was the equivalent of realizing your car is 5 meters off-center and instantly setting your steering wheel to "5 meters" instead of turning it 5 degrees.

---

### Phase 2: The "Frozen World" Era

**The Code:** We implemented **Safe Normalization**. We updated the code to calculate the standard deviation of both the noise and the error. We scaled the correction so it never exceeded ~15% of the noise prediction's magnitude. We also added a Velocity Gate so static backgrounds wouldn't cause division-by-zero errors.

**The Result:** The mathematical explosions stopped. The videos were clean, but absolutely nothing moved.

**The Diagnosis (Averaging = Braking):** Our smoothing logic calculated `(Frame 1 + Frame 2 + Frame 3) / 3`. However, motion *is* deviation. When a tiger's leg moves forward, it mathematically deviates from the previous frame. By forcing frames to look like the average of their neighbors, we mathematically punished deviation, effectively deleting the motion.

---

### Phase 3: The Debugging & "Muddy Trash" Era

**The Code:** To figure out why the RLC was still generating "trash" artifacts on certain settings while the Baseline (no RLC) was completely static, we wrote `apply_retinal_constraint_debug` to print out the exact forces being applied at every timestep (`t=999` to `t=0`).

**The Result:** The logs proved our normalization math was perfect (`Ratio: 0.20`), but the visual output was still terrible.

**The Diagnosis (The Breakthrough):** We discovered two fundamental structural flaws in our approach:

1. **The Noise Trap (Why RLC was trash):** At step `t=999`, the "image" is just pure Gaussian noise. Our algorithm was smoothing random noise, which just creates "blurry noise." By forcing the model to adhere to this smoothed noise, we were instructing the UNet to sculpt the image out of static. We were constraining the foundation before the house was built.

2. **Frame Starvation (Why Baseline was static):** To save GPU memory, we had set `num_frames = 8`. However, the AnimateDiff v1.5 adapter is explicitly trained on 16-frame clips. When we fed it 8 frames, its internal time-awareness broke. It squashed the timeline, defaulting to a static image because it didn't think it had enough "time" to execute a motion.

---

## 3. The Pivot & New Architecture: "Gradient Consensus"

**Conclusion from Phase 3:** Our core hypothesis — constraining the Latents (the image pixels) — was structurally flawed because it directly fought the diffusion denoising process.

**The New Strategy:** Instead of forcing the image pixels to average out, we decided to smooth the *Instructions* the model was giving.

We threw away the complex Trajectory/Orthogonal math and implemented a much simpler, safer approach:

1. **Upgraded to 16 Frames:** We fixed the static baseline by feeding the model the `num_frames=16` it required, using VAE Slicing and Tiling to ensure it fit into an 8GB VRAM GPU.

2. **`apply_gradient_smoothing`:** We wrote a new function that applies a temporal blur `[1, 2, 1]` exclusively to the `noise_pred` (the gradient update), not the latents.

**How it works:** If Frame 1 wants to color a pixel Red, and Frame 2 wants to color it Blue, Gradient Consensus forces them to agree on Purple for that specific timestep's update. This mathematically prevents flickering (inconsistent updates) while freely allowing the image to move across the screen. We also applied a schedule to shut off the constraint in the final 20% of steps so fine textures (like fur) could resolve without being blurred.

---

## 4. Quantitative Evaluation of Gradient Consensus

### Phase 4: The "Contrasting Colors" Era

**The Code:** We deployed `apply_gradient_smoothing` with a `[1, 2, 1]` kernel at `strength=0.3`, active for the first 80% of timesteps. We also added **Variance Normalization** — after smoothing, we rescaled the output so the standard deviation matched the original `noise_pred`, to prevent the mathematical averaging from shrinking the signal magnitude (which was causing hyper-saturated, high-contrast artifacts).

**The Result:** The deep-fried explosions stopped, but visual inspection was ambiguous — the tiger was visible and moving, but the colors still looked unnatural. We could no longer trust our eyes.

**The Fix — Objective Metrics:** We introduced two quantitative metrics to replace subjective visual inspection:

- **SSIM (Structural Similarity Index):** Measures how structurally similar Frame N is to Frame N+1. Higher = more temporally stable.
- **PTD (Pixel-wise Temporal Difference):** Measures the average absolute pixel change between consecutive frames. Lower = less random flickering.

**The Multi-Prompt Evaluation:** We ran the algorithm across **4 different prompts** (tiger in jungle, dog on beach, car on highway, person in snow) to ensure the results weren't prompt-specific.

**The Numbers:**

| Prompt | Base SSIM | RLC SSIM | Base PTD | RLC PTD | Winner |
|---|---|---|---|---|---|
| Tiger running through jungle | **0.6133** | 0.4272 | **12.90** | 22.63 | BASE |
| Golden retriever on beach | **0.6586** | 0.4505 | **10.74** | 17.49 | BASE |
| Car driving highway sunset | **0.8313** | 0.7656 | **6.32** | 8.50 | BASE |
| Person walking snowy forest | **0.7454** | 0.5756 | **9.25** | 11.98 | BASE |

**The Diagnosis (Gradient Contamination):** The baseline won on **every prompt, on both metrics**. The RLC made videos *less* temporally stable, not more. The root cause: by blending Frame 2's noise prediction with Frame 1's and Frame 3's predictions at `strength=0.3`, we were **contaminating** each frame's specific update instructions with instructions meant for neighboring frames. The model was receiving a *compromise* of three different sets of directions — producing frames that didn't match what *any* of the three predictions intended.

---

## 5. Phase 5: Anomaly-Based Selective Smoothing

### The Hypothesis Revision

Since blind smoothing was destructive, we hypothesized that a **targeted** approach could work: instead of smoothing every pixel's prediction, detect which spatial locations have anomalously inconsistent predictions across adjacent frames (actual flickering spots) and only smooth those. Leave everything else completely untouched.

### The Implementation

`apply_anomaly_rlc` was built with three safeguards:

1. **Anomaly Detection:** Compute the temporal difference of `noise_pred` between adjacent frames. Only flag locations where the difference exceeds **2× the median** — these are the statistical outliers (flickering pixels).
2. **Strict Schedule:** Only active between 40% and 90% of the diffusion process (skip noise phase + detail phase).
3. **Conservative Strength:** Maximum `0.1` (vs. the previous `0.3`) with a **cosine decay** within the active window.

### The Multi-Prompt Evaluation

| Prompt | Base SSIM | RLC SSIM | Base PTD | RLC PTD | Winner |
|---|---|---|---|---|---|
| Tiger running through jungle | 0.6133 | 0.6112 | 12.90 | 12.97 | BASE |
| Golden retriever on beach | 0.6586 | 0.6553 | 10.74 | 10.84 | BASE |
| Car driving highway sunset | 0.8313 | 0.8302 | 6.32 | 6.34 | BASE |
| Person walking snowy forest | 0.7454 | 0.7416 | 9.25 | 9.36 | BASE |

**The Result:** The algorithm was no longer destructive — differences were less than 0.5% — but it was also completely **inert**. The baseline still won on every metric by a marginal amount.

**The Diagnosis (Redundancy):** AnimateDiff already contains purpose-built **temporal attention layers** (motion modules) that were specifically *trained* to enforce frame-to-frame consistency. Our external smoothing was either:
- **Too aggressive** → overrode the model's learned temporal behavior → destroyed the video
- **Too conservative** → the model's own temporal attention already handled consistency → our intervention was redundant

There is no "sweet spot" because the model's internal modules already occupy the correct operating point.

---

## 6. Final Conclusions

### Conclusion 1: The RLC Hypothesis Is Structurally Incompatible with Diffusion Models

Every variant of external temporal constraint — whether applied to **Latents** (Phases 1–3) or to **Noise Predictions** (Phases 4–5) — either degraded or had zero effect on video quality. The fundamental issue is that diffusion models learn a precise mapping from noise to image across timesteps. Any external mathematical intervention disrupts this learned mapping. The model's internal temporal attention modules are already optimized for this task.

### Conclusion 2: The "Sweet Spot" Does Not Exist

We systematically swept the parameter space:

| Strength | Target | Schedule | Result |
|---|---|---|---|
| 0.15–0.20 | Latents | Full | Deep fried / Frozen |
| 0.15 | Latents (Orthogonal) | t=200–800 only | Muddy trash |
| 0.30 | Noise Pred (Global) | First 80% | Worse than baseline (all metrics) |
| 0.10 | Noise Pred (Anomaly-only) | 40–90% + Cosine | Identical to baseline |

### Conclusion 3: Proven Alternatives Exist

Techniques that work *with* the model's architecture rather than against it have been shown in literature to genuinely improve temporal consistency:

- **FreeInit** (built into `diffusers`): Re-initializes latent noise using low-frequency information from a first pass, giving the model a "warm start" on temporal structure without modifying any noise predictions.
- **Post-generation stabilization:** Optical flow-based video stabilization or frame interpolation (RIFE/FILM) applied after the video is fully generated.
- **Architecture-level improvements:** Newer models (AnimateDiff v3, SVD, CogVideoX) use improved temporal attention that natively reduces flickering.

### Conclusion 4: What We Learned

The experiment, while producing a negative result for the RLC hypothesis, provided valuable insights:

1. **Unit awareness matters:** Latent-space values and noise predictions operate on fundamentally different scales — mixing them without normalization causes catastrophic failure.
2. **Averaging kills motion:** Temporal smoothing and motion are mathematically opposing forces. Smoothing enforces similarity; motion requires difference.
3. **Quantitative metrics are essential:** Visual inspection of AI-generated video is unreliable. SSIM and PTD provided objective proof that contradicted visual intuition.
4. **Respect the model's learned behavior:** Pre-trained diffusion models have internally learned temporal dynamics. External constraints that conflict with this learned behavior are counterproductive.
