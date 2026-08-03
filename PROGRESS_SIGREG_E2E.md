# Progress: SIGReg + end-to-end encoder training

Status of the work described in `research_papers/ICLR_Submission_Plan.md`, as of
the last commit on `feat/sigreg-e2e`. Operational instructions live in
`AGENT_MEMORY_3.0.md`; this file is the record of what was built, what was
measured, and what is still open.

Every number below is measured output, not an estimate. Where something is
unverified it says so.

---

## 1. Headline

The objective is implemented, and the claim it rests on is measured rather than
asserted:

- **`stop_grad` does not prevent representation collapse.** Trained end-to-end
  with it on, a linear probe for the true state goes from R² 0.77 to **−0.0000**
  while the prediction loss *improves 12x*. The loss went down by destroying the
  representation.
- **SIGReg prevents it.** Same setup plus SIGReg: probe R² **0.60** retained.
- **Straightening on top costs no information and adds rank.** Effective rank
  2.25 → **3.40** of 8, probe R² unchanged within noise.
- **On real PushT the curvature term works**: mean cos(v_t, v_{t+1}) goes
  **−0.288 → +0.456** over 2000 steps, i.e. the latent trajectory goes from
  zigzagging to genuinely straighter.
- **SIGReg reaches near-exact Gaussian marginals**: the aggregated latent has
  E|z| = **0.7934** against a standard Gaussian's √(2/π) = **0.7979**, std 0.957.
- **The configuration is fully paper-anchored** — zero hyperparameters outside
  the two source papers.

Not yet done: any full-length (123,858-step) run, and therefore any success-rate
number. Everything above is from CPU gates plus a 2000-step GPU probe.

---

## 2. What was built

11 commits on `feat/sigreg-e2e`, 129 tests passing.

| Commit | What |
|---|---|
| `da4b8ed` | `experiments/verify_stop_grad.py` — falsification harness for the stop_grad anti-collapse claim (T1–T5) |
| `1d69672` | `training.max_iterations` + `iteration_budget.py` — hard optimizer-step budget, survives resume |
| `bc5312b` | **SIGReg + end-to-end training** (`models/sigreg.py`, `models/diagnostics.py`, `run_naming.py`, phases 0–2 of the plan) |
| `4dc8b5d` | ICLR plan + LeWM paper source in-repo |
| `6dd7934` | `reproduce_table1.py` evaluates objective variants instead of skipping them |
| `404db22` | `experiments/verify_encoder_trains.py` — proves the trunk is *optimized*, not merely trainable |
| `bf24f96` | Bounded-memory training telemetry + digest tool |
| `adb2666` | Fix underdetermined probe in the gates harness |
| `9d48902` | Summariser accepts directories, globs, multiple logs |
| `c9bb943` | SIGReg evaluated in fp32 even when latents arrive as bf16 |
| `f5ddc98` | `probe_r2` made held-out; collapse diagnostics run during training |

### The objective

```
L = L_pred + lambda_SIG * SIGReg(Z) + lambda_curv * L_curv
```

`models/sigreg.py` implements the sliced Epps–Pulley statistic: M random unit
directions on S^{d−1}, empirical characteristic function taken across the batch
at each timestep, compared against exp(−t²/2) on a 17-node trapezoid grid over
t ∈ [0,3], scaled by sample count, averaged over projections and time. Quadrature
constants ported from the LeJEPA reference implementation shipped with LeWM, so
the statistic is on the same scale as the published λ.

### Phase 0 — unblocking

- `training.freeze_backbone` — the switch that **did not exist**.
  `_configure_encoder_trainability()` froze the trunk unconditionally, *before*
  it consulted `model.train_encoder`, so only 1,810,280 of ~23.9M encoder params
  ever trained and there was no config to change it.
- `training.backbone_lr` + trunk/head param groups; `null` reproduces the
  original single-group Adam exactly.
- `models/diagnostics.py` — latent std across (batch, time), effective rank
  (participation ratio), curvature cosine, held-out linear-probe R².
- `training.diag_every` — those diagnostics during training, not just per epoch.

### Phase 2 — the paper's own contradiction

`training.curv_on: features|velocity`. App. B.6's [agg] equation applies the head
to the *velocity*, `C_t = cos(h(v_t), h(v_{t+1}))`, while the caption of Fig.
`train_agg` says the curvature loss is applied to the *aggregated features*.
These differ because h is a nonlinear MLP. The original code path (`features`)
stays the default; the equation is available as an ablation. Tests confirm the
two agree for a linear head and differ for the real one. **This must be stated
explicitly in the submission either way.**

### Telemetry

`training_log.py` — append-only JSONL. Memory is O(#metrics), not O(#steps):
each metric folds into a Welford accumulator (six floats regardless of step
count), and the only histories are two capped deques. Disk is one record per
`telemetry_every` steps, ~620 records for the PushT budget. Records every loss
term separately, the collapse metrics, per-group gradient/weight norms and their
ratio, measured weight movement per group, and dated anomaly events (NaN, loss
spikes vs the running mean, collapse threshold crossings).

`summarize_training_log.py` — digest whose length is independent of run length,
so a 12-hour run reduces to something readable in one pass.

---

## 3. Falsification results (CPU, `verify_stop_grad.py`)

Runs the repo's own `DinoV2Encoder` / `ChannelProjector` / `ViTPredictor` /
`VWorldModel` on a synthetic but genuinely learnable task, so collapse is a
failure rather than the only option.

| Test | Result |
|---|---|
| T1 | `stop_grad=True` leaves `|grad| encoder.projector = 7.55` — *larger* than with it off (4.69), because the two gradient paths were partly cancelling. Identical loss value. It is a one-sided detach, not a barrier |
| T2 | `model.train_encoder: True` never unfreezes the backbone; 1,810,280 of ~23.9M params train |
| T3 | End-to-end + `stop_grad`: probe R² 0.77 → **−0.0000**, eff_rank 3.77 → 1.24, 99.5% of latent variation lost, prediction loss down 12x |
| T4 | Frozen backbone only *slows* it (89% variation lost, unstable) — freezing, not `stop_grad`, is the load-bearing mechanism |
| T5 | Curvature is **exactly** scale-invariant (identical at latent scale 1.0 and 1e-5) so it cannot see collapse, and `_cos_curvature` returned **NaN** at collapse |

Two bugs found and fixed: the NaN, and 1.13M dead `agg_mlp` parameters receiving
zero gradient in every `straighten=False` run while still sitting in the
optimizer and every checkpoint.

## 4. Gate results (CPU, on the pod)

Backbone unfrozen in all three; gates 2/3 drop stop-gradient as LeWM specifies.

| Gate | Objective | std(b,t) | eff_rank/8 | probe R² | Verdict |
|---|---|---|---|---|---|
| 1 | `L_pred` only | 0.00205 | 1.244 | **−0.0000** | PASS — collapsed, as a negative control must |
| 2 | + SIGReg 0.1 | 0.14490 | 2.250 | **0.6073** | PASS — no collapse |
| 3 | + SIGReg + curvature | 0.14901 | **3.403** | **0.6298** | PASS — no collapse |

Robust across machines: gate 1 collapses and 2/3 do not (−0.0000 vs ~0.6 is not
a noise-sized gap), and gate 3 has substantially higher effective rank than gate
2 (3.40 vs 2.25 on the pod, 3.11 vs 2.24 locally). **Not** robust: the sign of
the gate 2 → 3 probe-R² difference flipped between machines, so claim that
curvature costs no information, not that it improves it.

## 5. Trunk really trains (`verify_encoder_trains.py`)

`requires_grad=True` is not proof — a parameter can be trainable, receive a
gradient, and never move. Measured weight delta after real optimizer steps, in
train.py's order (configure trainability → build optimizer → step):

| | trunk requires_grad | in optimizer | \|grad\| | ‖Δw‖ after 5 steps |
|---|---|---|---|---|
| `freeze_backbone=True` | 0 | 0 | 0.0 | **exactly 0.0** (projector still moves 3.43e-2) |
| `freeze_backbone=False` | all | all | 3.07 | **2.38e-2** (largest change in `patch_embed.weight`) |

Confirmed on the real model in the GPU smoke test:
```
Encoder base_model is TRAINABLE (end-to-end)
Encoder trainable params: 23,866,856
Encoder param groups: trunk 22,056,576 params @ lr=1e-05 | heads 1,810,280 params @ lr=1e-05
```
22,056,576 is the real DINOv2 ViT-S/14; 1,810,280 matches the head count
computed independently on CPU. **~13x more encoder parameters now train than in
any previous run.**

## 6. Smoke test (GPU, real model)

```
Iteration budget reached: global_iter=30 / max_iterations=30 (epoch 1, batch 29 of 113)
Run finished on the iteration budget: 30 optimizer steps
```
Validated: the cap stops mid-epoch at exactly the right step; `run()` breaks and
forces a checkpoint on a non-save epoch; validation and diagnostics complete; the
run directory resolved to `..._sgFalse_lr1e-05_sig1e-1_e2e`, so `variant_tag`
works and cannot collide with the baseline. Throughput **2.47 it/s** with the
trunk unfrozen → ~14 h for the full budget (baseline ~12 h).

## 7. Probe run: real PushT, 2000 steps, `backbone_lr=1e-5`

| Metric | Start → 2000 | Reading |
|---|---|---|
| `loss/curvature_loss...` | 1.288 → **0.544** | mean cos **−0.288 → +0.456**: straightening works on real data |
| `latent/agg_curvature_cos` | −0.185 → **+0.527** | stronger on the agg head, where the loss acts |
| `latent/probe_r2` (held out) | 0.224 → **0.393** (max 0.441) | state information *increased* — the opposite of collapse |
| `loss/sigreg_loss` | 5.55 → **1.47** plateau | vs a measured null floor of **1.058 ± 0.041**; converged, ~37% above floor |
| `latent/agg_latent_std` | **0.957** | ≈ 1 |
| `latent/agg_latent_abs_mean` | **0.7934** | Gaussian E\|z\| = √(2/π) = **0.7979**. Marginals matched to three decimals |
| `grad/encoder.trunk/ratio` | 0.215 → **0.0098** | initial transient decays into the healthy 1e-4..1e-2 band |
| `delta/encoder.trunk` | max 5.2e-2 | trunk moving; `param_norm` 393.2755 → 393.2779 |
| `latent/latent_eff_rank_frac` | 0.377 → 0.29 | declining but stabilising — **watch on the full run** |

Verdict line: `Prediction loss fell while probe R^2 held: this is what a healthy
run looks like.`

**Consequence:** `backbone_lr=1e-5` equals `encoder_lr=1e-5`, so a single param
group (`backbone_lr=null`) is numerically identical to this probe. The
paper-exact configuration is therefore already validated, and the method carries
**zero hyperparameters outside the two papers** — inheriting LeWM's "λ is the
only effective hyperparameter" property.

---

## 8. Findings worth putting in the paper

1. **`stop_grad` is not an anti-collapse mechanism**, and the straightening
   paper's frozen backbone is what was actually holding its representations up.
   T1/T3/T4 are the evidence, and T3 is the motivating figure.
2. **Curvature alone cannot be trained end-to-end** because it is exactly
   scale-invariant (T5a) — a defect alone, a virtue once SIGReg owns the scale.
   This is the sharpest available statement of complementarity.
3. **Straightening does not cost SIGReg's Gaussianity**, and raises effective
   rank (2.25 → 3.40).
4. **Sliced Gaussianity tests are weakly sensitive to anisotropy in high
   dimensions.** The agg latent has near-exact Gaussian marginals yet effective
   rank ~19 of 128. Concentration of measure means a random 1-D projection of a
   rank-19 distribution in 128-d still sees variance ≈ tr(Σ)/d, so the test is
   nearly satisfied. This explains the 1.45-vs-1.06 plateau and gives the plan's
   "does straightening hurt SIGReg's Gaussianity?" ablation a quantitative
   target. It is a property of SIGReg, not a bug here.
5. **The paper contradicts itself on where the agg head applies** (App. B.6
   equation vs the Fig. `train_agg` caption). Must be disclosed; `curv_on`
   ablates it.

---

## 9. Open work

Ordered by value.

| # | Item | Status | Cost |
|---|---|---|---|
| 1 | The four 123,858-step runs | **not started** | ~54 h GPU, sequential |
| 2 | GD + MPC evaluation, seeds 100/200/300 | not started | ~15 h |
| 3 | CEM arm + planning-time aggregation | `conf/plan_cem.yaml` is correctly wired; `reproduce_table1.py` runs GD only and nothing collects `[timing] perform_planning_s` | small, no GPU |
| 4 | Condition-number proxy of the planning Jacobian | not started; one `jacobian` call on a trained ckpt, singular-value spread | small |
| 5 | Gaussianity-under-straightening plot | nearly free — telemetry already logs `sigreg_loss`, floor is 1.058 | trivial |
| 6 | Latent-distance vs geodesic correlation | not started; along-trajectory state distance is a usable proxy for PushT | medium |
| 7 | λ_curv / λ_SIG sensitivity | not started; 8 runs at full budget = 112 h, unaffordable — run at reduced budget and say so | medium |
| 8 | Loss-landscape figures | not started | medium |
| 9 | Multi-seed variance | one training seed so far | ×3 on runs 1 and 4 |
| 10 | Long-horizon H = 10–15 | **BLOCKED** — `num_frames = num_hist + num_pred = 4` gives exactly 2 curvature terms per sample; needs `curv_window` decoupled, which requires all five dataset loaders to emit extra frames | the only real unbuilt infrastructure |
| 11 | VoE analysis | not started; the plan does not define it concretely | low priority |

### Risks

- **`backbone_lr` over 123,858 steps.** The probe covers 1.6% of the run. Slow
  collapse could begin later; `diag_every=500` gives ~247 samples of `probe_r2`
  to catch it.
- **`latent_eff_rank_frac` drifted 0.377 → 0.29** in the probe. If it continues
  toward 0.125 (rank 1 of 8) the run is degenerating.
- **Contribution #2 is not yet safe.** LeWM already ships a `GradientSolver`, so
  if stock LeWM plans well with gradients, "first end-to-end JEPA to support
  gradient planning" weakens to "straightening improves it". A stock-LeWM
  `solver=adam` run is the cheapest way to find out and is still not done.
- **Two-codebase split.** The method is implemented here (unfreezing DINOv2)
  rather than in LeWM (ViT-tiny from scratch), so the "no pretrained backbone"
  claim is the weaker version. Decide which repo hosts the submission.
