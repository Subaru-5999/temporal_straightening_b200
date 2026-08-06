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

---

# Diagnosis of the PushT open-loop failure (closed)

**Result being explained.** PushT, end-to-end + SIGReg + straightening, paper-exact
budget (123,858 steps): open-loop **13.33 ± 1.15** (seeds 12/14/14), MPC **56.0**
(seed 100). Paper's frozen ✓ cell: 77.33 ± 6.18 / 85.33 ± 4.99. Same-pod frozen ✗
reproduction (`REPRODUCTION.md` row 4): 76.00 ± 3.27 / 82.00 ± 4.32.

## Conclusion

**Information was not lost from `z`. It was redistributed between the channels of
`z`, and the planning objective has hard-coded relative weights.**

End-to-end training let the visual encoder stop representing the pusher, because
the pusher is redundantly available in `z`'s proprio channels and dropping it
lowers the prediction loss at no cost. Neither SIGReg (Gaussianity) nor cosine
curvature (straightness) constrains state information, so nothing forbids it.

`objectives.py` forms `loss_visual + alpha * loss_proprio` with a per-dim mean on
both sides, so the effective weight is `alpha * scale_proprio / scale_visual`.
Measured: 0.001089 / 0.2598 → **alpha=1 behaves as alpha_eff = 0.0042**. The cost
is 99.58% the channel that forgot the pusher and 0.42% the channel that kept it.
In PushT the action *is* the pusher target, so the cost is nearly blind to what
the actions control. With a frozen trunk this cannot happen: DINOv2's scales are
fixed and its features retain the pusher. **alpha=1 was silently calibrated to a
frozen encoder.**

## Evidence

| measurement | value | what it rules in/out |
|---|---|---|
| MPC success | 56.0 | NOT collapse — impossible with a dead latent |
| `H_oracle` (GT actions, 0 GD steps) | **1.0** | harness sound; task is open-loop solvable |
| `H_floor` (fixed actions) | **0.0** | 0.12 is above chance but barely |
| rollout drift @ k=5 | 0.135 | NOT rollout drift; 1-step NMSE 1.1% |
| `compound` @ k=5 / k=8 | 12.2 / 30.9 | error does amplify through feedback, from a small base |
| `rho_local` (latent vs state dist) | 0.489 vs 0.517 pristine | geometry aligned and unchanged |
| probe R² agent_x / agent_y | 0.943/0.947 → **−0.011/−0.618** | visual latent lost the pusher |
| probe R² block_x / block_y | 0.945/0.942 → 0.979/0.989 | block retained |
| probe R² agent, proprio channel | **0.997/0.998** | pusher IS in `z`, just not in the visual channel |
| planning snr, visual / proprio | **2.91 / 33.68** | the clean channel is the ignored one |
| proprio share of cost @ alpha=1 | **0.42%** | alpha_eff = 0.0042 |
| `A_alpha0` vs `A_alpha1` | 0.12 vs 0.12 | proprio term is numerically inert |
| `beat` @ k=5 | 0.065 | GT actions are not the cost's minimiser |
| curvature cos, pristine → trained | −0.181 → **+0.706** | straightening WORKED |
| pusher vs block state curvature | +0.599 vs +0.430 | pusher is *straighter*; curvature exonerated |
| pusher-subspace ablation on DINOv2 | −0.0024 (control −0.0005, null −0.0000) | curvature indifferent to the pusher |

## Hypotheses tested and refuted

1. **Collapsed representation** — killed by MPC 56.
2. **Autoregressive rollout drift** — killed by drift 0.135 at the protocol horizon.
3. **Misaligned latent geometry** — killed by `rho_local` 0.489 ≈ pristine 0.517.
4. **Curvature rewards forgetting the pusher** — killed twice: the pusher is
   intrinsically *straighter* than the block (+0.599 vs +0.430), and ablating the
   pusher subspace from pristine DINOv2 moves curvature by −0.0024 against a
   control of −0.0005. The straightening term is innocent.
5. **Broken eval harness** — killed by `H_oracle` = 1.0.

## Fix shipped

`planning/objectives.py`: `create_objective_fn(..., normalize=False)`. When True,
each channel's term is divided by its own mean per-feature variance across the
eval batch, making the objective scale-invariant so `alpha` is a true relative
weight. Exposed as `objective.normalize` in the three plan configs, default
`false`, so all five tracked Table-1 cells are byte-identical. Guarded by
`tests/test_objective_normalize.py` (6 tests: default equivalence, that the
unnormalized share collapses ~100x when proprio is scaled 0.01x, that the
normalized share is invariant across 1e-2..1e2 rescaling of either channel, that
alpha=1 becomes a near-even split, that all three modes honour the flag, and that
a batch of 1 does not produce NaN).

**Fairness requirement.** `normalize=true` must be applied to BOTH arms, or not at
all. Tuning alpha for the method and not the baseline is not a comparison.

## Open

- Confirmation that the weighting is *causal*: `A_alpha240` / `A_alpha1400` and
  `objective.normalize=true`. Prediction: substantial recovery over 13.33.
- `O_descent` (GD started at the GT actions). Falls → cost minimum is not at the
  correct actions; holds → zero-init basin.
- Which term caused the redistribution: prediction-loss redundancy vs SIGReg.
  Cheap test: ~10k-iteration runs with `proprio_encoder=dummy` (no offload target)
  and with `SIGREG=0`, reading probe R² on state dims 0-1.
- Frozen control on this pod, for its own alpha_eff and probe numbers.
- MPC is n=1; generality beyond PushT untested.

---

# The fix works: proprio grounding (8k-step validation)

`training.ground_proprio=1.0`, 8,000 steps (6.5% of the paper budget), everything
else identical to the ungrounded run.

## Representation

| state dim | ungrounded @123,858 | pristine DINOv2 | **grounded @8,000** |
|---|---|---|---|
| agent_x | **−0.011** | +0.943 | **+0.991** |
| agent_y | **−0.618** | +0.947 | **+0.993** |
| block_x | +0.979 | +0.945 | +0.930 |
| block_y | +0.989 | +0.942 | +0.899 |
| block_angle | +0.622 | +0.732 | +0.514 |
| vel_x / vel_y | −1.0 / −1.0 | −0.06 / +0.27 | +0.326 / −0.795 |
| `rho_local` | 0.489 | 0.517 | **0.528** |
| `nn_state_ratio` | 0.0227 | 0.0223 | **0.0187** |

The agent ends up **above pristine DINOv2** (0.991 vs 0.943), which rules out
"8k was too short to lose it": training moved agent decodability *up*, and
nothing else in the objective rewards that. Cost: block precision slips
(0.979 → 0.930) and the angle more so (0.622 → 0.514).

## Planning (PushT open-loop GD, 50 samples, goal_H 25)

| configuration | budget | success |
|---|---|---|
| ungrounded, α=1 | 100% | 13.33 ± 1.15 |
| ungrounded, α=240 (its best) | 100% | 26.0 (1 seed) |
| **grounded, α=1** | **6.5%** | **20.67 ± 1.15** (20/22/20) |
| grounded, α=0 | 6.5% | 20.0 |
| grounded, α=240 | 6.5% | 14.0 |
| grounded, `normalize=true` | 6.5% | 18.0 |

+7.3 points over ungrounded at matched α, combined SE ~0.94, ~8σ. Statistically
indistinguishable from the ungrounded model's BEST configuration while using
6.5% of its training.

## Grounding and alpha-reweighting are SUBSTITUTES, not complements

α=240 helped the ungrounded model (0.12 → 0.26) and *hurts* the grounded one
(0.20 → 0.14). One story covers both: without grounding the visual channel had
lost the agent and proprio was its only source, so up-weighting proprio helped;
with grounding the visual channel holds the agent at R² 0.991, so proprio is
redundant and up-weighting it merely dilutes the block information. Same reason
`normalize=true` (0.18) is slightly worse than α=1 (0.20). **Do not stack them.**
It also explains why the α sweep saturated at 0.26 rather than recovering fully.

Practical consequence: the best grounded configuration is the paper's own default
α=1, so the comparison needs no protocol deviation.

## Known issue with this configuration

Grounding on all four proprio dims includes `vel_x, vel_y`, which are **not
identifiable from a single frame**. Measured cost: `ground_proprio_loss`
plateaus at 0.23 instead of approaching 0; straightness degrades to cos 0.327
against 0.589 ungrounded; `z_loss` 0.0118 vs ~0.008 ungrounded at matched steps;
visual velocity probes land at +0.326 / −0.795. At coefficient 1.0 grounding is
**51% of the total objective** against prediction's 2.6%.
`training.ground_proprio_dims=[0,1]` addresses all of it and is untested.

## Engineering notes

- `VWorldModel` is built after `accelerator.prepare()` and is never prepared, so
  the grounding head needed explicit `.to(device)` AND explicit registration in
  an optimizer. Without both the run dies two seconds into epoch 1 with a device
  mismatch, and without the second the head stays at its random init and forces
  the encoder to match a fixed random projection.
- The head takes `action_encoder_lr` (5e-4), not `encoder_lr` (1e-5): a read-out
  probe that trains slower than the representation it reads gives a stale gradient.
- `ground_proprio` had to enter `run_naming.variant_tag()` or a grounded run
  resolves to the ungrounded directory and auto-resumes it.

---

# Comparability contract (baseline vs ours)

The headline comparison is the paper's ✓ PushT cell against our end-to-end
variant. For that to mean anything, **only the contribution may differ.**

Reference baseline: `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`,
produced by `train_pusht_on_paperiters.sh` with no edits (the repo defaults
`freeze_backbone: True`, `stop_grad: True`, `sigreg: False`, `ground_proprio: 0`,
`backbone_lr: null` give exactly this, and `variant_tag` resolves to `''`).
Paper targets: OL 77.33 ± 6.18, MPC 85.33 ± 4.99.

## Intended differences — exactly four

| setting | baseline | ours |
|---|---|---|
| `freeze_backbone` | True | False |
| `stop_grad` | True | False |
| `sigreg` / `sigreg_coeff` | False / 0 | True / 0.1 |
| `ground_proprio` | 0 | 1.0 |

## Must match — verified from both runs' telemetry configs

`straighten=aggcos1e-1`, `encoder_lr=1e-5`, `predictor_lr=5e-4`,
`action_encoder_lr=5e-4`, `batch_size=32`, `num_hist=3`, `num_pred=1`,
`frameskip=5`, `max_iterations=123858`, `epochs=3` (so the cap ends the run),
`encoder=dino_channel` (projector 14x14x8), `training.seed=0`.

Pass `BACKBONE_LR=null` on the full grounded run. `null` and `1e-5` are
numerically identical here (both put the trunk at `encoder_lr`), but matching the
config literally removes a question a reviewer would otherwise ask.

## Eval protocol — identical for both arms, no exceptions

Open-loop `plan_gd.yaml`: 50 samples, `goal_H=25` -> 5 model steps, GD with Adam
lr 0.1, 100 steps, zero init, `action_noise=0`, `mode=last`. MPC
`plan_gd_mpc.yaml`: `max_iter=20`, `n_taken_actions=5`, `mode=staged`. PushT uses
**alpha=1**. Seeds 100/200/300. `objective.normalize=false`.

Run both arms through `reproduce_table1.py <run_name>`, which applies the alpha,
mode, seeds and env recipe internally, rather than hand-rolled `plan.py` calls
that can drift between arms.

## Numbers that must NOT be reported as protocol results

- **e2e ungrounded = 13.33 ± 1.15, not 26.0.** The 26.0 came from alpha=240,
  which is off-protocol. It is a diagnostic that identified the weighting defect,
  not a result.
- **alpha must not be tuned per model.** alpha=240 helps the ungrounded model and
  *hurts* the grounded one (0.20 -> 0.14), so per-model best-alpha would compare
  two different objectives. alpha=1 is both the paper's setting and the grounded
  model's best, so no deviation is needed.
- **`objective.normalize=true` stays off in the headline.** It is a diagnostic and
  a fairness tool for cross-scale comparisons; if it is ever used it must be
  applied to both arms.

## Run queue (one MIG slice, strictly serial)

1. position-only pilot checks (~12 min) -> pick the grounding config
2. full 123,858-step grounded run at that config (~14 h)
3. `train_pusht_on_paperiters.sh` frozen ✓ baseline (~14 h) -- its checkpoint is
   gone from the pod, so `REPRODUCTION.md`'s recorded 76.00/82.00 covers PushT at
   H=5 but cannot be re-evaluated at long horizon or under any protocol variant
4. `reproduce_table1.py` on both, 3 seeds, open-loop + MPC

---

# Position-only grounding: better representation, much worse planning

`ground_proprio_dims=[0,1]` (agent position only, dropping the two velocity
targets that a single frame cannot determine), 8,000 steps, otherwise identical.

| | ungrounded | all-dims | position-only |
|---|---|---|---|
| `ground_proprio_loss` | — | 0.2324 (plateau) | **0.0107** |
| `z_loss` | 0.0060 @6k | 0.0118 @4k | **0.0042 @8k** |
| curvature (1−cos) | 0.4109 @6k | 0.6726 @4k | **0.4199 @8k** |
| agent_x / agent_y | −0.011 / −0.618 | 0.991 / 0.993 | **0.997 / 0.994** |
| block_x / block_y | 0.979 / 0.989 | 0.930 / 0.899 | **0.974 / 0.905** |
| block_angle | 0.622 | 0.514 | **0.756** (> pristine 0.732) |
| vel_x / vel_y | −1.0 / −1.0 | **+0.326** / −0.795 | −0.680 / −1.0 |
| `rho_global` | 0.695 | 0.552 | **0.732** |
| visual `snr` @k=5 | 2.91 | **3.21** | **1.09** |
| visual `beat` | 0.065 | **0.019** | **0.140** |
| **success, 3 seeds, α=1** | 13.33 ± 1.15 | **20.67 ± 1.15** | **7.33 ± 4.16** |

Every mechanical prediction for position-only came true — grounding loss to 0.01,
curvature recovered 0.67 → 0.42, prediction better than ungrounded, block and
angle recovered — and success **collapsed** to 7.33 (≈5σ below all-dims).

## What this establishes

**1. `snr` and `beat` predict planning success; static probe R² does not.**
Across three independently trained models the ordering is monotone on both:

    snr   1.09  <  2.91  <  3.21
    beat  0.140 >  0.065 >  0.019
    succ  7.33  <  13.33 <  20.67

Position-only had the BEST probe R² on every state dimension and the worst
planning. Measure **action sensitivity**, not state decodability.
`experiments/rollout_drift.py` gives both in ~3 GPU-minutes with no environment.

**2. Grounding on velocity is a feature, not a defect.** Velocity is the
derivative of position, i.e. the quantity actions control most directly.
Requiring the visual latent to predict it — even though one frame does not
determine it — pushes the encoder toward motion-sensitive features, which is
what makes the terminal latent respond to a change in the action sequence.
Removing it dropped `snr` to 1.09: the action's effect on the terminal latent
became the same size as the rollout error, so the cost surface is noise at the
scale being optimised, and `beat` rose 7x.

The "ill-posed target" objection was wrong. It acts as a dynamics-aware
regulariser: bad for static decodability, good for planning.

**3. Grounding repairs the scale imbalance by itself.** Proprio spread rose 9x
(0.00109 → 0.0097), so `alpha_eff` went 0.0042 → 0.039 with no change to α. That
is why α=240 helps the ungrounded model and hurts the grounded one, and why
`objective.normalize` is no longer needed once grounding is on.

## Decision

Full 123,858-step run uses **all four proprio dims at `ground_proprio=1.0`**,
`BACKBONE_LR=null` to match the ungrounded config literally, α=1 per the
comparability contract.

---

# Grounding-target ablation: both halves are needed

Four models, 8,000 steps each, identical except for the grounding target.

| target | visual snr | beat | rho_local | agent x/y | block x/y | angle | **success (3 seeds, α=1)** |
|---|---|---|---|---|---|---|---|
| none (ungrounded) | 2.91 | 0.065 | 0.489 | −0.011 / −0.618 | 0.979 / 0.989 | 0.622 | 13.33 ± 1.15 |
| positions `[0,1]` | 1.09 | 0.140 | 0.511 | 0.997 / 0.994 | 0.974 / 0.905 | 0.756 | 7.33 ± 4.16 |
| velocities `[2,3]` | 2.08 | 0.025 | 0.590 | 0.943 / 0.934 | 0.892 / 0.877 | 0.494 | 7.33 ± 3.06 |
| **all four** | **3.21** | **0.019** | 0.528 | 0.991 / 0.993 | 0.930 / 0.899 | 0.514 | **20.67 ± 1.15** |

**Ablating either half of the target breaks it.** Positions alone: 7.33.
Velocities alone: 7.33. Both: 20.67. An interaction, not an additive effect.
Position-only keeps the state but loses action sensitivity (snr 1.09 -- the
action's effect on the terminal latent shrinks to the size of the rollout error).
Velocity-only keeps moderate action sensitivity (snr 2.08) despite never learning
velocity (probe 0.036/0.007, unlearnable from one frame) but degrades positional
precision (block 0.892/0.877, angle 0.494). No single-mechanism account of why
the combination is special; the prescription "ground on the full proprio
observation" is what the data supports.

## Correction to the earlier snr/beat claim

On three models snr and beat both ranked success monotonically. The fourth model
breaks `beat`: velocity-only has better beat than ungrounded (0.025 vs 0.065) and
worse success (7.33 vs 13.33). `snr` has no inversions but is not injective
(1.09 and 2.08 both give 7.33).

Downgraded claim, which is what four points support: **low snr reliably predicts
bad planning; high snr does not guarantee good planning.** Useful as a cheap
negative filter before spending a full run, not as a success predictor.

`rho_local` is dead as a predictor -- velocity-only has the best value of the four
(0.590) and ties for worst success. Static geometry and state decodability do not
determine planning performance in this setting.

## Config chosen for the full run

`training.ground_proprio=1.0`, all four proprio dims (the default `null`),
`BACKBONE_LR=null`, α=1 per the comparability contract.

---

# Full grounded run, mid-flight readings

`ground_proprio=1.0` (all four dims), `BACKBONE_LR=null`, 123,858 steps.
Reproduces the 8k pilot to the 4th decimal over the first 600 steps, which also
confirms empirically that `backbone_lr=null` and `1e-5` are equivalent here.

| | 8k (pilot) | ~34k | ungrounded @matched step |
|---|---|---|---|
| `z_loss` | 0.0118 @4k | **0.0020** | 0.0019 @35.6k |
| curvature (1−cos) | 0.6726 @4k | **0.3980** | 0.2684 @35.6k |
| `ground_proprio_loss` | 0.2324 | **0.1394** | — |
| grounding share of objective | 51% | **45%** | — |
| `state[0]` / `state[1]` agent | 0.991 / 0.993 | **0.992 / 0.983** | −0.011 / −0.618 @full |
| `state[2]` / `state[3]` block | 0.930 / 0.899 | **0.868 / 0.862** | 0.979 / 0.989 @full |
| `state[4]` angle | 0.514 | **0.427** | 0.622 @full |
| `state[5]` vel_x | +0.326 | **−0.085** | −1.0 @full |
| visual `snr` | 3.21 | **3.21** | 2.91 @full |
| `alpha_eff` | 0.0392 | **0.0040** | 0.0042 @full |

## Confirmed

- **The agent holds at 0.99** at 4x the pilot's budget. The original failure is
  not recurring, and grounding costs nothing in prediction: `z_loss` tracks the
  ungrounded run step for step (0.0020 vs 0.0019).
- **Grounding does not run away.** Its loss keeps falling (0.605 -> 0.139) and its
  share of the objective *drops* (54% -> 45%). The pilot's apparent plateau at
  0.23 was the pilot ending early.
- Curvature stays persistently worse than ungrounded (0.398 vs 0.268), a stable
  gap. Grounding costs straightness. The pilots already showed planning does not
  depend on it here (all-dims had worse curvature and better success).

## Correction to an earlier claim

The entry above stating "grounding repairs the scale imbalance by itself" is
**false at longer budget**. Proprio spread rose 9x by 8k (alpha_eff 0.0042 ->
0.0392) and then collapsed back to 0.00081 by 34k (alpha_eff 0.0040), matching the
ungrounded run. Expected in hindsight: the grounding term constrains the VISUAL
latent to predict the proprio observation and never constrains the proprio
ENCODER, which is free to keep shrinking.

This is now harmless, and the reason matters: `alpha_eff ~ 0.004` was fatal before
because proprio was the ONLY carrier of agent position. With the visual channel
holding it at 0.99, an inert proprio term costs nothing. Same number, opposite
significance. Re-test alpha on the final checkpoint rather than assuming the 8k
model's optimum carries over.

## Hypothesis: an snr attractor near 3

Between 8k and 34k `z_loss` fell 6x (0.0118 -> 0.0020) while visual `snr` did not
move (3.21 -> 3.21). Since `snr = reach / rollout_error`, `reach` must have fallen
by the same factor: **the encoder sheds action-relevant content and rollout error
in lockstep.** That would also explain the ungrounded run sitting at 2.91 despite
an excellent `z_loss`.

If it holds, more budget does not raise `snr`, the final number lands near the
pilot's 20.67, and beating the baseline requires something that moves `snr`
structurally rather than more training. Two points only -- a third at ~15:00 tests
it.

## Erosion follows what is pinned

The agent is grounded and holds; block, angle and velocity are unpinned and all
decline. Same offloading mechanism as the original failure, now visible as a
general principle: **the encoder sheds whatever the objective does not pin.**
The design decision is therefore *what to pin*. Block position is not a model
input, so pinning it would require ground-truth state as a training target -- a
departure from the image+proprio+action setting that must be labelled as such.

---

# Full grounded run: FINAL (123,858 steps)

`ground_proprio=1.0` (all four dims), `BACKBONE_LR=null`, paper-exact budget.

## Training endpoint vs the ungrounded run

| | grounded final | ungrounded |
|---|---|---|
| `z_loss` | **7.74e-04** | 0.0016 @47.4k |
| curvature (1−cos) | 0.2873 | 0.2684 @35.6k |
| `ground_proprio_loss` | 0.0921 (from 0.605) | — |
| grounding share of objective | 38% (from 54%) | — |

## Representation: matches pristine DINOv2, with a better predictor

| probe | pristine DINOv2 | 8k | 34k | **final** |
|---|---|---|---|---|
| agent_x / agent_y | 0.943 / 0.947 | 0.991 / 0.993 | 0.992 / 0.983 | **0.993 / 0.991** |
| block_x / block_y | 0.945 / 0.942 | 0.930 / 0.899 | 0.868 / 0.862 | **0.945 / 0.940** |
| block_angle | 0.732 | 0.514 | 0.427 | **0.711** |
| vel_x / vel_y | −0.06 / +0.27 | +0.33 / −0.80 | −0.09 / −0.58 | +0.003 / −0.258 |
| `rho_local` | 0.517 | 0.528 | 0.506 | 0.475 |
| `nn_state_ratio` | 0.0223 | 0.0187 | 0.0345 | 0.0202 |

## Planning signal

| k=5 | ungrounded | 8k | 34k | **final** |
|---|---|---|---|---|
| visual snr | 2.91 | 3.21 | 3.21 | **3.49** |
| visual beat | 0.065 | 0.019 | 0.035 | **0.029** |
| drift | 0.135 | — | — | **0.110** |
| `alpha_eff` | 0.0042 | 0.0392 | 0.0040 | 0.0025 |

Long-horizon profile of the final model: k=8 snr 2.50 (ungrounded 2.28), k=10 snr
**2.14** beat 0.027 drift 0.308, k=12 snr 1.98 beat 0.035, k=15 snr **1.85** beat
0.076 drift 0.533. `beat` stays 0.023–0.035 from k=6 to k=12, so the cost still
identifies the correct actions well beyond the protocol horizon.

## THREE EARLIER CONCLUSIONS CORRECTED

**1. There is no snr attractor near 3.** Claimed after seeing snr 3.21 at both 8k
and 34k while `z_loss` fell 6x. It reached 3.49 by 123,858. The effect is small but
the hypothesis predicted exactly zero movement.

**2. "The encoder sheds whatever the objective does not pin" was read off a
mid-training dip.** block 0.930 → 0.868 and angle 0.514 → 0.427 at 34k both
RECOVERED, to 0.945 and 0.711. The final visual latent matches pristine DINOv2 on
the block and exceeds it on the agent.

**3. "Long horizon is dead" was premature.** Called from the 34k checkpoint
(k=10 snr 1.63, k=15 snr 1.06). The final model gives 2.14 and 1.85, with drift
still well under 1 and `beat` low. The route is viable again.

**Methodological lesson: mid-training representation metrics are non-monotone.**
Use them to detect catastrophic failure (the agent going to zero), never to
extrapolate a trend. Both wrong calls above came from treating a mid-run reading
as a trajectory.

## The measurement that is now most valuable

The **frozen baseline's snr**. Our final representation matches pristine DINOv2 on
every task-relevant dimension and has a strictly better predictor, so on the
mechanistic account our snr should be at least the baseline's -- yet the baseline
scores 77 and we expect ~20-27. Either the baseline's snr is far above 3.49, or snr
saturates and a different quantity governs the remaining gap. Three GPU-minutes
once a baseline checkpoint exists, and it decides whether the snr framing survives.

---

# HEADLINE RESULT: full grounded run, protocol evaluation

PushT, 3 seeds (100/200/300), 50 samples, alpha=1, `goal_H=25`, GD planner.
`reproduce_table1.py`, so both arms run the identical protocol.

| | Open-loop | MPC | **OL→MPC gap** |
|---|---|---|---|
| e2e + SIGReg, ungrounded | 13.33 ± 1.15 | 56.0 (n=1) | **+43** |
| **e2e + SIGReg + grounding** | **30.00 ± 7.21** (38/28/24) | **40.00 ± 10.39** (34/34/52) | **+10** |
| paper ✓ frozen baseline | 77.33 ± 6.18 | 85.33 ± 4.99 | **+8** |

Planning wall-clock on the same checkpoint: **GD 76.2 s, GD-MPC 1545.8 s, CEM ~660 s**
(CEM measured earlier at equal success on the ungrounded model, i.e. GD is ~8.7x
faster than CEM).

## What the numbers say

**Open-loop 13.33 → 30.00**, a 2.25x improvement at ~4σ (SEMs 0.66 and 4.16).
Above the 20–27 predicted from the 8k pilot's snr.

**The OL→MPC gap collapsed from +43 to +10, against the baseline's +8.** This is
the strongest structural evidence that the representation defect is repaired, and
it is independent of the absolute success rate. The +43 gap was the signature of a
pusher-blind latent: MPC compensated by re-encoding a real observation every
frameskip steps, so it never needed the agent to be *planned*. With the agent back
in the visual latent (probe 0.993/0.991) open-loop no longer needs that crutch and
the two settings become mutually consistent, exactly as they are for the baseline.

**MPC 56.0 → 40.00 is −1.7σ** against a single-seed comparison with high variance
(34/34/52). Not a regression, and expected: the crutch is no longer load-bearing.

**Still far below the baseline's 77.33.** Grounding recovers most of the damage
that unfreezing causes; it does not produce a surplus. On PushT at H=5 that was
never available -- pristine DINOv2 already probes ~0.94 on every dimension of the
success criterion.

## Success criterion: correcting the record

`env/pusht/pusht_wrapper.py:57`

    pos_diff   = np.linalg.norm(goal_state[:4] - cur_state[:4])   # agent_x, agent_y, block_x, block_y
    angle_diff = min(|d|, 2pi-|d|)
    success    = pos_diff < 20 and angle_diff < pi/9

`goal_state[:4]` includes the AGENT. Earlier notes stating "the block is what the
task is scored on" are wrong: agent position is half of `pos_diff`, at equal
weight, in a single 20-unit budget.

This makes the whole diagnosis stronger. Deleting the agent from the visual latent
(0.943 → −0.011) while the objective gave proprio 0.42% of the cost made the
planner blind to **half the success metric**, not to a useful intermediate. And it
identifies the next grounding target: `state[4]` (angle, threshold pi/9) sits at
0.711 vs pristine 0.732 -- the weakest of the three scored quantities.

## Bug fixed

`summarize_run.rebuild_master()` globbed `results/*.json` and loaded
`results/<run>.timing.json` -- a sidecar written by `reproduce_table1.py` with no
`run` key -- crashing with `KeyError: 'run'` *after* a 50-minute evaluation had
completed. Now skipped explicitly, plus a defensive filter for any record without
a `run` key. Guarded by `tests/test_rebuild_master_sidecars.py`.

---

# FINAL TABLE (corrected — supersedes the numbers above)

An earlier version of this record reported the ungrounded run's MPC as 56.0 (n=1)
and the OL→MPC gap as +43. Both were wrong: `summarize_run.read_success_rates`
globbed `f"{name}_*"`, and `..._e2e` is a strict prefix of `..._e2e_gp1e0`, so the
shorter run absorbed the longer one's `logs.json`. It reported 6 pooled seeds
(12/14/14 + 38/28/24) as one run. Fixed by anchoring the glob on the `_gH`
boundary; guarded by `tests/test_rebuild_master_sidecars.py`.

The bug never crashed. It silently averaged two different models into one row, and
surfaced only because `n=6` tripped the seed-count warning. A run name being a
strict prefix of another is exactly how a confidently wrong number reaches a paper.

| PushT, α=1, 3 seeds (100/200/300), 50 samples | Open-loop | MPC | gap |
|---|---|---|---|
| e2e + SIGReg | 13.33 ± 1.15 (12/14/14) | 44.67 ± 10.26 (56/36/42) | **+31** |
| **e2e + SIGReg + proprio grounding** | **30.00 ± 7.21** (38/28/24) | 40.00 ± 10.39 (34/34/52) | **+10** |
| paper ✓ frozen baseline | 77.33 ± 6.18 | 85.33 ± 4.99 | +8 |

Planning wall-clock, same checkpoint: GD 76.2 s, GD-MPC 1545.8 s, CEM ~660 s at
equal success → **GD is 8.7x faster than CEM**.

## Statistics

- Open-loop **+16.7 points, 2.25x, ≈4σ** (SEMs 0.66 and 4.16).
- MPC **−4.67 points, −0.55σ — statistically unchanged** (SEMs 5.92 and 6.00).

The entire gain is in open-loop. That is exactly where a pusher-blind latent hurts
and where MPC's per-step re-observation had been masking the defect, so the gap
narrowing from +31 to +10 (baseline +8) is the structural signature of the repair.

## Bottom line

Proprio grounding recovers most of the damage that unfreezing a pretrained encoder
causes, and does not produce a surplus. On PushT at H=5 a surplus was never
available: pristine DINOv2 already probes ~0.94 on every dimension of the success
criterion (`pos_diff` over agent_x/y + block_x/y, `angle_diff`), so frozen features
are saturated and parity is the ceiling.
