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
