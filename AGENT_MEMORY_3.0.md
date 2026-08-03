# Agent memory 3.0 — SIGReg + end-to-end encoder training

Operational context for whoever (human or agent) picks this up next. Terse by
design. `AGENT_MEMORY_2.0.md` covers the *earlier* work: reproducing the frozen
DINOv2 Table-1 cells. This file covers the **new objective**.

Companion: `PROGRESS_SIGREG_E2E.md` (what has been done and verified, with
numbers). Read that for results; read this for how to operate.

---

## 1. What this project is now

Original repo: reproduce Table 1 of *Temporal Straightening for Latent Planning*
(arXiv 2603.12231v2) — frozen DINOv2 + trainable projector + curvature loss.

New goal, per `research_papers/ICLR_Submission_Plan.md`: train the encoder
**end-to-end** (no frozen backbone) by adding SIGReg, giving

```
L = L_pred + lambda_SIG * SIGReg(Z) + lambda_curv * L_curv
```

Why SIGReg is *required* rather than optional: `stop_grad` does not prevent
collapse (it detaches only `z_tgt`; the `z_src` path stays open and a constant
encoder is still a global minimum of `L_pred`), and the cosine curvature term is
exactly scale-invariant so it cannot see collapse either. SIGReg pins the
distribution and hence the scale; curvature pins the direction. All three claims
are measured, not argued — see §"Falsification results" in the progress file.

Branch: **`feat/sigreg-e2e`**. `main` holds the validated frozen baseline; do not
merge until a full GPU run confirms the new objective.

---

## 2. Non-negotiable design constraint

**Every default stays byte-identical.** The five Table-1 run directories that
`reproduce_table1.py` looks up by name must not change, or the earlier
reproduction stops being comparable. New behaviour is opt-in only, and
`run_naming.variant_tag()` returns `''` for the original settings so paths are
unchanged (asserted in `tests/test_run_naming.py`).

Consequence: when adding anything, default it to off/None and add a test that the
five existing names are untouched.

---

## 3. Paper anchors — do not invent hyperparameters

| Setting | Value | Source |
|---|---|---|
| `sigreg_coeff` (lambda_SIG) | 0.1 | LeWM sec. 3 |
| `sigreg_num_proj` (M) | 1024 | LeWM sec. 3 |
| `straighten` (lambda_curv) | `aggcos1e-1` = 0.1 | Straightening App. B.6, "0.1 for agg" |
| `stop_grad` | **False** | LeWM sec. 3: no stop-gradient, no EMA, no heuristics |
| batch / num_hist / frameskip | 32 / 3 / 5 | Straightening Table 3 |
| predictor, action, proprio lr | 5e-4 | Table 3 |
| projector (`encoder_lr`) | 1e-5 with straightening, 1e-6 without | Table 3 + footnote |
| PushT step budget | **123,858** = 61,929 x 2 epochs | App. A.3 (2 epochs) |
| planning | H=5, Adam, zero init, lr 0.1, 100 steps, 50 samples, seeds 100/200/300, OL executes 25 / MPC 5, `mode=last` / `staged`, alpha=1 for PushT | Table 4 + sec. 5.3 |
| `backbone_lr` | **null** (single group at `encoder_lr`) | RESOLVED — see below |

`backbone_lr` was the one un-anchored knob (neither paper fine-tunes a pretrained
trunk). The 2000-step probe at 1e-5 was healthy, and since `encoder_lr` is also
1e-5, `backbone_lr=null` is *numerically identical* to that probe. So use `null`
and the configuration has **zero hyperparameters outside the two papers**. Only
revisit if a long run misbehaves; fallbacks 3e-6 / 1e-6 were probed.

Sources are in-repo: `arXiv-2603.12231v2.tar.gz` (straightening; `sec/1_main.tex`
Table 1 ~line 408, `sec/2_appendix.tex` Tables 3/4 and App. B.6) and
`research_papers/arXiv-2603.19312v3.tar.gz` (LeWM; `sections/3-method.tex`).
Re-read the `.tex`, not the OCR'd `_paper.txt`.

---

## 4. The four runs (all at 123,858 steps, PushT)

| # | Variant | `straighten` | `sigreg` | `freeze_backbone` | `stop_grad` | Run dir suffix |
|---|---|---|---|---|---|---|
| 1 | Frozen baseline (= paper's Table-1 ✓ cell) | `aggcos1e-1` | off | True | True | *(none)* |
| 2 | Only SIGReg | `False` | 0.1 | False | False | `_sig1e-1_e2e` |
| 3 | Only straightening (predicted to collapse) | `aggcos1e-1` | off | False | False | `_e2e` |
| 4 | **Full method** | `aggcos1e-1` | 0.1 | False | False | `_sig1e-1_e2e` |

All four directory names are distinct, so nothing auto-resumes into the wrong
run. Order by information value in case GPU time runs out: **4, 1, 2, 3**.

Drivers: `train_pusht_e2e_sigreg.sh` (2/3/4 via `SIGREG=` / `STRAIGHTEN=` env
vars) and `train_pusht_on_paperiters.sh` (1). Roughly 14 h each, 12 h for run 1.
One MIG slice = one job, so strictly sequential.

---

## 5. Environment recipe (B200 / MIG) — apply for train AND eval

```bash
cd /workspace/arun/ts_newloss
source /workspace/arun/envs/ts_newloss/bin/activate
unset CUDA_VISIBLE_DEVICES          # MIG UUID breaks mujoco-py's int() parse
export DATASET_DIR=/workspace/arun/data
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=offline
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync   # MIG NVML allocator assert
export PLAN_SERIAL_ENV=1                                  # eval only: MIG fork safety
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
```

Fresh-pod setup: `python3 -m venv --system-site-packages <env>` then
`DATASET_ROOT=/workspace/arun/data bash setup_b200.sh`, then
**`pip uninstall -y transformer-engine transformer_engine`** (NGC's copy is
ABI-mismatched against torch 2.7 and `accelerate` auto-imports it; bites at train
time only, on a fresh pod only). Planning tier: `pip install -r requirements-plan.txt`
(PushT only needs this; `setup_planning.sh` is for the maze cells).

Pre-flight before every GPU job — the 45 GB slice holds exactly one:
```bash
ps -eo pid,etime,rss,cmd | grep -i python | grep -v grep    # kill -9 strays
nvidia-smi | sed -n '/MIG dev/,/Processes/p'                # want MiB, not ~41 GB
```
`nvidia-smi` cannot list MIG processes — trust `ps`. Never Ctrl-Z a GPU job.

---

## 6. Verify before spending GPU (~5 min, no GPU needed)

```bash
python -m pytest tests -q                                   # 129 tests
python -c "import experiments.verify_stop_grad as v; v.gates()"
python experiments/verify_encoder_trains.py
python check_dataset_sync.py
```

Expected: gate 1 PASS (collapsed, probe R^2 ~ -0.0000), gates 2/3 PASS (~0.60 /
~0.57); both trunk checks PASS. `check_dataset_sync.py` FAILing on
`point_maze_medium` is fine — that env is unused.

Then a smoke test on the real model before any long run:
```bash
python train.py --config-name train.yaml env=pusht encoder=dino_channel \
  training.straighten=aggcos1e-1 training.sigreg=True training.sigreg_coeff=0.1 \
  training.stop_grad=False training.freeze_backbone=False training.backbone_lr=null \
  training.epochs=1 training.max_iterations=30 training.telemetry_every=5 \
  training.save_every_x_iterations=0 \
  env.dataset.n_rollout=40 env.num_workers=4 \
  ckpt_base_path=$PWD/checkpoints/smoke 2>&1 | tee smoke.log
```
Must print `Encoder param groups: trunk 22,056,576 ... | heads 1,810,280 ...`.
Anything other than ~22M for the trunk means the real DINOv2 is not in the
optimizer. **Delete `checkpoints/smoke` afterwards** or a later run with the same
flags silently auto-resumes from it.

---

## 7. Telemetry — how to know a run is alive

`training.telemetry` (default on) writes
`<run_dir>/telemetry/train_<ts>.jsonl`; memory is O(#metrics) via Welford
accumulators, disk is one record per `telemetry_every` (200) steps.

```bash
python summarize_training_log.py <run_dir> --latest
python summarize_training_log.py <run_dir> --latest --metrics loss,latent,grad,delta
python summarize_training_log.py <run_dir> --latest --csv out.csv
```
Digest length is independent of run length, so it is safe to paste in full.

**The kill criterion**: `latent/probe_r2` heading to 0 means the encoder
collapsed and the run is dead regardless of the loss curve. `diag_every` (500)
gives ~247 samples of it across the PushT budget. Also watch
`latent/latent_eff_rank_frac` sliding toward 0.125 (rank 1 of 8).

Reference values measured on the healthy 1e-5 probe (2000 steps, real PushT):

| Metric | Value | Note |
|---|---|---|
| `loss/sigreg_loss` | 5.55 -> **1.47** plateau | finite-sample null floor at (T=4,B=32,d=128) is **1.058 ± 0.041**, so ~37% above floor |
| `loss/curvature_loss...` | 1.288 -> **0.544** | mean cos -0.288 -> +0.456 |
| `latent/probe_r2` (held out) | 0.224 -> **0.393** | rising = healthy |
| `latent/latent_eff_rank_frac` | 0.377 -> 0.29 | declining but stabilising; WATCH |
| `latent/agg_latent_eff_rank_frac` | ~0.13-0.16 flat | ~19 effective dims of 128 |
| `grad/encoder.trunk/ratio` | 0.215 -> **0.0098** | transient decays to the top of the healthy 1e-4..1e-2 band |
| `delta/encoder.trunk` | max 5.2e-2 | non-zero = trunk really moving |

The `## Verdict` OK/FAIL thresholds were calibrated on the synthetic gate task
(`probe_r2 > 0.2`). On real PushT the healthy value sits near 0.2-0.44, so the
threshold is marginal — **compare runs against each other, not against the
threshold.**

---

## 8. Evaluation

```bash
pip install -r requirements-plan.txt
RUN=pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgFalse_lr1e-05_sig1e-1_e2e
nohup python reproduce_table1.py <baseline_run> $RUN > eval_all.log 2>&1 &
python summarize_run.py --all
cat results/table1_reproduction.md
```

`reproduce_table1.py` falls back to the leading env token when a run name is not
one of the five tracked cells, so the `_sig1e-1_e2e` variants get the right
protocol (PushT -> alpha=1 / `mode=staged`) instead of being skipped. Unknown env
prefixes still fail closed.

Paper targets for the frozen baseline cell: **OL 70.00±1.63, MPC 78.67±0.94**.

---

## 9. Pitfalls (symptom -> cause -> fix)

Inherited from 2.0 and still live:
- `transformer_engine` ImportError at train time -> ABI mismatch -> uninstall it.
- `NVML_SUCCESS == r` assert -> MIG allocator -> `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`.
- `int('MIG-...')` -> mujoco-py parses `CUDA_VISIBLE_DEVICES` -> keep it unset.
- `OutOfMemoryError` -> almost always a leaked python holding 41.5 GB -> `ps` + `kill -9`.
- ~250 s "hang" in `setup_model` -> CPU thread oversubscription in DINOv2 init -> thread caps = 8.
- gym + NumPy 2 -> `pip install "numpy<2"`; `wandb TypeIs` -> `typing_extensions>=4.12`.

New to this work:
- **Underdetermined linear probes silently report R^2 = 1.0.** Bit us twice: the
  gate harness (393 features from 64 samples) and the production `agg_probe_r2`
  (129 from 128). `probe_r2` is now a held-out split. If you add a probe, check
  samples >> features or expect fiction.
- **bf16 quietly degrades SIGReg.** The statistic squares a ~1e-2 deviation
  between order-1 numbers; `models/sigreg.py` upcasts to fp32 for that reason.
  Do not "optimise" it back.
- **`VWorldModel` is never moved to the accelerator.** It is built after
  `accelerator.prepare()`, so any buffer you register on it stays on CPU while
  the latents are on GPU. `SIGReg.forward` follows the input's device/dtype
  because of this. Same trap awaits any new module you attach there.
- **`_cos_curvature` used to return NaN** once every velocity fell under
  `step_thresh`, reachable exactly when the encoder is trainable. Guarded now.
- **Diagnostics once per epoch is useless** on a 123,858-step run (2-3 points).
  Hence `diag_every`.
- **`encoder=dino` has zero trainable encoder params** yet still builds and steps
  an Adam. That is correct for the paper's frozen baseline row, so it warns
  rather than raising.
- Telemetry must never kill a run: every entry point is guarded and a write
  failure disables the logger.

---

## 10. Current state and next actions

Done and verified: see `PROGRESS_SIGREG_E2E.md`. 129 tests pass. 11 commits on
`feat/sigreg-e2e`, pushed.

Immediate:
1. Launch run 4 — `BACKBONE_LR=null bash train_pusht_e2e_sigreg.sh` (~14 h).
2. Then runs 1, 2, 3 in that order.
3. Evaluate all four, GD open-loop + GD-MPC, seeds 100/200/300.

Known gaps, ordered by value (details in the progress file):
CEM arm + planning-time aggregation; condition-number proxy of the planning
Jacobian; latent-vs-geodesic correlation; lambda sensitivity sweeps (unaffordable
at full budget — run reduced); loss-landscape figures; long-horizon H=10-15
(**blocked**: needs `curv_window` decoupled from `num_hist`, which requires all
five dataset loaders to emit extra frames); multi-seed variance.
