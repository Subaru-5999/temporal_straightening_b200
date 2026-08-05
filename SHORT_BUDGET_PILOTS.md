# Short-budget pilots

How to test a training-side hypothesis in ~1 hour instead of ~14, and when the
answer is trustworthy.

Worked example throughout: the PushT proprio-grounding fix, validated at **8,000
of 123,858 steps (6.5% of budget)** in 54 minutes, which produced a ~8σ result.

---

## 1. The escalation ladder

Never jump straight to a full run. Each rung is ~10x cheaper than the next and
kills most bad ideas.

| rung | cost | answers |
|---|---|---|
| offline probe on an existing checkpoint | 2–5 min, CPU | is the mechanism I suspect actually present? |
| **short-budget pilot** | **~1 h, 1 GPU** | does my change move the mechanism? |
| full-budget run | ~14 h | what is the reportable number? |

The middle rung is the one people skip. It is also the only one that tests a
*training* change, because rung 1 cannot see a change that has not been trained
and rung 3 costs a day per attempt.

Real numbers from this project: 4 pilot runs at 54 min each cost 3.6 h and
settled the coefficient, the target dims, the device bug and the α interaction.
Discovering those from full runs would have cost ~56 h.

---

## 2. Setting the budget

The repo has a hard iteration cap (`iteration_budget.py`) that stops mid-epoch,
which is what makes sub-epoch pilots possible at all.

```bash
# 8,000 of 123,858 steps == 6.5% of the PushT paper budget
MAX_ITERS=8000 bash train_pusht_e2e_sigreg.sh training.ground_proprio=1.0
```

or directly:

```bash
python train.py --config-name train.yaml env=pusht \
  training.max_iterations=8000 training.epochs=3 ...
```

Three things make this work and all three matter:

- **`training.epochs` is set higher than needed** (3, not 2). The iteration cap
  can only ever *shorten* a run, so leaving epochs generous guarantees the cap is
  what ends it rather than an epoch boundary.
- **`save_every_x_iterations: 1000`** writes `model_latest.pth` mid-epoch, and
  because the condition is `i % 1000 == 0` it also fires at `i == 0`. A
  checkpoint exists within seconds. If the directory is empty a minute in, the
  run has **crashed** — that is a signal, not a wait.
- **`global_iter` is checkpointed**, so a pilot can be resumed or extended.

Budget arithmetic for PushT: 18,685 rollouts x 0.9 train split, windows of
`num_frames(4) x frameskip(5)`, batch 32 -> **61,929 steps/epoch**, and the paper
trains 2 epochs -> **123,858**. The exact figure for any env is logged at startup
as `Iteration budget:`.

---

## 3. Isolating the run directory

The Hydra run dir is derived from the objective, so **two pilots with different
settings can collide and silently auto-resume each other**. This bit us once:
`ground_proprio` was not in the run name, so a grounded pilot resolved to the
completed ungrounded run's directory and would have continued and overwritten it.

Two ways to separate them, in order of preference:

1. **Add the knob to `run_naming.variant_tag()`.** Correct and self-documenting.
   Keep the contract that defaults return `''` so pre-existing paths stay
   byte-identical, and add a test asserting the grounded and ungrounded names
   differ.
2. **Override `CKPT_BASE`.** Fine for a throwaway sweep over a knob that is not
   in the name:
   ```bash
   CKPT_BASE=$PWD/checkpoints_gp_pos MAX_ITERS=8000 bash train_....sh \
     training.ground_proprio=1.0 'training.ground_proprio_dims=[0,1]'
   ```
   The cost is that you must remember which directory is which. Prefer (1) for
   anything you will cite.

Always confirm where it landed. Do not trust a launcher's printed path — ours
had a hardcoded string that went stale:

```bash
grep -m1 -oE "$PWD/checkpoints[^ ]*" "$LOG"
ls -d checkpoints*/test/*/ -t | head -3
```

---

## 4. Choosing the readout: the part that decides whether the pilot works

A pilot is only as good as its metric. Pick one that is

1. **causally upstream** of the outcome you care about,
2. has a **known reference value** to compare against, and
3. **moves early**, within the pilot's budget.

The failure mode is an aggregate metric. In the failing 123,858-step run,
aggregate `probe_r2` (patch-mean vs full state) went **0.244 -> 0.280** — it
*improved* while the model was being destroyed, because it was dominated by the
block, which was never lost. The **per-state-dimension** probe showed
`agent_x: +0.943 -> -0.011`. Same data, same code, opposite conclusions.

> Disaggregate before you trust a metric. If a quantity is a mean over
> heterogeneous components, at least one component can collapse without moving it.

Reference values come from three places, cheapest first:

- **An untrained/pretrained reference.** Pristine DINOv2 from the hub cache gave
  us "what the representation looked like at initialisation" for free — no
  training, no baseline checkpoint. This is what let us claim the grounded run's
  0.991 was an *improvement* (over pristine 0.943) rather than merely "not yet
  degraded".
- **The long run's own telemetry over its first N steps.** This is a **free
  matched-budget control**. Our JSONL covered steps 200 -> 123,858, so its first
  8,000 steps were already on disk and no control run was needed. We cancelled a
  queued 55-minute job because of this.
- **A matched-budget control run.** Only if the two above are unavailable.

---

## 5. Define the gate before you run

Write the pass/fail threshold down first, or you will rationalise whatever comes
back. Ours:

> Three seeds at the pilot's best α reaching ≈0.26 justifies the full run, since
> that is what the *full* ungrounded run achieved — a 6.5%-budget model matching a
> 100%-budget one implies scaling should carry it further.

Result: 20.67 ± 1.15, indistinguishable from 26.0 ± 6.2 at 6.5% of the budget.
Gate passed on the comparison that mattered, and stated as such rather than
rounded up.

---

## 6. Reading a pilot: loss shares, not loss values

A falling total loss says nothing. Read each term's **share of the objective**,
which the telemetry gives directly:

```bash
python summarize_training_log.py <run_dir> --latest --metrics loss
```

From the grounding pilot at step 4,000:

| term | scaled | share |
|---|---|---|
| grounding | 0.2324 | **51%** |
| SIGReg | 0.1391 | 31% |
| curvature | 0.0673 | 15% |
| **prediction** | 0.0118 | **2.6%** |

That table said immediately that a new term at coefficient 1.0 was 20x the
prediction loss and dominating training — something no single loss curve shows.
It also flagged that the grounding target was partly ill-posed: the loss
plateaued at 0.23 instead of approaching 0, because 2 of its 4 target dims were
velocities, which are not identifiable from a single frame.

Sanity checks worth running on every pilot:

- **step rate unchanged** vs a known run (2.47 vs 2.48 it/s) => the new module is
  not secretly expensive.
- **step-200 rows match** the reference run => the arms really are comparable.
- **the new term is not collapsing to ~0 immediately** => otherwise its own head
  absorbed the task and the encoder was never pressured, which looks like success
  and is not.

---

## 7. Evaluating a pilot checkpoint

Worth doing, with a caveat attached. A pilot's *predictor* is genuinely worse
(`z_loss` 0.0118 at 8k vs 0.0016 at full budget, ~7x), and planning depends on
rollout accuracy, so **a low number is not evidence against the change**. A high
one is strong evidence for it.

Use 1 seed to triage, 3 seeds before believing a difference. At n=50 samples the
binomial standard error near p=0.2 is **5.7 points**, so single-seed gaps under
~10 points are noise. Our 0.20 vs 0.12 was 1.2σ on one seed and ~8σ on three.

---

## 7b. Mid-run readings are NOT a trajectory

Representation metrics during training are non-monotone, and treating a mid-run
reading as a trend produced two wrong calls on this project:

| metric | 8k | 34k | final (123,858) |
|---|---|---|---|
| block_x probe | 0.930 | 0.868 | **0.945** |
| block_angle probe | 0.514 | 0.427 | **0.711** |
| visual snr @k=10 | — | 1.63 | **2.14** |
| visual snr @k=15 | — | 1.06 | **1.85** |

From the 34k column I concluded "the encoder sheds whatever the objective does not
pin" and "long horizon is dead". Both dimensions recovered and both conclusions
were wrong. The dip was real and transient.

Use mid-run probes for exactly one thing: **detecting catastrophic failure** — the
quantity you pinned collapsing toward zero. That is a step change, not a slope, and
it justifies killing a run. Anything gentler is noise until the run finishes.

## 8. What a pilot cannot do

- **Produce a reportable number.** Full budget only.
- **Rank configurations reliably.** Pilot ordering may not survive scaling.
- **Detect late-emerging failures.** If the mechanism only appears after the
  pilot's horizon, the pilot is blind to it. Check this by confirming the
  mechanism was already visible early in the long run's telemetry — ours diverged
  by step 200–4,000, which is why 8k sufficed.
- **Substitute for a baseline.** A pilot compares against a reference, not
  against a trained baseline at the same budget.

---

## 9. Pitfalls hit while doing this

| symptom | cause |
|---|---|
| dies ~2 s into epoch 1, `Expected all tensors on the same device` | a module created in `VWorldModel.__init__` keeps CPU params: the model is built *after* `accelerator.prepare()` and is never itself prepared. Needs explicit `.to(device)` **and** explicit registration in an optimizer, or the head also never learns. |
| checkpoint dir empty a minute in | the run crashed. `save_ckpt` fires at `i == 0`. |
| `Predictor not found in model checkpoint` | the checkpoint file does not exist; `load_model` misreports a missing file. Pre-check the path. |
| loss values absent from stdout | `WANDB_MODE=offline` sends them to wandb. Read the JSONL telemetry instead. |
| a shell that prints nothing | a `sleep`/`until` loop blocking, with the remaining pasted lines queued behind it. Ctrl-C is safe — it only sleeps, and detached jobs are unaffected. |
| `grep ... \| head -3 \|\| echo "none"` never prints "none" | `\|\|` binds to `head`, which succeeds on empty input. Use an `if grep -q`. |
| two jobs land on one GPU | polling `pgrep` has gaps between a driver's sequential jobs. Chain on the **driver's PID**, not on the absence of its children. `setsid` exits immediately, so `$!` is not the script's PID. |

---

## 10. Checklist

```
[ ] mechanism confirmed present offline first (rung 1)
[ ] readout chosen: upstream, has a reference, moves early — and DISAGGREGATED
[ ] reference identified: pretrained init / long run's early telemetry / control
[ ] gate threshold written down BEFORE launching
[ ] budget set via training.max_iterations, epochs left generous
[ ] run dir isolated: variant_tag entry, or CKPT_BASE override
[ ] slice free; job chained on a PID, not on a poll
[ ] within 2 min: new term logged as enabled, on the right device, checkpoint on disk
[ ] step rate and step-200 losses match the reference run
[ ] loss SHARES read, not just values
[ ] 3 seeds before believing any eval difference (SE ~5.7 pts at n=50)
[ ] result recorded with its caveats, in PROGRESS_*.md
```
