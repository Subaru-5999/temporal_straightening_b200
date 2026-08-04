#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_alpha_sweep.sh -- rebalance the planning objective. No retraining.
#
# experiments/rollout_drift.py on the e2e+SIGReg PushT checkpoint, at the
# protocol horizon k=5:
#
#            reach      rollout err    snr     beat    share of cost
#   visual   0.102      0.035          2.91    0.065   99.58 %
#   proprio  0.000844   0.0000251     33.68    0.026    0.42 %
#
# The proprio channel is a 12x cleaner planning signal and gets 0.42% of the
# weight. objectives.py computes `loss_visual + alpha * loss_proprio` with a
# per-dim mean on both sides, so the spread ratio IS the effective weight:
# 0.2598 / 0.001089 -> alpha_eff = 0.0042 at the configured alpha=1.
#
#   alpha ~ 240   equal contribution        (spread_visual / spread_proprio)
#   alpha ~ 1400  inverse-noise weighting   (rollout_v / rollout_p)
#
# alpha=1 was calibrated for a FROZEN DINOv2 encoder, where the two channels sat
# at comparable scale. Training the trunk end-to-end moves the visual scale and
# silently re-weights the objective by ~3 orders of magnitude. That is a pitfall
# for anyone who unfreezes the encoder, and it costs nothing to correct at
# planning time.
#
# Seed 100 only, open-loop GD, so this is ~5 cheap jobs. If any alpha lifts
# open-loop well above the 13.33 +/- 1.15 baseline, run the full 3 seeds.
#
# NOTE the hydra output dir for plan_gd keys on lr/action_noise/opt_steps/
# objective.mode/sample_type but NOT on alpha, so every alpha would land in the
# same directory and append to one logs.json. Each job gets an explicit
# hydra.run.dir; results are read per-directory, never pooled.
#
# Usage:  setsid nohup bash run_alpha_sweep.sh > alpha_sweep.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

unset CUDA_VISIBLE_DEVICES
export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled WANDB_SILENT=true
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export PLAN_SERIAL_ENV=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

METHOD="${METHOD:-pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgFalse_lr1e-05_sig1e-1_e2e}"
CK="$PWD/checkpoints/test/$METHOD"
ALPHAS="${ALPHAS:-1 50 240 1400 100000}"
COMMON="--config-name plan_gd.yaml ckpt_base_path=$CK model_name=$METHOD \
model_epoch=latest decode_for_viz=false seed=100"

echo "waiting for the slice to free..."
while pgrep -f "[p]lan.py" > /dev/null; do sleep 60; done
sleep 20
echo "slice free at $(date)"

for a in $ALPHAS; do
  echo "=== alpha=$a ==="
  python plan.py $COMMON "objective.alpha=$a" \
    hydra.run.dir="plan_alpha/a$a" 2>&1 \
    | grep -E "Success rate|Error executing|OutOfMemory|Traceback"
done

cat <<'EOF'

=== how to read it ===
  alpha=1                 reproduces 0.12-0.14, the number already measured.
  alpha=100000            proprio-only: the visual term is numerically absent.
  any alpha >> 13.33      the failure was objective WEIGHTING, not the encoder,
                          the rollout, or the planner. Then rerun 3 seeds at the
                          best alpha and report it as a scale-calibration step.
  all of them ~0.13       weighting is not the cause; fall back to O1/O2 in
                          run_oracle_diag.sh.
EOF
echo "=== ALPHA SWEEP DONE $(date) ==="
