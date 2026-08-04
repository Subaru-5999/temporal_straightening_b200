#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_oracle_diag.sh -- decide WHERE the 0.12 open-loop PushT number comes from.
#
# The A-E sweep (alpha / init / opt budget) all vary the *planner*. None of them
# can tell you whether the harness is sound or whether the learned cost is
# actively misleading. These three do, and two of them cost almost nothing.
#
#   O3  floor      no optimisation, no ground truth  -> what a NON-adaptive
#                  action sequence scores on this exact protocol. If 0.12 is
#                  within noise of this, the planner is contributing nothing.
#   O1  oracle     ground-truth actions, 0 GD steps  -> replays the actions that
#                  generated the goal. Should be ~1.0. Model quality is
#                  irrelevant here, so anything low means the bug is in the
#                  harness (goal/state/seed/success metric), NOT the encoder.
#   O2  descent    ground-truth actions, 100 GD steps -> starts AT the answer and
#                  lets the learned cost optimise. High => the representation is
#                  fine and the failure is initialisation/optimisation. Falls to
#                  ~0.12 => the learned latent cost walks *away* from the correct
#                  actions, i.e. the cost is misaligned with task success. That
#                  is the single most diagnostic outcome available.
#
# O1/O2 rely on plan.py's `debug_dset_init` flag, which was a silent no-op for
# any MPCPlanner-topped config (plan_gd.yaml included) until the initial actions
# were forwarded to the sub-planner. Pull before running.
#
# Usage:  setsid nohup bash run_oracle_diag.sh > oracle.log 2>&1 < /dev/null &
#         tail -f oracle.log
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# --- the env recipe reproduce_table1.py normally applies internally ---
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
COMMON="--config-name plan_gd.yaml ckpt_base_path=$CK model_name=$METHOD \
model_epoch=latest decode_for_viz=false seed=100 objective.alpha=1"

# --- one job per MIG slice: wait, never pre-empt ---
echo "waiting for the slice to free..."
while pgrep -f "[p]lan.py" > /dev/null; do sleep 60; done
sleep 20                     # let CUDA memory actually release
echo "slice free at $(date)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

run () {
  name=$1; shift
  echo "=== $name : $* ==="
  python plan.py $COMMON "$@" hydra.run.dir="plan_oracle/$name" 2>&1 \
    | grep -E "Success rate|Error executing|OutOfMemory|Traceback"
}

run O3_floor    planner.sub_planner.opt_steps=0
run O1_oracle   planner.sub_planner.opt_steps=0   debug_dset_init=true
run O2_descent  planner.sub_planner.opt_steps=100 debug_dset_init=true

cat <<'EOF'

=== how to read it ===
  O1 low (<0.8)               -> harness bug. Stop looking at the encoder.
  O1 ~1.0, O3 ~= 0.12         -> the planner adds nothing over a fixed sequence.
  O1 ~1.0, O2 ~1.0            -> representation OK; zero-init basin / optimisation.
  O1 ~1.0, O2 collapses       -> the learned latent cost is misaligned with task
                                 success. Fix the objective/representation, not
                                 the planner.
EOF
echo "=== ORACLE DIAGNOSTICS DONE $(date) ==="
