#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_pusht_on_paperiters.sh
#
# PushT, DINOv2 (patch) + channel projector 14x14x8, straightening ON,
# pinned to the paper's exact update budget.
#
# Paper settings used (arXiv 2603.12231v2):
#   App. A.3 ............ PushT trains for 2 epochs
#   Table 3 ............. batch 32, num_hist 3, frameskip 5, predictor lr 5e-4
#   Table 3 footnote .... projector lr 1e-5 WITH straightening
#   App. B.6 ............ straightening on the agg head, lambda = 0.1 -> aggcos1e-1
#
# Iteration budget:
#   pusht_noise = 18,685 rollouts, 0.9 train split, windows of
#   num_frames(4) x frameskip(5) = 20 steps, batch 32
#     -> 61,929 optimizer steps / epoch      (logged as "Iteration budget:")
#     -> 2 epochs = 123,858 steps            <- training.max_iterations
#
# training.epochs is set to 3 on purpose: it makes the ITERATION CAP the thing
# that ends the run, not the epoch boundary. The cap can only ever shorten a
# run, so the run still stops at exactly 123,858 steps -- mid-way through
# epoch 2 -- which is the paper-exact budget. Set epochs=2 instead if you want
# the epoch boundary and the cap to coincide.
#
# Usage:
#   bash train_pusht_on_paperiters.sh                     # detached, paper-exact
#   bash train_pusht_on_paperiters.sh training.epochs=2   # any extra hydra override
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# --- Blackwell B200 / MIG env recipe (see .kiro steering + REPRODUCTION.md) ---
unset CUDA_VISIBLE_DEVICES                  # MIG UUID breaks mujoco-py's int() parse
export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE="${WANDB_MODE:-offline}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync   # MIG NVML allocator assert
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

MAX_ITERS="${MAX_ITERS:-123858}"            # 61,929 x 2  (paper: 2 epochs)
CKPT_BASE="${CKPT_BASE:-$PWD/checkpoints}"
LOG="train_pusht_on_paperiters_$(date +%Y%m%d_%H%M%S).log"

echo "DATASET_DIR            = ${DATASET_DIR}"
echo "ckpt_base_path         = ${CKPT_BASE}"
echo "training.max_iterations= ${MAX_ITERS}"
echo "log                    = ${LOG}"
echo

# --- pre-flight: the 45 GB MIG slice holds exactly one job ---
# This is a HARD GATE, not a printout. Launching a 12 h training run on top of a
# live plan.py has happened repeatedly: the eval holds ~41 GB of the 45 GB slice,
# so training either OOMs on the spot or trips the allocator's NVML assert and
# takes the eval down with it. Refuse instead. FORCE=1 overrides.
BUSY=""
for pat in "[t]rain.py --config-name" "[p]lan.py --config-name" "[r]eproduce_table1.py"; do
  hits=$(pgrep -af "$pat" || true)
  [ -n "$hits" ] && BUSY="${BUSY}${hits}"$'\n'
done
if [ -n "${BUSY}" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "REFUSING TO START: the MIG slice already has a job." >&2
  echo "${BUSY}" >&2
  echo "Wait for it, or chain on its PID:" >&2
  echo "  setsid nohup bash -c 'while kill -0 <PID> 2>/dev/null; do sleep 60; done;" \
       "sleep 45; bash $(basename "$0")' > train_queue.log 2>&1 < /dev/null &" >&2
  echo "Override with FORCE=1 only if you know the slice is free." >&2
  exit 1
fi

echo "== stray python processes (kill -9 anything left over) =="
ps -eo pid,etime,rss,cmd | grep -i python | grep -v grep || echo "  none"
echo "== MIG memory (want a few MiB used, not ~41 GB) =="
nvidia-smi | sed -n '/MIG dev/,/Processes/p' || true
echo

setsid nohup python train.py --config-name train.yaml \
  env=pusht \
  encoder=dino_channel \
  training.straighten=aggcos1e-1 \
  training.encoder_lr=1e-5 \
  training.epochs=3 \
  training.max_iterations="${MAX_ITERS}" \
  env.num_workers=4 \
  ckpt_base_path="${CKPT_BASE}" \
  "$@" \
  > "${LOG}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > .train_pid
cat <<EOF

Started PID ${PID} (saved to .train_pid). Detached: survives disconnects.

  Watch:              tail -f ${LOG}
  Confirm the budget: grep -A6 "Iteration budget" ${LOG}
  Confirm the config: grep -E "Straightening enabled|Stop-grad|base_model is frozen" ${LOG}
  Progress:           grep -o "global_iter=[0-9]*" ${LOG} | tail -1
  Stop:               kill ${PID}

Expect the run to end with:
  "Iteration budget reached: global_iter=${MAX_ITERS} / max_iterations=${MAX_ITERS}"
  "Run finished on the iteration budget: ${MAX_ITERS} optimizer steps"

Checkpoint lands in:
  ${CKPT_BASE}/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05/checkpoints/model_latest.pth
EOF
