#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_pusht_e2e_sigreg.sh
#
# The proposed method on PushT: prediction + SIGReg + temporal straightening,
# with the DINOv2 trunk TRAINED (no frozen encoder), at the paper's exact
# optimizer-step budget.
#
#   L = L_pred  +  lambda_SIG * SIGReg(Z)  +  lambda_curv * L_curv
#
# Anchors for every hyperparameter:
#   lambda_SIG   = 0.1        LeWM sec. 3 ("we use M = 1024 projections and
#                             lambda = 0.1"); M = 1024 likewise.
#   lambda_curv  = 0.1        Straightening App. B.6: "lambda=0.1 for agg"
#                             -> training.straighten=aggcos1e-1
#   stop_grad    = False      LeWM sec. 3: "We do not employ stop-gradient,
#                             exponential moving averages, or additional
#                             stabilization heuristics." Also: stop_grad
#                             demonstrably does not prevent collapse here
#                             (experiments/verify_stop_grad.py, T1/T3).
#   epochs/iters = 123,858    Straightening App. A.3 (PushT = 2 epochs) at
#                             61,929 iters/epoch. Same budget as the baseline.
#   encoder_lr   = 1e-5       Straightening Table 3 (projector lr, with
#                             straightening).
#   backbone_lr  = 1e-5       *** NOT anchored in either paper. *** The
#                             straightening paper freezes DINOv2; LeWM trains a
#                             ViT-tiny from scratch at 5e-5 AdamW. This is the
#                             one free knob -- sweep it first (see below).
#
# Variants (all at the same step budget, for the ablation table):
#   baseline   bash train_pusht_on_paperiters.sh          frozen trunk, no SIGReg
#   gate 1     SIGREG=0  FREEZE=False                     negative control: collapses
#   only SIG   STRAIGHTEN=False                           SIGReg without curvature
#   full       (this script's defaults)                   SIGReg + curvature
#
# Usage:
#   bash train_pusht_e2e_sigreg.sh
#   BACKBONE_LR=3e-6 bash train_pusht_e2e_sigreg.sh       # sweep the free knob
#   SIGREG=0 bash train_pusht_e2e_sigreg.sh               # gate-1 control
#   bash train_pusht_e2e_sigreg.sh training.curv_on=velocity   # any hydra override
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# --- Blackwell B200 / MIG env recipe ---
unset CUDA_VISIBLE_DEVICES                  # MIG UUID breaks mujoco-py's int() parse
export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE="${WANDB_MODE:-offline}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync   # MIG NVML allocator assert
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

MAX_ITERS="${MAX_ITERS:-123858}"            # 61,929 x 2 == paper's PushT budget
SIGREG="${SIGREG:-1}"                       # 0 -> gate-1 negative control
SIGREG_COEFF="${SIGREG_COEFF:-0.1}"
STRAIGHTEN="${STRAIGHTEN:-aggcos1e-1}"      # False -> SIGReg-only ablation
FREEZE="${FREEZE:-False}"                   # False == end-to-end (the point)
ENCODER_LR="${ENCODER_LR:-1e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
CKPT_BASE="${CKPT_BASE:-$PWD/checkpoints}"
LOG="train_pusht_e2e_sigreg_$(date +%Y%m%d_%H%M%S).log"

SIGREG_FLAG=$([ "${SIGREG}" = "0" ] && echo "False" || echo "True")
[ "${SIGREG}" = "0" ] && SIGREG_COEFF=0.0

echo "DATASET_DIR   = ${DATASET_DIR}"
echo "objective     = L_pred + ${SIGREG_COEFF} * SIGReg + straighten=${STRAIGHTEN}"
echo "freeze_backbone = ${FREEZE}   encoder_lr=${ENCODER_LR}  backbone_lr=${BACKBONE_LR}"
echo "max_iterations  = ${MAX_ITERS}"
echo "log             = ${LOG}"
echo

# --- pre-flight: the 45 GB MIG slice holds exactly one job ---
# HARD GATE, not a printout -- see train_pusht_on_paperiters.sh for why.
BUSY=""
for pat in "[t]rain.py --config-name" "[p]lan.py --config-name" "[r]eproduce_table1.py"; do
  hits=$(pgrep -af "$pat" || true)
  [ -n "$hits" ] && BUSY="${BUSY}${hits}"$'\n'
done
if [ -n "${BUSY}" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "REFUSING TO START: the MIG slice already has a job." >&2
  echo "${BUSY}" >&2
  echo "Chain on its PID instead, or set FORCE=1 if you know the slice is free." >&2
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
  training.straighten="${STRAIGHTEN}" \
  training.sigreg="${SIGREG_FLAG}" \
  training.sigreg_coeff="${SIGREG_COEFF}" \
  training.sigreg_num_proj=1024 \
  training.sigreg_apply_to=agg \
  training.stop_grad=False \
  training.freeze_backbone="${FREEZE}" \
  training.encoder_lr="${ENCODER_LR}" \
  training.backbone_lr="${BACKBONE_LR}" \
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

  Watch:            tail -f ${LOG}
  Objective check:  grep -E "SIGReg enabled|Straightening enabled|Stop-grad|base_model is" ${LOG}
  Budget check:     grep -A6 "Iteration budget" ${LOG}
  Param groups:     grep "Encoder param groups" ${LOG}

  *** THE metric to watch -- collapse, logged once per epoch: ***
    grep -oE "val_probe_r2[^,]*|val_latent_eff_rank[^,]*|val_agg_probe_r2[^,]*" ${LOG}
  If val_probe_r2 heads to 0 the encoder has collapsed and the run is dead,
  no matter how nicely the training loss falls. Lower backbone_lr and retry.

Run directory: derived by hydra from the objective, so DO NOT assume it -- read it
back from the log (a hardcoded path here was stale and misleading once already):
  grep -m1 -oE "${CKPT_BASE}/test/[^ ]*" "${LOG}"
  ls -d ${CKPT_BASE}/test/*/ -t | head -1
The suffix encodes the variant: _sig1e-1_e2e for SIGReg end-to-end, plus _gp<coeff>
when training.ground_proprio > 0, so each variant keeps its own checkpoint.

Before burning ~12 h of GPU on this, confirm the objective on CPU in ~2 min:
  python -c "import experiments.verify_stop_grad as v; v.gates()"
EOF
