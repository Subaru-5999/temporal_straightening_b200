#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_diagnose_all.sh -- root-cause the PushT end-to-end + SIGReg failure.
#
# Established so far (method run: OL 13.33 +/- 1.15, MPC 56; paper 77.33 / 85.33):
#   * NOT collapse            MPC reaches 56, impossible with a dead latent
#   * NOT rollout drift       drift 0.135 at the protocol horizon k=5,
#                             one-step NMSE 1.1%
#   * objective mis-weighted  proprio carries 0.42% of the cost at alpha=1,
#                             i.e. alpha_eff = 0.0042, while proprio has 12x the
#                             planning SNR of the visual channel (33.7 vs 2.9)
#
# Four candidates remain. Each stage below kills or confirms exactly one, and the
# stages are ordered cheapest-and-most-decisive first.
#
#   STAGE 1  geometry      offline, no GPU contention beyond a few minutes.
#                          Is latent distance a proxy for state distance? The
#                          paper's whole claim. Reference is pristine DINOv2, so
#                          no baseline checkpoint is needed.
#   STAGE 2  harness       does replaying the ground-truth actions succeed? If
#                          not, every number including 13.33 is meaningless.
#   STAGE 3  weighting     alpha sweep. alpha=0 vs alpha=1 identical proves the
#                          proprio term is inert; a jump at 240/1400 proves the
#                          weighting IS the cause.
#   STAGE 4  optimiser     random init and 10x budget, plus GD started AT the
#                          ground-truth optimum. Separates "cost is wrong" from
#                          "search is wrong".
#
# Everything runs on seed 100 only -- this is diagnosis, not a reportable table.
# One job at a time: the 45 GB MIG slice holds exactly one.
#
# Usage:
#   setsid nohup bash run_diagnose_all.sh > diagnose.log 2>&1 < /dev/null &
#   tail -f diagnose.log
#
#   STAGES=1     bash run_diagnose_all.sh      # geometry only (fast)
#   STAGES=3,4   bash run_diagnose_all.sh      # skip what you already have
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
RUN="$PWD/checkpoints/test/$METHOD"
STAGES="${STAGES:-1,2,3,4}"
mkdir -p results

COMMON="--config-name plan_gd.yaml ckpt_base_path=$RUN model_name=$METHOD \
model_epoch=latest decode_for_viz=false seed=100"

want () { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

wait_for_slice () {
  if pgrep -f "[p]lan.py --config-name" > /dev/null \
     || pgrep -f "[t]rain.py --config-name" > /dev/null; then
    echo ">>> slice busy, waiting..."
    while pgrep -f "[p]lan.py --config-name" > /dev/null \
       || pgrep -f "[t]rain.py --config-name" > /dev/null; do sleep 60; done
    sleep 30
  fi
  echo ">>> slice free at $(date)"
}

# plan.py's hydra output dir keys on lr/action_noise/opt_steps/objective.mode/
# sample_type but NOT on alpha, so without an explicit run dir every alpha would
# append to one shared logs.json and the results would be pooled silently.
plan () {
  name=$1; shift
  echo "=== $name : $* ==="
  python plan.py $COMMON "$@" hydra.run.dir="plan_diag/$name" 2>&1 \
    | grep -E "Success rate|Error executing|OutOfMemory|Traceback|MemoryError"
}

echo "############ DIAGNOSIS $(date) ############"
echo "run    = $RUN"
echo "stages = $STAGES"
[ -f "$RUN/checkpoints/model_latest.pth" ] || { echo "FATAL: no checkpoint at $RUN"; exit 1; }

# ------------------------------------------------------------------ STAGE 1
if want 1; then
  echo
  echo "################ STAGE 1: LATENT GEOMETRY ################"
  echo "# Is Euclidean latent distance a proxy for state distance? Read rho_LOCAL"
  echo "# and nn_state_ratio; the pristine-DINOv2 row is the untrained reference."
  wait_for_slice
  python experiments/metric_alignment.py "$RUN" \
    --frames 768 --json results/metric_alignment_method.json

  echo
  echo "################ STAGE 1b: ROLLOUT + PLANNING SNR ################"
  wait_for_slice
  python experiments/rollout_drift.py "$RUN" \
    --json results/rollout_drift_method.json
fi

# ------------------------------------------------------------------ STAGE 2
if want 2; then
  echo
  echo "################ STAGE 2: HARNESS SANITY ################"
  echo "# H_oracle replays the ground-truth actions with ZERO optimisation."
  echo "# It must be ~1.0. If it is not, the goal/state/seed/success path is"
  echo "# broken and no other number in this file means anything."
  wait_for_slice; plan H_floor  planner.sub_planner.opt_steps=0 objective.alpha=1
  wait_for_slice; plan H_oracle planner.sub_planner.opt_steps=0 objective.alpha=1 debug_dset_init=true
fi

# ------------------------------------------------------------------ STAGE 3
if want 3; then
  echo
  echo "################ STAGE 3: OBJECTIVE WEIGHTING ################"
  echo "# alpha_eff = spread_proprio / spread_visual = 0.0042 at alpha=1."
  echo "# equal contribution ~240, inverse-noise weighting ~1400."
  echo "# A0 == A1 proves the proprio term is inert. A240/A1400 >> 13.33 proves"
  echo "# the weighting is the cause and costs no retraining to fix."
  for a in 0 1 50 240 1400 100000; do
    wait_for_slice; plan "A_alpha$a" "objective.alpha=$a"
  done
fi

# ------------------------------------------------------------------ STAGE 4
if want 4; then
  echo
  echo "################ STAGE 4: OPTIMISER vs OBJECTIVE ################"
  echo "# O_descent starts GD AT the ground-truth actions. If it FALLS from"
  echo "# H_oracle's score, the learned cost actively walks away from the correct"
  echo "# answer -- the objective is wrong, not the search."
  wait_for_slice; plan O_descent objective.alpha=1 planner.sub_planner.opt_steps=100 debug_dset_init=true
  wait_for_slice; plan O_randn   objective.alpha=1 \
      planner.sub_planner.sample_type=randn planner.sub_planner.action_noise=0.003
  wait_for_slice; plan O_opt1000 objective.alpha=1 planner.sub_planner.opt_steps=1000
fi

# ------------------------------------------------------------------ STAGE 5
if want 5; then
  echo
  echo "################ STAGE 5: IS THE PLANNER EXPLOITING THE MODEL? ################"
  echo "# Stage 3 result: alpha 0/1/50/240/1400/1e5 -> 0.12/0.12/0.24/0.26/0.16/0.00."
  echo "# Reweighting doubles it and saturates, so weighting was an amplifier, not"
  echo "# the binding constraint. Meanwhile H_oracle=1.0 (task-optimal actions exist"
  echo "# and work open-loop), beat=0.065 (6.5% of REAL action sequences already"
  echo "# score lower cost than those optimal actions) and visual snr=2.91 (the"
  echo "# controllable signal is 2.9x the rollout-error floor)."
  echo "#"
  echo "# That combination says GD is minimising the cost correctly and the cost is"
  echo "# wrong: 100 Adam steps over 50 free parameters find off-manifold actions"
  echo "# the model believes reach the goal. Two tests, either of which is decisive."

  echo
  echo "--- 5a: success vs GD steps, at the best alpha ---"
  echo "# eval_every makes the GD planner evaluate in the REAL env during"
  echo "# optimisation, so one job yields the whole curve. If success PEAKS EARLY"
  echo "# and then DECLINES as the cost keeps falling, the planner is provably"
  echo "# optimising against model error rather than task progress. A monotone"
  echo "# rise instead means the cost is sound and the budget is the limit."
  wait_for_slice
  plan X_steps_curve objective.alpha=240 planner.sub_planner.eval_every=10

  echo
  echo "--- 5b: CEM on the SAME checkpoint ---"
  echo "# CEM samples inside the action distribution and structurally cannot"
  echo "# exploit off-manifold gradient directions. If CEM >> GD's 0.26 on this"
  echo "# very model, the representation is adequate and the PLANNER is the"
  echo "# failure -- which is exactly the claim the submission rests on, since"
  echo "# LeWM plans with CEM and we are proposing to replace it with GD."
  wait_for_slice
  echo "=== C_cem_alpha240 ==="
  python plan.py --config-name plan_cem.yaml ckpt_base_path="$RUN" \
    model_name="$METHOD" model_epoch=latest decode_for_viz=false seed=100 \
    objective.alpha=240 objective.mode=last \
    hydra.run.dir="plan_diag/C_cem_alpha240" 2>&1 \
    | grep -E "Success rate|Error executing|OutOfMemory|Traceback"

  echo
  echo "--- 5c: does a trust region help GD? ---"
  echo "# If 5a declines, the cheap mitigation is to stop before the exploit:"
  echo "# fewer steps, and AdamW weight decay pulling actions toward the zero-init"
  echo "# prior so the search cannot wander off-distribution."
  for st in 10 25; do
    wait_for_slice
    plan "X_steps$st" objective.alpha=240 "planner.sub_planner.opt_steps=$st"
  done
  wait_for_slice
  plan X_trustregion objective.alpha=240 \
    planner.sub_planner.optimizer=adamw planner.sub_planner.adamw_weight_decay=0.1
fi

cat <<'EOF'

################ DECISION TABLE ################

 STAGE 1  rho_local < 0.2            -> ROOT CAUSE is latent geometry. The cost
                                        has no usable gradient toward the goal.
                                        Fix the representation; alpha and the
                                        optimiser are irrelevant.
          rho_local worse than the
          pristine DINOv2 row        -> end-to-end training DEGRADED alignment
                                        while straightness improved. Curvature
                                        was satisfied, its proxy target was not.
          rho_local >= 0.5           -> geometry is fine, go to stage 3.

 STAGE 2  H_oracle << 1.0            -> harness bug, discard every other number.
          H_floor ~= 0.13            -> the planner adds nothing over a fixed
                                        action sequence.

 STAGE 3  A0 == A1                   -> the proprio term is numerically inert,
                                        confirming alpha_eff = 0.004.
          A240 or A1400 >> 13.33     -> ROOT CAUSE is objective weighting.
                                        alpha=1 is not scale-free and unfreezing
                                        the trunk silently rescaled it.
          all alphas ~ 0.13          -> weighting is not the cause.

 STAGE 4  O_descent << H_oracle      -> the cost's minimum is not at the correct
                                        actions. Objective, not optimiser.
          O_descent ~ H_oracle       -> the cost is fine locally; zero-init lands
                                        in a bad basin. Cheapest fix of all.
          O_randn / O_opt1000 >> A1  -> search budget / initialisation.

 STAGE 5  5a peaks early then falls  -> MODEL EXPLOITATION. GD minimises the cost
                                        correctly; the cost stops tracking the
                                        task past some step count. Fix the
                                        planner (trust region / early stop), and
                                        note that CEM hides this, which is why
                                        LeWM never hit it.
          5a rises monotonically     -> the cost is sound; the ceiling is the
                                        representation, not the search.
          5b CEM >> 0.26             -> the model is adequate and GD is the
                                        failure. This is the central obstacle for
                                        a paper whose contribution is GD-for-JEPA.
          5b CEM ~= 0.26             -> the model itself cannot support planning
                                        at this horizon. Retraining is required;
                                        no planner-side fix will do.
          5c short/trust-region > A240 -> a cheap planner-side mitigation exists.

EOF
echo "############ DIAGNOSIS DONE $(date) ############"
