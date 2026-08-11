#!/usr/bin/env python3
"""
reproduce_table1.py  --  pure-Python (no bash) driver for the 5 Table-1 cells.

Runs the paper's exact GD protocol for each tracked run, with NO result mixing:
  - Planner: GD (open-loop = plan_gd; MPC = plan_gd_mpc, GD subplanner)
  - 50 samples, goal_H=25 env-steps -> H=5 model steps (frameskip 5)
  - 3 data seeds: 100 / 200 / 300
  - Table 4 hyperparams come from the plan configs (horizon 25, zero init, Adam,
    lr 0.1, 100 steps; open-loop executes 25, MPC executes 5)
  - Objectives (Sec 5.3):
      UMaze (images only)          -> open-loop mode=last, MPC mode=all,    alpha=0
      PushT (images + proprio)     -> open-loop mode=last, MPC mode=staged, alpha=1

No mixing: before each run, ONLY that run's plan_outputs are removed (basename-
scoped), so its logs.json holds exactly its 3 seeds; summarize_run reads ONLY that
run's logs.json and stores results/<run>.json + rebuilds results/table1_reproduction.*

Usage:
    python reproduce_table1.py                       # all 5 runs
    python reproduce_table1.py <run> [<run> ...]     # only the named run(s)
    python reproduce_table1.py --base /abs/checkpoints/test
Detached (survives disconnects; nohup is POSIX, not bash-specific):
    nohup python reproduce_table1.py > eval_all.log 2>&1 &
    tail -f eval_all.log
"""
import os
import re
import sys
import glob
import json
import shutil
import argparse
import subprocess

# ---- env defaults so NO shell exports are needed (export beforehand to override) ----
os.environ.setdefault("DATASET_DIR", "/workspace/arun/data")
# Fully disable wandb for headless eval: no login, no background service, no repo
# scanning. Results come from logs.json, not wandb. (offline still does work/threads.)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
# On this B200 MIG slice, torch 2.7's default caching allocator NVML-asserts;
# cudaMallocAsync avoids that NVML path and works. (Do NOT set CUDA_VISIBLE_DEVICES
# to the MIG UUID -- mujoco-py int()-parses it and crashes; leave it unset.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PLAN_SERIAL_ENV", "1")
# Cap CPU threads: on many-core nodes torch's default thread pool makes tiny CPU
# ops (e.g. DINOv2 trunc_normal_ weight init) pathologically slow due to
# thread-launch/sync overhead. 8 is plenty for dataloading/env; GPU does the math.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

# mujoco-py needs MuJoCo 210 + nvidia libs on LD_LIBRARY_PATH at import time.
# Set it here so the plan.py subprocess (which inherits os.environ) can import gym/env.
_ld = os.environ.get("LD_LIBRARY_PATH", "")
for _p in (os.path.expanduser("~/.mujoco/mujoco210/bin"), "/usr/lib/nvidia"):
    if _p not in _ld.split(":"):
        _ld = (_ld + ":" + _p) if _ld else _p
os.environ["LD_LIBRARY_PATH"] = _ld

# mujoco-py does int(CUDA_VISIBLE_DEVICES) to pick its render device; a MIG UUID
# (non-integer) crashes it. If it's set to something non-integer, unset it so
# rendering works (torch still sees the MIG device via the container).
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cvd and not all(p.strip().isdigit() for p in _cvd.split(",") if p.strip()):
    print(f"[driver] unsetting non-integer CUDA_VISIBLE_DEVICES={_cvd!r} (breaks mujoco-py)", flush=True)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import summarize_run  # reuse the run-scoped summarizer (must sit next to this file)

SEEDS = [100, 200, 300]

# name -> (alpha, mpc_mode). alpha 0 = target images only; 1 = images + proprio.
CFG = {
 "umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05":          (0, "all"),
 "umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06":         (0, "all"),
 "umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05": (0, "all"),
 "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06":         (1, "staged"),
 "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05": (1, "staged"),
}
ORDER = list(CFG)


# (alpha, mpc_mode) is a property of the ENVIRONMENT, not of the training
# objective: alpha=1 and mode=staged for PushT (images + proprio, terminal loss
# within H), alpha=0 and mode=all elsewhere. Sec 5.3.
ENV_DEFAULTS = {
    "umaze": (0, "all"),
    "medium": (0, "all"),
    "wall": (0, "all"),
    "pusht": (1, "staged"),
}


def base_cell(name):
    """Map a run name onto the Table-1 cell whose eval protocol it should use.

    Exact match first, so the five tracked cells behave exactly as before. Then
    strip a trailing _seed<N> for training-seed variants. Otherwise fall back to
    the leading environment token, which is what actually determines the eval
    protocol -- this is what lets objective variants (e.g. the SIGReg /
    end-to-end runs, '..._sgFalse_lr1e-05_sig1e-1_e2e') be evaluated with the
    right objective instead of being skipped as unknown.
    """
    if name in CFG:
        return name
    stripped = re.sub(r"_seed\d+$", "", name)
    if stripped in CFG:
        return stripped
    return None


def eval_protocol(name):
    """(alpha, mpc_mode) for a run, or None if the environment is unrecognised."""
    cell = base_cell(name)
    if cell is not None:
        return CFG[cell]
    env = name.split("_", 1)[0]
    if env in ENV_DEFAULTS:
        alpha, mode = ENV_DEFAULTS[env]
        print(f"   [driver] '{name}' is not a tracked cell; using the {env} "
              f"protocol (alpha={alpha}, mpc={mode}) from its env prefix.", flush=True)
        return alpha, mode
    return None


PLAN_ROOTS = {
    # arm -> (hydra output root, root config, extra overrides beyond alpha/seed)
    "gd":     ("plan_outputs_gd",     "plan_gd.yaml",     []),                 # open-loop GD
    "gd_mpc": ("plan_outputs_gd_mpc", "plan_gd_mpc.yaml", []),                 # MPC, GD subplanner
    "cem":    ("plan_outputs_cem",    "plan_cem.yaml",    ["objective.mode=last"]),  # open-loop CEM
}


def clean_scoped(name, arms=None):
    """Remove ONLY this run's plan_outputs so its logs.json holds exactly its 3 seeds."""
    roots = [PLAN_ROOTS[a][0] for a in (arms or PLAN_ROOTS)]
    for root in roots:
        for d in glob.glob(os.path.join(root, f"{name}_*")):
            shutil.rmtree(d, ignore_errors=True)


# plan.py already prints "[timing] perform_planning_s=<float>"; capture it so the
# GD-vs-CEM speed comparison has numbers instead of anecdote.
TIMING_RE = re.compile(r"\[timing\]\s*perform_planning_s=([0-9.]+)")


def run_plan(cfg_name, run_dir, name, extra):
    """Run one plan.py job. Returns (returncode, perform_planning_seconds|None).

    stdout is piped so the timing line can be parsed and echoed; stderr is left
    attached to the terminal so tqdm progress bars still render live.
    """
    cmd = [sys.executable, "plan.py", "--config-name", cfg_name,
           f"ckpt_base_path={run_dir}", f"model_name={name}", "model_epoch=latest",
           "decode_for_viz=false"] + extra
    print("   $ " + " ".join(cmd), flush=True)
    secs = None
    proc = subprocess.Popen(cmd, env=os.environ, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        m = TIMING_RE.search(line)
        if m:
            secs = float(m.group(1))
    proc.stdout.close()
    return proc.wait(), secs


def run_eval(name, base, arms=("gd", "gd_mpc")):
    protocol = eval_protocol(name)
    if protocol is None:
        print(f"!!! SKIP {name}: unknown env prefix (no alpha/mode mapping). "
              f"Known: {sorted(ENV_DEFAULTS)}", flush=True)
        return
    alpha, mpc_mode = protocol
    run_dir = os.path.join(base, name)
    print("\n" + "#" * 60, flush=True)
    print(f"# RUN: {name}\n#   alpha={alpha}  open-loop=last  mpc={mpc_mode}", flush=True)
    print("#" * 60, flush=True)
    if not (os.path.isfile(os.path.join(run_dir, "hydra.yaml"))
            and os.path.isfile(os.path.join(run_dir, "checkpoints", "model_latest.pth"))):
        print(f"!!! SKIP {name}: missing hydra.yaml or checkpoints/model_latest.pth under {run_dir}", flush=True)
        return

    clean_scoped(name, arms)

    timings = {}
    for arm in arms:
        root, cfg_name, extra = PLAN_ROOTS[arm]
        over = list(extra) + [f"objective.alpha={alpha}"]
        if arm == "gd_mpc":
            over.append(f"objective.mode={mpc_mode}")
        label = {"gd": "OPEN-LOOP GD (execute 25)",
                 "gd_mpc": f"MPC GD (mode={mpc_mode}, execute 5)",
                 "cem": "OPEN-LOOP CEM (execute 25)"}[arm]
        print(f">>> {label}  [{cfg_name}]", flush=True)
        secs = []
        for s in SEEDS:
            rc, t = run_plan(cfg_name, run_dir, name, over + [f"seed={s}"])
            if rc:
                print(f"FAIL {arm} {name} seed={s}", flush=True)
            if t is not None:
                secs.append(t)
        if secs:
            timings[arm] = secs
            print(f"   [time] {arm}: {sum(secs)/len(secs):.1f} s mean over "
                  f"{len(secs)} seed(s)", flush=True)

    if timings:
        os.makedirs("results", exist_ok=True)
        tpath = os.path.join("results", f"{name}.timing.json")
        # Merge, don't clobber: a later `--planners cem` re-run must keep the
        # gd/gd_mpc timings recorded by the earlier full pass.
        if os.path.isfile(tpath):
            try:
                with open(tpath) as f:
                    prev = json.load(f)
            except Exception:
                prev = {}
            prev.update(timings)
            timings = prev
        with open(tpath, "w") as f:
            json.dump(timings, f, indent=2)

    # Immediate, run-scoped summary + results/<name>.json
    summarize_run.summarize_one(name)


def main():
    ap = argparse.ArgumentParser(description="Pure-Python Table-1 reproduction driver (no bash).")
    ap.add_argument("runs", nargs="*", help="run basenames to evaluate (default: all 5 tracked runs)")
    ap.add_argument("--base", default=os.path.join(os.getcwd(), "checkpoints", "test"),
                    help="folder containing the run dirs (default ./checkpoints/test)")
    ap.add_argument("--planners", default="gd,gd_mpc",
                    help="comma-separated arms from " + ",".join(PLAN_ROOTS) +
                         ". Default reproduces the paper's GD-only Table 1; add "
                         "'cem' for the GD-vs-CEM speed/success comparison.")
    args = ap.parse_args()
    arms = [a.strip() for a in args.planners.split(",") if a.strip()]
    bad = [a for a in arms if a not in PLAN_ROOTS]
    if bad:
        ap.error(f"unknown planner arm(s) {bad}; choose from {sorted(PLAN_ROOTS)}")
    runs = args.runs if args.runs else ORDER
    print(f"BASE={args.base}  DATASET_DIR={os.environ['DATASET_DIR']}  "
          f"runs={len(runs)}  arms={arms}", flush=True)
    for name in runs:
        run_eval(name, args.base, arms=arms)
    print("\n############### FINAL TABLE-1 REPRODUCTION ###############", flush=True)
    summarize_run.rebuild_master()
    print("\nALL EVALS DONE", flush=True)


if __name__ == "__main__":
    main()
