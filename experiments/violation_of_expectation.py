#!/usr/bin/env python3
"""violation_of_expectation.py -- VoE analysis for the submission plan.

Why this exists
---------------
Contribution 5 of the submission plan lists "violation-of-expectation" among
the clean analyses. The idea (as in world-model evaluations): a model that has
learned the environment's physics should be *surprised* -- i.e. assign a large
latent prediction error -- to continuations that violate those physics, and
calmly predict continuations that obey them.

For each eval window we take the model's own rollout future z_pred (predicted
from the true history and true actions) and measure its latent distance to
four encoded continuations that share the SAME history:

    true     the real future frames and actions (expectation confirmed)
    static   the world freezes: last history frame repeated, true actions
    reverse  the future played backwards, true actions
    foreign  another trajectory's future frames AND its actions

A working world model ranks err(true) << err(static), err(reverse),
err(foreign). We report the per-violation error ratio (viol / true) and the
fraction of windows where the violation surprised the model more than the
true continuation. A ratio near 1 means the predictor is not modelling the
physics -- it is memorising visual texture.

Usage
-----
    python experiments/violation_of_expectation.py checkpoints/test/<run> [<run2> ...]
    python experiments/violation_of_expectation.py <run> --samples 64 --out final_eval

Writes <out>/voe_<runname>.json and prints a compact table.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra  # noqa: E402
import custom_resolvers  # noqa: F401,E402
from plan import load_model  # noqa: E402
from rollout_drift import collect_windows  # noqa: E402


def future_visual(model, z, num_hist, K):
    """Visual-only latent slice of the K future steps, flattened per frame."""
    v = model.visual_only(z)[:, num_hist : num_hist + K]
    return v.flatten(1)                                  # (b, K*p*d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="checkpoint run dirs (containing hydra.yaml)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--goal-h", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output dir (default: <run>/final_eval)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for run in args.runs:
        run = os.path.abspath(run.rstrip("/\\"))
        cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))
        frameskip = int(cfg.frameskip)
        num_hist = int(cfg.num_hist)
        K = args.goal_h // frameskip

        _, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                   num_pred=cfg.num_pred, frameskip=frameskip)
        dset = traj["valid"]

        from pathlib import Path
        ckpt = Path(os.path.join(run, "checkpoints", f"model_{args.epoch}.pth"))
        if not ckpt.exists():
            raise SystemExit(f"No checkpoint at {ckpt}.")
        model = load_model(ckpt, cfg, cfg.num_action_repeat, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        rng = np.random.default_rng(args.seed)
        obs, act = collect_windows(dset, num_hist, K, frameskip, args.samples, rng)
        obs = {k: v.to(device) for k, v in obs.items()}
        act = act.to(device)
        n = act.shape[0]

        hist_obs = {k: v[:, :num_hist] for k, v in obs.items()}
        with torch.no_grad():
            # The model's own expectation: rollout future under true actions.
            # rollout() returns (separated obs dict, concatenated z tensor);
            # visual_only() needs the concatenated tensor.
            _, z_pred = model.rollout(hist_obs, act)
            z_pred = future_visual(model, z_pred, num_hist, K)

            # Encoded continuations sharing the same history.
            z_true = future_visual(model, model.encode(obs, act), num_hist, K)

            def enc_with(fut_obs, fut_act):
                o = {k: torch.cat([hist_obs[k], v], dim=1) for k, v in fut_obs.items()}
                a = torch.cat([act[:, :num_hist], fut_act], dim=1)
                return future_visual(model, model.encode(o, a), num_hist, K)

            last = obs["visual"][:, num_hist - 1 : num_hist]
            z_static = enc_with(
                {"visual": last.repeat(1, K, 1, 1, 1),
                 "proprio": obs["proprio"][:, num_hist - 1 : num_hist].repeat(1, K, 1)},
                act[:, num_hist:])
            z_reverse = enc_with(
                {k: torch.flip(obs[k][:, num_hist:], [1]) for k in obs},
                act[:, num_hist:])
            perm = torch.randperm(n, device=device)
            z_foreign = enc_with(
                {k: obs[k][:, num_hist:][perm] for k in obs},
                act[:, num_hist:][perm])

        def stats(z_viol, name):
            err_v = (z_pred - z_viol).pow(2).mean(1)
            err_t = (z_pred - z_true).pow(2).mean(1)
            ratio = (err_v / err_t.clamp_min(1e-12))
            return {
                "name": name,
                "err_true_mean": float(err_t.mean()),
                "err_viol_mean": float(err_v.mean()),
                "ratio_mean": float(ratio.mean()),
                "ratio_std": float(ratio.std()),
                "frac_viol_greater": float((err_v > err_t).float().mean()),
            }

        rows = [stats(z_true, "true"),
                stats(z_static, "static"),
                stats(z_reverse, "reverse"),
                stats(z_foreign, "foreign")]

        runname = os.path.basename(run)
        print(f"\n=== VoE: {runname} (n={n}, K={K}) ===")
        print(f"{'continuation':<10} {'err_true':>10} {'err_viol':>10} "
              f"{'ratio':>10} {'P(viol>true)':>12}")
        for r in rows:
            print(f"{r['name']:<10} {r['err_true_mean']:>10.5f} "
                  f"{r['err_viol_mean']:>10.5f} "
                  f"{r['ratio_mean']:>8.2f}±{r['ratio_std']:.2f} "
                  f"{r['frac_viol_greater']:>12.3f}")

        out_dir = args.out or os.path.join(run, "final_eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"voe_{runname}.json")
        with open(out_path, "w") as f:
            json.dump({"run": runname, "n": n, "K": K, "seed": args.seed,
                       "rows": rows}, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
