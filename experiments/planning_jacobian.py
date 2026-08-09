#!/usr/bin/env python3
"""planning_jacobian.py -- the condition number of the planning map a -> z_H.

Why this exists
---------------
The Hessian of the planning cost J(a) = || z_H(a) - z_g ||^2 splits into

    d^2J/da^2 = 2 (dz_H/da)^T (dz_H/da)  +  2 (z_H - z_g) . d^2z_H/da^2 .

Temporal straightening (the cosine-curvature term) only addresses the SECOND
factor, and only along data trajectories. The FIRST factor -- the action
Jacobian of the terminal latent -- is what decides whether gradient planning
can move at all, and it was never measured in this project. This script
measures it, plus the two offline quantities the tailored training terms
(cf_curv, act_sens) are supposed to improve.

What it measures, per run
-------------------------
  jacobian        J = dz_H/da at the ground-truth actions, central finite
                  differences over every future-action dimension.
  cond_proxy      sigma_max / sigma_min of J. A huge value means the cost
                  surface is a needle: gradient descent crawls along the flat
                  directions and overshoots along the steep one. CEM does not
                  care; that is exactly why straightening alone may lift GD
                  less than hoped.
  eff_rank        entropy-effective rank of J's singular spectrum: how many
                  action directions the terminal latent responds to at all.
  snr / beat      from rollout_drift.measure, for direct cross-reading.
  cf_cos_true     straightness (mean cos between consecutive velocities) of
                  rollouts under the TRUE actions.
  cf_cos_alt      straightness under WRONG actions (another window's future
                  held constant) -- the counterfactual map the cf_curv term
                  trains and the planner actually lives in.

Usage
-----
    python experiments/planning_jacobian.py checkpoints/test/<run> [<run2> ...]
    python experiments/planning_jacobian.py <run> --windows 24 --json out.json

Read-only. Finite differences: 2 * K * action_dim rollouts per window, so
keep --windows modest (24 ~= a few minutes on the MIG slice for PushT).
"""

import argparse
import json
import math
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
from rollout_drift import collect_windows, measure  # noqa: E402


@torch.no_grad()
def terminal_visual(model, obs0, act_row, num_hist, K):
    """Flattened visual tokens at horizon K. Indexing matches rollout_drift:
    latent j lines up with real frame j, so horizon K is num_hist + K - 1."""
    z_obs, _ = model.rollout(obs0, act_row)
    j = num_hist + K - 1
    return z_obs["visual"][:, j].flatten(1)


@torch.no_grad()
def jacobian_fd(model, obs0, act_row, num_hist, K, delta):
    """(D, M) Jacobian of the terminal visual latent w.r.t. future actions,
    central finite differences. M = K * action_dim."""
    a = act_row
    M = K * a.shape[-1]
    cols = []
    for j in range(M):
        outs = []
        for s in (+delta, -delta):
            pert = a.clone()
            pert[:, num_hist : num_hist + K].reshape(1, M)[0, j] += s
            outs.append(terminal_visual(model, obs0, pert, num_hist, K))
        cols.append((outs[0] - outs[1]) / (2.0 * delta))
    return torch.cat(cols, dim=0).T                       # (D, M)


def jac_stats(J):
    s = torch.linalg.svdvals(J.float())
    smax = float(s[0])
    smin = float(s[-1])
    cond = smax / max(smin, 1e-12)
    e = s.pow(2)
    p = e / e.sum().clamp_min(1e-30)
    p = p[p > 0]
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    top1 = float(e[0] / e.sum().clamp_min(1e-30))
    active = int((s > 1e-3 * smax).sum())
    return {"sigma_max": smax, "sigma_min": smin, "cond": cond,
            "eff_rank": eff_rank, "top1_energy": top1, "active_dims": active}


@torch.no_grad()
def traj_cos(model, obs0, act_row, num_hist, K, thresh=1e-6):
    """Mean cos between consecutive velocities of the visual-token rollout."""
    z_obs, _ = model.rollout(obs0, act_row)
    z = z_obs["visual"][:, : num_hist + K].flatten(2)     # (1, T, D)
    v = z[:, 1:] - z[:, :-1]
    n1, n2 = v[:, :-1].norm(dim=-1), v[:, 1:].norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)
    mask = (n1 > thresh) & (n2 > thresh)
    if not bool(mask.any()):
        return float("nan")
    return float(cos[mask].mean())


@torch.no_grad()
def counterfactual_straightness(model, obs, act, num_hist, K, device,
                                n_alts=8, gen=None):
    """Straightness under the true future actions vs under someone else's."""
    cos_true, cos_alt = [], []
    n = act.shape[0]
    for i in range(n):
        obs0 = {k: v[i : i + 1, :num_hist].to(device) for k, v in obs.items()}
        row = act[i : i + 1, : num_hist + K].to(device)
        cos_true.append(traj_cos(model, obs0, row, num_hist, K))
        for _ in range(n_alts):
            o = int(torch.randint(0, n, (1,), generator=gen))
            if o == i and n > 1:
                o = (o + 1) % n
            alt = row.clone()
            alt[:, num_hist:] = act[o : o + 1, num_hist : num_hist + K].to(device)
            cos_alt.append(traj_cos(model, obs0, alt, num_hist, K))
    return (float(np.nanmean(cos_true)), float(np.nanmean(cos_alt)))


def report(name, res, K):
    print("\n" + "=" * 78)
    print(f"RUN  {name}    (horizon K={K} model steps)")
    print("=" * 78)
    j = res["jacobian"]
    print(f"  jacobian:      sigma_max={j['sigma_max']:.4g}  "
          f"sigma_min={j['sigma_min']:.4g}")
    print(f"  cond_proxy:    {j['cond']:.4g}   (lower = rounder cost surface)")
    print(f"  eff_rank:      {j['eff_rank']:.2f} / {j['active_dims']} active "
          f"dims   top-1 energy {j['top1_energy'] * 100:.1f}%")
    v = res["rollout_drift"]["visual"]
    h = min(res["horizon"], res["K"]) - 1
    print(f"  snr@K={h+1}:    {v['snr'][h]:.2f}   beat={v['beat'][h]:.3f}   "
          f"drift={v['drift'][h]:.3f}")
    print(f"  cf_cos true:   {res['cf_cos_true']:+.3f}   "
          f"cf_cos alt: {res['cf_cos_alt']:+.3f}")
    if j["cond"] > 1e3:
        print("  -> the cost surface is a needle at the ground-truth actions: "
              "GD will crawl/overshoot regardless of the optimiser settings. "
              "This is the first-order gap straightening does NOT touch.")
    if res["cf_cos_alt"] < res["cf_cos_true"] - 0.2:
        print("  -> counterfactual rollouts are much more curved than data "
              "rollouts: the data-trajectory curvature term did not generalise "
              "off the data distribution. cf_curv is the term for that.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="checkpoint run dirs (containing hydra.yaml)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--windows", type=int, default=24,
                    help="windows for the jacobian (2*K*ad rollouts each)")
    ap.add_argument("--alts", type=int, default=8,
                    help="counterfactual action sequences per window")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="finite-difference step (actions are normalised)")
    ap.add_argument("--goal-h", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_res = {}

    for run in args.runs:
        run = os.path.abspath(run.rstrip("/\\"))
        cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))
        frameskip = int(cfg.frameskip)
        num_hist = int(cfg.num_hist)
        horizon = args.goal_h // frameskip
        K = horizon

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
        obs, act = collect_windows(dset, num_hist, K, frameskip, args.windows, rng)

        rd = measure(model, obs, act, num_hist, K, device,
                     n_alts=args.alts, rng=rng)

        js = []
        for i in range(act.shape[0]):
            obs0 = {k: v[i : i + 1, :num_hist].to(device) for k, v in obs.items()}
            row = act[i : i + 1, : num_hist + K].to(device)
            J = jacobian_fd(model, obs0, row, num_hist, K, args.delta)
            js.append(jac_stats(J))
        agg = {k: float(np.mean([x[k] for x in js])) for k in js[0]}

        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        cos_t, cos_a = counterfactual_straightness(
            model, obs, act, num_hist, K, device, n_alts=args.alts, gen=gen)

        res = {
            "jacobian": agg,
            "jacobian_per_window": js,
            "rollout_drift": rd,
            "cf_cos_true": cos_t,
            "cf_cos_alt": cos_a,
            "horizon": horizon, "K": K,
            "windows": act.shape[0], "delta": args.delta,
        }
        report(os.path.basename(run), res, K)
        all_res[os.path.basename(run)] = res

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"\nwrote {args.json}")

    if len(all_res) > 1:
        print("\n" + "=" * 78)
        print("COMPARISON (cond lower is better, cf_cos higher is straighter)")
        print("=" * 78)
        print(f"  {'cond':>10}  {'eff_rank':>8}  {'cf_true':>8}  {'cf_alt':>8}   run")
        for name, r in all_res.items():
            j = r["jacobian"]
            c = j["cond"]
            c_str = f"{c:.3g}" if math.isfinite(c) else "inf"
            print(f"  {c_str:>10}  {j['eff_rank']:>8.2f}  "
                  f"{r['cf_cos_true']:>+8.3f}  {r['cf_cos_alt']:>+8.3f}   {name}")


if __name__ == "__main__":
    main()
