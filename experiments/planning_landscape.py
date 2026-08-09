#!/usr/bin/env python3
"""planning_landscape.py -- 2-D slices of the planning cost J(a).

Why this exists
---------------
The submission plan promises loss-landscape figures as evidence that
straightening makes the planning objective easier to optimise. This script
produces the per-run data: for each eval window it GD-optimises the action
sequence from the protocol's zero initialisation, then slices the terminal
cost J(a) = || z_H(a) - z_g ||^2 (+ alpha * proprio) on a 2-D plane through
the solution:

    axis 1:  toward the ground-truth actions (the direction that matters)
    axis 2:  a random orthogonal direction (the direction it should not matter)

A well conditioned landscape is round: the curvature along both axes is
comparable and the GD solution sits near the bottom of the slice. A needle
landscape (huge anisotropy) or one where a* sits on a ridge says gradient
planning is fighting the geometry, no matter how good the optimiser is.

Usage
-----
    python experiments/planning_landscape.py checkpoints/test/<run> [<run2> ...]
    python experiments/planning_landscape.py <run> --samples 12 --grid 15 --out final_eval

Writes <out>/landscape_<run>.json and (if matplotlib is installed) .png.
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


def terminal_cost(model, obs0, act_row, num_hist, K, goal, alpha):
    """Terminal (mode='last') cost: visual MSE + alpha * proprio MSE, per sample."""
    z_obs, _ = model.rollout(obs0, act_row)
    j = num_hist + K - 1
    vis = z_obs["visual"][:, j].flatten(1)
    c = (vis - goal["visual"]).pow(2).mean(1)
    if alpha > 0:
        prop = z_obs["proprio"][:, j]
        c = c + alpha * (prop - goal["proprio"]).pow(2).mean(1)
    return c                                            # (b,)


def gd_solve(model, obs0, ctx, num_hist, K, goal, alpha, ad,
             steps=100, lr=0.1, device="cpu"):
    """Protocol GD: zero init, Adam, 100 steps. Returns optimised future actions."""
    fut = torch.zeros(1, K, ad, device=device, requires_grad=True)
    opt = torch.optim.Adam([fut], lr=lr)
    for _ in range(steps):
        act_row = torch.cat([ctx, fut], dim=1)
        c = terminal_cost(model, obs0, act_row, num_hist, K, goal, alpha).mean()
        opt.zero_grad()
        c.backward()
        opt.step()
    return fut.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="checkpoint run dirs (containing hydra.yaml)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--gd-steps", type=int, default=100)
    ap.add_argument("--gd-lr", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=None,
                    help="proprio weight (default: env convention, 1 for pusht)")
    ap.add_argument("--goal-h", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output dir (default: <run>/final_eval)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = np.random.default_rng(args.seed)

    for run in args.runs:
        run = os.path.abspath(run.rstrip("/\\"))
        cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))
        frameskip = int(cfg.frameskip)
        num_hist = int(cfg.num_hist)
        K = args.goal_h // frameskip
        alpha = args.alpha
        if alpha is None:
            alpha = 1.0 if "pusht" in str(cfg.env.name).lower() else 0.0

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
        n = act.shape[0]
        ad = act.shape[-1]
        M = K * ad

        t = np.linspace(-1.5, 1.5, args.grid)
        grids = []
        scal = {"J_star": [], "J_gt": [], "J_zero": [], "J_min": [],
                "curv_d1": [], "curv_d2": [], "dist_gt_star": []}

        for i in range(n):
            obs0 = {k: v[i : i + 1, :num_hist].to(device) for k, v in obs.items()}
            row = act[i : i + 1, : num_hist + K].to(device)
            ctx = row[:, :num_hist]
            j = num_hist + K - 1
            with torch.no_grad():
                z_obs, _ = model.rollout(obs0, row)
                goal = {"visual": z_obs["visual"][:, j].flatten(1),
                        "proprio": z_obs["proprio"][:, j]}

            a_star = gd_solve(model, obs0, ctx, num_hist, K, goal, alpha, ad,
                              steps=args.gd_steps, lr=args.gd_lr, device=device)
            gt = row[:, num_hist:].reshape(1, M)
            a_star_f = a_star.reshape(1, M)

            with torch.no_grad():
                def J_of(fut_flat):
                    act_row = torch.cat(
                        [ctx, fut_flat.reshape(1, K, ad)], dim=1)
                    return float(terminal_cost(model, obs0, act_row, num_hist,
                                               K, goal, alpha).mean())

                J_star = J_of(a_star_f)
                J_gt = J_of(gt)
                J_zero = J_of(torch.zeros_like(gt))

            d1 = (gt - a_star_f).reshape(-1)
            dist = float(d1.norm())
            if dist < 1e-6:
                d1 = torch.tensor(gen.standard_normal(M),
                                  device=device, dtype=torch.float32)
                dist = 1.0
            d1 = d1 / d1.norm()
            r = torch.tensor(gen.standard_normal(M),
                             device=device, dtype=torch.float32)
            d2 = r - (r @ d1) * d1
            d2 = d2 / d2.norm().clamp_min(1e-12)

            r_scale = max(dist, 1e-3)
            grid = np.zeros((args.grid, args.grid))
            with torch.no_grad():
                for xi, tx in enumerate(t):
                    for yi, ty in enumerate(t):
                        fut = a_star_f + r_scale * (tx * d1 + ty * d2)
                        grid[xi, yi] = J_of(fut)
            grids.append(grid)
            scal["J_star"].append(J_star)
            scal["J_gt"].append(J_gt)
            scal["J_zero"].append(J_zero)
            scal["J_min"].append(float(grid.min()))
            scal["dist_gt_star"].append(dist)
            c = int(args.grid // 2)
            scal["curv_d1"].append((grid[0, c] + grid[-1, c] - 2 * grid[c, c])
                                   / max(r_scale ** 2, 1e-12))
            scal["curv_d2"].append((grid[c, 0] + grid[c, -1] - 2 * grid[c, c])
                                   / max(r_scale ** 2, 1e-12))

        mean_grid = np.mean(grids, axis=0)
        scal = {k: float(np.mean(v)) for k, v in scal.items()}
        aniso = scal["curv_d1"] / scal["curv_d2"] if scal["curv_d2"] > 0 else float("inf")

        print("\n" + "=" * 78)
        print(f"RUN  {os.path.basename(run)}   (K={K}, alpha={alpha}, n={n})")
        print("=" * 78)
        print(f"  J(zero)={scal['J_zero']:.4g}  J(a*)={scal['J_star']:.4g}  "
              f"J(gt)={scal['J_gt']:.4g}  min on grid={scal['J_min']:.4g}")
        print(f"  curvature toward gt: {scal['curv_d1']:.4g}   "
              f"orthogonal: {scal['curv_d2']:.4g}   anisotropy={aniso:.3g}")
        print(f"  mean ||gt - a*|| = {scal['dist_gt_star']:.4g}")
        if aniso > 100:
            print("  -> needle landscape: the cost only bends toward the true "
                  "actions in ~1 direction. GD from zero init will struggle "
                  "whatever the lr; this is the geometry claim to fix/report.")

        out_dir = args.out or os.path.join(run, "final_eval")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(run)
        rec = {"run": base, "K": K, "alpha": alpha, "n_samples": n,
               "grid_t": t.tolist(), "mean_grid": mean_grid.tolist(),
               "scalars": scal, "anisotropy": aniso}
        jpath = os.path.join(out_dir, f"landscape_{base}.json")
        with open(jpath, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  wrote {jpath}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            levels = np.linspace(mean_grid.min(),
                                 np.quantile(mean_grid, 0.98), 24)
            cs = ax.contourf(t, t, mean_grid.T, levels=levels, cmap="viridis")
            ax.plot(0, 0, "w*", ms=14, label="a* (GD solution)")
            if scal["dist_gt_star"] > 1e-6:
                ax.plot(1, 0, "r^", ms=10, label="ground-truth actions")
            fig.colorbar(cs, ax=ax, label="terminal cost J")
            ax.set_xlabel("toward ground-truth actions")
            ax.set_ylabel("random orthogonal direction")
            ax.set_title(f"planning cost slice -- {base}\n"
                         f"aniso={aniso:.3g}  J*={scal['J_star']:.3g}")
            ax.legend(loc="upper right")
            png = os.path.join(out_dir, f"landscape_{base}.png")
            fig.savefig(png, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {png}")
        except Exception as e:                          # matplotlib optional
            print(f"  (no figure: {e})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
