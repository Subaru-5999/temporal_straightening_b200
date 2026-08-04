#!/usr/bin/env python3
"""rollout_drift.py -- where does the latent rollout stop being usable?

Why this exists
---------------
The PushT end-to-end + SIGReg run scores 13.3 open-loop and 56 MPC. The paper's
open-loop -> MPC gap is ~+8 (70.0 -> 78.7 and 77.3 -> 85.3); ours is +43. MPC
re-encodes a real observation every frameskip steps and commits only the first
model step of each plan, while open-loop commits all `goal_H / frameskip` model
steps from a single encode. A +43 gap therefore says one-step prediction is
roughly sound and the AUTOREGRESSIVE multi-step rollout is not -- which is not
something a collapsed representation can produce.

That is a claim about `VWorldModel.rollout`, and it is measurable directly on a
checkpoint with no environment, no MuJoCo and no planner.

What it measures, per horizon k = 1..K
--------------------------------------
  rollout_mse[k]   || z_rollout[k] - z_real[k] ||^2, autoregressive: the
                   predictor is fed its OWN previous output, exactly as planning
                   does it.
  teacher_mse[k]   the same target, but predicted one step from the REAL encoded
                   window ending at k-1. This is the only thing training ever
                   optimises. No compounding.
  spread[k]        E|| z_real[k] - z_real'[k] ||^2 over pairs of DIFFERENT
                   trajectories at the same time index: how far apart two
                   genuinely different states are in this latent space.
  drift[k]         rollout_mse[k] / spread[k]
  compound[k]      rollout_mse[k] / teacher_mse[k]

  reach[k]         E|| z_rollout[k](a) - z_rollout[k](a') ||^2 for a DIFFERENT
                   real action sequence a' from the SAME initial frames: how
                   much of the latent the planner can actually move. This, not
                   `spread`, is the right denominator -- `spread` is inflated by
                   initial-state variation that no action can control.
  snr[k]           reach[k] / rollout_mse[k]       <-- the number that matters
  beat[k]          fraction of alternative action sequences whose terminal cost
                   MSE(z_rollout[k](a'), z_real[k]) is LOWER than the ground
                   truth actions' own cost. 0 = the true actions are the
                   minimiser. ~0.5 = the cost cannot tell right actions from
                   wrong ones, and no optimiser can rescue that.

How to read it
--------------
`drift` is scale-free, so it is comparable across checkpoints with different
latent scales (which frozen-trunk and end-to-end runs certainly have).

`snr[k]` is the planning signal-to-noise ratio at horizon k: how far actions can
move the terminal latent, divided by how wrong the rollout is when it gets there.

  snr[k] >> 1   the cost surface is dominated by the action's effect. Planning
                is well posed at horizon k.
  snr[k] ~ 1    action effect and rollout error are the same size. The cost
                surface is noise at the scale of the thing being optimised.
  beat[k] -> 0  the true actions minimise the cost, so a good optimiser finds
                them. beat[k] -> 0.5 means the cost is blind to the actions and
                the failure is the objective, not the planner.

`drift[k]` uses `spread` instead and so answers a weaker question -- whether the
rollout still carries state information at all. It can look healthy while snr is
~1, because most of `spread` is initial-state variation the planner never
controls. Read snr and beat first.

`compound` separates two failure modes: compound ~ 1 means multi-step is no
worse than one-step and the predictor is simply inaccurate; compound >> 1 means
error is amplified by feeding predictions back, i.e. the predictor's own outputs
are off the encoder's manifold.

Per-key scale matters too. The planning objective is
`loss_visual + alpha * loss_proprio`, so comparing `spread` between the visual
and proprio rows gives the term's EFFECTIVE weight, which can be orders of
magnitude away from the configured alpha if end-to-end training shrank the
proprio embedding.

Usage
-----
    python experiments/rollout_drift.py checkpoints/test/<run> [<run2> ...]
    python experiments/rollout_drift.py <run> --k 8 --windows 256 --json out.json

Runs on whatever device is available; a few hundred windows takes ~1-2 min on
the MIG slice. Read-only: it never writes into the run directory.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra  # noqa: E402
import custom_resolvers  # noqa: F401,E402  (registers OmegaConf resolvers)
from plan import load_model  # noqa: E402  same loader the real evaluation uses


def _mse_per_sample(a, b):
    """(b, ...) -> (b,) mean squared error over every non-batch axis."""
    return (a.float() - b.float()).pow(2).flatten(1).mean(1)


@torch.no_grad()
def collect_windows(dset, num_hist, K, frameskip, n_windows, rng):
    """Sample fixed-length windows of (obs, act) at the model's frameskip.

    Returns obs dict of (n, num_hist+K, ...) and act of (n, num_hist+K, f*d).

    Both must have the SAME number of model steps: VWorldModel.encode()
    concatenates the tiled action embedding onto the visual tokens along the
    channel axis, so a mismatched time axis is a shape error, not a warning.
    rollout() then takes act[:, :num_hist] as the context actions and rolls the
    remaining K, emitting num_hist+K+1 frames; latent index j lines up with real
    frame j, and j >= num_hist is the autoregressive part.
    """
    n_frames = num_hist + K                      # subsampled frames needed
    raw_span = n_frames * frameskip + 1          # raw frames needed (act binds)
    obs_parts, act_parts = [], []
    tries = 0
    while len(act_parts) < n_windows and tries < 50 * n_windows:
        tries += 1
        i = int(rng.integers(0, len(dset)))
        obs, act, _state, _info = dset[i]
        T = min(obs["visual"].shape[0], act.shape[0] + 1)
        if T < raw_span:
            continue
        start = int(rng.integers(0, T - raw_span + 1))
        w_obs = {
            k: v[start : start + (n_frames - 1) * frameskip + 1 : frameskip]
            for k, v in obs.items()
        }
        if w_obs["visual"].shape[0] != n_frames:
            continue
        a = act[start : start + n_frames * frameskip]
        if a.shape[0] != n_frames * frameskip:
            continue
        a = rearrange(a, "(h f) d -> h (f d)", f=frameskip)
        obs_parts.append(w_obs)
        act_parts.append(a)
    if not act_parts:
        raise RuntimeError(
            f"No trajectory in the valid split is long enough for "
            f"num_hist={num_hist} + K={K} at frameskip={frameskip} "
            f"({raw_span} raw frames). Lower --k."
        )
    obs = {k: torch.stack([w[k] for w in obs_parts]) for k in obs_parts[0]}
    return obs, torch.stack(act_parts)


@torch.no_grad()
def measure(model, obs, act, num_hist, K, device, chunk=16, n_alts=8, rng=None):
    """Per-horizon rollout / teacher-forced / spread / reachability statistics."""
    keys = ("visual", "proprio")
    names = ("rollout", "teacher", "spread", "reach")
    acc = {k: {n: np.zeros(K) for n in names} for k in keys}
    # beat: alternative action sequences that score BETTER than the true ones
    beat = {k: np.zeros(K) for k in keys}
    beat_n = np.zeros(K)
    counts = np.zeros(K)
    n = act.shape[0]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0 if rng is None else int(rng.integers(0, 2**31 - 1)))

    for s in range(0, n, chunk):
        o = {k: v[s : s + chunk].to(device) for k, v in obs.items()}
        a = act[s : s + chunk].to(device)
        b = a.shape[0]
        if b < 2:                                 # spread/reach need a pair
            continue

        # ground truth: encode every real frame once
        z_real_obs = model.encode_obs(o)                       # dict of (b, T, ...)
        z_real = model.encode(o, a)                            # (b, T, p, d) full token

        # autoregressive rollout from the first num_hist frames, real actions
        o0 = {k: v[:, :num_hist] for k, v in o.items()}
        z_roll_obs, _ = model.rollout(o0, a)                   # (b, T, ...)

        # --- counterfactual rollouts: same start, DIFFERENT real action seqs ---
        # Real sequences from other windows, not Gaussian noise, so the
        # alternatives stay inside the action distribution the model was
        # trained on. Context actions are held fixed: they are part of the
        # encode, so changing them would change the starting latent too.
        alt_rolls = []
        for _ in range(n_alts):
            p = torch.randperm(b, generator=gen).to(device)
            if bool((p == torch.arange(b, device=device)).all()):
                continue
            a_alt = a.clone()
            a_alt[:, num_hist:] = a[p][:, num_hist:]
            z_alt_obs, _ = model.rollout(o0, a_alt)
            alt_rolls.append(z_alt_obs)

        for k in range(1, K + 1):
            j = num_hist + k - 1                               # frame index
            # --- teacher forced: one step from the REAL window ending at j-1 ---
            src = z_real[:, j - num_hist : j]                  # (b, num_hist, p, d)
            z_tf_obs, _ = model.separate_emb(model.predict(src))
            perm = torch.roll(torch.arange(b, device=device), 1)
            for key in keys:
                tgt = z_real_obs[key][:, j]
                roll = z_roll_obs[key][:, j]
                tf = z_tf_obs[key][:, -1]
                cost_true = _mse_per_sample(roll, tgt)
                acc[key]["rollout"][k - 1] += float(cost_true.sum())
                acc[key]["teacher"][k - 1] += float(_mse_per_sample(tf, tgt).sum())
                acc[key]["spread"][k - 1] += float(_mse_per_sample(tgt[perm], tgt).sum())
                for z_alt_obs in alt_rolls:
                    alt = z_alt_obs[key][:, j]
                    acc[key]["reach"][k - 1] += float(
                        _mse_per_sample(alt, roll).sum()
                    ) / len(alt_rolls)
                    beat[key][k - 1] += float(
                        (_mse_per_sample(alt, tgt) < cost_true).sum()
                    )
            counts[k - 1] += b
            beat_n[k - 1] += b * max(1, len(alt_rolls))

    out = {}
    for key in keys:
        m = {n: acc[key][n] / np.maximum(counts, 1) for n in acc[key]}
        with np.errstate(divide="ignore", invalid="ignore"):
            m["drift"] = m["rollout"] / m["spread"]
            m["compound"] = m["rollout"] / m["teacher"]
            m["snr"] = m["reach"] / m["rollout"]
        m["beat"] = beat[key] / np.maximum(beat_n, 1)
        out[key] = {n: v.tolist() for n, v in m.items()}
    out["n_windows"] = int(counts[0]) if len(counts) else 0
    out["n_alts"] = int(n_alts)
    return out


def first_crossing(drift, thresh=1.0):
    """Smallest k (1-based) with drift[k] >= thresh, or None."""
    for i, v in enumerate(drift):
        if v == v and v >= thresh:
            return i + 1
    return None


def report(name, res, K, horizon):
    print("\n" + "=" * 78)
    print(f"RUN  {name}")
    print(f"     windows={res['n_windows']}  alts={res.get('n_alts')}  "
          f"protocol horizon = {horizon} model steps")
    print("=" * 78)
    for key in ("visual", "proprio"):
        m = res[key]
        print(f"\n[{key}]")
        print(f"  {'k':>2}  {'rollout':>11}  {'teacher':>11}  {'reach':>11}"
              f"  {'spread':>11}  {'snr':>7}  {'beat':>6}  {'drift':>7}  {'compound':>8}")
        for i in range(K):
            print(f"  {i+1:>2}  {m['rollout'][i]:>11.4g}  {m['teacher'][i]:>11.4g}"
                  f"  {m['reach'][i]:>11.4g}  {m['spread'][i]:>11.4g}"
                  f"  {m['snr'][i]:>7.2f}  {m['beat'][i]:>6.3f}"
                  f"  {m['drift'][i]:>7.3f}  {m['compound'][i]:>8.2f}")
        h = min(horizon, K) - 1
        snr_h, beat_h = m["snr"][h], m["beat"][h]
        print(f"  at the protocol horizon k={h+1}: snr={snr_h:.2f}  beat={beat_h:.3f}")
        if snr_h < 3:
            print(f"  -> the action's effect on the terminal latent is within "
                  f"{snr_h:.1f}x of the rollout error. The cost surface is noise "
                  f"at the scale being optimised; this alone can explain "
                  f"open-loop failure with healthy MPC.")
        if beat_h > 0.1:
            print(f"  -> {beat_h*100:.0f}% of alternative action sequences score "
                  f"BETTER than the true ones. The objective, not the optimiser, "
                  f"is the problem.")
        if snr_h >= 3 and beat_h <= 0.1:
            print(f"  -> the objective does identify the correct actions at this "
                  f"horizon. Look at the optimiser / initialisation instead.")

    sv = res["visual"]["spread"][min(horizon, K) - 1]
    sp = res["proprio"]["spread"][min(horizon, K) - 1]
    if sp > 0:
        print(f"\n[objective balance at k={min(horizon, K)}]")
        print(f"  visual spread {sv:.4g} vs proprio spread {sp:.4g}"
              f"  ->  proprio carries {100*sp/(sv+sp):.2f}% of the cost at alpha=1")
        print(f"  configured alpha=1 behaves like alpha_eff={sp/sv:.4g} "
              f"relative to a balanced objective")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="checkpoint run dirs (containing hydra.yaml)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--k", type=int, default=None,
                    help="max horizon in model steps (default: goal_H/frameskip + 3)")
    ap.add_argument("--windows", type=int, default=192)
    ap.add_argument("--alts", type=int, default=8,
                    help="counterfactual action sequences per window, for reach/beat")
    ap.add_argument("--goal-h", type=int, default=25,
                    help="protocol goal_H in env steps, for the horizon marker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write results here")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_res = {}

    for run in args.runs:
        run = os.path.abspath(run.rstrip("/\\"))
        cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))
        frameskip = int(cfg.frameskip)
        num_hist = int(cfg.num_hist)
        horizon = args.goal_h // frameskip
        K = args.k if args.k else horizon + 3

        _, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                   num_pred=cfg.num_pred, frameskip=frameskip)
        dset = traj["valid"]

        from pathlib import Path
        ckpt = Path(os.path.join(run, "checkpoints", f"model_{args.epoch}.pth"))
        if not ckpt.exists():
            raise SystemExit(
                f"No checkpoint at {ckpt}. train.py writes model_latest.pth only "
                "every training.save_every_x_iterations steps (default 1000), so a "
                "freshly launched run has none yet. Check progress with:\n"
                "  grep -o 'global_iter=[0-9]*' <train log> | tail -1"
            )
        model = load_model(ckpt, cfg, cfg.num_action_repeat, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        rng = np.random.default_rng(args.seed)
        obs, act = collect_windows(dset, num_hist, K, frameskip, args.windows, rng)
        res = measure(model, obs, act, num_hist, K, device,
                      n_alts=args.alts, rng=rng)
        res["horizon"] = horizon
        res["K"] = K
        report(os.path.basename(run), res, K, horizon)
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
        print("COMPARISON at the protocol horizon (snr high is good, beat low is good)")
        print("=" * 78)
        print(f"  {'snr':>7}  {'beat':>6}  {'drift@1.0':>9}   run")
        for name, r in all_res.items():
            h = min(r["horizon"], r["K"]) - 1
            v = r["visual"]
            x = first_crossing(v["drift"])
            print(f"  {v['snr'][h]:>7.2f}  {v['beat'][h]:>6.3f}"
                  f"  {(x if x else '>' + str(r['K'])):>9}   {name}")


if __name__ == "__main__":
    main()
