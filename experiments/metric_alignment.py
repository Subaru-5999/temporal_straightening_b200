#!/usr/bin/env python3
"""metric_alignment.py -- is latent distance a proxy for state distance?

Why this exists
---------------
The straightening paper's claim is geometric: a curvature-regularized latent
makes Euclidean distance a better proxy for geodesic/state distance, which is
what conditions the planning objective. Everything the planner does rests on
that. `MSE(z_rollout, z_goal)` is only a useful cost if latent proximity implies
state proximity.

On the PushT end-to-end + SIGReg checkpoint we have already ruled out the two
obvious explanations for 13.33 open-loop:
  * the representation is not collapsed  (MPC reaches 56)
  * the rollout is accurate              (drift 0.135 at the protocol horizon)
and found the objective mis-weighted (proprio carries 0.42% of the cost).

What has never been checked is the geometry itself. A latent can be perfectly
straight, perfectly Gaussian, perfectly predictable AND geometrically useless
for planning, because none of those three properties mention the state.

This script needs no simulator, no planner and no baseline checkpoint: the
reference is PRISTINE DINOv2 from the torch hub cache, i.e. the geometry the
encoder started from. If the trained encoder aligns WORSE than untrained
DINOv2, training destroyed the property the method depends on.

What it measures, per representation
------------------------------------
  probe_r2         held-out ridge R^2, latent -> full state, and per state dim.
                   Information content. Reported per dimension because "the
                   latent knows where the pusher is but not the block angle" and
                   "the latent knows nothing" are completely different failures
                   with the same aggregate number.

  rho_global       Spearman correlation between latent distance and state
                   distance over random pairs. Is the metric monotone at all?

  rho_local        the same, restricted to the closest decile of pairs by state
                   distance. THIS is what gradient descent actually walks on: a
                   metric can be globally monotone and locally flat or noisy,
                   and only the local number predicts whether GD converges to
                   the right actions.

  nn_state_ratio   median state distance to a frame's latent nearest neighbour,
                   divided by the median state distance between random frames.
                   0 = latent neighbourhoods are state neighbourhoods.
                   1 = latent neighbours are no better than random frames.

How to read it
--------------
  rho_local >= ~0.5 and nn_state_ratio <= ~0.3   the geometry supports planning;
                    the failure is elsewhere (weighting, optimiser, rollout).
  rho_local ~ 0     latent proximity does not imply state proximity at the scale
                    GD operates on. The cost has no usable gradient toward the
                    goal, and no amount of optimiser tuning fixes it. This is a
                    representation-geometry failure and it is the deepest
                    diagnosis available offline.
  trained WORSE than pristine DINOv2   end-to-end training actively degraded the
                    geometry. Combined with a straightness metric that IMPROVED
                    (0.037 -> 0.78) this is the paper-relevant finding: the
                    curvature objective was satisfied while the property it is
                    supposed to be a proxy FOR got worse.

Usage
-----
    python experiments/metric_alignment.py checkpoints/test/<run>
    python experiments/metric_alignment.py <run> --frames 768 --json out.json
    python experiments/metric_alignment.py <run> --no-reference   # skip DINOv2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# hydra / plan.py / gym are imported lazily inside main() so the statistics below
# can be imported and checked on a machine without the planning stack installed.


# ---------------------------------------------------------------- statistics


def _require_ckpt(path):
    """Fail with the real reason when the checkpoint is not there yet.

    plan.py's load_model treats a missing file as 'no checkpoint supplied' and
    then reports `Predictor not found in model checkpoint`, which sends you
    looking at the model instead of at the clock: train.py only writes
    model_latest.pth every training.save_every_x_iterations steps, so a freshly
    launched run has no checkpoint for the first few minutes.
    """
    import glob
    if path.exists():
        return
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(
        os.path.dirname(str(path)), "*.pth")))
    raise SystemExit(
        f"No checkpoint at {path}\n"
        + (f"Available in that directory: {', '.join(have)}\n" if have else
           "That directory has no .pth files at all.\n")
        + "If the run just started, train.py writes model_latest.pth only every\n"
          "training.save_every_x_iterations steps (default 1000). Check progress\n"
          "with:  grep -o 'global_iter=[0-9]*' <train log> | tail -1"
    )


def _rank(x):
    """Average-tie-free rank transform (ordinal ranks are enough for Spearman)."""
    order = np.argsort(x, kind="stable")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def spearman(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def held_out_r2(X, Y, ridge=1e-3):
    """Held-out R^2 of a ridge probe X -> Y, overall and per output dim.

    Interleaved even/odd split so both halves span the sample, and the fit is
    scored on data it never saw -- an in-sample probe on a 1568-dim latent from
    a few hundred rows returns 1.0 and means nothing.

    Solved in whichever space is smaller. The primal normal equations are
    (D+1)x(D+1), and D is 196*384 = 75,264 for pristine DINOv2 patch tokens,
    i.e. a 45 GB float64 Gram matrix. The dual form
        pred_held = K_hf (K_ff + ridge I)^-1 Y_fit,   K = X X^T
    is the SAME ridge estimator at n_fit x n_fit, a few hundred squared. Use it
    whenever D >= n_fit, which is every high-dimensional representation here.
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)
    fit, held = X[0::2], X[1::2]
    yf, yh = Y[0::2], Y[1::2]
    n_fit, d = fit.shape
    if n_fit < 4 or held.shape[0] < 4:
        return float("nan"), []
    try:
        if d >= n_fit:
            k_ff = fit @ fit.T + ridge * torch.eye(n_fit, dtype=X.dtype)
            alpha = torch.linalg.solve(k_ff, yf)
            pred = (held @ fit.T) @ alpha
        else:
            gram = fit.T @ fit + ridge * torch.eye(d, dtype=X.dtype)
            pred = held @ torch.linalg.lstsq(gram, fit.T @ yf).solution
    except Exception:
        return float("nan"), []
    ss_res = ((pred - yh) ** 2).sum(0)
    ss_tot = ((yh - yh.mean(0, keepdim=True)) ** 2).sum(0)
    per = [float(v) for v in (1 - ss_res / ss_tot.clamp_min(1e-12)).clamp(-1, 1)]
    overall = float((1 - ss_res.sum() / ss_tot.sum().clamp_min(1e-12)).clamp(-1, 1))
    return overall, per


def pair_stats(Z, S, n_pairs, rng, local_frac=0.1):
    """Spearman(latent dist, state dist) globally and among the closest pairs."""
    n = Z.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    # same reduction as the planning objective: mean squared error over dims
    dz = ((Z[i] - Z[j]) ** 2).mean(axis=1)
    ds = ((S[i] - S[j]) ** 2).mean(axis=1)
    rho_g = spearman(dz, ds)
    k = max(16, int(local_frac * len(ds)))
    near = np.argsort(ds)[:k]
    rho_l = spearman(dz[near], ds[near])
    return rho_g, rho_l, float(np.median(ds))


def nn_state_ratio(Z, S, median_ds, chunk=256):
    """Median state distance to each frame's LATENT nearest neighbour / random."""
    Zt = torch.as_tensor(Z, dtype=torch.float32)
    St = torch.as_tensor(S, dtype=torch.float32)
    n = Zt.shape[0]
    got = []
    for s in range(0, n, chunk):
        d = torch.cdist(Zt[s : s + chunk], Zt)
        d[torch.arange(d.shape[0]), torch.arange(s, min(s + chunk, n))] = float("inf")
        nn = d.argmin(dim=1)
        got.append(((St[s : s + chunk] - St[nn]) ** 2).mean(dim=1))
    nn_ds = float(torch.cat(got).median())
    return nn_ds / median_ds if median_ds > 0 else float("nan")


# ---------------------------------------------------------------- sampling


@torch.no_grad()
def sample_frames(dset, n_frames, rng):
    """Random (trajectory, timestep) frames with their ground-truth states."""
    vis, prop, states = [], [], []
    tries = 0
    while len(states) < n_frames and tries < 50 * n_frames:
        tries += 1
        i = int(rng.integers(0, len(dset)))
        obs, _act, state, _info = dset[i]
        T = min(obs["visual"].shape[0], state.shape[0])
        if T < 1:
            continue
        t = int(rng.integers(0, T))
        vis.append(obs["visual"][t])
        prop.append(obs["proprio"][t])
        states.append(state[t])
    if not states:
        raise RuntimeError("could not sample any frames from the valid split")
    return torch.stack(vis), torch.stack(prop), torch.stack(states)


@torch.no_grad()
def encode_all(encoder, transform, vis, device, chunk=32):
    out = []
    for s in range(0, vis.shape[0], chunk):
        x = transform(vis[s : s + chunk].to(device))
        out.append(encoder(x).float().flatten(1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- reporting


def analyse(name, Z, S, rng, n_pairs, state_names):
    r2, per = held_out_r2(Z, S)
    rho_g, rho_l, med = pair_stats(Z, S, n_pairs, rng)
    ratio = nn_state_ratio(Z, S, med)
    print(f"\n[{name}]  dim={Z.shape[1]}")
    print(f"  probe_r2 (held out, latent -> state) : {r2:+.4f}")
    for nm, v in zip(state_names, per):
        print(f"      {nm:<14s} {v:+.4f}")
    print(f"  rho_global   (latent dist vs state dist) : {rho_g:+.4f}")
    print(f"  rho_LOCAL    (closest decile of pairs)   : {rho_l:+.4f}   <-- what GD walks on")
    print(f"  nn_state_ratio (0 good, 1 = random)      : {ratio:.4f}")
    return {"dim": int(Z.shape[1]), "probe_r2": r2, "probe_r2_per_dim": per,
            "rho_global": rho_g, "rho_local": rho_l, "nn_state_ratio": ratio}


def verdict(res):
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    m = res.get("method.visual_patch")
    if not m:
        return
    rl, nn = m["rho_local"], m["nn_state_ratio"]
    if rl >= 0.5 and nn <= 0.3:
        print("  The latent geometry SUPPORTS planning: local latent distance tracks")
        print("  state distance. The open-loop failure is NOT geometric -- look at")
        print("  objective weighting (alpha_eff 0.0042) and the optimiser.")
    elif rl < 0.2:
        print("  The latent geometry DOES NOT support planning: at the scale GD")
        print("  operates on, latent proximity does not imply state proximity.")
        print("  MSE(z_rollout, z_goal) has no usable gradient toward the goal, and")
        print("  no optimiser or alpha fixes that. This is the root cause.")
    else:
        print("  Borderline geometry: locally weak but not absent. Expect the")
        print("  objective weighting to matter a lot, since the visual term is")
        print("  carrying 99.58% of a cost it can only weakly justify.")

    ref = res.get("reference.dinov2_patch")
    if ref:
        print()
        if m["rho_local"] < ref["rho_local"] - 0.05:
            print(f"  AND end-to-end training made it WORSE than untrained DINOv2")
            print(f"  (rho_local {m['rho_local']:+.3f} vs {ref['rho_local']:+.3f}).")
            print(f"  Straightness improved over the same run (0.037 -> ~0.78), so the")
            print(f"  curvature objective was satisfied while the property it proxies")
            print(f"  for degraded. That is the finding worth writing up.")
        elif m["rho_local"] > ref["rho_local"] + 0.05:
            print(f"  Training IMPROVED alignment over untrained DINOv2 "
                  f"({m['rho_local']:+.3f} vs {ref['rho_local']:+.3f}), so the")
            print(f"  representation is not the regression. Look at the objective.")
        else:
            print(f"  Alignment is unchanged from untrained DINOv2 "
                  f"({m['rho_local']:+.3f} vs {ref['rho_local']:+.3f}): training")
            print(f"  neither helped nor hurt the geometry.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="checkpoint run dir containing hydra.yaml")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--frames", type=int, default=768)
    ap.add_argument("--pairs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the pristine-DINOv2 comparison")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # Piping into tee makes stdout block-buffered, so a long CPU run looks hung
    # while only stderr warnings appear. Line-buffer instead of relying on -u.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    import hydra
    from omegaconf import OmegaConf
    import custom_resolvers  # noqa: F401  (registers OmegaConf resolvers)
    from plan import load_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run = os.path.abspath(args.run.rstrip("/\\"))
    cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))

    _, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                              num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    dset = traj["valid"]

    from pathlib import Path
    ckpt = Path(os.path.join(run, "checkpoints", f"model_{args.epoch}.pth"))
    _require_ckpt(ckpt)
    model = load_model(ckpt, cfg, cfg.num_action_repeat, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    rng = np.random.default_rng(args.seed)
    vis, prop, state = sample_frames(dset, args.frames, rng)
    S = state.float().numpy()
    S = (S - S.mean(0)) / np.maximum(S.std(0), 1e-8)      # z-score per dim
    state_names = [f"state[{i}]" for i in range(S.shape[1])]

    print("=" * 78)
    print(f"RUN  {os.path.basename(run)}")
    print(f"     frames={vis.shape[0]}  pairs={args.pairs}  state_dim={S.shape[1]}")
    print("=" * 78)

    res = {}
    tf = model.encoder_transform

    print("encoding with the trained encoder...")
    Zv = encode_all(model.encoder, tf, vis, device)
    res["method.visual_patch"] = analyse("method: visual patch tokens (the planning latent)",
                                        Zv, S, rng, args.pairs, state_names)

    with torch.no_grad():
        Zp = model.encode_proprio(prop.to(device).unsqueeze(1)).squeeze(1).float().cpu().numpy()
    res["method.proprio"] = analyse("method: proprio embedding", Zp, S, rng,
                                    args.pairs, state_names)

    if hasattr(model.encoder, "agg"):
        with torch.no_grad():
            d = int(model.encoder.emb_dim)          # per-patch channels after projection
            p = Zv.shape[1] // d                    # number of patches
            za = torch.as_tensor(Zv, dtype=torch.float32).reshape(-1, p, d).to(device)
            Za = model.encoder.agg(za).float().cpu().numpy()
        res["method.agg"] = analyse("method: agg head (what SIGReg + curvature act on)",
                                    Za, S, rng, args.pairs, state_names)

    if not args.no_reference:
        from models.dino import DinoV2Encoder
        print("\n--- reference: pristine DINOv2 from the hub cache (untrained geometry) ---")
        ref = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").to(device).eval()
        print("encoding with pristine DINOv2 (75,264-dim patch tokens)...")
        Zr = encode_all(ref, tf, vis, device)
        res["reference.dinov2_patch"] = analyse("reference: pristine DINOv2 patch tokens",
                                               Zr, S, rng, args.pairs, state_names)
        del ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    verdict(res)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
