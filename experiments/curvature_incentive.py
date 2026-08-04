#!/usr/bin/env python3
"""curvature_incentive.py -- does the straightening objective reward forgetting?

The finding this tests
----------------------
On the PushT end-to-end + SIGReg run, `metric_alignment.py` shows the visual
latent LOST the pusher while keeping the block:

    state dim              pristine DINOv2   after e2e training
    0  agent_x (pusher)         +0.943            -0.011
    1  agent_y (pusher)         +0.947            -0.618
    2  block_x                  +0.945            +0.979
    3  block_y                  +0.942            +0.989
    4  block_angle              +0.732            +0.622

In PushT the action IS the pusher's target position, so a cost built on a latent
that cannot see the pusher has almost no gradient with respect to the actions.
That explains open-loop 13.33 alongside MPC 56, where re-observation re-grounds
the pusher every frameskip steps.

Dropping the pusher lowers the prediction loss, and neither SIGReg (Gaussianity)
nor cosine curvature (straightness) mentions state information, so nothing in
the objective forbids it. But there is a stronger possibility: the pusher is the
fastest and most direction-changing thing in the scene, so REMOVING it should
make the latent trajectory straighter. If so the curvature term does not merely
permit the forgetting, it rewards it -- which turns a symptom into a cause and
indicts the objective rather than the optimiser.

The test
--------
Everything below uses PRISTINE DINOv2, i.e. the representation training STARTED
from, so it measures the incentive that existed at initialisation rather than the
outcome. No trained checkpoint is involved and nothing is fitted to the result.

  1. Raw-state curvature per component. cos(v_t, v_t+1) on the true trajectory of
     the pusher (dims 0,1), the block (dims 2,3) and the block angle (dim 4). If
     the pusher is intrinsically more curved than the block, the incentive exists
     independently of any encoder.

  2. Latent curvature under linear ablation. Fit a ridge probe from DINOv2
     features to the pusher position, take the (2-dimensional) row space of that
     probe as the "pusher subspace", and project it out of the features. Then
     compare cosine curvature of
         full features                              (baseline)
         features with the PUSHER subspace removed  (the hypothesis)
         features with the BLOCK subspace removed   (the control)
         features with a RANDOM 2-d subspace removed (the null)
     Removing any 2 of 75,264 directions is a tiny perturbation, so the null and
     control pin down how much change is nothing.

Reading it
----------
  pusher-removed curvature > full, and > both control and null
      -> the curvature objective is MINIMISED by discarding the pusher. The
         regularizer actively drives the information loss. Cause, not symptom.
  pusher-removed ~= control ~= null
      -> curvature is indifferent to the pusher; the forgetting comes from the
         prediction loss finding proprio redundancy, and curvature is innocent.

Usage
-----
    python experiments/curvature_incentive.py checkpoints/test/<run>
    python experiments/curvature_incentive.py <run> --windows 96 --json out.json

The run dir is only used for its hydra.yaml (dataset, frameskip, num_hist) and
image transform. Add --trained to additionally report the same curvatures under
the trained encoder for comparison.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE_NAMES = ["agent_x", "agent_y", "block_x", "block_y", "block_angle",
               "vel_x", "vel_y"]
GROUPS = {"pusher (dims 0,1)": [0, 1],
          "block pos (dims 2,3)": [2, 3],
          "block angle (dim 4)": [4]}


def cos_curvature(x):
    """mean cos(v_t, v_{t+1}) over a (n, T, d) trajectory batch. 1.0 = straight."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.shape[1] < 3:
        return float("nan")
    v = x[:, 1:] - x[:, :-1]
    c = F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1, eps=1e-12)
    return float(c.mean())


def ridge_rowspace(X, Y, ridge=1e-3):
    """Orthonormal basis of the subspace of X that linearly predicts Y.

    Solved in the dual because X here is 75,264-dimensional: the row space of the
    ridge solution X^T (X X^T + lambda I)^-1 Y is spanned by the columns of
    X^T A, which is n x dim(Y) and cheap to orthonormalise.
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    Xc = X - X.mean(0, keepdim=True)
    n = Xc.shape[0]
    K = Xc @ Xc.T + ridge * torch.eye(n, dtype=Xc.dtype)
    A = torch.linalg.solve(K, Y - Y.mean(0, keepdim=True))     # (n, dY)
    B = Xc.T @ A                                              # (D, dY)
    Q, _ = torch.linalg.qr(B)
    return Q                                                  # (D, dY) orthonormal


def project_out(X, Q):
    """Remove the span of Q (orthonormal columns) from the rows of X."""
    X = torch.as_tensor(X, dtype=torch.float64)
    return X - (X @ Q) @ Q.T


@torch.no_grad()
def sample_traj_windows(dset, n_windows, length, frameskip, rng):
    """(n, length) consecutive frames at the model's frameskip, plus states."""
    vis, states = [], []
    tries = 0
    span = (length - 1) * frameskip + 1
    while len(states) < n_windows and tries < 50 * n_windows:
        tries += 1
        i = int(rng.integers(0, len(dset)))
        obs, _act, st, _info = dset[i]
        T = min(obs["visual"].shape[0], st.shape[0])
        if T < span:
            continue
        s = int(rng.integers(0, T - span + 1))
        sl = slice(s, s + span, frameskip)
        v, q = obs["visual"][sl], st[sl]
        if v.shape[0] != length:
            continue
        vis.append(v)
        states.append(q)
    if not states:
        raise RuntimeError(f"no trajectory long enough for {span} raw frames")
    return torch.stack(vis), torch.stack(states)


@torch.no_grad()
def encode_windows(encoder, transform, vis, device, chunk=16):
    n, T = vis.shape[0], vis.shape[1]
    flat = vis.reshape(n * T, *vis.shape[2:])
    out = []
    for s in range(0, flat.shape[0], chunk):
        out.append(encoder(transform(flat[s : s + chunk].to(device)))
                   .float().flatten(1).cpu())
    return torch.cat(out).reshape(n, T, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--windows", type=int, default=96)
    ap.add_argument("--length", type=int, default=8, help="frames per window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trained", action="store_true",
                    help="also report curvatures under the trained encoder")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    import hydra
    from omegaconf import OmegaConf
    import custom_resolvers  # noqa: F401
    from models.dino import DinoV2Encoder

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run = os.path.abspath(args.run.rstrip("/\\"))
    cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))
    _, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                              num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    dset = traj["valid"]

    rng = np.random.default_rng(args.seed)
    vis, states = sample_traj_windows(dset, args.windows, args.length,
                                      int(cfg.frameskip), rng)
    S = states.float()
    res = {}

    print("=" * 78)
    print("PART 1: curvature of the TRUE state trajectory, per component")
    print("        (cos of consecutive velocities; 1.0 = straight, 0 = a random walk)")
    print("=" * 78)
    part1 = {}
    for name, dims in GROUPS.items():
        sub = S[:, :, dims]
        if len(dims) == 1:                      # a scalar has no direction
            print(f"  {name:<22s} skipped (1-D)")
            continue
        c = cos_curvature(sub)
        part1[name] = c
        print(f"  {name:<22s} cos = {c:+.4f}")
    if "pusher (dims 0,1)" in part1 and "block pos (dims 2,3)" in part1:
        pu, bl = part1["pusher (dims 0,1)"], part1["block pos (dims 2,3)"]
        print(f"\n  pusher is {'MORE' if pu < bl else 'LESS'} curved than the block "
              f"({pu:+.4f} vs {bl:+.4f})")
        if pu < bl:
            print("  -> an encoder that drops the pusher has a straighter latent "
                  "trajectory\n     before any latent is even computed. The "
                  "incentive is intrinsic to the task.")
    res["state_curvature"] = part1

    # ---------------------------------------------------------------- part 2
    def analyse_encoder(tag, encoder, transform):
        print("\n" + "=" * 78)
        print(f"PART 2 [{tag}]: latent curvature under linear ablation")
        print("=" * 78)
        Z = encode_windows(encoder, transform, vis, device)
        n, T, D = Z.shape
        flat = Z.reshape(n * T, D)
        sflat = S.reshape(n * T, -1)
        base = cos_curvature(Z)
        print(f"  full features                          cos = {base:+.4f}   (dim {D})")
        out = {"full": base, "dim": int(D)}
        for name, dims in (("pusher subspace removed", [0, 1]),
                           ("block subspace removed (control)", [2, 3])):
            Q = ridge_rowspace(flat, sflat[:, dims])
            Za = project_out(flat, Q).reshape(n, T, D)
            c = cos_curvature(Za)
            out[name] = c
            print(f"  {name:<38s} cos = {c:+.4f}   ({c - base:+.4f})")
        g = torch.Generator().manual_seed(args.seed)
        Qr, _ = torch.linalg.qr(torch.randn(D, 2, generator=g, dtype=torch.float64))
        c = cos_curvature(project_out(flat, Qr).reshape(n, T, D))
        out["random subspace removed (null)"] = c
        print(f"  {'random 2-d subspace removed (null)':<38s} cos = {c:+.4f}"
              f"   ({c - base:+.4f})")

        dp = out["pusher subspace removed"] - base
        dc = out["block subspace removed (control)"] - base
        dn = c - base
        # An ABSOLUTE floor as well as a relative one. Without it, when every
        # delta is ~0 (which is what a latent with no recoverable pusher
        # subspace gives) the relative test fires on floating-point noise and
        # reports a spurious effect. 0.01 = one point of cosine, ~20x the
        # observed control/null perturbation.
        EPS = 0.01
        print()
        if dp > EPS and dp > max(abs(dc), abs(dn)) * 2:
            print("  VERDICT: removing the pusher makes the latent trajectory")
            print("  STRAIGHTER, by more than the control and null perturbations.")
            print("  The curvature objective is minimised by discarding the pusher:")
            print("  the regularizer REWARDS the information loss that breaks")
            print("  planning. Cause, not symptom.")
        elif dp <= EPS:
            print("  VERDICT: curvature is INDIFFERENT to the pusher -- removing it")
            print(f"  changes cos by {dp:+.4f}, below the {EPS} threshold and on the")
            print(f"  order of the control ({dc:+.4f}) and null ({dn:+.4f}).")
            print("  The curvature term does not drive the forgetting. Look to the")
            print("  prediction loss exploiting proprio redundancy instead.")
            if abs(dp) < 1e-4 and out["dim"] < 5000:
                print("  NOTE: on the TRAINED latent a null result is also expected for")
                print("  a second reason -- there is no linearly recoverable pusher")
                print("  subspace left to remove (probe R^2 is already ~0), so the")
                print("  projection is fitted to noise. Only the pristine-DINOv2 row")
                print("  answers the incentive question.")
        else:
            print("  VERDICT: weak effect in the predicted direction. Curvature mildly")
            print("  favours dropping the pusher but is not the dominant driver.")
        return out

    ref = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").to(device).eval()
    # same resize the world model applies before the encoder
    from torchvision import transforms
    side = (224 // 16) * ref.patch_size
    tf = transforms.Compose([transforms.Resize(side)])
    res["pristine_dinov2"] = analyse_encoder("pristine DINOv2 = training's starting point",
                                             ref, tf)
    del ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.trained:
        from pathlib import Path
        from plan import load_model
        model = load_model(Path(os.path.join(run, "checkpoints", f"model_{args.epoch}.pth")),
                           cfg, cfg.num_action_repeat, device=device)
        model.eval()
        res["trained"] = analyse_encoder("trained encoder = the outcome",
                                         model.encoder, model.encoder_transform)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
