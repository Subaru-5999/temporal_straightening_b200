#!/usr/bin/env python3
"""Verify that the DINOv2 trunk is ACTUALLY trained with the new config.

`requires_grad=True` is not proof. A parameter can be trainable, receive a
gradient, and still never move -- if it was left out of the optimizer's param
groups, if the optimizer was built before trainability was configured, or if
something re-freezes it later. The only conclusive check is: did the weights
change after a step?

This measures exactly that, on the repo's own classes, for both settings:

    freeze_backbone=True   (baseline)     trunk delta MUST be exactly 0
    freeze_backbone=False  (end-to-end)   trunk delta MUST be > 0

The optimizer is built with the same logic as
train.py::Trainer._encoder_param_groups(), and in the same order as
train.py (configure trainability -> build optimizer -> step), so an ordering
bug in the real trainer would reproduce here.

Usage:  python experiments/verify_encoder_trains.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.verify_stop_grad import build, rollout, banner


def encoder_param_groups(encoder, encoder_lr, backbone_lr):
    """Mirror of train.py::Trainer._encoder_param_groups()."""
    trunk, heads = [], []
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        (trunk if name.startswith("base_model.") else heads).append(param)

    if backbone_lr is None or not trunk:
        return [{"params": list(encoder.parameters()), "lr": encoder_lr}], trunk, heads
    groups = [{"params": trunk, "lr": float(backbone_lr)}]
    if heads:
        groups.append({"params": heads, "lr": encoder_lr})
    return groups, trunk, heads


def snapshot(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def total_delta(module, before):
    """L2 norm of the total parameter change, and the worst single tensor."""
    tot, worst, worst_name = 0.0, 0.0, None
    for n, p in module.named_parameters():
        d = (p.detach() - before[n]).norm().item()
        tot += d ** 2
        if d > worst:
            worst, worst_name = d, n
    return tot ** 0.5, worst, worst_name


def check(freeze_backbone, sigreg, straighten, encoder_lr=1e-5, backbone_lr=1e-5,
          steps=5, seed=0):
    torch.manual_seed(seed)
    wm = build(freeze_backbone=freeze_backbone, sigreg=sigreg, sigreg_coeff=0.1,
               straighten=straighten, stop_grad=not sigreg)
    enc = wm.encoder

    # --- order matters, and matches train.py: trainability is configured inside
    # --- build() (mirroring _configure_encoder_trainability), THEN the optimizer.
    groups, trunk, heads = encoder_param_groups(enc, encoder_lr, backbone_lr)
    opt = torch.optim.Adam(groups, lr=encoder_lr)
    opt_other = torch.optim.Adam(
        list(wm.predictor.parameters())
        + list(wm.proprio_encoder.parameters())
        + list(wm.action_encoder.parameters()),
        lr=5e-4,
    )

    n_trunk_total = sum(p.numel() for p in enc.base_model.parameters())
    n_trunk_train = sum(p.numel() for p in enc.base_model.parameters() if p.requires_grad)
    n_head_train = sum(p.numel() for p in heads)
    in_opt = {id(p) for g in groups for p in g["params"] if p.requires_grad}
    trunk_in_opt = sum(
        p.numel() for p in enc.base_model.parameters()
        if p.requires_grad and id(p) in in_opt
    )

    before_trunk = snapshot(enc.base_model)
    before_proj = snapshot(enc.projector)

    wm.train()
    g = torch.Generator().manual_seed(seed + 7)
    grad_norm = 0.0
    for _ in range(steps):
        obs, act, _ = rollout(8, 4, gen=g)
        *_, loss, comp = wm(obs, act)
        opt.zero_grad(set_to_none=True)
        opt_other.zero_grad(set_to_none=True)
        loss.backward()
        gs = [p.grad.flatten() for p in enc.base_model.parameters() if p.grad is not None]
        grad_norm = torch.cat(gs).norm().item() if gs else 0.0
        opt.step()
        opt_other.step()

    d_trunk, worst, worst_name = total_delta(enc.base_model, before_trunk)
    d_proj, _, _ = total_delta(enc.projector, before_proj)

    label = "freeze_backbone=False (END-TO-END)" if not freeze_backbone \
        else "freeze_backbone=True  (baseline)"
    print(f"\n  {label}   sigreg={sigreg}  straighten={straighten}")
    print(f"    trunk params                 : {n_trunk_total:,} total")
    print(f"    trunk params requires_grad   : {n_trunk_train:,}")
    print(f"    trunk params IN the optimizer: {trunk_in_opt:,}")
    print(f"    head  params requires_grad   : {n_head_train:,}")
    print(f"    |grad| trunk (last step)     : {grad_norm:.6e}")
    print(f"    ||delta trunk|| after {steps} steps   : {d_trunk:.6e}"
          f"   (largest single tensor: {worst:.3e} @ {worst_name})")
    print(f"    ||delta projector||              : {d_proj:.6e}")
    return dict(trunk_train=n_trunk_train, trunk_in_opt=trunk_in_opt,
                grad=grad_norm, d_trunk=d_trunk, d_proj=d_proj)


def main():
    banner("Is the DINOv2 trunk actually being trained?")
    print("  A parameter can be 'trainable', get a gradient, and still never move.")
    print("  The conclusive test is whether the weights CHANGED after a step.")

    frozen = check(freeze_backbone=True, sigreg=False, straighten=False)
    e2e = check(freeze_backbone=False, sigreg=True, straighten="aggcos1e-1")

    banner("VERDICT")
    ok_frozen = (frozen["trunk_train"] == 0 and frozen["d_trunk"] == 0.0
                 and frozen["d_proj"] > 0)
    ok_e2e = (e2e["trunk_train"] > 0 and e2e["trunk_in_opt"] == e2e["trunk_train"]
              and e2e["grad"] > 0 and e2e["d_trunk"] > 0 and e2e["d_proj"] > 0)

    print(f"  baseline   trunk frozen and unchanged, head still learns : "
          f"{'PASS' if ok_frozen else 'FAIL'}")
    print(f"  end-to-end trunk trainable, in optimizer, gets grad, MOVES: "
          f"{'PASS' if ok_e2e else 'FAIL'}")
    print(f"\n  trunk movement, baseline vs end-to-end: "
          f"{frozen['d_trunk']:.3e}  ->  {e2e['d_trunk']:.3e}")
    if not (ok_frozen and ok_e2e):
        raise SystemExit("VERIFICATION FAILED")
    print("\n  => Yes: with training.freeze_backbone=False the visual trunk itself")
    print("     is optimized, not just the projector and agg head.")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    main()
