#!/usr/bin/env python3
"""
verify_stop_grad.py -- falsification test for the claim
"stop_grad=True prevents representation collapse of the encoder".

Runs on CPU. No dataset and no DINOv2 download required: torch.hub.load is
patched to return a small *trainable* stand-in trunk exposing the interface
DinoV2Encoder expects (num_features / patch_size / forward_features).
Everything else is the repo's own code: ChannelProjector, DinoV2Encoder,
ViTPredictor, VWorldModel, its prediction loss and its curvature loss.

The task is a synthetic but genuinely learnable one: a 2-D point moves by the
action, the observation is a Gaussian blob rendered at that point, proprio is
[pos, vel]. So a non-collapsed encoder is both possible and useful -- collapse
here is a real failure, not the only available solution.

Tests
  T1  Does stop_grad actually stop gradient into the encoder?
  T2  Which encoder params are trainable under `model.train_encoder: True`?
  T3  End-to-end (backbone unfrozen) + stop_grad: does the encoder collapse?
  T4  Control: backbone frozen (repo default) -- collapse or not?
  T5  Does the curvature (straightening) loss provide a collapse barrier?

Usage:  python experiments/verify_stop_grad.py
"""
import os
import sys
import types
import contextlib

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------- cuda-free ViT
# models/vit.py Attention.__init__ does generate_mask_matrix(...).to('cuda').
# Redirect cuda->cpu for the whole script so the real predictor can be built.
_orig_to = torch.Tensor.to


def _cpu_to(self, *args, **kwargs):
    args = tuple("cpu" if (isinstance(a, str) and a.startswith("cuda")) else a for a in args)
    if isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("cuda"):
        kwargs["device"] = "cpu"
    return _orig_to(self, *args, **kwargs)


torch.Tensor.to = _cpu_to


# ------------------------------------------------------- stand-in DINOv2 trunk
class FakeDinoTrunk(nn.Module):
    """Same interface as dinov2_vits14, but small and trainable."""

    def __init__(self, dim=384, patch_size=14):
        super().__init__()
        self.num_features = dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward_features(self, x):
        t = self.patch_embed(x)                       # B, D, h, w
        t = t.flatten(2).transpose(1, 2)              # B, N, D
        t = self.norm(t + self.mlp(t))
        return {"x_norm_patchtokens": t, "x_norm_clstoken": t.mean(1)}


@contextlib.contextmanager
def patched_hub(dim=384, patch_size=14):
    import torch.hub as hub

    orig = hub.load
    hub.load = lambda *a, **k: FakeDinoTrunk(dim, patch_size)
    try:
        yield
    finally:
        hub.load = orig


# ------------------------------------------------- synthetic learnable dynamics
IMAGE_SIZE = 112


def rollout(b, t, image_size=IMAGE_SIZE, gen=None):
    """2-D point pushed by the action; obs = Gaussian blob at the point.

    Returns (obs_dict, act, state) with state = true (x, y) in [-1, 1].
    A perfect encoder recovers the point position; a collapsed one recovers
    nothing. Dynamics are smooth and fully determined by the actions.
    """
    pos = (torch.rand(b, 2, generator=gen) * 1.2 - 0.6)
    act = torch.randn(b, t, 2, generator=gen) * 0.35
    states, vels = [], []
    v = torch.zeros(b, 2)
    for i in range(t):
        states.append(pos.clone())
        vels.append(v.clone())
        v = 0.6 * v + 0.25 * act[:, i]
        pos = (pos + v).clamp(-0.9, 0.9)
    state = torch.stack(states, 1)                          # b, t, 2
    vel = torch.stack(vels, 1)

    lin = torch.linspace(-1, 1, image_size)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    cx = state[..., 0].reshape(-1, 1, 1)
    cy = state[..., 1].reshape(-1, 1, 1)
    blob = torch.exp(-(((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 0.12 ** 2)))
    img = blob.unsqueeze(1).repeat(1, 3, 1, 1)
    img[:, 1] *= 0.4                                        # give channels structure
    img = img.reshape(b, t, 3, image_size, image_size)

    obs = {"visual": img, "proprio": torch.cat([state, vel], -1)}
    return obs, act, state


# --------------------------------------------------------------- model builder
def build(image_size=IMAGE_SIZE, proj_out=8, stop_grad=True, straighten=False,
          freeze_backbone=True, depth=2, heads=4, mlp_dim=128, agg=True,
          agg_hidden=64, sigreg=False, sigreg_coeff=0.0, sigreg_num_proj=256,
          sigreg_apply_to="agg", curv_on="features"):
    """Mirror train.py::init_models() wiring for encoder=dino_channel."""
    from models.dino import DinoV2Encoder, ChannelProjector
    from models.proprio import ProprioceptiveEmbedding
    from models.vit import ViTPredictor
    from models.visual_world_model import VWorldModel

    # stand-in for the OmegaConf nodes in conf/encoder/dino_channel.yaml
    conv_layers = [
        types.SimpleNamespace(kernel_size=3, stride=1, padding=1, in_dim=384, out_dim=192),
        types.SimpleNamespace(kernel_size=3, stride=1, padding=1, in_dim=192, out_dim=proj_out),
    ]
    projector = ChannelProjector(in_dim=384, out_dim=proj_out, norm_type="layer",
                                 conv_layers=conv_layers)

    num_side = image_size // 16                 # VWorldModel's own convention
    num_patches = num_side ** 2

    with patched_hub():
        enc = DinoV2Encoder(
            name="dinov2_vits14",
            feature_key="x_norm_patchtokens",
            projector="channel",
            projector_config=projector,
            agg_type="mlp" if agg else "flatten",
            agg_mlp_hidden_dim=agg_hidden,
            agg_out_dim=128,
        )
    # the agg MLP input dim is hardcoded to 196 tokens upstream; resize for this grid
    if agg:
        enc._agg_mlp_in_dim = num_patches * proj_out
        enc.agg_mlp[0] = nn.Linear(enc._agg_mlp_in_dim, agg_hidden)

    # --- exactly train.py::_configure_encoder_trainability() -------------------
    for p in enc.base_model.parameters():
        p.requires_grad = not freeze_backbone
    for n, p in enc.named_parameters():
        if not n.startswith("base_model."):
            p.requires_grad = True

    prop = ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=4, emb_dim=10)
    act = ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=2, emb_dim=10)
    dim = proj_out + (10 + 10)                  # concat_dim=1, repeats=1
    pred = ViTPredictor(num_patches=num_patches, num_frames=3, dim=dim, depth=depth,
                        heads=heads, mlp_dim=mlp_dim, pool="mean", dropout=0.0)

    wm = VWorldModel(
        image_size=image_size, num_hist=3, num_pred=1,
        encoder=enc, proprio_encoder=prop, action_encoder=act,
        decoder=None, predictor=pred,
        proprio_dim=10, action_dim=10, concat_dim=1,
        num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=True, train_predictor=True, train_decoder=False,
        straighten=straighten, stop_grad=stop_grad, curv_on=curv_on,
        sigreg=sigreg, sigreg_coeff=sigreg_coeff,
        sigreg_num_proj=sigreg_num_proj, sigreg_apply_to=sigreg_apply_to,
    )
    return wm


# ------------------------------------------------------------------- diagnostics
def gnorm(params):
    gs = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return torch.cat(gs).norm().item() if gs else 0.0


@torch.no_grad()
def diagnostics(wm, obs, act, state):
    """std across (b,t), effective rank, and linear-probe R^2 for true position."""
    zv = wm.visual_only(wm.encode(obs, act))            # b, t, p, d
    b, t, p, d = zv.shape
    std_bt = zv.reshape(b * t, p, d).std(dim=0).mean().item()

    x = zv.reshape(b * t * p, d)
    xc = x - x.mean(0)
    ev = torch.linalg.eigvalsh(xc.T @ xc / max(1, xc.shape[0] - 1)).clamp_min(0)
    eff_rank = (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)).item()

    # Linear probe: patch-MEAN latent -> true (x, y), closed-form ridge.
    # The patch mean (d features) is used rather than the flattened latent
    # (p*d features) because b*t samples cannot determine p*d coefficients:
    # at 49 patches x 8 dims that is 393 unknowns from 64 samples, which fits
    # anything and makes R^2 both inflated and LAPACK-driver dependent (it
    # printed 0.98/0.68/0.9993 at step 0 across three identically-seeded runs).
    # This matches models/diagnostics.latent_diagnostics, which already pools.
    X = zv.mean(dim=2).reshape(b * t, d)
    X = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    Y = state.reshape(b * t, 2)
    lam = 1e-4 * torch.eye(X.shape[1])
    W = torch.linalg.lstsq(X.T @ X + lam, X.T @ Y).solution
    resid = ((X @ W - Y) ** 2).sum()
    total = ((Y - Y.mean(0)) ** 2).sum()
    r2 = (1 - resid / total).clamp(-1, 1).item()
    return std_bt, eff_rank, r2


def banner(s):
    print("\n" + "=" * 92)
    print(s)
    print("=" * 92)


# ============================================================== T1 + T2
def t1_t2():
    banner("T1  Does stop_grad=True stop gradient into the encoder?")
    obs, act, _ = rollout(4, 4, gen=torch.Generator().manual_seed(1))
    for sg in (True, False):
        torch.manual_seed(0)
        wm = build(stop_grad=sg, freeze_backbone=True)
        wm.train()
        *_, loss, comp = wm(obs, act)
        wm.zero_grad(set_to_none=True)
        loss.backward()
        print(f"\n  stop_grad={sg}")
        print(f"    |grad| encoder.base_model (frozen) : "
              f"{gnorm(list(wm.encoder.base_model.parameters())):.4e}")
        print(f"    |grad| encoder.projector          : "
              f"{gnorm(list(wm.encoder.projector.parameters())):.4e}   <-- ENCODER TRAINS")
        print(f"    |grad| encoder.agg_mlp            : "
              f"{gnorm(list(wm.encoder.agg_mlp.parameters())):.4e}")
        print(f"    |grad| predictor                  : "
              f"{gnorm(list(wm.predictor.parameters())):.4e}")
        print(f"    z_loss = {comp['z_loss'].item():.6f}   "
              f"z_visual_loss = {comp['z_visual_loss'].item():.6f}")

    banner("T2  What does `model.train_encoder: True` actually train?")
    torch.manual_seed(0)
    # image_size=224 + agg_hidden=512 == the real conf/encoder/dino_channel.yaml
    # shapes (14x14=196 tokens, projector 384->192->8, agg MLP 1568->512->512->128)
    wm = build(image_size=224, freeze_backbone=True, agg_hidden=512)
    enc = wm.encoder
    proj = sum(p.numel() for p in enc.projector.parameters())
    aggp = sum(p.numel() for p in enc.agg_mlp.parameters()) \
        + sum(p.numel() for p in enc.agg_post_norm.parameters())
    print(f"    trainable: projector {proj:,} + agg head {aggp:,} = {proj + aggp:,}")
    print(f"    frozen   : base_model (real dinov2_vits14 is ~21-22 M params)")
    print(f"    base_model any requires_grad: "
          f"{any(p.requires_grad for p in enc.base_model.parameters())}")
    print("    => `train_encoder: True` never unfreezes the visual backbone;")
    print("       _configure_encoder_trainability() freezes it unconditionally.")


# ============================================================== T3/T4/T5
def train_loop(tag, steps=300, lr=1e-3, freeze_backbone=True, straighten=False,
               stop_grad=True, seed=0, b=8, sigreg=False, sigreg_coeff=0.0,
               curv_on="features", telemetry_dir=None):
    torch.manual_seed(seed)
    wm = build(stop_grad=stop_grad, straighten=straighten, freeze_backbone=freeze_backbone,
               sigreg=sigreg, sigreg_coeff=sigreg_coeff, curv_on=curv_on)
    enc_params = [p for p in wm.encoder.parameters() if p.requires_grad]
    other = list(wm.predictor.parameters()) + list(wm.proprio_encoder.parameters()) \
        + list(wm.action_encoder.parameters())
    opt = torch.optim.Adam([{"params": enc_params, "lr": lr},
                            {"params": other, "lr": 5e-4}])

    g = torch.Generator().manual_seed(seed + 7)
    probe = rollout(16, 4, gen=torch.Generator().manual_seed(999))   # held out
    wm.train()

    # Optional: emit the same bounded telemetry the real trainer writes, so the
    # whole log -> digest pipeline can be exercised on CPU in seconds.
    tl = None
    if telemetry_dir:
        from training_log import TrainingLogger
        slug = tag.split()[0].lower() + "_" + tag.split()[1].strip("_").lower()
        tl = TrainingLogger(
            os.path.join(telemetry_dir, f"{slug}.jsonl"),
            run_name=tag,
            config={"freeze_backbone": freeze_backbone, "sigreg": sigreg,
                    "sigreg_coeff": sigreg_coeff, "straighten": straighten,
                    "stop_grad": stop_grad, "curv_on": curv_on, "lr": lr},
            log_every=max(1, steps // 20),
        )

    print(f"\n  {tag}")
    print(f"    {'step':>5} {'z_visual_loss':>14} {'std(b,t)':>11} {'eff_rank/8':>11} "
          f"{'probe R^2':>10} {'curv':>9} {'sigreg':>10}")
    nan = torch.tensor(float("nan"))
    for i in range(steps + 1):
        obs, act, _ = rollout(b, 4, gen=g)               # fresh data every step
        *_, loss, comp = wm(obs, act)
        if tl is not None:
            tl.record(i, **{f"loss/{k}": float(v.item()) for k, v in comp.items()})
        if i % (steps // 6) == 0:
            wm.eval()
            s, r, r2 = diagnostics(wm, *probe)
            wm.train()
            c = comp.get("curvature_loss_used_for_training", nan)
            sg_val = comp.get("sigreg_loss", nan)
            print(f"    {i:>5} {comp['z_visual_loss'].item():>14.6f} {s:>11.5f} "
                  f"{r:>11.3f} {r2:>10.4f} {c.item():>9.4f} {sg_val.item():>10.3f}")
            if i == 0:
                first = (s, r, r2)
            if tl is not None:
                tl.probe_latents(i, {"latent/std": s, "latent/eff_rank": r,
                                     "latent/eff_rank_frac": r / 8.0,
                                     "latent/probe_r2": r2})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tl is not None and i % tl.log_every == 0:
            tl.probe_modules(i, {"encoder.trunk": wm.encoder.base_model,
                                 "encoder.projector": wm.encoder.projector,
                                 "predictor": wm.predictor},
                             lr_by_group={"encoder.trunk": lr,
                                          "encoder.projector": lr,
                                          "predictor": 5e-4})
        opt.step()
        if tl is not None:
            tl.maybe_flush(i)

    wm.eval()
    s, r, r2 = diagnostics(wm, *probe)
    print(f"    -> std(b,t) {first[0]:.5f} -> {s:.5f} "
          f"({100 * (1 - s / max(first[0], 1e-12)):.1f}% of latent variation lost)")
    print(f"       eff_rank {first[1]:.2f} -> {r:.2f} | probe R^2 {first[2]:.4f} -> {r2:.4f}")
    if tl is not None:
        tl.probe_latents(steps, {"latent/std": s, "latent/eff_rank": r,
                                 "latent/eff_rank_frac": r / 8.0,
                                 "latent/probe_r2": r2})
        tl.close(steps, status="completed", memory=tl.memory_report())
        print(f"       telemetry -> {tl.path}")
    return s, r, r2


def t3_t4():
    banner("T3/T4  Does stop_grad prevent collapse once the encoder is trained?")
    print("  Task has learnable structure, so a non-collapsed encoder IS available.")
    print("  Collapse indicators, all on a HELD-OUT batch:")
    print("    std(b,t)  -> does z_visual still vary across samples/time?")
    print("    eff_rank  -> participation ratio of the 8-dim latent covariance")
    print("    probe R^2 -> can a linear map read the true position out of z?")
    train_loop("T3  end-to-end (backbone UNFROZEN) + stop_grad=True, no straightening",
               freeze_backbone=False, lr=1e-3)
    train_loop("T4  control: backbone FROZEN (repo default) + stop_grad=True",
               freeze_backbone=True, lr=1e-3)


def t5():
    banner("T5  Does the curvature (straightening) loss provide a collapse barrier?")
    wm = build(straighten="aggcos1e-1", freeze_backbone=False)
    print(f"  straighten parsed -> mode={wm.curvature_mode} scale={wm.straighten_scale}")

    v1 = torch.randn(2, 2, 8)
    v2 = v1 * 1.1 + 0.2 * torch.randn(2, 2, 8)
    print("\n  (a) cosine curvature is scale-invariant, so shrinking the latents is free:")
    for s in (1.0, 1e-1, 1e-3, 1e-5):
        print(f"      latent scale={s:<8} curvature="
              f"{wm._cos_curvature(v1 * s, v2 * s).item():.6f}")

    print("\n  (b) at exact collapse all velocities are ~0, the step_thresh mask empties")
    print("      the tensor and the loss becomes NaN:")
    tiny = torch.full((2, 2, 8), 1e-9)
    out = wm._cos_curvature(tiny, tiny)
    print(f"      _cos_curvature(||v||=1e-9) = {out.item()}  (isnan={bool(torch.isnan(out))})")

    train_loop("T5  end-to-end (UNFROZEN) + stop_grad=True + straightening aggcos1e-1",
               freeze_backbone=False, straighten="aggcos1e-1", lr=1e-3)


def gates():
    """Phase-3 gates 1-3 on CPU, before spending any GPU hour.

    Gate 1  unfrozen + stop_grad, no SIGReg  -> MUST collapse (negative control)
    Gate 2  unfrozen + SIGReg, no curvature  -> must NOT collapse
    Gate 3  unfrozen + SIGReg + curvature    -> straighter, still no collapse
    """
    banner("GATES  Does SIGReg make end-to-end encoder training viable?")
    print("  All runs: backbone UNFROZEN. Gates 2/3 follow LeJEPA/LeWM and drop")
    print("  stop-gradient entirely. Pass = probe R^2 and eff_rank retained.")

    results = {}
    results["gate1_pred_only"] = train_loop(
        "GATE 1  L_pred only, stop_grad=True (negative control -- must collapse)",
        freeze_backbone=False, lr=1e-3, stop_grad=True)
    results["gate2_sigreg"] = train_loop(
        "GATE 2  L_pred + SIGReg (lambda=0.1), stop_grad=False",
        freeze_backbone=False, lr=1e-3, stop_grad=False,
        sigreg=True, sigreg_coeff=0.1)
    results["gate3_sigreg_curv"] = train_loop(
        "GATE 3  L_pred + SIGReg (0.1) + curvature (aggcos1e-1), stop_grad=False",
        freeze_backbone=False, lr=1e-3, stop_grad=False,
        sigreg=True, sigreg_coeff=0.1, straighten="aggcos1e-1")

    banner("GATE VERDICT")
    hdr = f"  {'gate':<22} {'std(b,t)':>10} {'eff_rank/8':>11} {'probe R^2':>10}   verdict"
    print(hdr)
    g1 = results["gate1_pred_only"]
    for name, (s, r, r2) in results.items():
        if name == "gate1_pred_only":
            ok = r2 < 0.1
            verdict = "PASS (collapsed, as required)" if ok else "FAIL (did not collapse)"
        else:
            # must retain information relative to the collapsed control
            ok = r2 > max(0.2, 2 * g1[2]) and r > 1.5 * g1[1]
            verdict = "PASS (no collapse)" if ok else "FAIL (collapsed)"
        print(f"  {name:<22} {s:>10.5f} {r:>11.3f} {r2:>10.4f}   {verdict}")

    c2, c3 = results["gate2_sigreg"], results["gate3_sigreg_curv"]
    print(f"\n  Gate 3 vs Gate 2 (does curvature cost information?): "
          f"probe R^2 {c2[2]:.4f} -> {c3[2]:.4f}, eff_rank {c2[1]:.3f} -> {c3[1]:.3f}")
    return results


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    t1_t2()
    t3_t4()
    t5()
    gates()
    banner("done")
