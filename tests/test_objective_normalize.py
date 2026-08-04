"""The scale-invariant planning objective.

Guards the fix for the failure diagnosed on the PushT end-to-end + SIGReg run:
`loss_visual + alpha * loss_proprio` uses a per-dim mean on both sides, so the
effective weight of the proprio term is alpha * (scale_proprio / scale_visual).
Those scales belong to the trained encoder, not the task. Measured on that run:
0.001089 / 0.2598, i.e. alpha=1 behaved as alpha_eff = 0.0042 and planning at
alpha=0 and alpha=1 returned an identical 0.12 success rate.
"""

import torch

from planning.objectives import create_objective_fn


def _obs(b=16, p=4, dv=8, dp=3, visual_scale=1.0, proprio_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "visual": torch.randn(b, 1, p, dv, generator=g) * visual_scale,
        "proprio": torch.randn(b, 1, dp, generator=g) * proprio_scale,
    }


def _pred(tgt, b=16, p=4, dv=8, dp=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "visual": torch.randn(b, 1, p, dv, generator=g) * tgt["visual"].std(),
        "proprio": torch.randn(b, 1, dp, generator=g) * tgt["proprio"].std(),
    }


def test_default_is_unchanged():
    """normalize=False must reproduce the original formula exactly."""
    tgt = _obs()
    pred = _pred(tgt)
    for mode in ("last", "all", "staged"):
        fn = create_objective_fn(alpha=1, base=2, mode=mode)
        ref = create_objective_fn(alpha=1, base=2, mode=mode, normalize=False)
        torch.testing.assert_close(fn(pred, tgt, step=9), ref(pred, tgt, step=9))


def test_unnormalized_weight_follows_the_channel_scales():
    """The bug: shrinking the proprio channel silently removes its term.

    Two setups with the SAME alpha and the SAME *relative* prediction error, but
    proprio scaled down 100x. Unnormalized, the proprio share of the cost falls
    by ~1e-4; that is exactly how alpha=1 became alpha_eff=0.0042.
    """
    fn = create_objective_fn(alpha=1, base=2, mode="last")

    def proprio_share(proprio_scale):
        tgt = _obs(proprio_scale=proprio_scale)
        pred = _pred(tgt)
        total = fn(pred, tgt).mean()
        # same objective with the proprio term switched off
        visual_only = create_objective_fn(alpha=0, base=2, mode="last")(pred, tgt).mean()
        return float((total - visual_only) / total)

    balanced = proprio_share(1.0)
    shrunk = proprio_share(0.01)
    assert balanced > 0.1, f"proprio should matter when scales match, got {balanced}"
    assert shrunk < balanced / 100, (
        f"shrinking proprio 100x must gut its contribution: {shrunk} vs {balanced}"
    )


def test_normalized_weight_is_invariant_to_channel_scale():
    """The fix: the proprio share stays put when the channel is rescaled."""
    fn = create_objective_fn(alpha=1, base=2, mode="last", normalize=True)
    off = create_objective_fn(alpha=0, base=2, mode="last", normalize=True)

    def proprio_share(proprio_scale, visual_scale=1.0):
        tgt = _obs(proprio_scale=proprio_scale, visual_scale=visual_scale)
        pred = _pred(tgt)
        total = fn(pred, tgt).mean()
        return float((total - off(pred, tgt).mean()) / total)

    base = proprio_share(1.0)
    for s in (0.01, 0.1, 10.0, 100.0):
        got = proprio_share(s)
        assert abs(got - base) < 0.05, (
            f"proprio share moved from {base} to {got} when proprio was scaled {s}x; "
            "the normalized objective must be scale-invariant"
        )
    # and invariant to the visual channel's scale too
    for s in (0.01, 100.0):
        got = proprio_share(1.0, visual_scale=s)
        assert abs(got - base) < 0.05, (
            f"proprio share moved to {got} when visual was scaled {s}x"
        )


def test_alpha_means_what_it_says_when_normalized():
    """With normalize=True, alpha=1 splits the cost roughly evenly."""
    tgt = _obs(proprio_scale=0.01)          # the pathological case
    pred = _pred(tgt)
    fn = create_objective_fn(alpha=1, base=2, mode="last", normalize=True)
    off = create_objective_fn(alpha=0, base=2, mode="last", normalize=True)
    total = fn(pred, tgt).mean()
    share = float((total - off(pred, tgt).mean()) / total)
    assert 0.2 < share < 0.8, f"alpha=1 should be a near-even split, got {share}"


def test_normalize_applies_to_all_modes():
    tgt = _obs(proprio_scale=0.01)
    pred = _pred(tgt)
    for mode in ("last", "all", "staged"):
        plain = create_objective_fn(alpha=1, base=2, mode=mode)(pred, tgt, step=9)
        norm = create_objective_fn(alpha=1, base=2, mode=mode,
                                   normalize=True)(pred, tgt, step=9)
        assert not torch.allclose(plain, norm), f"mode={mode} ignored normalize"


def test_single_sample_batch_does_not_produce_nan():
    """Variance over a batch of 1 is undefined; the scale must fall back to 1."""
    tgt = _obs(b=1)
    pred = _pred(tgt, b=1)
    out = create_objective_fn(alpha=1, base=2, mode="last", normalize=True)(pred, tgt)
    assert torch.isfinite(out).all(), out
