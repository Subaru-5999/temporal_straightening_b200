"""Tests for the curvature-loss fixes and models/diagnostics.py.

The curvature tests exercise VWorldModel._cos_curvature / total_curvature
directly through a bare instance, so no encoder, dataset or GPU is needed.

Regression under test: _cos_curvature used to return NaN when every velocity
fell below step_thresh (loss[mask].mean() on an empty tensor), which is exactly
the regime an end-to-end trainable encoder enters as its latents shrink.

Run:  pytest tests/test_curvature_and_diagnostics.py -q
"""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.visual_world_model import VWorldModel
from models import diagnostics as dg


# ------------------------------------------------------------------- bare model
def bare(curv_on="features", agg=None):
    """A VWorldModel shell with only what the curvature helpers touch."""
    m = object.__new__(VWorldModel)
    m.curv_on = curv_on
    m.encoder = types.SimpleNamespace()
    if agg is not None:
        m.encoder.agg = agg
    return m


def straight_traj(b=4, t=5, d=8, seed=0):
    """Constant-velocity latent trajectory: curvature must be exactly 0."""
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(b, 1, d, generator=g)
    v = torch.randn(b, 1, d, generator=g)
    steps = torch.arange(t, dtype=torch.float32).reshape(1, t, 1)
    return z0 + v * steps


# --------------------------------------------------------------- the NaN guard
def test_no_nan_when_every_velocity_is_below_threshold():
    """The regression. Was NaN; must now be a finite, differentiable 0."""
    m = bare()
    tiny = torch.full((2, 3, 8), 1e-9, requires_grad=True)
    v = tiny[:, 1:] - tiny[:, :-1]
    out = m._cos_curvature(v[:, :-1], v[:, 1:])
    assert torch.isfinite(out), out
    assert out.item() == 0.0
    out.backward()                      # graph must survive, not detach
    assert tiny.grad is not None


def test_no_nan_for_exactly_zero_velocities():
    m = bare()
    z = torch.zeros(2, 4, 8, requires_grad=True)
    out = m.total_curvature(z, mode="cos")
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_partial_mask_still_averages_over_survivors():
    """Mixed batch: only the moving samples should contribute."""
    m = bare()
    moving = straight_traj(b=3, t=4)                 # curvature 0
    frozen = torch.zeros(1, 4, 8)
    z = torch.cat([moving, frozen], dim=0)
    out = m.total_curvature(z, mode="cos")
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------- loss semantics
def test_straight_trajectory_has_zero_curvature():
    m = bare()
    out = m.total_curvature(straight_traj(), mode="cos")
    assert out.item() == pytest.approx(0.0, abs=1e-5)


def test_reversing_trajectory_is_maximal():
    """v_{t+1} = -v_t gives cos = -1, so 1 - cos = 2, the upper bound."""
    m = bare()
    d = 8
    z = torch.zeros(1, 4, d)
    z[0, 1, 0] = 1.0        # +e0
    z[0, 2, 0] = 0.0        # -e0
    z[0, 3, 0] = 1.0        # +e0
    out = m.total_curvature(z, mode="cos")
    assert out.item() == pytest.approx(2.0, abs=1e-4)


def test_curvature_is_scale_invariant():
    """Documents the defect that motivates SIGReg: shrinking latents is free."""
    m = bare()
    g = torch.Generator().manual_seed(3)
    z = torch.randn(4, 5, 8, generator=g)
    vals = [m.total_curvature(z * s, mode="cos").item() for s in (1.0, 1e-2, 1e-4)]
    # step_thresh masks the tiniest scale; compare the two that survive
    assert vals[0] == pytest.approx(vals[1], rel=1e-4)


def test_requires_at_least_three_frames():
    m = bare()
    with pytest.raises(ValueError, match="at least 3 frames"):
        m.total_curvature(torch.randn(2, 2, 8), mode="cos")


def test_unknown_mode_raises():
    m = bare()
    with pytest.raises(ValueError, match="Unknown curvature mode"):
        m.total_curvature(torch.randn(2, 4, 8), mode="nope")


def test_aggcos_requires_an_agg_head():
    m = bare()
    with pytest.raises(ValueError, match="requires encoder.agg"):
        m.total_curvature(torch.randn(2, 4, 6, 8), mode="aggcos")


# ------------------------------------------------------------- curv_on ablation
def test_curv_on_features_vs_velocity_differ_for_a_nonlinear_head():
    """They must differ, otherwise the ablation axis is meaningless."""
    torch.manual_seed(0)
    head = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(6 * 8, 16),
                               torch.nn.ReLU(), torch.nn.Linear(16, 5))
    z = torch.randn(4, 5, 6, 8)
    a = bare("features", agg=head).total_curvature(z, mode="aggcos").item()
    b = bare("velocity", agg=head).total_curvature(z, mode="aggcos").item()
    assert a != pytest.approx(b, rel=1e-3)


def test_curv_on_agree_for_a_linear_head():
    """Sanity check on the implementation: for linear h, aggregate-then-difference
    and difference-then-aggregate are the same operation."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(6 * 8, 5, bias=False)
    head = torch.nn.Sequential(torch.nn.Flatten(1), lin)
    z = torch.randn(4, 5, 6, 8)
    a = bare("features", agg=head).total_curvature(z, mode="aggcos").item()
    b = bare("velocity", agg=head).total_curvature(z, mode="aggcos").item()
    assert a == pytest.approx(b, rel=1e-4)


def test_invalid_curv_on_is_rejected_at_construction():
    with pytest.raises(ValueError, match="curv_on must be"):
        VWorldModel(
            image_size=224, num_hist=3, num_pred=1,
            encoder=types.SimpleNamespace(emb_dim=8, name="dummy"),
            proprio_encoder=torch.nn.Identity(), action_encoder=torch.nn.Identity(),
            decoder=None, predictor=None, curv_on="sideways",
        )


# ------------------------------------------------------------------ diagnostics
def test_diagnostics_flag_a_collapsed_latent():
    healthy = torch.randn(8, 4, 16)
    collapsed = torch.ones(8, 4, 16) * 0.5

    h = dg.latent_diagnostics(healthy)
    c = dg.latent_diagnostics(collapsed)

    assert h["latent_std"] > 0.5 and c["latent_std"] == pytest.approx(0.0, abs=1e-6)
    assert h["latent_eff_rank"] > 5
    assert not (c["latent_eff_rank"] == c["latent_eff_rank"]) or c["latent_eff_rank"] < 2


def test_probe_r2_detects_information_loss():
    g = torch.Generator().manual_seed(0)
    state = torch.randn(32, 4, 2, generator=g)
    informative = torch.cat([state, torch.randn(32, 4, 6, generator=g)], dim=-1)
    assert dg.probe_r2(informative, state) > 0.99
    # A constant latent carries nothing. Held out, that reads as "<= 0" rather
    # than exactly 0: predicting the fit half's mean on the held-out half is
    # slightly worse than that half's own mean, so a small negative is correct.
    assert dg.probe_r2(torch.zeros(32, 4, 8), state) < 0.05


def test_curvature_cos_is_one_for_a_straight_trajectory():
    assert dg.curvature_cos(straight_traj()) == pytest.approx(1.0, abs=1e-5)


def test_diagnostics_handle_spatial_latents_and_states():
    z = torch.randn(6, 4, 9, 8)
    state = torch.randn(6, 4, 3)
    out = dg.latent_diagnostics(z, state=state, prefix="val_")
    assert set(out) == {
        "val_latent_std", "val_latent_eff_rank", "val_latent_abs_mean",
        "val_curvature_cos", "val_latent_eff_rank_frac", "val_probe_r2",
    }
    assert all(isinstance(v, float) for v in out.values())
    assert out["val_latent_eff_rank_frac"] <= 1.0 + 1e-6


def test_diagnostics_never_raise_on_degenerate_input():
    assert dg.latent_std_across_bt(torch.randn(1, 1, 4)) != dg.latent_std_across_bt(
        torch.randn(1, 1, 4)
    ) or True                                  # NaN != NaN, just must not raise
    assert dg.effective_rank(torch.randn(1, 4)) != dg.effective_rank(torch.randn(1, 4)) or True
    out = dg.curvature_cos(torch.randn(2, 2, 4))
    assert out != out                          # NaN for T < 3, no exception
    assert dg.probe_r2(torch.randn(2, 2, 4), torch.randn(3, 2, 1)) != 0.5


# ------------------------------------------- held-out probe (overfitting guard)
def test_probe_r2_is_held_out_not_in_sample():
    """The bug this fixes: a 128-dim latent probed from 128 samples fitted 129
    coefficients and returned exactly R^2 = 1.0 on the real smoke run.

    Pure noise carries no information about the state, so a held-out score must
    be near zero or negative even when the in-sample fit is perfect.
    """
    g = torch.Generator().manual_seed(0)
    b, t, d = 32, 4, 128                       # 128 samples, 128 features
    noise = torch.randn(b, t, d, generator=g)
    state = torch.randn(b, t, 2, generator=g)
    r2 = dg.probe_r2(noise, state)
    assert r2 < 0.5, f"held-out R^2 on pure noise should not be high, got {r2}"


def test_probe_r2_still_finds_real_information():
    """A latent that genuinely contains the state must still score high."""
    g = torch.Generator().manual_seed(0)
    state = torch.randn(48, 4, 2, generator=g)
    z = torch.cat([state, torch.randn(48, 4, 6, generator=g)], dim=-1)
    assert dg.probe_r2(z, state) > 0.95


def test_probe_r2_zero_for_a_constant_latent():
    """No information -> held-out R^2 at or just below zero, never high."""
    state = torch.randn(32, 4, 2, generator=torch.Generator().manual_seed(0))
    r2 = dg.probe_r2(torch.ones(32, 4, 8), state)
    assert -0.5 < r2 < 0.05, r2


def test_probe_r2_needs_enough_samples_to_split():
    state = torch.randn(2, 2, 2)
    out = dg.probe_r2(torch.randn(2, 2, 8), state)
    assert out != out                          # NaN, not a fabricated number


def test_probe_r2_split_is_interleaved_not_contiguous():
    """A contiguous split would put all late-time samples in the held-out half.

    Build a latent whose scale drifts with the batch index: an interleaved split
    keeps both halves comparable, so the score stays high.
    """
    g = torch.Generator().manual_seed(0)
    b, t = 40, 4
    state = torch.randn(b, t, 2, generator=g)
    drift = torch.linspace(1.0, 5.0, b).reshape(b, 1, 1)
    z = torch.cat([state * drift, torch.randn(b, t, 4, generator=g)], dim=-1)
    assert dg.probe_r2(z, state) > 0.5


def test_diagnostics_dict_uses_the_held_out_probe():
    g = torch.Generator().manual_seed(0)
    state = torch.randn(32, 4, 2, generator=g)
    noise = torch.randn(32, 4, 128, generator=g)
    out = dg.latent_diagnostics(noise, state=state)
    assert out["probe_r2"] < 0.5               # would have been 1.0 before
