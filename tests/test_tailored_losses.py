"""Tests for the tailored counterfactual loss terms (cf_curv, act_sens).

These terms close the two degrees of freedom the rest of the objective leaves
free for planning: the data-trajectory curvature term never sees rollouts OFF
the data distribution, and nothing pins how strongly the terminal latent
responds to actions (the first-order factor dz_H/da of the planning Hessian).

Run:  pytest tests/test_tailored_losses.py -q
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.visual_world_model import VWorldModel
from run_naming import variant_tag


class _Enc(nn.Module):
    """Stand-in DINOv2 patch encoder: (n, 3, H, W) -> (n, num_patches, emb_dim)."""

    def __init__(self, emb_dim=4, patch_size=14, num_patches=9):
        super().__init__()
        self.name = "dinov2_vits14"
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.latent_ndim = 2
        self.lin = nn.Linear(3 * 4 * 4, emb_dim)

    def forward(self, x):
        n = x.shape[0]
        pooled = nn.functional.adaptive_avg_pool2d(x, 4).reshape(n, -1)
        return self.lin(pooled).unsqueeze(1).expand(n, self.num_patches, self.emb_dim)


class _Prop(nn.Module):
    def __init__(self, in_chans=4, emb_dim=10):
        super().__init__()
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.lin = nn.Linear(in_chans, emb_dim)

    def forward(self, x):
        return self.lin(x)


class _Act(_Prop):
    pass


def _model(predictor=None, image_size=48, **kwargs):
    """Small end-to-end model. num_patches = (48 // 16)**2 = 9."""
    num_patches = (image_size // 16) ** 2
    return VWorldModel(
        image_size=image_size,
        num_hist=2,
        num_pred=1,
        encoder=_Enc(emb_dim=4, num_patches=num_patches),
        proprio_encoder=_Prop(),
        action_encoder=_Act(in_chans=2, emb_dim=10),
        decoder=None,
        predictor=nn.Identity() if predictor is None else predictor,
        proprio_dim=10,
        action_dim=10,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        **kwargs,
    )


def _batch(b=6, t=3):
    torch.manual_seed(0)
    obs = {
        "visual": torch.rand(b, t, 3, 48, 48),
        "proprio": torch.randn(b, t, 4),
    }
    act = torch.randn(b, t, 2)
    return obs, act


# ------------------------------------------------------------------ defaults
def test_off_by_default():
    m = _model()
    assert m.cf_curv_coeff == 0.0
    assert m.act_sens_coeff == 0.0
    obs, act = _batch()
    _, _, _, loss, comp = m.forward(obs, act)
    assert "cf_curv_loss" not in comp
    assert "act_sens_loss" not in comp
    assert torch.isfinite(loss)
    cf_loss, cf_comp = m.counterfactual_loss(obs, act)
    assert cf_loss is None and cf_comp == {}


def test_invalid_cf_mode_rejected():
    with pytest.raises(ValueError, match="cf_mode"):
        _model(cf_curv=0.1, cf_mode="nope")


def test_cf_H_below_two_rejected():
    with pytest.raises(ValueError, match="cf_H"):
        _model(cf_curv=0.1, cf_H=1)


def test_nonpositive_margin_rejected():
    with pytest.raises(ValueError, match="act_sens_margin"):
        _model(act_sens=0.1, act_sens_margin=0.0)


# -------------------------------------------------------------- term semantics
def test_identity_predictor_gives_zero_cf_curvature():
    """An identity predictor rolls out a CONSTANT latent trajectory: velocities
    are zero, so the counterfactual curvature must be exactly 0 (the NaN guard),
    not NaN."""
    m = _model(cf_curv=0.1, cf_H=3)
    obs, act = _batch()
    loss, comp = m.counterfactual_loss(obs, act)
    assert loss is not None and "cf_curv_loss" in comp
    assert comp["cf_curv_loss"].item() == 0.0
    assert torch.isfinite(comp["cf_curv_loss"])


def test_action_blind_rollout_pays_the_full_margin():
    """With an identity predictor the terminal latent cannot depend on the
    actions at all: move = 0, so the hinge must be exactly the margin. This is
    the degenerate regime the term exists to punish."""
    margin = 0.25
    m = _model(act_sens=0.1, act_sens_margin=margin)
    obs, act = _batch()
    loss, comp = m.counterfactual_loss(obs, act)
    assert loss is not None
    assert comp["act_sens_loss"].item() == pytest.approx(margin, abs=1e-6)


def test_action_responsive_predictor_reduces_the_hinge():
    """A predictor that mixes the action dims into the state dims makes the
    terminal latent move under different actions: ratio > 0, hinge < margin."""
    torch.manual_seed(1)
    m = _model(predictor=nn.Linear(24, 24, bias=False),
               act_sens=0.1, act_sens_margin=0.1)
    obs, act = _batch()
    loss, comp = m.counterfactual_loss(obs, act)
    assert loss is not None
    assert torch.isfinite(comp["act_sens_loss"])
    assert comp["act_sens_loss"].item() < 0.1


def test_terms_are_differentiable_and_reach_predictor_and_action_encoder():
    torch.manual_seed(2)
    pred = nn.Linear(24, 24, bias=False)
    m = _model(predictor=pred, cf_curv=0.1, cf_H=3, act_sens=0.1)
    obs, act = _batch()
    loss, comp = m.counterfactual_loss(obs, act)
    assert loss is not None
    assert "cf_curv_loss_scaled" in comp and "act_sens_loss_scaled" in comp
    loss.backward()
    assert pred.weight.grad is not None and pred.weight.grad.abs().sum() > 0, (
        "the counterfactual terms must push gradient into the predictor, or "
        "they cannot straighten the rollout map"
    )
    assert m.action_encoder.lin.weight.grad is not None
    # The initial latent is encoded DETACHED by design: the arms train the
    # rollout map (predictor + action encoder), never the encoder -- that is
    # exactly what keeps their memory peak standalone on the GPU.
    assert m.encoder.lin.weight.grad is None


def test_small_batch_skips_terms_instead_of_crashing():
    """The batch shuffle needs >= 3 elements; a short batch must skip the terms
    silently rather than loop forever or crash (mirrors the curvature NaN guard
    philosophy: degenerate inputs contribute exactly nothing)."""
    m = _model(cf_curv=0.1, act_sens=0.1)
    obs, act = _batch(b=2)
    loss, comp = m.counterfactual_loss(obs, act)
    assert loss is None and comp == {}
    _, _, _, main_loss, main_comp = m.forward(obs, act)
    assert torch.isfinite(main_loss)
    assert "cf_curv_loss" not in main_comp


def test_scaled_components_match_coefficients():
    torch.manual_seed(3)
    m = _model(predictor=nn.Linear(24, 24, bias=False),
               cf_curv=0.3, cf_H=3, act_sens=0.7)
    obs, act = _batch()
    loss, comp = m.counterfactual_loss(obs, act)
    assert loss is not None
    torch.testing.assert_close(comp["cf_curv_loss_scaled"],
                               comp["cf_curv_loss"] * 0.3)
    torch.testing.assert_close(comp["act_sens_loss_scaled"],
                               comp["act_sens_loss"] * 0.7)


# ------------------------------------------------------------------ run naming
def test_defaults_still_produce_an_empty_tag():
    assert variant_tag(False, 0.0, True, "features", 0.0, 0.0, 0.0) == ""
    assert variant_tag(False, 0.0, True, "features") == ""       # old callers


def test_cf_and_as_change_the_run_directory():
    base = variant_tag(True, 0.1, False, "features", 1.0)
    tailored = variant_tag(True, 0.1, False, "features", 1.0, 0.1, 0.1)
    assert base == "_sig1e-1_e2e_gp1e0"
    assert tailored == "_sig1e-1_e2e_gp1e0_cf1e-1_as1e-1"
    assert tailored != base, (
        "a tailored-objective run MUST get its own directory or it will resume "
        "the previous objective's checkpoint"
    )


@pytest.mark.parametrize("cf,ass,expected", [
    (0.1, 0.0, "_cf1e-1"),
    (0.0, 0.1, "_as1e-1"),
    (0.1, 0.1, "_cf1e-1_as1e-1"),
    (0.0, 0.0, ""),
    (None, None, ""),
])
def test_cf_as_coefficient_formatting(cf, ass, expected):
    assert variant_tag(False, 0.0, True, "features", 0.0, cf, ass) == expected


# ------------------------------------------------- counterfactual batch fraction
def test_cf_batch_frac_subsamples_the_arms():
    """The arms may run on a subsample of the batch (memory knob); the terms
    must still be produced and finite."""
    torch.manual_seed(4)
    m = _model(predictor=nn.Linear(24, 24, bias=False),
               cf_curv=0.1, cf_H=3, act_sens=0.1, cf_batch_frac=0.5)
    obs, act = _batch(b=6)
    loss, comp = m.counterfactual_loss(obs, act)
    assert loss is not None
    assert torch.isfinite(loss)
    assert "cf_curv_loss" in comp and "act_sens_loss" in comp


def test_cf_batch_frac_is_validated():
    with pytest.raises(ValueError):
        _model(cf_curv=0.1, cf_batch_frac=0.0)
    with pytest.raises(ValueError):
        _model(cf_curv=0.1, cf_batch_frac=1.5)
