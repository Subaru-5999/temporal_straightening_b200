"""Tests for models/sigreg.py.

The properties that matter for the objective:
  * it is minimised by an isotropic standard Gaussian and large otherwise,
  * it is NOT scale-invariant (this is the whole point -- it pins the latent
    scale, which the prediction loss and the cosine curvature term leave free),
  * a collapsed encoder (constant output) is heavily penalised,
  * gradients flow, and it survives bf16/float32 and odd shapes.

Run:  pytest tests/test_sigreg.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sigreg import SIGReg, to_time_major

T, B, D = 4, 256, 16


def sig(num_proj=256, knots=17, seed=0):
    torch.manual_seed(seed)
    return SIGReg(knots=knots, num_proj=num_proj)


def gaussian(t=T, b=B, d=D, seed=0):
    return torch.randn(t, b, d, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------ basic shape
def test_returns_finite_scalar():
    out = sig()(gaussian())
    assert out.shape == ()
    assert torch.isfinite(out)
    assert out.item() >= 0


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(T, B, D\)"):
        sig()(torch.randn(B, D))


@pytest.mark.parametrize("bad", [dict(knots=1), dict(num_proj=0)])
def test_rejects_degenerate_config(bad):
    with pytest.raises(ValueError):
        SIGReg(**bad)


# ------------------------------------------- it is a Gaussianity goodness-of-fit
def test_standard_gaussian_scores_lowest():
    """N(0,1) should score below a shifted, scaled, or degenerate alternative."""
    z = gaussian()
    loss = sig()
    base = loss(z).item()
    assert loss(z * 3.0).item() > base          # wrong variance
    assert loss(z + 3.0).item() > base          # wrong mean
    assert loss(z * 0.05).item() > base         # shrunk


def test_not_scale_invariant():
    """Contrast with the curvature term, which IS scale-invariant (see T5).

    This is precisely why SIGReg can stop the encoder from shrinking its
    latents to drive the prediction loss down.
    """
    z = gaussian()
    loss = sig()
    vals = [loss(z * s).item() for s in (1.0, 0.1, 0.01, 1e-4)]
    # monotonically worse as the latents shrink towards a point mass
    assert vals == sorted(vals), vals
    assert vals[-1] > 10 * vals[0]


def test_collapse_is_heavily_penalised():
    """A constant encoder output -- the global minimum of L_pred -- is the worst case.

    Measured with B=256, D=16: collapsed ~= 102.9 vs N(0,1) ~= 1.18, i.e. ~87x.
    """
    collapsed = torch.zeros(T, B, D)
    healthy = gaussian()
    loss = sig()
    assert loss(collapsed).item() > 50 * loss(healthy).item()


def test_collapse_matches_the_closed_form():
    """For a constant embedding every projection is 0, so the statistic is exact.

    phi_hat(t) = 1 for all t, hence err = (1 - e^{-t^2/2})^2 and
    T = B * sum_k w_k (1 - phi_k)^2 with no Monte-Carlo variance at all.
    """
    loss = sig(num_proj=32)
    expected = float(
        (((1.0 - loss.phi) ** 2) * loss.weights).sum().item() * B
    )
    got = loss(torch.zeros(T, B, D)).item()
    assert got == pytest.approx(expected, rel=1e-5)


def test_statistic_scales_with_sample_count():
    """Epps-Pulley is scaled by the number of samples, so a fixed deviation
    from N(0,1) grows with the batch -- the term does not wash out at scale."""
    loss = sig(num_proj=64)
    small = loss(torch.zeros(T, 32, D)).item()
    large = loss(torch.zeros(T, 128, D)).item()
    assert large == pytest.approx(4 * small, rel=1e-4)


def test_low_rank_is_penalised():
    """Variance funnelled into one direction is not isotropic."""
    z = gaussian()
    z_lowrank = z.clone()
    z_lowrank[..., 1:] *= 0.01
    loss = sig()
    assert loss(z_lowrank).item() > loss(z).item()


# ------------------------------------------------------------------- optimisation
def test_gradients_flow_and_reduce_the_statistic():
    """Descending SIGReg should actually make the embedding more Gaussian."""
    torch.manual_seed(0)
    z = (torch.randn(T, B, D) * 4.0 + 2.0).requires_grad_(True)
    loss = sig(num_proj=512)
    opt = torch.optim.Adam([z], lr=0.1)
    first = loss(z).item()
    for _ in range(60):
        opt.zero_grad()
        out = loss(z)
        out.backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()
        opt.step()
    assert loss(z).item() < first / 2


def test_more_projections_reduce_variance_of_the_estimate():
    z = gaussian()
    few = torch.tensor([SIGReg(num_proj=8)(z).item() for _ in range(12)])
    many = torch.tensor([SIGReg(num_proj=1024)(z).item() for _ in range(12)])
    assert many.std() < few.std()


def test_directions_are_resampled_every_call():
    """Sketching: fresh random projections each step, so calls differ slightly."""
    z = gaussian()
    loss = SIGReg(num_proj=8)
    assert loss(z).item() != loss(z).item()


# ------------------------------------------------------------------- to_time_major
def test_to_time_major_pooled():
    z = torch.randn(5, 3, 7)                     # b t d
    assert to_time_major(z).shape == (3, 5, 7)


def test_to_time_major_spatial_folds_patches_into_batch():
    z = torch.randn(5, 3, 196, 8)                # b t p d
    assert to_time_major(z).shape == (3, 5 * 196, 8)


def test_to_time_major_preserves_values():
    z = torch.randn(2, 3, 4, 5)
    tm = to_time_major(z)
    # element (b=1, t=2, p=3) must survive at (t=2, b*p = 1*4+3)
    assert torch.allclose(tm[2, 1 * 4 + 3], z[1, 2, 3])


def test_to_time_major_rejects_other_ranks():
    with pytest.raises(ValueError):
        to_time_major(torch.randn(4, 5))


# ------------------------------------------------------------------------- dtypes
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_dtypes(dtype):
    z = gaussian().to(dtype)
    out = SIGReg(num_proj=64)(z)
    assert torch.isfinite(out.float())


# ----------------------------------------------------------- device/dtype safety
def test_buffers_follow_the_input_not_the_module():
    """Regression: VWorldModel is built after accelerator.prepare() and is never
    itself moved to the accelerator, so SIGReg's buffers can sit on a different
    device/dtype than the latents. forward() must follow the input, not `self`.

    Simulated here with dtype, which exercises the same code path as device.
    """
    loss = SIGReg(num_proj=32)
    assert loss.t.dtype == torch.float32
    out = loss(gaussian().to(torch.float64))
    assert torch.isfinite(out)
    assert loss.t.dtype == torch.float32          # buffers left untouched


def test_forward_does_not_mutate_buffers():
    loss = sig(num_proj=16)
    before = [b.clone() for b in (loss.t, loss.phi, loss.weights)]
    loss(gaussian())
    for b, a in zip(before, (loss.t, loss.phi, loss.weights)):
        assert torch.equal(b, a)


def test_bf16_input_is_evaluated_in_fp32():
    """Training runs under mixed_precision=bf16, so the latents arrive as bf16.

    The statistic must not be computed at bf16 precision: it averages cosines
    over the batch and squares a ~1e-2 deviation from phi, so 3 significant
    digits is not enough and lambda_SIG would be effectively noisy.
    """
    loss = sig(num_proj=256, seed=1)
    z = gaussian(seed=1)
    torch.manual_seed(0)
    ref = loss(z).item()
    torch.manual_seed(0)
    got = loss(z.to(torch.bfloat16)).item()
    # bf16 rounding of the *input* still shifts things slightly, but the result
    # must be close to the fp32 answer, not merely finite
    assert got == pytest.approx(ref, rel=0.05), (got, ref)


def test_bf16_gradients_flow_back_into_the_bf16_graph():
    z = (torch.randn(T, 64, D) * 2).to(torch.bfloat16).requires_grad_(True)
    out = SIGReg(num_proj=64)(z)
    out.backward()
    assert z.grad is not None
    assert z.grad.dtype == torch.bfloat16
    assert torch.isfinite(z.grad.float()).all()


def test_fp32_and_fp64_are_left_alone():
    for dt in (torch.float32, torch.float64):
        out = SIGReg(num_proj=32)(gaussian().to(dt))
        assert out.dtype == dt
