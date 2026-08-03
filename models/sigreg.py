"""SIGReg -- Sketched Isotropic Gaussian Regularization.

The anti-collapse term of LeJEPA (Balestriero & LeCun, 2025), as used by
LeWorldModel. It replaces stop-gradient / EMA / teacher-student heuristics with
a single distributional constraint: the embeddings must look like an isotropic
Gaussian.

Method (LeWM sec. 3, "Training Objective"):
    Let Z in R^{N x B x d} be the latent embeddings over history length N, batch
    B and embedding dim d. Assessing normality directly in d dimensions is hard,
    so SIGReg projects onto M random unit-norm directions u^(m) in S^{d-1} and
    sums the univariate Epps-Pulley statistic T over the resulting 1-D
    projections h^(m) = Z u^(m):

        SIGReg(Z) = (1/M) sum_m T(h^(m))

    By Cramer-Wold, matching every 1-D marginal matches the joint distribution.
    Reference hyperparameters: M = 1024 projections, lambda = 0.1.

The Epps-Pulley statistic compares the empirical characteristic function of a
projection against the standard normal's, phi(t) = exp(-t^2/2), integrated over
t with a Gaussian window:

        T(h) = B * integral | E[e^{i t h}] - e^{-t^2/2} |^2 exp(-t^2/2) dt

evaluated by the trapezoid rule on `knots` nodes over t in [0, t_max]. Cost is
linear in B, M and knots -- no BxB Gram matrix.

Two properties matter for this codebase (see experiments/verify_stop_grad.py):
  * It pins the *scale* of the latents, which the prediction loss and the
    cosine curvature term both leave free. A shrinking encoder is no longer a
    free way to drive the prediction loss down.
  * The expectation is taken over the BATCH at each time-step, so a collapsed
    encoder (identical output for every input) is maximally penalised: all mass
    at one point is as far from N(0,1) as it gets.

Numerics: the quadrature nodes/weights follow the LeJEPA reference
implementation used by LeWorldModel (`research_papers/le-wm/module.py`) rather
than being re-derived here, so the statistic is on the same scale as the
published lambda values.
"""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (single-process).

    Args:
        knots: quadrature nodes for the characteristic-function integral (17).
        num_proj: number of random 1-D projections M (1024).
        t_max: upper limit of the integration range (3.0).

    Forward input is ``(T, B, D)`` -- time-major, matching the reference. The
    empirical characteristic function is estimated across ``B`` at each ``T``,
    then averaged over projections and time.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024, t_max: float = 3.0):
        super().__init__()
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        if num_proj < 1:
            raise ValueError(f"num_proj must be >= 1, got {num_proj}")
        self.knots = int(knots)
        self.num_proj = int(num_proj)
        self.t_max = float(t_max)

        t = torch.linspace(0, self.t_max, self.knots, dtype=torch.float32)
        dt = self.t_max / (self.knots - 1)
        # trapezoid weights: half-width at the two endpoints
        weights = torch.full((self.knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)   # = phi(t), also the Gaussian weight

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def extra_repr(self) -> str:
        return f"knots={self.knots}, num_proj={self.num_proj}, t_max={self.t_max}"

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: ``(T, B, D)`` latent embeddings, time-major.
        Returns:
            Scalar statistic, >= 0, zero iff every 1-D marginal is exactly N(0,1).
        """
        if proj.dim() != 3:
            raise ValueError(
                f"SIGReg expects (T, B, D); got shape {tuple(proj.shape)}. "
                "Flatten any extra axes into B before calling."
            )
        # Always evaluate the statistic in at least float32. Training runs under
        # accelerate with mixed_precision=bf16, so the encoder hands us bf16
        # latents, and this statistic averages cosines over the batch and then
        # squares the deviation from phi -- differences of order 1e-2 between
        # numbers of order 1, which is where bf16's ~3 significant digits bite.
        # The tensors here are tiny (T x B x M x knots on an 8- or 128-dim
        # latent), so the upcast is free, and the gradient still flows back into
        # the bf16 graph. Without it, lambda_SIG would effectively be noisy.
        if proj.dtype not in (torch.float32, torch.float64):
            proj = proj.float()

        d = proj.size(-1)
        # fresh random directions on the unit sphere every step (sketching)
        A = torch.randn(d, self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        # Follow the input's device AND dtype. VWorldModel is instantiated after
        # accelerator.prepare() and is never itself moved to the accelerator, so
        # these buffers can still be on CPU while the latents are on GPU. They
        # are `knots` floats, so the copy is free.
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)

        # (T, B, M) -> (T, B, M, knots)
        x_t = (proj @ A).unsqueeze(-1) * t
        # empirical CF over the batch axis (-3) vs the standard-normal CF
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        # integrate over t, scale by the sample count -> Epps-Pulley statistic
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()   # average over projections and time


def to_time_major(z: torch.Tensor) -> torch.Tensor:
    """Reshape latents to the ``(T, B, D)`` layout SIGReg expects.

    Accepts ``(b, t, d)`` (pooled/global features) or ``(b, t, p, d)`` (spatial
    tokens, where the ``p`` patches are folded into the batch axis so each token
    is treated as an independent sample of the same distribution).
    """
    if z.dim() == 3:
        return z.transpose(0, 1)                       # b t d -> t b d
    if z.dim() == 4:
        b, t, p, d = z.shape
        return z.permute(1, 0, 2, 3).reshape(t, b * p, d)   # t (b p) d
    raise ValueError(f"Expected a 3-D or 4-D latent tensor, got {tuple(z.shape)}")
