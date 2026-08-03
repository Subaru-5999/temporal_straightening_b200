"""Collapse and geometry diagnostics for latent world-model training.

A falling prediction loss is NOT evidence that training is working: the
cheapest way to minimise it is for the encoder to stop responding to its input
(see experiments/verify_stop_grad.py, T3 -- loss fell 13x while a linear probe
went from R^2 0.85 to 0.00). These metrics separate the two cases, and are the
minimum needed to tell a healthy end-to-end run from a collapsed one.

Metrics
  latent_std          std of the latents across (batch, time), averaged over
                      the remaining axes. -> 0 means the encoder has stopped
                      distinguishing inputs.
  latent_eff_rank     participation ratio of the feature covariance spectrum,
                      (sum l)^2 / sum l^2, in [1, d]. -> 1 means all variance
                      has funnelled into a single direction.
  latent_eff_rank_frac eff_rank / d, so runs with different d are comparable.
  latent_abs_mean     mean |z|; catches the pure-shrinkage failure mode that
                      cosine curvature is blind to.
  curvature_cos       mean cos(v_t, v_{t+1}) over consecutive latent
                      velocities: the paper's straightness measure. 1.0 = a
                      perfectly straight latent trajectory.
  probe_r2            R^2 of a ridge linear probe from the latent to the
                      ground-truth state, when states are available. The
                      information-preservation check.

All functions are no-grad and safe to call on any tensor shape of the form
(b, t, ...); they never raise on degenerate input, returning NaN instead so a
collapsed run logs rather than crashes.
"""

import logging

import torch

log = logging.getLogger(__name__)

_NAN = float("nan")


@torch.no_grad()
def _flatten_features(z: torch.Tensor) -> torch.Tensor:
    """(b, t, ..., d) -> (n, d), folding every leading axis into n."""
    return z.reshape(-1, z.shape[-1])


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Participation ratio of the covariance spectrum, in [1, d]."""
    x = _flatten_features(z).float()
    if x.shape[0] < 2:
        return _NAN
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    try:
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    except Exception:                                  # pragma: no cover
        return _NAN
    denom = ev.pow(2).sum()
    if denom <= 0:
        return _NAN
    return float((ev.sum() ** 2 / denom).item())


@torch.no_grad()
def latent_std_across_bt(z: torch.Tensor) -> float:
    """std over the (batch, time) axes, averaged over the rest.

    This is the collapse indicator: it asks whether the latent still *varies*
    from input to input, which is exactly what a constant encoder destroys.
    """
    if z.dim() < 3:
        raise ValueError(f"expected (b, t, ...), got {tuple(z.shape)}")
    b, t = z.shape[0], z.shape[1]
    if b * t < 2:
        return _NAN
    flat = z.reshape(b * t, *z.shape[2:]).float()
    return float(flat.std(dim=0).mean().item())


@torch.no_grad()
def curvature_cos(z: torch.Tensor) -> float:
    """mean cos(v_t, v_{t+1}) over consecutive velocities; 1.0 == straight.

    Reported as the cosine itself (not 1 - cosine) so it reads as the paper's
    straightness measure rather than as a loss.
    """
    if z.shape[1] < 3:
        return _NAN
    v = z[:, 1:] - z[:, :-1]
    v1, v2 = v[:, :-1], v[:, 1:]
    cos = torch.nn.functional.cosine_similarity(v1.float(), v2.float(), dim=-1, eps=1e-8)
    return float(cos.mean().item())


@torch.no_grad()
def probe_r2(z: torch.Tensor, state: torch.Tensor, ridge: float = 1e-3) -> float:
    """HELD-OUT R^2 of a ridge linear probe from the latent to the true state.

    Fit on half the samples, scored on the other half. The in-sample version of
    this metric is worthless whenever the latent has many dimensions relative to
    the batch: a 128-dim aggregated latent probed from a 32x4 val batch fits 129
    coefficients from 128 rows and returns exactly R^2 = 1.0, which is what it
    did before this was a split fit. A held-out score exposes that instead of
    hiding it -- an overfitted probe simply scores badly on the other half.

    The split is interleaved (even/odd samples) rather than contiguous, so the
    two halves both span the batch and the time axis.

    Args:
        z: (b, t, ...) latents.
        state: (b, t, s) ground-truth state.
    Returns:
        Held-out R^2, clamped to [-1, 1]. ~0 or below means the latent carries
        no usable linear information about the state -- the signature of
        collapse. NaN when there are too few samples to split.
    """
    if z.dim() < 3 or state.dim() != 3:
        return _NAN
    b, t = z.shape[0], z.shape[1]
    if state.shape[0] != b or state.shape[1] != t or b * t < 8:
        return _NAN
    X = z.reshape(b * t, -1).float()
    Y = state.reshape(b * t, -1).float()
    X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)

    fit, held = X[0::2], X[1::2]
    y_fit, y_held = Y[0::2], Y[1::2]
    if fit.shape[0] < 2 or held.shape[0] < 2:
        return _NAN

    gram = fit.T @ fit + ridge * torch.eye(fit.shape[1], device=X.device)
    try:
        W = torch.linalg.lstsq(gram, fit.T @ y_fit).solution
    except Exception:                                  # pragma: no cover
        return _NAN
    resid = ((held @ W - y_held) ** 2).sum()
    total = ((y_held - y_held.mean(0, keepdim=True)) ** 2).sum()
    if total <= 0:
        return _NAN
    return float((1 - resid / total).clamp(-1, 1).item())


@torch.no_grad()
def latent_diagnostics(z: torch.Tensor, state: torch.Tensor = None, prefix: str = "") -> dict:
    """All of the above in one dict, ready to hand to a logger.

    Args:
        z: (b, t, d) or (b, t, p, d) latents. For 4-D input the curvature and
           probe metrics use the patch-mean, matching how a global/aggregated
           trajectory representation would be read out.
        state: optional (b, t, s) ground truth for the linear probe.
        prefix: prepended to every key, e.g. "train_".
    """
    z = z.detach().float()
    z_global = z.mean(dim=2) if z.dim() == 4 else z

    out = {
        f"{prefix}latent_std": latent_std_across_bt(z),
        f"{prefix}latent_eff_rank": effective_rank(z),
        f"{prefix}latent_abs_mean": float(z.abs().mean().item()),
        f"{prefix}curvature_cos": curvature_cos(z_global),
    }
    d = z.shape[-1]
    er = out[f"{prefix}latent_eff_rank"]
    out[f"{prefix}latent_eff_rank_frac"] = er / d if er == er else _NAN  # NaN-safe
    if state is not None:
        out[f"{prefix}probe_r2"] = probe_r2(z_global, state.detach())
    return out
