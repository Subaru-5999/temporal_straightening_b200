import numpy as np
import torch
import torch.nn as nn


def _channel_scale(z, eps=1e-12):
    """Mean per-feature variance of a target latent, across the eval batch.

    This is the natural unit of the channel: how much that latent varies between
    the goals actually being planned for. Dividing each term by it makes the
    objective scale-invariant.
    """
    if z.shape[0] < 2:
        return torch.ones((), device=z.device, dtype=torch.float32)
    x = z.detach().float().reshape(z.shape[0], -1)
    return x.var(dim=0).mean().clamp_min(eps)


def create_objective_fn(alpha, base, mode="last", normalize=False):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int. relative weight of the proprio term.
        base: int. only used for objective_fn_all
        normalize: bool. Divide each channel's term by its own spread before
            combining, so `alpha` is a true relative weight.

            WHY THIS EXISTS. `loss_visual + alpha * loss_proprio` takes a per-dim
            MEAN on both sides, so the effective weight of the proprio term is
            alpha * (scale_proprio / scale_visual). Those scales are properties of
            the trained encoder, not of the task. With a frozen DINOv2 trunk they
            happen to be comparable and alpha=1 means roughly what it says. Train
            the trunk end-to-end and they diverge: on the PushT SIGReg run the
            measured ratio is 0.001089 / 0.2598, so alpha=1 behaves like
            alpha_eff = 0.0042 and the proprio term is numerically inert
            (planning at alpha=0 and alpha=1 gives an identical 0.12 success).

            That matters because end-to-end training redistributes information
            between the channels of z rather than destroying it: on that run the
            visual latent stopped representing the pusher (probe R^2 0.943 ->
            -0.011) while the proprio channel kept it at 0.997. In PushT the
            action IS the pusher target, so a cost weighted 99.58/0.42 toward the
            channel that forgot the pusher is nearly blind to what actions
            control. The information is present in z; the objective just cannot
            see it.

            Normalizing removes the coupling between encoder scale and objective
            weighting. Default False, so every existing run is unchanged.
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def _weights(z_obs_tgt):
        if not normalize:
            return 1.0, 1.0
        return (_channel_scale(z_obs_tgt["visual"]),
                _channel_scale(z_obs_tgt["proprio"]))

    def objective_fn_last(z_obs_pred, z_obs_tgt, step=None):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        w_v, w_p = _weights(z_obs_tgt)
        loss = loss_visual / w_v + alpha * loss_proprio / w_p
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, step=None, coeffs=None, base=base):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        if coeffs is None:
            coeffs = np.array([base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32)
            coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        else:
            coeffs = coeffs.to(z_obs_pred["visual"].device)

        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        w_v, w_p = _weights(z_obs_tgt)
        loss = loss_visual / w_v + alpha * loss_proprio / w_p
        return loss

    def objective_fn_staged(z_obs_pred, z_obs_tgt, step=None):
        if step is None:
            return objective_fn_all(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)
        # stage 1: optimize only terminal match
        if step < z_obs_pred["visual"].shape[1] - 1:
            return objective_fn_last(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)
        # stage 2: use the full-horizon weighted objective
        else:
            return objective_fn_all(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt, coeffs=None)

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    elif mode == "staged":
        return objective_fn_staged
    else:
        raise NotImplementedError
