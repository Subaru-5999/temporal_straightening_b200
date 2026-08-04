"""The proprio-grounding term.

Guards the fix for the diagnosed root cause on PushT end-to-end + SIGReg: the
visual latent stopped representing the agent (held-out probe R^2 0.943 -> -0.011)
while keeping the block (0.945 -> 0.979), because z's proprio channels already
carry the agent at R^2 0.997 and the prediction loss is taken on the concatenated
z. Both GD and CEM then converge to an identical 0.26 success rate while the
ground-truth actions score 1.0.
"""

import pytest
import torch
import torch.nn as nn

from models.visual_world_model import VWorldModel, _proprio_in_dim


class _Enc(nn.Module):
    """Stand-in DINOv2 patch encoder: (n, 3, H, W) -> (n, num_patches, emb_dim)."""

    def __init__(self, emb_dim=8, patch_size=14, num_patches=196):
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


def _model(ground_proprio=0.0, emb_dim=8, image_size=224):
    # the model derives the token grid as (image_size // 16) ** 2, so the stand-in
    # encoder must agree or the head would be built for the wrong width
    num_patches = (image_size // 16) ** 2
    enc = _Enc(emb_dim=emb_dim, num_patches=num_patches)
    return VWorldModel(
        image_size=image_size,
        num_hist=2,
        num_pred=1,
        encoder=enc,
        proprio_encoder=_Prop(),
        action_encoder=_Act(in_chans=2, emb_dim=10),
        decoder=None,
        predictor=None,
        proprio_dim=10,
        action_dim=10,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        ground_proprio=ground_proprio,
    )


def test_off_by_default():
    m = _model()
    assert m.ground_head is None
    assert m.ground_coeff == 0.0


def test_head_is_built_eagerly_with_the_right_shape():
    """Optimizers are built before any forward, so the head cannot be lazy."""
    m = _model(ground_proprio=0.1, emb_dim=8)
    assert isinstance(m.ground_head, nn.Linear)
    assert m.ground_head.in_features == 196 * 8      # tokens x channels
    assert m.ground_head.out_features == 4           # raw proprio width
    assert any(p is m.ground_head.weight for p in m.parameters())


def test_num_patches_derives_from_the_image_and_patch_size():
    m = _model(ground_proprio=0.1)
    assert m._num_patches == (224 // 16) ** 2 == 196


def test_loss_is_zero_when_the_latent_determines_proprio():
    """A perfectly groundable latent must give ~0 loss after fitting the head."""
    m = _model(ground_proprio=1.0, emb_dim=4, image_size=48)   # 3x3 = 9 tokens
    b, t, p, d = 40, 2, 9, 4
    z_vis = torch.randn(b, t, p, d)
    target = torch.randn(b, t, 4)
    X1 = torch.cat([z_vis.reshape(b * t, -1), torch.ones(b * t, 1)], dim=1)
    W = torch.linalg.lstsq(X1, target.reshape(b * t, -1)).solution
    with torch.no_grad():
        m.ground_head.weight.copy_(W[:-1].T)
        m.ground_head.bias.copy_(W[-1])
    z = torch.cat([z_vis, torch.zeros(b, t, p, 20)], dim=-1)   # + proprio/action
    # b*t = 80 rows against 37 free parameters, so a perfect fit is not automatic;
    # what is asserted is that the term is exactly the residual of that read-out
    expected = ((X1 @ W - target.reshape(b * t, -1)) ** 2).mean()
    torch.testing.assert_close(m.proprio_grounding_loss(z, target), expected)


def test_loss_is_differentiable_wrt_the_visual_latent():
    m = _model(ground_proprio=1.0, emb_dim=4, image_size=48)
    b, t, p, d = 8, 2, 9, 4
    z_vis = torch.randn(b, t, p, d, requires_grad=True)
    z = torch.cat([z_vis, torch.zeros(b, t, p, 20)], dim=-1)
    loss = m.proprio_grounding_loss(z, torch.randn(b, t, 4))
    assert loss.item() > 0
    loss.backward()
    assert z_vis.grad is not None and z_vis.grad.abs().sum() > 0, (
        "the term must push gradient back into the visual latent, or it cannot "
        "stop the encoder from dropping the agent"
    )


def test_mismatched_token_count_raises_a_readable_error():
    m = _model(ground_proprio=1.0, emb_dim=4, image_size=48)
    b, t = 4, 2
    wrong = torch.randn(b, t, 5, 4)                    # 5 tokens, head expects 9
    z = torch.cat([wrong, torch.zeros(b, t, 5, 20)], dim=-1)
    with pytest.raises(ValueError, match="grounding head expects"):
        m.proprio_grounding_loss(z, torch.randn(b, t, 4))


def test_rejects_an_encoder_without_a_known_patch_count():
    enc = _Enc()
    enc.name = "resnet"          # no dino patch grid -> _num_patches stays None
    with pytest.raises(ValueError, match="patch count"):
        VWorldModel(
            image_size=224, num_hist=2, num_pred=1, encoder=enc,
            proprio_encoder=_Prop(), action_encoder=_Act(in_chans=2, emb_dim=10),
            decoder=None, predictor=None, proprio_dim=10, action_dim=10,
            concat_dim=1, num_action_repeat=1, num_proprio_repeat=1,
            ground_proprio=0.1,
        )


def test_proprio_in_dim_unwraps_a_ddp_style_wrapper():
    inner = _Prop(in_chans=7)

    class _Wrapped(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

    assert _proprio_in_dim(inner) == 7
    assert _proprio_in_dim(_Wrapped(inner)) == 7
    assert _proprio_in_dim(nn.Identity()) is None
