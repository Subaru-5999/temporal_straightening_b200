import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torchvision import transforms
from einops import rearrange, repeat

log = logging.getLogger(__name__)


def _proprio_in_dim(proprio_encoder):
    """Raw proprio observation width, unwrapping an accelerate/DDP wrapper."""
    for mod in (proprio_encoder, getattr(proprio_encoder, "module", None)):
        dim = getattr(mod, "in_chans", None)
        if dim:
            return int(dim)
    return None


class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        straighten=False,
        curv_on="features",
        stop_grad=True,
        vcreg=False,
        vcreg_std_coeff=0,
        vcreg_cov_coeff=0,
        vcreg_apply_to="enc",
        sigreg=False,
        sigreg_coeff=0.0,
        sigreg_num_proj=1024,
        sigreg_knots=17,
        sigreg_apply_to="agg",
        ground_proprio=0.0,
        ground_proprio_dims=None,
        cf_curv=0.0,
        cf_H=4,
        cf_mode="cos",
        act_sens=0.0,
        act_sens_margin=0.1,
        cf_batch_frac=0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
        self.straighten = False
        self.straighten_scale = 0.0
        self.curvature_mode = None
        self.stop_grad = bool(stop_grad)
        self.vcreg = bool(vcreg)
        self.std_coeff = float(vcreg_std_coeff)
        self.cov_coeff = float(vcreg_cov_coeff)
        if vcreg_apply_to != "enc":
            raise ValueError(
                f"Only encoder VCReg is supported, got vcreg_apply_to='{vcreg_apply_to}'."
            )

        if isinstance(straighten, str):
            if straighten.startswith("aggcos"):
                suffix = straighten.replace("aggcos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "aggcos"
            elif straighten.startswith("cos"):
                suffix = straighten.replace("cos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "cos"

        self.straighten = self.curvature_mode is not None and self.straighten_scale > 0

        # Where the curvature is measured. The paper is internally inconsistent
        # here: App. B.6's [agg] equation applies the head to the velocity,
        #   C_t = cos(h(v_t), h(v_{t+1})),
        # while Fig. "train_agg"'s caption says the curvature loss is applied to
        # the *aggregated features*. These differ because h is a nonlinear MLP.
        # "features" (aggregate, then difference) is the original code path and
        # stays the default; "velocity" (difference, then aggregate) implements
        # the equation. Ablate, and state which one the submission used.
        if curv_on not in ("features", "velocity"):
            raise ValueError(
                f"curv_on must be 'features' or 'velocity', got {curv_on!r}"
            )
        self.curv_on = curv_on

        # SIGReg: the distributional anti-collapse term (LeJEPA / LeWM).
        self.sigreg_coeff = float(sigreg_coeff)
        self.sigreg_enabled = bool(sigreg) and self.sigreg_coeff > 0
        if sigreg_apply_to not in ("agg", "patch"):
            raise ValueError(
                f"sigreg_apply_to must be 'agg' or 'patch', got {sigreg_apply_to!r}"
            )
        self.sigreg_apply_to = sigreg_apply_to
        self.sigreg = None
        if self.sigreg_enabled:
            from models.sigreg import SIGReg

            self.sigreg = SIGReg(knots=int(sigreg_knots), num_proj=int(sigreg_num_proj))
            if sigreg_apply_to == "agg" and not hasattr(encoder, "agg"):
                raise ValueError(
                    "sigreg_apply_to='agg' requires an encoder exposing .agg(); "
                    "use sigreg_apply_to='patch' or an encoder with an agg head."
                )

        log.info("num_action_repeat: %s", self.num_action_repeat)
        log.info("num_proprio_repeat: %s", self.num_proprio_repeat)
        log.info("proprio encoder: %s", proprio_encoder)
        log.info("action encoder: %s", action_encoder)
        log.info("proprio_dim: %s, after repeat: %s", proprio_dim, self.proprio_dim)
        log.info("action_dim: %s, after repeat: %s", action_dim, self.action_dim)
        log.info("emb_dim: %s", self.emb_dim)
        if self.straighten:
            log.info(
                "Straightening enabled: mode=%s, scale=%s, curv_on=%s",
                self.curvature_mode,
                self.straighten_scale,
                self.curv_on,
            )
        else:
            log.info("Straightening disabled")
        if self.sigreg_enabled:
            log.info(
                "SIGReg enabled: coeff=%s, num_proj=%s, knots=%s, apply_to=%s",
                self.sigreg_coeff,
                sigreg_num_proj,
                sigreg_knots,
                self.sigreg_apply_to,
            )
        else:
            log.info("SIGReg disabled")
        log.info("Stop-grad enabled: %s", self.stop_grad)
        if self.sigreg_enabled and self.stop_grad:
            log.warning(
                "SIGReg is on together with stop_grad=True. LeJEPA/LeWM train "
                "fully end-to-end with no stop-gradient; stop_grad=False is the "
                "intended setting for the SIGReg variant."
            )
        log.info(
            "VCReg enabled: %s, apply_to=enc, std_coeff=%s, cov_coeff=%s",
            self.vcreg,
            self.std_coeff,
            self.cov_coeff,
        )

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        log.info("Model emb_dim: %s", self.emb_dim)

        self._num_patches = None
        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            # the encoder sees encoder_image_size pixels at patch_size stride, so
            # the token grid is num_side_patches on a side
            self._num_patches = num_side_patches * num_side_patches
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        # ---- proprio grounding: keep the pusher IN the visual latent ----------
        # Diagnosed failure (PushT, end-to-end + SIGReg, see PROGRESS_SIGREG_E2E.md):
        # the visual latent stopped representing the agent/pusher (held-out probe
        # R^2 0.943 -> -0.011) while KEEPING the block (0.945 -> 0.979). The
        # information was not destroyed, it moved: z's proprio channels carry the
        # agent at R^2 0.997. Because the prediction loss is taken on the
        # concatenated z, the visual tokens are never required to encode anything
        # the proprio channels already provide, and dropping it lowers the loss.
        # SIGReg constrains only the distribution and the curvature term only the
        # direction, so neither forbids it. With a frozen trunk it cannot happen.
        #
        # The consequence is fatal for planning but invisible in training metrics:
        # in PushT the action IS the pusher target, so a cost dominated by the
        # visual channel cannot tell whether the pusher went where it was sent.
        # Both GD and CEM converge to an identical 0.26 success rate on that
        # checkpoint, while the ground-truth actions score 1.0 -- the optimiser is
        # fine, the cost's optimum is simply in the wrong place.
        #
        # This term closes the loophole: a LINEAR read-out of the visual tokens
        # must reproduce the proprio observation. Linear on purpose -- the planning
        # cost is Euclidean on those tokens, so information that is only
        # non-linearly decodable would not make the cost sensitive to the pusher.
        # The target is an existing model INPUT, not a new label.
        self.ground_coeff = float(ground_proprio)
        self.ground_head = None
        if self.ground_coeff > 0:
            if self._num_patches is None:
                raise ValueError(
                    "ground_proprio > 0 requires an encoder with a known patch "
                    "count (a DINOv2 patch-token encoder); got "
                    f"{getattr(self.encoder, 'name', type(self.encoder).__name__)!r}."
                )
            # The head is built here, not lazily on the first forward, because the
            # optimizers are constructed before any forward pass. Both sizes are
            # therefore read from the modules: the token count and channel width
            # from the encoder, the target width from the proprio encoder's input.
            proprio_width = _proprio_in_dim(self.proprio_encoder)
            if not proprio_width:
                raise ValueError(
                    "ground_proprio > 0 needs the proprio encoder to expose its "
                    "input width (`in_chans`); use proprio_encoder=proprio."
                )
            # Which proprio dimensions to ground on. Grounding on ALL of them is
            # wrong for PushT, whose proprio is [agent_x, agent_y, vel_x, vel_y]:
            # velocity is not identifiable from a SINGLE frame, so half the target
            # is ill-posed. Measured cost of including it (8k-step runs): the
            # grounding loss plateaus near 0.23 instead of approaching 0, and
            # straightness degrades to cos 0.327 against 0.589 ungrounded --
            # forcing unpredictable high-frequency content into the latent is
            # exactly what raises curvature. Ground on positions only.
            if ground_proprio_dims is None:
                dims = list(range(proprio_width))
            else:
                dims = [int(d) for d in ground_proprio_dims]
                bad = [d for d in dims if not 0 <= d < proprio_width]
                if bad or not dims:
                    raise ValueError(
                        f"ground_proprio_dims={ground_proprio_dims} is invalid for a "
                        f"{proprio_width}-dim proprio observation."
                    )
            self.register_buffer(
                "ground_dims", torch.as_tensor(dims, dtype=torch.long), persistent=False
            )
            in_features = self._num_patches * self.encoder.emb_dim
            self.ground_head = nn.Linear(in_features, len(dims))
            log.info(
                "Proprio grounding enabled: coeff=%s, linear head %d -> %d, "
                "proprio dims %s of %d",
                self.ground_coeff, in_features, len(dims), dims, proprio_width,
            )
        else:
            log.info("Proprio grounding disabled")

        # ---- counterfactual geometry: what the planner actually needs --------
        # The Hessian of the planning cost J(a) = || z_H(a) - z_g ||^2 splits as
        #   d^2J/da^2 = 2 (dz_H/da)^T (dz_H/da)  +  2 (z_H - z_g) . d^2z_H/da^2 .
        # SIGReg pins the latent distribution, the curvature term pins the
        # DIRECTION of data velocities, grounding pins the content -- but the
        # data-trajectory curvature term never sees rollouts OFF the data
        # distribution, and nothing pins the first-order factor dz_H/da (how far
        # the terminal latent moves when actions change). Both gaps were measured
        # to bite (PROGRESS_SIGREG_E2E.md: snr ~3, representation matching
        # pristine DINOv2 on probes yet 47 points short on planning). Two terms:
        #   cf_curv   straightens predictor rollouts under WRONG action sequences
        #             (another sample's actions held constant), measured in the
        #             cost space. GD initialises actions at zero and refines, so
        #             early planner iterates are near-constant action sequences --
        #             exactly the regime this term covers.
        #   act_sens  hinge requiring the terminal latent to MOVE when the action
        #             sequence changes, normalised by the batch spread so it is
        #             scale-invariant. Guards against an action-blind rollout map,
        #             which no distributional or directional term can see.
        self.cf_curv_coeff = float(cf_curv)
        self.cf_H = int(cf_H)
        if cf_mode not in ("cos", "aggcos"):
            raise ValueError(f"cf_mode must be 'cos' or 'aggcos', got {cf_mode!r}")
        self.cf_mode = cf_mode
        self.act_sens_coeff = float(act_sens)
        self.act_sens_margin = float(act_sens_margin)
        # Memory knob: counterfactual arms run extra encoder+predictor passes,
        # so they may subsample the batch (stochastic regulariser; a half-batch
        # estimate is fine). 1.0 = full batch.
        self.cf_batch_frac = float(cf_batch_frac)
        if self.cf_batch_frac <= 0 or self.cf_batch_frac > 1:
            raise ValueError(
                f"cf_batch_frac must be in (0, 1], got {self.cf_batch_frac}"
            )
        if self.cf_curv_coeff > 0 or self.act_sens_coeff > 0:
            if self.cf_H < 2:
                raise ValueError(
                    f"cf_H must be >= 2 (curvature needs 3 frames), got {self.cf_H}"
                )
            if self.act_sens_coeff > 0 and self.act_sens_margin <= 0:
                raise ValueError(
                    f"act_sens_margin must be > 0, got {self.act_sens_margin}"
                )
            log.info(
                "Counterfactual terms enabled: cf_curv=%s (H=%s, mode=%s), "
                "act_sens=%s (margin=%s), batch_frac=%s",
                self.cf_curv_coeff, self.cf_H, self.cf_mode,
                self.act_sens_coeff, self.act_sens_margin, self.cf_batch_frac,
            )

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def visual_only(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-2, :]
        drop = self.proprio_dim + self.action_dim
        return z[..., :-drop] if drop > 0 else z

    def visual_prop(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-1, :]
        return z[..., :-self.action_dim]

    def vcreg_std_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std_x))

    def vcreg_cov_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        _, d = x.shape
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum() / d
        return cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _cos_curvature(self, v1, v2, eps=1e-6, step_thresh=1e-6):
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
        loss = 1.0 - cos
        if step_thresh > 0:
            step1 = v1.norm(dim=-1)
            step2 = v2.norm(dim=-1)
            mask = (step1 > step_thresh) & (step2 > step_thresh)
            # NaN guard: once the encoder shrinks, EVERY velocity can fall under
            # step_thresh, and loss[mask].mean() on an empty tensor is NaN, which
            # then poisons the whole objective. That regime is reachable exactly
            # when the encoder is trainable, so return a hard 0 instead: a
            # motionless latent trajectory has no curvature to penalise. Note
            # that this is *not* a collapse barrier -- cosine curvature is
            # scale-invariant and cannot see collapse. SIGReg is what pins the
            # scale (see experiments/verify_stop_grad.py, T5).
            if not bool(mask.any()):
                return loss.sum() * 0.0        # keeps the graph, value exactly 0
            loss = loss[mask]
        return loss.mean()

    def _agg_tokens(self, x):
        """Apply the encoder aggregation head to a (b, t, p, d) tensor."""
        b, t, p, d = x.shape
        return self.encoder.agg(x.reshape(b * t, p, d)).reshape(b, t, -1)

    def total_curvature(self, features, mode="cos"):
        if features.shape[1] < 3:
            raise ValueError(f"Features must have at least 3 frames for curvature calculation, got {features.shape[1]}")

        if mode == "aggcos":
            if not hasattr(self.encoder, "agg"):
                raise ValueError("curvature mode 'aggcos' requires encoder.agg().")
            if self.curv_on == "velocity":
                # App. B.6 equation: aggregate the velocities, C_t = cos(h(v_t), h(v_{t+1}))
                vel = features[:, 1:] - features[:, :-1]        # (b, t-1, p, d)
                hv = self._agg_tokens(vel)                      # (b, t-1, d_h)
                v1, v2 = hv[:, :-1], hv[:, 1:]
            else:
                # default ("features"): aggregate, then difference
                z = self._agg_tokens(features)
                v1 = z[:, 1:-1] - z[:, :-2]
                v2 = z[:, 2:] - z[:, 1:-1]
        elif mode == "cos":
            v1 = features[:, 1:-1] - features[:, :-2]
            v2 = features[:, 2:] - features[:, 1:-1]
        else:
            raise ValueError(f"Unknown curvature mode '{mode}'. Use 'cos' or 'aggcos'.")

        return self._cos_curvature(v1, v2)

    def proprio_grounding_loss(self, z, proprio):
        """MSE of a linear read-out of the visual tokens against the proprio obs.

        Args:
            z: (b, t, p, d_total) the full concatenated latent.
            proprio: (b, t, d_proprio) the raw (normalised) proprio observation,
                i.e. an existing model input rather than a new label.
        """
        feats = self.visual_only(z)                    # (b, t, p, d)
        b, t = feats.shape[0], feats.shape[1]
        flat = feats.reshape(b, t, -1)
        if flat.shape[-1] != self.ground_head.in_features:
            raise ValueError(
                f"grounding head expects {self.ground_head.in_features} visual "
                f"features ({self._num_patches} tokens x {self.encoder.emb_dim} "
                f"channels) but the latent has {flat.shape[-1]}. The token count "
                "is derived as (image_size // 16) ** 2; an encoder with a "
                "different token grid needs that made explicit."
            )
        pred = self.ground_head(flat)
        target = proprio[:, : pred.shape[1]].to(pred.dtype)
        target = target.index_select(-1, self.ground_dims.to(target.device))
        return F.mse_loss(pred, target)

    def _counterfactual_terms(self, obs, act):
        """Curvature and action-sensitivity on rollouts under WRONG actions.

        Training batches only hold num_hist + num_pred frames, so the future
        action sequence for each arm is built by HOLDING another sample's
        future actions for cf_H steps (wrapping when the batch window is
        shorter). Each sample's own context actions are kept: they are part of
        the initial encode, so every arm starts from the SAME initial latent
        and the only difference between arms is the future action sequence --
        the clean counterfactual, exactly as experiments/rollout_drift.py does
        it offline.

        Returns a dict with whichever of cf_curv_loss / act_sens_loss is
        enabled. The initial latent is encoded ONCE under no_grad and held
        fixed -- the counterfactual varies only the actions -- so gradients
        reach the predictor and the action encoder (the rollout map), i.e.
        exactly the first-order Hessian factor the planner needs.
        """
        b, t_act = act.shape[:2]
        avail = t_act - self.num_hist              # future action frames in batch
        # Subsample the batch for the arms (memory knob; the terms are
        # stochastic regularisers, a partial-batch estimate is fine).
        m = b if self.cf_batch_frac >= 1.0 else max(3, int(b * self.cf_batch_frac))
        sel = torch.randperm(b, device=act.device)[:m]
        obs0 = {k: v[sel, : self.num_hist] for k, v in obs.items()}
        act_s = act[sel]
        arange_m = torch.arange(m, device=act.device)
        idx = torch.arange(self.cf_H, device=act.device) % avail

        # Fixed initial latent: encode once, detached. The arms then run
        # predictor-only rollouts (_rollout_from_z) -- no encoder memory at
        # all in the arms, which is what keeps the slice inside its budget.
        with torch.no_grad():
            z0 = self.encode(obs0, act_s[:, : self.num_hist])

        perms, rolls = [], []
        while len(rolls) < 2:
            p = torch.randperm(m, device=act.device)
            if bool((p == arange_m).all()):
                continue
            if perms and bool((p == perms[0]).all()):
                continue
            fut = act_s[p][:, self.num_hist:]      # (m, avail, d), someone else's
            perms.append(p)
            # Activation-checkpoint the predictor loop; recompute on backward.
            rolls.append(
                torch_checkpoint(
                    self._rollout_from_z, z0, fut[:, idx], use_reentrant=False,
                )
            )   # (m, T, p, d)

        out = {}
        if self.cf_curv_coeff > 0:
            out["cf_curv_loss"] = self.total_curvature(rolls[0], mode=self.cf_mode)
        if self.act_sens_coeff > 0:
            z1, z2 = rolls[0][:, -1], rolls[1][:, -1]
            move = (z1 - z2).pow(2).flatten(1).mean(1)          # (m,) per sample
            q = torch.randperm(m, device=act.device)
            spread = (z1[q] - z1).pow(2).flatten(1).mean(1)     # batch spread
            ratio = move.mean() / (spread.mean().detach() + 1e-8)
            out["act_sens_loss"] = F.relu(self.act_sens_margin - ratio)
        return out

    def counterfactual_loss(self, obs, act):
        """Scaled counterfactual objective as a SEPARATE loss for train.py.

        Returns (loss, components): `loss` is a scalar tensor (or None when
        the terms are disabled or the batch is too small to shuffle), and
        `components` carries the raw/scaled terms for telemetry. train.py
        must backward this AFTER the main optimizer step, once the main graph
        is freed, so the two backward peaks never stack on the GPU.

        Gradients reach only the predictor and the action encoder (the
        initial latent z0 is encoded detached), so stepping just those two
        optimizers on this loss is exact.
        """
        if self.cf_curv_coeff <= 0 and self.act_sens_coeff <= 0:
            return None, {}
        if act.shape[0] < 3 or act.shape[1] <= self.num_hist:
            return None, {}
        cf = self._counterfactual_terms(obs, act)
        loss = None
        comp = {}
        if "cf_curv_loss" in cf:
            term = cf["cf_curv_loss"] * self.cf_curv_coeff
            loss = term if loss is None else loss + term
            comp["cf_curv_loss"] = cf["cf_curv_loss"]
            comp["cf_curv_loss_scaled"] = term
        if "act_sens_loss" in cf:
            term = cf["act_sens_loss"] * self.act_sens_coeff
            loss = term if loss is None else loss + term
            comp["act_sens_loss"] = cf["act_sens_loss"]
            comp["act_sens_loss_scaled"] = term
        return loss, comp

    def sigreg_loss(self, feats):
        """SIGReg on the visual latents.

        Args:
            feats: (b, t, p, d) visual latents (proprio/action dims removed).
        """
        from models.sigreg import to_time_major

        if self.sigreg_apply_to == "agg":
            z = self._agg_tokens(feats)          # (b, t, d_h) global trajectory repr
        else:
            z = feats                            # (b, t, p, d), tokens -> batch axis
        return self.sigreg(to_time_major(z))

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        decoder_enabled = self.decoder is not None and self.train_decoder
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if decoder_enabled:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            z_tgt_for_loss = z_tgt.detach() if self.stop_grad else z_tgt
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt_for_loss[:, :, :-2, :])
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt_for_loss[:, :, -2, :])
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt_for_loss[:, :, :-1, :])
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt_for_loss[:, :, :, :-(self.proprio_dim + self.action_dim)]
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt_for_loss[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim]
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt_for_loss[:, :, :, :-self.action_dim]
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss

            if self.vcreg:
                z_vic_in = self.visual_prop(z)
                z_std_loss = self.vcreg_std_loss(z_vic_in)
                z_cov_loss = self.vcreg_cov_loss(z_vic_in)
                z_reg_loss = z_std_loss * self.std_coeff + z_cov_loss * self.cov_coeff
                loss_components["z_vicreg_std_loss"] = z_std_loss
                loss_components["z_vicreg_cov_loss"] = z_cov_loss
                loss_components["z_vcreg_loss_scaled"] = z_reg_loss
                loss = loss + z_reg_loss

            # L = L_pred + lambda_SIG * SIGReg(Z) + lambda_curv * L_curv
            # SIGReg pins the distribution (and hence the scale); the curvature
            # term pins the direction. They act on orthogonal degrees of freedom,
            # which is why neither alone is sufficient for end-to-end training.
            if self.sigreg_enabled:
                feats = self.visual_only(z)
                sig = self.sigreg_loss(feats)
                loss = loss + sig * self.sigreg_coeff
                loss_components["sigreg_loss"] = sig
                loss_components["sigreg_loss_scaled"] = sig * self.sigreg_coeff

            if self.straighten and self.straighten_scale > 0:
                feats = self.visual_only(z)
                curvature_loss = self.total_curvature(feats, mode=self.curvature_mode)
                loss = loss + curvature_loss * self.straighten_scale
                loss_components["curvature_loss_used_for_training"] = curvature_loss

            # Require the VISUAL tokens alone to linearly reproduce the proprio
            # observation, so the encoder cannot offload the agent's position onto
            # z's proprio channels and stop representing it. See __init__.
            if self.ground_head is not None:
                ground_loss = self.proprio_grounding_loss(z, obs["proprio"])
                loss = loss + ground_loss * self.ground_coeff
                loss_components["ground_proprio_loss"] = ground_loss
                loss_components["ground_proprio_loss_scaled"] = (
                    ground_loss * self.ground_coeff
                )

            # Counterfactual geometry (see __init__) lives in
            # counterfactual_loss(), which train.py applies as a SEPARATE
            # forward/backward AFTER the main optimizer step. Adding it to the
            # main loss instead would stack the arms' backward recompute on top
            # of the live main graph and overflow the GPU at e2e batch sizes.
        else:
            visual_pred = None
            z_pred = None

        if decoder_enabled:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z

    def _rollout_from_z(self, z0, action):
        """Predictor-only rollout from a FIXED initial latent (no encoder).

        Mirrors rollout()'s predict loop minus encode(): z0 already carries
        the context frames and their action tokens; `action` supplies only
        the future action frames. Used by the counterfactual terms, which
        hold the initial latent constant and vary only the actions -- so the
        encoder never runs (or backprops) for the arms.

        input:  z0: (b, n, num_patches, emb_dim) initial latent (detached)
                action: (b, t, action_dim) future actions
        output: visual latents (b, t+n+1, num_patches, emb_dim)
        """
        z = z0
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z = torch.cat([z, z_pred[:, -1:, ...]], dim=1)
        z_obses, _ = self.separate_emb(z)
        return z_obses["visual"]