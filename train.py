import os
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from models.diagnostics import latent_diagnostics
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from iteration_budget import IterationBudget
from training_log import TrainingLogger
import custom_resolvers  # noqa: F401  # Registers OmegaConf resolvers at import time.

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("checkpoints/")[-1]
        model_name += f"_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        mixed_precision = self.cfg.training.get("mixed_precision", "no")
        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=mixed_precision,
        )
        log.info(f"Accelerate mixed precision: {mixed_precision}")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        # Total optimizer steps taken across all epochs. Checkpointed, so the
        # training.max_iterations budget survives a resume.
        self.global_iter = 0
        self._stop_requested = False
        self.decoder_start_epoch = int(self.cfg.training.get("decoder_start_epoch", 1))
        if self.decoder_start_epoch < 1:
            log.warning(
                f"decoder_start_epoch={self.decoder_start_epoch} is invalid; clamping to 1"
            )
            self.decoder_start_epoch = 1
        log.info(f"Decoder training will start at epoch {self.decoder_start_epoch}")

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project=f"temporal_straightening_{self.cfg.env.name}",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project=f"temporal_straightening_{self.cfg.env.name}",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
                pin_memory=True,
                persistent_workers=True,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        # len() of the *prepared* loader is this process's batches per epoch, and
        # there is exactly one optimizer step per batch, so it is the step unit.
        self.budget = IterationBudget(
            iters_per_epoch=len(self.dataloaders["train"]),
            epochs=self.total_epochs,
            max_iterations=self.cfg.training.get("max_iterations", None),
        )
        log.info(self.budget.describe())

        # Bounded-memory telemetry: a replayable traceback of the run. Memory is
        # O(#metrics), not O(#steps); one JSON record per telemetry_every steps.
        tel_every = int(self.cfg.training.get("telemetry_every", 200))
        self.telemetry = TrainingLogger(
            path=os.path.join(
                self.cfg.saved_folder, "telemetry",
                f"train_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
            ),
            run_name=model_name,
            config={
                "env": self.cfg.env.name,
                "encoder": self.cfg.encoder.get("_target_", "?"),
                "straighten": self.cfg.training.get("straighten", False),
                "curv_on": self.cfg.training.get("curv_on", "features"),
                "sigreg": self.cfg.training.get("sigreg", False),
                "sigreg_coeff": self.cfg.training.get("sigreg_coeff", 0.0),
                "stop_grad": self.cfg.training.get("stop_grad", True),
                "freeze_backbone": self.cfg.training.get("freeze_backbone", True),
                "encoder_lr": self.cfg.training.encoder_lr,
                "backbone_lr": self.cfg.training.get("backbone_lr", None),
                "predictor_lr": self.cfg.training.predictor_lr,
                "batch_size": self.cfg.training.batch_size,
                "epochs": self.total_epochs,
                "max_iterations": self.cfg.training.get("max_iterations", None),
                "iters_per_epoch": self.budget.iters_per_epoch,
                "seed": self.cfg.training.seed,
            },
            log_every=tel_every,
            enabled=bool(self.cfg.training.get("telemetry", True))
                     and self.accelerator.is_main_process,
        )
        if self.telemetry.enabled:
            log.info("Telemetry -> %s (every %s steps)", self.telemetry.path, tel_every)

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
            "global_iter",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()

    def _configure_encoder_trainability(self):
        # training.freeze_backbone controls whether the pretrained visual trunk
        # (DINOv2) is trainable. Default True == the original behaviour, where
        # the trunk was frozen unconditionally regardless of model.train_encoder.
        # Set it False for the end-to-end variant; note that the frozen trunk is
        # the only thing currently suppressing collapse (see
        # experiments/verify_stop_grad.py T3/T4), so unfreezing without SIGReg
        # will collapse the representation.
        freeze_backbone = bool(self.cfg.training.get("freeze_backbone", True))
        base_model = getattr(self.encoder, "base_model", None)
        if base_model is not None:
            for param in base_model.parameters():
                param.requires_grad = not freeze_backbone
            log.info(
                "Encoder base_model is %s (training.freeze_backbone=%s).",
                "frozen" if freeze_backbone else "TRAINABLE (end-to-end)",
                freeze_backbone,
            )
            if not freeze_backbone and not self.cfg.training.get("sigreg", False):
                log.warning(
                    "freeze_backbone=False with SIGReg off: nothing in the "
                    "objective bounds the latent scale, so the encoder is free "
                    "to collapse (probe R^2 -> 0). This is only valid as a "
                    "deliberate negative control."
                )

        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            log.info("Encoder is fully frozen (train_encoder=False).")
            return

        # train_encoder=True: keep non-backbone encoder modules trainable.
        for name, param in self.encoder.named_parameters():
            if not name.startswith("base_model."):
                param.requires_grad = True
        log.info(
            "Encoder: base_model %s; non-backbone encoder modules are trainable.",
            "frozen" if freeze_backbone else "trainable",
        )

        # Surface the silent no-op instead of leaving it invisible: with
        # encoder=dino there is no projector and no agg head, so the encoder has
        # zero trainable params and encoder_optimizer.step() does nothing every
        # iteration. That is *correct* for the paper's frozen DINOv2 baseline
        # row, so warn rather than raise -- raising would break that cell.
        n_trainable = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        log.info("Encoder trainable params: %s", f"{n_trainable:,}")
        if n_trainable == 0:
            log.warning(
                "model.train_encoder=True but the encoder has 0 trainable "
                "parameters: encoder_optimizer.step() is a no-op and "
                "training.encoder_lr has no effect. Expected for the frozen "
                "DINOv2 baseline (encoder=dino); otherwise use an encoder with a "
                "trainable projector/head, or set training.freeze_backbone=False."
            )

    def _log_trainable_params(self, module, module_name):
        if not self.accelerator.is_main_process:
            return
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        log.info(f"[{module_name}] trainable params: {trainable} / {total}")
        for name, param in module.named_parameters():
            if param.requires_grad:
                log.info(f"[{module_name}] trainable: {name} shape={tuple(param.shape)}")

    def decoder_training_active(self):
        return (
            self.cfg.has_decoder
            and self.train_decoder
            and self.epoch >= self.decoder_start_epoch
        )

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                v = self.__dict__.get(k, None)
                if k.endswith("_optimizer") and v is not None:
                    ckpt[k] = v.state_dict()
                elif hasattr(v, "module"):
                    ckpt[k] = self.accelerator.unwrap_model(v)
                else:
                    ckpt[k] = v
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        # weights_only=False: checkpoints store full nn.Module objects, not just
        # state-dicts. Required on torch>=2.6 (where weights_only defaults to True).
        ckpt = torch.load(filename, weights_only=False)
        self._loaded_optim_state = {}
        for k, v in ckpt.items():
            if k.endswith("_optimizer") and isinstance(v, dict):
                self._loaded_optim_state[k] = v
            else:
                self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        # Resume priority:
        #   1. An explicit checkpoint via training.resume_from=/abs/path/model_X.pth
        #      (useful to continue offline from a specific saved epoch).
        #   2. Otherwise auto-resume from model_latest.pth in this run's folder.
        # Because the encoder (incl. the DINOv2 backbone weights) is stored in the
        # checkpoint, resuming does NOT re-download anything and works with no internet.
        resume_from = self.cfg.training.get("resume_from", None)
        if resume_from:
            model_ckpt = Path(resume_from).expanduser()
            if not model_ckpt.exists():
                raise FileNotFoundError(
                    f"training.resume_from='{model_ckpt}' does not exist. "
                    "Point it at a valid model_<epoch>.pth checkpoint."
                )
        else:
            model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"

        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")
            # Checkpoints written before training.max_iterations existed carry no
            # global_iter; infer it from the completed epochs so the cap is not
            # silently restarted from zero.
            if self.global_iter == 0 and self.epoch > 0:
                self.global_iter = self.budget.resume_estimate(self.epoch)
                log.warning(
                    "Checkpoint has no global_iter; estimating %s from %s completed "
                    "epoch(s) x %s iters/epoch.",
                    self.global_iter,
                    self.epoch,
                    self.budget.iters_per_epoch,
                )
            log.info(
                "Resumed at global_iter=%s (budget remaining: %s)",
                self.global_iter,
                self.budget.remaining(self.global_iter),
            )
        else:
            log.info("No checkpoint found; starting training from scratch.")

        # initialize encoder
        if self.encoder is None:
            encoder_kwargs = {}
            if (
                hasattr(self.cfg.encoder, "projector_config")
                and self.cfg.encoder.projector_config is not None
            ):
                encoder_kwargs["projector_config"] = hydra.utils.instantiate(
                    self.cfg.encoder.projector_config
                )
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
                **encoder_kwargs,
            )
        self._configure_encoder_trainability()

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path, weights_only=False)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path, weights_only=False)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    decoder_kwargs = {
                        "emb_dim": self.encoder.emb_dim,  
                    }
                    if (
                        hasattr(self.cfg.encoder, "projector_config")
                        and self.cfg.encoder.projector_config is not None
                        and "conv_layers" in self.cfg.encoder.projector_config
                    ):
                        decoder_kwargs["projector_cfg"] = self.cfg.encoder.projector_config
                        log.info(f"Passing projector_cfg to decoder")
                    decoder_kwargs["_recursive_"] = False
                    self.decoder = hydra.utils.instantiate(self.cfg.decoder, **decoder_kwargs)
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            straighten=self.cfg.training.get("straighten", False),
            curv_on=self.cfg.training.get("curv_on", "features"),
            stop_grad=self.cfg.training.get("stop_grad", True),
            vcreg=self.cfg.training.get("vcreg", False),
            vcreg_std_coeff=self.cfg.training.get("vcreg_std_coeff", 0),
            vcreg_cov_coeff=self.cfg.training.get("vcreg_cov_coeff", 0),
            vcreg_apply_to=self.cfg.training.get("vcreg_apply_to", "enc"),
            ground_proprio=self.cfg.training.get("ground_proprio", 0.0),
            ground_proprio_dims=self.cfg.training.get("ground_proprio_dims", None),
            sigreg=self.cfg.training.get("sigreg", False),
            sigreg_coeff=self.cfg.training.get("sigreg_coeff", 0.0),
            sigreg_num_proj=self.cfg.training.get("sigreg_num_proj", 1024),
            sigreg_knots=self.cfg.training.get("sigreg_knots", 17),
            sigreg_apply_to=self.cfg.training.get("sigreg_apply_to", "agg"),
            cf_curv=self.cfg.training.get("cf_curv", 0.0),
            cf_H=self.cfg.training.get("cf_H", 4),
            cf_mode=self.cfg.training.get("cf_mode", "cos"),
            act_sens=self.cfg.training.get("act_sens", 0.0),
            act_sens_margin=self.cfg.training.get("act_sens_margin", 0.1),
            cf_batch_frac=self.cfg.training.get("cf_batch_frac", 0.5),
        )
        self._log_trainable_params(self.model, "model")

    def _encoder_param_groups(self):
        """Split the encoder into trunk vs head param groups.

        The paper's encoder lr (1e-6 / 1e-5) was tuned for a small projector on
        top of a FROZEN trunk. Fine-tuning a ViT trunk at that rate is not a
        sensible default, so training.backbone_lr gives the trunk its own rate.
        When backbone_lr is null (or the trunk is frozen) this returns a single
        group and reproduces the original single-group Adam exactly.
        """
        enc_lr = self.cfg.training.encoder_lr
        backbone_lr = self.cfg.training.get("backbone_lr", None)
        trunk, heads = [], []
        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            (trunk if name.startswith("base_model.") else heads).append(param)

        if backbone_lr is None or not trunk:
            return [{"params": list(self.encoder.parameters()), "lr": enc_lr}]

        log.info(
            "Encoder param groups: trunk %s params @ lr=%s | heads %s params @ lr=%s",
            f"{sum(p.numel() for p in trunk):,}",
            backbone_lr,
            f"{sum(p.numel() for p in heads):,}",
            enc_lr,
        )
        groups = [{"params": trunk, "lr": float(backbone_lr)}]
        if heads:
            groups.append({"params": heads, "lr": enc_lr})
        return groups

    def _register_ground_head(self, groups):
        """Place the proprio-grounding head on the device and into an optimizer.

        VWorldModel is built AFTER accelerator.prepare() and is never itself
        prepared, so a module created in its __init__ keeps CPU parameters that
        no optimizer owns. SIGReg only has buffers and relocates them per call;
        the grounding head has parameters, so it needs both fixes or the run
        dies on the first forward with a device mismatch and the head never
        learns.

        It gets action_encoder_lr rather than encoder_lr: it is a read-out probe,
        and a probe that trains slower than the representation it reads supplies
        a stale gradient to the encoder. Appended LAST so the trunk/head group
        indices that _probe_module_health relies on keep their meaning.
        """
        head = getattr(self.model, "ground_head", None)
        self._n_encoder_groups = len(groups)
        if head is None:
            return groups
        head.to(self.accelerator.device)
        lr = self.cfg.training.action_encoder_lr
        log.info(
            "Grounding head: %s params @ lr=%s, on %s (not checkpointed: it is a "
            "training-only read-out and relearns in a few hundred steps, so a "
            "resume shows a brief transient in ground_proprio_loss)",
            f"{sum(p.numel() for p in head.parameters()):,}", lr,
            self.accelerator.device,
        )
        return list(groups) + [{"params": list(head.parameters()), "lr": lr}]

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self._register_ground_head(self._encoder_param_groups()),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if getattr(self, "_loaded_optim_state", None) and "encoder_optimizer" in self._loaded_optim_state:
            try:
                self.encoder_optimizer.load_state_dict(self._loaded_optim_state["encoder_optimizer"])
                log.info(f"Loaded encoder optimizer state from checkpoint.")
            except Exception as e:
                log.warning(f"Failed to load encoder optimizer state: {e}")
        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )
            if getattr(self, "_loaded_optim_state", None) and "predictor_optimizer" in self._loaded_optim_state:
                try:
                    self.predictor_optimizer.load_state_dict(self._loaded_optim_state["predictor_optimizer"])
                    log.info(f"Loaded predictor optimizer state from checkpoint.")
                except Exception as e:
                    log.warning(f"Failed to load predictor optimizer state: {e}")

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )
            if getattr(self, "_loaded_optim_state", None) and "action_encoder_optimizer" in self._loaded_optim_state:
                try:
                    self.action_encoder_optimizer.load_state_dict(self._loaded_optim_state["action_encoder_optimizer"])
                    log.info(f"Loaded action/proprio optimizer state from checkpoint.")
                except Exception as e:
                    log.warning(f"Failed to load action/proprio optimizer state: {e}")

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)
            if getattr(self, "_loaded_optim_state", None) and "decoder_optimizer" in self._loaded_optim_state:
                try:
                    self.decoder_optimizer.load_state_dict(self._loaded_optim_state["decoder_optimizer"])
                    log.info(f"Loaded decoder optimizer state from checkpoint.")
                except Exception as e:
                    log.warning(f"Failed to load decoder optimizer state: {e}")

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        if self.budget.enabled and not self.budget.is_reachable:
            log.warning(
                "training.max_iterations=%s is unreachable with training.epochs=%s "
                "(%s iters/epoch -> %s steps max). Raise epochs to >= %s to spend the "
                "full budget.",
                self.budget.max_iterations,
                self.total_epochs,
                self.budget.iters_per_epoch,
                self.budget.epoch_bounded_total,
                self.budget.epochs_needed,
            )

        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            if self.accelerator.is_main_process:
                decoder_active = self.decoder_training_active()
                log.info(
                    "Epoch %s decoder_active=%s (train_decoder=%s, decoder_start_epoch=%s)",
                    self.epoch,
                    decoder_active,
                    self.train_decoder,
                    self.decoder_start_epoch,
                )
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            # `or self._stop_requested`: always checkpoint the final model when the
            # iteration budget ends the run mid-epoch, even on a non-save epoch.
            if (
                self.epoch % self.cfg.training.save_every_x_epoch == 0
                or self._stop_requested
            ):
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only: launch planning jobs on the saved ckpt
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

            self.telemetry.event(self.global_iter, "epoch_end",
                                 f"epoch {self.epoch} finished")
            if self._stop_requested:
                log.info(
                    "Run finished on the iteration budget: %s optimizer steps "
                    "(training.max_iterations=%s), stopped during epoch %s of %s.",
                    self.global_iter,
                    self.budget.max_iterations,
                    self.epoch,
                    self.total_epochs,
                )
                break

        self.telemetry.close(
            step=self.global_iter,
            status="budget_reached" if self._stop_requested else "epochs_completed",
            epochs_run=self.epoch,
            memory=self.telemetry.memory_report(),
        )
        if self.telemetry.enabled:
            log.info("Telemetry written: %s", self.telemetry.path)
            log.info("Digest it with: python summarize_training_log.py %s",
                     self.telemetry.path)

    def _telemetry_groups(self):
        """The module groups worth tracking separately in the telemetry.

        Split so the visual trunk is visible on its own: it is the part that was
        never trained before, and the part most likely to be mis-tuned now.
        """
        enc = self.encoder
        groups = {}
        base = getattr(enc, "base_model", None)
        if base is not None:
            groups["encoder.trunk"] = base
        for attr in ("projector", "agg_mlp"):
            mod = getattr(enc, attr, None)
            if mod is not None:
                groups[f"encoder.{attr}"] = mod
        if self.cfg.has_predictor and self.predictor is not None:
            groups["predictor"] = self.predictor
        if self.action_encoder is not None:
            groups["action_encoder"] = self.action_encoder
        if self.proprio_encoder is not None:
            groups["proprio_encoder"] = self.proprio_encoder
        return groups

    def _probe_module_health(self, step):
        """Gradient/weight norms, their ratio, and measured weight movement."""
        try:
            lrs = {}
            enc_groups = self.encoder_optimizer.param_groups
            # _encoder_param_groups puts the trunk first when backbone_lr is set.
            # Count the ENCODER groups only: the grounding head is appended after
            # them, and counting it would mislabel a single-group encoder's lr.
            if getattr(self, "_n_encoder_groups", len(enc_groups)) > 1:
                lrs["encoder.trunk"] = enc_groups[0]["lr"]
                lrs["encoder.projector"] = enc_groups[1]["lr"]
                lrs["encoder.agg_mlp"] = enc_groups[1]["lr"]
            else:
                for name in ("encoder.trunk", "encoder.projector", "encoder.agg_mlp"):
                    lrs[name] = enc_groups[0]["lr"]
            if self.cfg.has_predictor:
                lrs["predictor"] = self.predictor_optimizer.param_groups[0]["lr"]
            self.telemetry.probe_modules(step, self._telemetry_groups(), lr_by_group=lrs)
            if torch.cuda.is_available():
                self.telemetry.record(
                    step,
                    **{"sys/gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                       "sys/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9},
                )
        except Exception as e:  # pragma: no cover - telemetry never breaks a run
            log.warning("Module-health probe failed (skipping): %s", e)

    @torch.no_grad()
    def _latent_diagnostics(self, obs, act, state=None):
        """Collapse / straightness metrics on one batch, as {key: [value]} logs.

        Never allowed to break training: any failure is logged and skipped.
        """
        try:
            z = self.model.encode(obs, act)
            z_visual = self.model.visual_only(z)
            logs = latent_diagnostics(z_visual, state=state, prefix="val_")
            # the aggregated trajectory representation is what both regularizers
            # act on, so report its geometry too when the head exists
            if hasattr(self.model.encoder, "agg"):
                z_agg = self.model._agg_tokens(z_visual)
                logs.update(latent_diagnostics(z_agg, state=state, prefix="val_agg_"))
            return {k: [v] for k, v in logs.items()}
        except Exception as e:  # pragma: no cover - diagnostics must never crash a run
            log.warning("Latent diagnostics failed (skipping): %s", e)
            return {}

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            decoder_active = self.decoder_training_active()
            self.model.train_decoder = decoder_active
            self.model.train()
            if self.cfg.has_decoder:
                self.decoder.train(decoder_active)
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            self.encoder_optimizer.zero_grad()
            if decoder_active:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            # Probe gradients AFTER backward and BEFORE step, on the telemetry
            # cadence only, so the cost is amortised to ~nothing.
            if self.telemetry.enabled and self.global_iter % self.telemetry.log_every == 0:
                self._probe_module_health(self.global_iter)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if decoder_active:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            self.global_iter += 1

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            if decoder_active and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            # Telemetry: every loss term separately, every step, into O(1)-memory
            # accumulators. A falling total says nothing about which term moved.
            self.telemetry.record(
                self.global_iter,
                **{f"loss/{k}": v for k, v in loss_components.items()},
            )
            self.telemetry.record(self.global_iter, **{"progress/epoch": self.epoch})

            # Collapse diagnostics DURING training, not just once per epoch.
            # val() runs them per epoch, which over a 123,858-step run is two or
            # three data points -- far too coarse to catch collapse while there
            # is still time to react. Cost is one extra encoder forward every
            # diag_every steps.
            diag_every = int(self.cfg.training.get("diag_every", 500))
            if diag_every > 0 and self.global_iter % diag_every == 0:
                self.model.eval()
                diag = self._latent_diagnostics(obs, act, state)
                self.model.train()
                self.telemetry.probe_latents(
                    self.global_iter,
                    {"latent/" + k.replace("val_", "", 1): v[0]
                     for k, v in diag.items()},
                )

            self.telemetry.maybe_flush(self.global_iter)

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

            # Auxiliary counterfactual objective (cf_curv / act_sens), applied
            # as its own forward/backward at the END of the iteration, after
            # the main graph is explicitly freed below -- so the two backward
            # peaks never stack on the GPU (the recurring OOM). Its gradients
            # reach only the predictor + action encoder (the initial latent is
            # encoded detached), so stepping just those two on it is exact.
            del z_out, visual_out, visual_reconstructed
            cf_comp = {}
            if self.cfg.has_predictor:
                cf_loss, cf_comp = self.model.counterfactual_loss(obs, act)
                if cf_loss is not None:
                    self.predictor_optimizer.zero_grad()
                    self.action_encoder_optimizer.zero_grad()
                    self.accelerator.backward(cf_loss)
                    if self.model.train_predictor:
                        self.predictor_optimizer.step()
                        self.action_encoder_optimizer.step()
                    self.telemetry.record(
                        self.global_iter,
                        **{f"loss/{k}": (v.item() if torch.is_tensor(v) else float(v))
                           for k, v in cf_comp.items()},
                    )

            if (
                self.cfg.training.save_every_x_iterations > 0
                and i % self.cfg.training.save_every_x_iterations == 0
            ):
                self.logs_flash_iter(iteration=i)
                self.save_ckpt()

            # Hard iteration budget: stop the moment it is reached, mid-epoch.
            # run() still runs val(), flushes the epoch log and saves a ckpt.
            if self.budget.reached(self.global_iter):
                self._stop_requested = True
                log.info(
                    "Iteration budget reached: global_iter=%s / max_iterations=%s "
                    "(epoch %s, batch %s of %s). Stopping training.",
                    self.global_iter,
                    self.budget.max_iterations,
                    self.epoch,
                    i,
                    self.budget.iters_per_epoch,
                )
                break

    @torch.no_grad()
    def val(self):
        decoder_active = self.decoder_training_active()
        self.model.train_decoder = decoder_active
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            train_rollout_logs = self.openloop_rollout(
                self.train_traj_dset, mode="train"
            )
            train_rollout_logs = {
                f"train_{k}": [v] for k, v in train_rollout_logs.items()
            }
            self.logs_update(train_rollout_logs)
            val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
            val_rollout_logs = {
                f"val_{k}": [v] for k, v in val_rollout_logs.items()
            }
            self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            # Latent health, once per epoch on the first val batch. A falling
            # prediction loss is not evidence of learning -- the cheapest way to
            # minimise it is to stop responding to the input. These metrics are
            # what separates the two cases.
            if plot and self.cfg.training.get("log_diagnostics", True):
                diag = self._latent_diagnostics(obs, act, state)
                self.logs_update(diag)
                # same numbers into the telemetry, renamed to latent/* and
                # threshold-checked so a collapse raises a dated event
                self.telemetry.probe_latents(
                    self.global_iter,
                    {"latent/" + k.replace("val_", "", 1): v[0]
                     for k, v in diag.items()},
                )
                self.telemetry.flush(self.global_iter)

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if decoder_active and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    def logs_flash_iter(self, iteration):
        iter_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            iter_log[key] = to_log
        iter_log["iter"] = iteration
        iter_log["epoch"] = self.epoch

        if self.accelerator.is_main_process:
            self.wandb_run.log(iter_log)

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
