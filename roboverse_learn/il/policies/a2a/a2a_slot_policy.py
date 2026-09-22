"""
Action-to-Action Flow Matching Policy with optional slot start/condition branches.

Supports two visual backbones:
- ResNet (default): slot attention over ResNet temporal tokens
- DINOv3 (dino_enable=True): pre-trained DINOv3 + frozen slot attention
"""

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from roboverse_learn.il.policies.a2a.a2a_policy import A2AImagePolicy
from roboverse_learn.il.utils.pytorch_util import dict_apply

try:
    from pytorch3d import transforms as _pt3d_transforms
except ImportError:
    _pt3d_transforms = None


class LowFreqPosEnc(nn.Module):
    """
    Low-frequency Fourier encoding for 2D normalized coordinates.
    Input p must be in [-1, 1], shape [B, 2].
    """

    def __init__(self, L: int = 4, out_dim: int = 256, include_raw_xy: bool = True):
        super().__init__()
        self.include_raw_xy = bool(include_raw_xy)
        L = int(L)
        if L <= 0:
            raise ValueError(f"LowFreqPosEnc expects L > 0, got {L}")
        freqs = (2.0 ** torch.arange(L).float()) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

        in_dim = (2 if self.include_raw_xy else 0) + 4 * L
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        if p.shape[-1] != 2:
            raise ValueError(f"LowFreqPosEnc expects last dim=2, got shape={tuple(p.shape)}")
        x = p.unsqueeze(-1) * self.freqs.view(1, 1, -1)  # [B,2,L]
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).flatten(start_dim=-2)  # [B,4L]
        if self.include_raw_xy:
            enc = torch.cat([p, enc], dim=-1)
        return self.mlp(enc)


class A2ASlotImagePolicy(A2AImagePolicy):
    """A2A with optional slot-guided start correction and condition fusion."""

    def __init__(
        self,
        *args,
        num_slots: int = 4,
        slot_dim: int = 512,
        slot_num_heads: int = 4,
        slot_dropout: float = 0.0,
        history_noise_std: float = 0.05,
        gate_reg_weight: float = 1e-4,
        cond_obj_weight: float = 1.0,
        flow_start_mode: str = "history",
        flow_start_noise_std: float = 1.0,
        flow_start_scale: float = 1.0,
        slot_start_enable: bool = True,
        slot_start_mode: str = "residual",
        start_use_centroid: bool = True,
        centroid_n_freqs: int = 16,
        history_jitter_prob: float = 0.2,
        history_jitter_std: float = 0.5,
        start_detach_offset: bool = False,
        start_detach_for_sampling_losses: bool = True,
        start_cross_pair_prob: float = 0.0,
        start_offset_l2_weight: float = 0.05,
        start_offset_align_weight: float = 0.05,
        train_offset_scale: float = 1.0,
        train_offset_clamp: float = 0.35,
        train_offset_ratio_cap: float = 0.35,
        eval_offset_scale: float = 0.5,
        eval_offset_clamp: float = 0.25,
        eval_offset_ratio_cap: float = 0.25,
        slot_cond_enable: bool = True,
        slot_cond_mode: str = "add",
        slot_semantic_enable: bool = True,
        slot_geometry_enable: bool = True,
        object_condition_source: str = "slot",
        slot_adapter_enable: bool = False,
        slot_adapter_hidden_dim: int = 512,
        slot_adapter_dropout: float = 0.0,
        slot_cond_image_dropout: float = 0.0,
        slot_cond_gate_enable: bool = True,
        cond_split_ln_enable: bool = True,
        cond_fuse_mlp_enable: bool = True,
        cond_fuse_mlp_hidden_dim: int = 512,
        cond_fuse_mlp_dropout: float = 0.1,
        slot_selection_mode: str = "history_softmax",
        slot_obj_index: int = 0,
        slot_selection_temperature: float = 1.0,
        saliency_reg_enable: bool = False,
        saliency_reg_weight: float = 0.0,
        saliency_noise_std: float = 0.05,
        saliency_apply_prob: float = 1.0,
        saliency_num_steps: int = 1,
        slot_mask_supervise_enable: bool = False,
        slot_mask_supervise_weight: float = 0.0,
        slot_distill_ckpt: str = "",
        slot_distill_load_backbone: bool = True,
        slot_distill_load_slot: bool = True,
        slot_distill_freeze_backbone: bool = False,
        slot_distill_freeze_slot: bool = False,
        # ── DINOv3 slot encoder (replaces ResNet→slot path) ──
        dino_enable: bool = False,
        dino_model_path: str = "",
        dino_slot_ckpt_path: str = "",
        dino_num_slots: int = 2,
        dino_slot_dim: int = 256,
        dino_slot_heads: int = 4,
        dino_img_size: int = 224,
        dino_two_scale_enable: bool = False,
        dino_two_scale_crop_prob: float = 0.5,
        cond_centroid_enable: bool = True,
        cond_centroid_low_freq_enable: bool = False,
        cond_centroid_low_freq_L: int = 16,
        cond_centroid_low_freq_use_raw_xy: bool = True,
        cond_centroid_scale_init: float = 0.0,
        cond_centroid_ema_enable: bool = False,
        cond_centroid_ema_decay: float = 0.7,
        cond_centroid_conf_gate_enable: bool = False,
        cond_centroid_conf_min: float = 0.2,
        # Where-conditioning: geometry-as-controller (FiLM + residual gate).
        cond_where_enable: bool = False,
        cond_where_film_enable: bool = False,
        cond_where_stats_enable: bool = True,
        cond_where_conf_gate_enable: bool = True,
        cond_where_gate_min: float = 0.0,
        cond_where_film_max_scale: float = 0.25,
        cond_where_stats_hidden_dim: int = 128,
        dino_freeze: bool = True,
        dino_layer: int = 12,
        robot_dim: int = None,  # auto-set by il_run.sh for UR3
        delta_action_history_enable: bool = False,
        # Stage 2: freeze flow backbone, only train start correction
        **kwargs,
    ):
        super().__init__(*args, robot_dim=robot_dim, **kwargs)

        if slot_dim % slot_num_heads != 0:
            raise ValueError(
                f"slot_dim ({slot_dim}) must be divisible by slot_num_heads ({slot_num_heads})"
            )

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slot_num_heads = slot_num_heads
        self.slot_dropout = slot_dropout
        self.history_noise_std = history_noise_std
        self.gate_reg_weight = gate_reg_weight
        self.cond_obj_weight = cond_obj_weight
        self.flow_start_mode = str(flow_start_mode).lower()
        self.flow_start_noise_std = float(flow_start_noise_std)
        self.flow_start_scale = float(flow_start_scale)
        self.slot_start_enable = slot_start_enable
        self.slot_start_mode = slot_start_mode
        self.start_use_centroid = bool(start_use_centroid) if slot_start_enable else False
        self.centroid_n_freqs = int(centroid_n_freqs)
        self.history_jitter_prob = float(history_jitter_prob) if slot_start_enable else 0.0
        self.history_jitter_std = float(history_jitter_std) if slot_start_enable else 0.0
        self.start_detach_offset = bool(start_detach_offset)
        self.start_detach_for_sampling_losses = bool(start_detach_for_sampling_losses)
        self.start_cross_pair_prob = float(start_cross_pair_prob)
        self.start_offset_l2_weight = float(start_offset_l2_weight)
        self.start_offset_align_weight = float(start_offset_align_weight)
        self.train_offset_scale = float(train_offset_scale)
        self.train_offset_clamp = float(train_offset_clamp)
        self.train_offset_ratio_cap = float(train_offset_ratio_cap)
        self.eval_offset_scale = float(eval_offset_scale)
        self.eval_offset_clamp = float(eval_offset_clamp)
        self.eval_offset_ratio_cap = float(eval_offset_ratio_cap)
        self.slot_cond_enable = slot_cond_enable
        self.slot_cond_mode = slot_cond_mode
        self.slot_semantic_enable = bool(slot_semantic_enable)
        self.slot_geometry_enable = bool(slot_geometry_enable)
        self.object_condition_source = str(object_condition_source).lower()
        self.slot_adapter_enable = bool(slot_adapter_enable)
        self.slot_adapter_hidden_dim = int(slot_adapter_hidden_dim)
        self.slot_adapter_dropout = float(slot_adapter_dropout)
        self.slot_cond_image_dropout = float(slot_cond_image_dropout)
        self.slot_cond_gate_enable = bool(slot_cond_gate_enable)
        self.cond_split_ln_enable = bool(cond_split_ln_enable)
        self.cond_fuse_mlp_enable = bool(cond_fuse_mlp_enable)
        self.cond_fuse_mlp_hidden_dim = int(cond_fuse_mlp_hidden_dim)
        self.cond_fuse_mlp_dropout = float(cond_fuse_mlp_dropout)
        self.slot_selection_mode = slot_selection_mode
        self.slot_obj_index = int(slot_obj_index)
        self.slot_selection_temperature = float(slot_selection_temperature)
        self.dino_two_scale_enable = bool(dino_two_scale_enable)
        self.delta_action_history_enable = bool(delta_action_history_enable)
        self.cond_centroid_enable = bool(cond_centroid_enable)
        self.cond_centroid_low_freq_enable = bool(cond_centroid_low_freq_enable)
        self.cond_centroid_low_freq_L = int(cond_centroid_low_freq_L)
        self.cond_centroid_low_freq_use_raw_xy = bool(cond_centroid_low_freq_use_raw_xy)
        self.cond_centroid_scale_init = float(cond_centroid_scale_init)
        self.cond_centroid_ema_enable = bool(cond_centroid_ema_enable)
        self.cond_centroid_ema_decay = float(cond_centroid_ema_decay)
        self.cond_centroid_conf_gate_enable = bool(cond_centroid_conf_gate_enable)
        self.cond_centroid_conf_min = float(cond_centroid_conf_min)
        self.cond_where_enable = bool(cond_where_enable)
        self.cond_where_film_enable = bool(cond_where_film_enable)
        self.cond_where_stats_enable = bool(cond_where_stats_enable)
        self.cond_where_conf_gate_enable = bool(cond_where_conf_gate_enable)
        self.cond_where_gate_min = float(cond_where_gate_min)
        self.cond_where_film_max_scale = float(cond_where_film_max_scale)
        self.cond_where_stats_hidden_dim = int(cond_where_stats_hidden_dim)
        self._debug_step = 0
        self.output_dir = './il_outputs'  # overridden by runner
        self._last_where_gate_mean = 0.0
        if not (0.0 <= self.cond_centroid_ema_decay <= 1.0):
            raise ValueError("cond_centroid_ema_decay must be in [0,1]")
        if not (0.0 <= self.cond_centroid_conf_min <= 1.0):
            raise ValueError("cond_centroid_conf_min must be in [0,1]")
        if not (0.0 <= self.cond_where_gate_min <= 1.0):
            raise ValueError("cond_where_gate_min must be in [0,1]")
        if self.cond_where_film_max_scale <= 0:
            raise ValueError("cond_where_film_max_scale must be > 0")
        if self.cond_where_stats_hidden_dim <= 0:
            raise ValueError("cond_where_stats_hidden_dim must be > 0")
        if self.flow_start_mode not in ("history", "gaussian", "gaussian_scaled", "zero", "shuffle"):
            raise ValueError(
                f"Unsupported flow_start_mode={self.flow_start_mode}, "
                "expected one of ['history', 'gaussian', 'gaussian_scaled', 'zero', 'shuffle']"
            )
        if self.flow_start_noise_std <= 0:
            raise ValueError("flow_start_noise_std must be > 0")
        if self.flow_start_scale < 0:
            raise ValueError("flow_start_scale must be >= 0")
        if self.slot_start_mode not in ("residual", "modulation"):
            raise ValueError(
                f"Unsupported slot_start_mode={self.slot_start_mode}, expected one of ['residual', 'modulation']"
            )
        if self.slot_cond_mode not in ("add", "replace"):
            raise ValueError(
                f"Unsupported slot_cond_mode={self.slot_cond_mode}, expected one of ['add', 'replace']"
            )
        if self.object_condition_source not in ("slot", "mask_pool"):
            raise ValueError(
                f"Unsupported object_condition_source={self.object_condition_source}, "
                "expected one of ['slot', 'mask_pool']"
            )
        if self.slot_selection_mode not in ("history_softmax", "fixed_index"):
            raise ValueError(
                f"Unsupported slot_selection_mode={self.slot_selection_mode}, "
                "expected one of ['history_softmax', 'fixed_index']"
            )
        if self.slot_obj_index < 0 or self.slot_obj_index >= self.num_slots:
            raise ValueError(
                f"slot_obj_index={self.slot_obj_index} out of range for num_slots={self.num_slots}"
            )
        if self.slot_selection_temperature <= 0:
            raise ValueError("slot_selection_temperature must be > 0")
        self.saliency_reg_enable = saliency_reg_enable
        self.saliency_reg_weight = saliency_reg_weight
        self.saliency_noise_std = saliency_noise_std
        self.saliency_apply_prob = saliency_apply_prob
        self.saliency_num_steps = saliency_num_steps
        self.slot_mask_supervise_enable = slot_mask_supervise_enable
        self.slot_mask_supervise_weight = slot_mask_supervise_weight
        self.slot_distill_ckpt = slot_distill_ckpt
        self.slot_distill_load_backbone = slot_distill_load_backbone
        self.slot_distill_load_slot = slot_distill_load_slot
        self.slot_distill_freeze_backbone = slot_distill_freeze_backbone
        self.slot_distill_freeze_slot = slot_distill_freeze_slot
        self._last_loss_metrics: Dict[str, float] = {}
        shape_meta_cfg = kwargs.get("shape_meta", None)
        if shape_meta_cfg is None and len(args) > 0:
            shape_meta_cfg = args[0]
        self.proprio_dim = 0
        self.visual_dim = int(self.obs_feature_dim - self.proprio_dim)
        self.cond_parallel_enable = self.visual_dim > 0 and self.proprio_dim > 0
        print(f"[COND] cond_parallel_enable={self.cond_parallel_enable} visual_dim={self.visual_dim} proprio_dim={self.proprio_dim} obs_feature_dim={self.obs_feature_dim}")
        print(f"[COND] → {'THREE-BRANCH: vis+proprio+slot' if self.cond_parallel_enable else 'TWO-BRANCH: vis+slot (proprio in vis)'}")
        print(f"[DIMS] action_dim={self.action_dim} obs_feature_dim={self.obs_feature_dim} visual_dim={self.visual_dim} proprio_dim={self.proprio_dim}")
        if self.delta_action_history_enable:
            print("[A2A] delta_action_history_enable=True: encode history as TCP/EE deltas for delta-action policy.")
        if self.visual_dim <= 0 or self.proprio_dim <= 0:
            self.cond_split_ln_enable = False
            self.cond_fuse_mlp_enable = False

        token_dim = slot_dim
        # Bridge current obs token dim to distill slot token dim when needed.
        # Identity by default; can be replaced during checkpoint loading.
        self.slot_input_adapter: nn.Module = nn.Identity()
        self.slot_token_proj = nn.Linear(self.obs_feature_dim, slot_dim)

        # Learned slot queries attend to temporal observation tokens.
        self.slot_queries = nn.Parameter(torch.randn(num_slots, token_dim) * 0.02)
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=slot_num_heads,
            dropout=slot_dropout,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(token_dim)
        self.slot_ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        self.hist_query_proj = nn.Linear(self.latent_dim, token_dim)
        self.slot_to_latent = nn.Linear(token_dim, self.latent_dim)
        # Project mask-pooled visual prototype (from spatial feature map) to latent space.
        self.mask_proto_to_latent = nn.Linear(token_dim, self.latent_dim)

        # ── Latent-space start correction: centroid_enc ⊖ proprio_latents ──
        # centroid_enc: sin/cos encoding of slot attention centroid → 2D visual position
        # proprio_latents: MLP-encoded robot state → already jointly trained by cond fusion
        # Δ_latent = centroid_enc - proprio_latents → semantic displacement in latent space
        # NO pre-projection to 2D, NO clean_loss, NO centroid noise
        if self.slot_start_enable:
            # Align centroid encoding to proprio latent dim
            self.centroid_align = nn.Linear(4 * self.centroid_n_freqs, self.latent_dim)
            self.offset_mlp = nn.Sequential(
                nn.LayerNorm(4 * self.centroid_n_freqs + self.action_dim * self.n_obs_steps),
                nn.Linear(4 * self.centroid_n_freqs + self.action_dim * self.n_obs_steps, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 9),  # 9D joint offset (UR3:7D handled by robot_dim)
            )
            # Separate copy for start correction: same structure, isolated from Condition pathway
# Old start correction modules removed — using offset_mlp instead

        # Centroid-based condition encoding (for slot_cond and dino_two_scale).
        self._centroid_needed = (self.start_use_centroid and self.slot_start_enable) or self.dino_two_scale_enable
        if self._centroid_needed:
            self.centroid_freqs = nn.Parameter(
                torch.logspace(0, 3, self.centroid_n_freqs), requires_grad=False
            )  # [n_freqs] — log-spaced from 1 to 1000
            cent_enc_dim = 4 * self.centroid_n_freqs  # sin+cos for x, sin+cos for y
            self.centroid_to_cond = nn.Linear(cent_enc_dim, self.latent_dim)
            self.centroid_cond_scale = nn.Parameter(
                torch.tensor([self.cond_centroid_scale_init], dtype=torch.float32)
            )
            if self.cond_centroid_low_freq_enable:
                self.cond_centroid_encoder = LowFreqPosEnc(
                    L=self.cond_centroid_low_freq_L,
                    out_dim=self.latent_dim,
                    include_raw_xy=self.cond_centroid_low_freq_use_raw_xy,
                )
            else:
                self.cond_centroid_encoder = None
            if self.cond_where_enable:
                self.cond_where_stats_encoder = nn.Sequential(
                    nn.Linear(3, self.cond_where_stats_hidden_dim),
                    nn.LayerNorm(self.cond_where_stats_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.cond_where_stats_hidden_dim, self.latent_dim),
                )
                self.cond_where_merge = nn.Sequential(
                    nn.LayerNorm(self.latent_dim * 2),
                    nn.Linear(self.latent_dim * 2, self.latent_dim),
                    nn.SiLU(),
                )
                if self.cond_where_film_enable:
                    self.cond_where_to_film = nn.Sequential(
                        nn.Linear(self.latent_dim, self.latent_dim),
                        nn.SiLU(),
                        nn.Linear(self.latent_dim, self.latent_dim * 2),
                    )
                else:
                    self.cond_where_to_film = None
                self.cond_where_gate = nn.Sequential(
                    nn.Linear(self.latent_dim + 1, self.cond_where_stats_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.cond_where_stats_hidden_dim, 1),
                    nn.Sigmoid(),
                )
            else:
                self.cond_where_stats_encoder = None
                self.cond_where_merge = None
                self.cond_where_to_film = None
                self.cond_where_gate = None

        # Slot-conditioned condition projection.
        self.obj_to_cond = nn.Linear(self.latent_dim, self.latent_dim)
        if self.slot_adapter_enable:
            self.slot_adapter = nn.Sequential(
                nn.Linear(self.latent_dim, self.slot_adapter_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(self.slot_adapter_hidden_dim),
                nn.Dropout(self.slot_adapter_dropout),
                nn.Linear(self.slot_adapter_hidden_dim, self.latent_dim),
            )
        else:
            self.slot_adapter = None
        self.slot_cond_ln = nn.LayerNorm(self.latent_dim)
        if self.slot_cond_gate_enable:
            self.slot_cond_gate = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.Sigmoid(),
            )
        else:
            self.slot_cond_gate = None
        self.slot_cond_image_dropout_layer = nn.Dropout(self.slot_cond_image_dropout)

        if self.cond_split_ln_enable:
            self.cond_visual_ln = nn.LayerNorm(self.visual_dim)
            self.cond_proprio_ln = nn.LayerNorm(self.proprio_dim)
        else:
            self.cond_visual_ln = nn.Identity()
            self.cond_proprio_ln = nn.Identity()

        if self.cond_fuse_mlp_enable:
            self.cond_fuse_mlp = nn.Sequential(
                nn.Linear(self.obs_feature_dim, self.cond_fuse_mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.cond_fuse_mlp_dropout),
                nn.Linear(self.cond_fuse_mlp_hidden_dim, self.obs_feature_dim),
            )
        else:
            self.cond_fuse_mlp = nn.Identity()

        # Parallel branch fusion (A2A-style condition fusion)
        # add-mode: [visual, proprio, slot]
        # replace-mode: [slot, proprio] (replace visual branch with slot branch)
        if self.cond_parallel_enable:
            self.cond_visual_proj = nn.Linear(self.visual_dim * self.n_obs_steps, self.latent_dim)
            self.cond_proprio_proj = nn.Linear(self.proprio_dim * self.n_obs_steps, self.latent_dim)
            self.cond_visual_latent_ln = nn.LayerNorm(self.latent_dim)
            self.cond_proprio_latent_ln = nn.LayerNorm(self.latent_dim)
        else:
            self.cond_visual_proj = None
            self.cond_proprio_proj = None
            self.cond_visual_latent_ln = nn.Identity()
            self.cond_proprio_latent_ln = nn.Identity()
        self.cond_slot_latent_ln = nn.LayerNorm(self.latent_dim)
        self.cond_joint_fuse_add = nn.Sequential(
            nn.Linear(self.latent_dim * 3, self.cond_fuse_mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cond_fuse_mlp_dropout),
            nn.Linear(self.cond_fuse_mlp_hidden_dim, self.latent_dim),
        )
        self.cond_joint_fuse_replace = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.cond_fuse_mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cond_fuse_mlp_dropout),
            nn.Linear(self.cond_fuse_mlp_hidden_dim, self.latent_dim),
        )

        # ── DINOv3 slot encoder (replaces ResNet→slot path) ──
        self.dino_enable = bool(dino_enable)
        self.dino_freeze = bool(dino_freeze)
        if self.dino_enable:
            from roboverse_learn.il.policies.a2a.dino_slot_encoder import DinoSlotEncoder
            self.dino_slot_encoder = DinoSlotEncoder(
                dino_model_path=dino_model_path,
                slot_ckpt_path=dino_slot_ckpt_path,
                num_slots=dino_num_slots,
                slot_dim=dino_slot_dim,
                slot_heads=dino_slot_heads,
                img_size=dino_img_size,
                freeze_dino=dino_freeze,
                freeze_slot=dino_freeze,
                dino_layer=dino_layer,
            )
            # Project slot feature (slot_dim → latent_dim) for policy conditioning
            self.dino_slot_proj = nn.Linear(dino_slot_dim, self.latent_dim)
            if self.slot_distill_ckpt:
                print(
                    "[A2ASlotPolicy] DINOv3 enabled, skipping ResNet-based "
                    "slot_distill_ckpt loading"
                )
        else:
            self.dino_slot_encoder = None

        # Optional: initialize from standalone slot-distill checkpoint.
        # This is an additive path and is disabled when ckpt path is empty.
        # NOTE: skip when DINOv3 is active (slot weights loaded by DinoSlotEncoder).

        if self.slot_distill_ckpt and not self.dino_enable:
            self._load_slot_distill_ckpt(self.slot_distill_ckpt)

    def _needs_object_path(self) -> bool:
        return (
            self.slot_start_enable
            or (
                self.slot_cond_enable
                and (self.slot_semantic_enable or self.slot_geometry_enable)
            )
            or (
                self.slot_mask_supervise_enable
                and self.slot_mask_supervise_weight > 0
            )
        )

    def _select_flow_start(self, reference_start: torch.Tensor) -> torch.Tensor:
        """Choose the flow source for history-prior ablation experiments."""
        if self.flow_start_mode == "history":
            return reference_start * self.flow_start_scale
        if self.flow_start_mode == "gaussian":
            return torch.randn_like(reference_start) * self.flow_start_noise_std
        if self.flow_start_mode == "gaussian_scaled":
            ref_rms = reference_start.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
            return torch.randn_like(reference_start) * ref_rms.clamp_min(1e-6) * self.flow_start_noise_std
        if self.flow_start_mode == "zero":
            return torch.zeros_like(reference_start)
        if self.flow_start_mode == "shuffle" and reference_start.shape[0] > 1:
            return torch.roll(reference_start, shifts=1, dims=0) * self.flow_start_scale
        return reference_start * self.flow_start_scale

    def _pool_tokens_with_alpha(
        self,
        proj_tokens: torch.Tensor,
        alpha: torch.Tensor,
        slot_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask-pooled DINO object prototype for object-centric baseline.

        Args:
            proj_tokens: [B, N, D] DINO patch tokens after token projection.
            alpha: [B, S, N] slot-to-patch attention/mask weights.
            slot_weights: [B, S] selected foreground-slot weights.
        """
        if alpha.ndim != 3:
            raise ValueError(f"Expected alpha [B,S,N], got shape={tuple(alpha.shape)}")
        fg_alpha = torch.einsum("bs,bsn->bn", slot_weights, alpha)
        fg_alpha = fg_alpha.clamp(min=0.0)
        fg_alpha = fg_alpha / fg_alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.einsum("bn,bnd->bd", fg_alpha, proj_tokens)

    def _load_slot_distill_ckpt(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"slot_distill_ckpt not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        state = payload.get("state_dict", payload)
        if not isinstance(state, dict):
            raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

        loaded = []
        skipped = []

        # Structural alignment for slot distillation checkpoints:
        # if ckpt token_proj expects a different input dim (e.g., 384 vs current 521),
        # insert an adapter so slot weights can still be loaded and used.
        if self.slot_distill_load_slot and "token_proj.weight" in state:
            ckpt_token_w = state["token_proj.weight"]
            if isinstance(ckpt_token_w, torch.Tensor) and ckpt_token_w.ndim == 2:
                ckpt_in_dim = int(ckpt_token_w.shape[1])
                cur_in_dim = int(self.slot_token_proj.in_features)
                if ckpt_in_dim != cur_in_dim:
                    self.slot_input_adapter = nn.Linear(cur_in_dim, ckpt_in_dim)
                    self.slot_token_proj = nn.Linear(ckpt_in_dim, self.slot_dim)
                    print(
                        f"[SlotDistillLoad] inserted slot_input_adapter {cur_in_dim}->{ckpt_in_dim} "
                        "for token_proj alignment"
                    )

        def _copy_param(module: nn.Module, key: str, tensor: torch.Tensor):
            target = module.state_dict()
            if key not in target:
                skipped.append(key)
                return
            if target[key].shape != tensor.shape:
                skipped.append(f"{key}:shape {tuple(tensor.shape)}->{tuple(target[key].shape)}")
                return
            module_state = module.state_dict()
            module_state[key] = tensor
            module.load_state_dict(module_state, strict=False)
            loaded.append(key)

        if self.slot_distill_load_backbone:
            if "head_cam" in self.obs_encoder.key_model_map:
                head_model = self.obs_encoder.key_model_map["head_cam"]
                map_pairs = [
                    ("stem.0.", "conv1."),
                    ("stem.1.", "bn1."),
                    ("layers.0.", "layer1."),
                    ("layers.1.", "layer2."),
                    ("layers.2.", "layer3."),
                    ("layers.3.", "layer4."),
                ]
                for s_key, tensor in state.items():
                    for src_pref, dst_pref in map_pairs:
                        if s_key.startswith(src_pref):
                            d_key = s_key.replace(src_pref, dst_pref, 1)
                            _copy_param(head_model, d_key, tensor)
                            break

        if self.slot_distill_load_slot:
            slot_map_prefix = {
                "token_proj.": "slot_token_proj.",
                "slot_queries": "slot_queries",
                "slot_attn.": "slot_attn.",
                "slot_ffn.": "slot_ffn.",
                "slot_norm.": "slot_norm.",
            }
            for s_key, tensor in state.items():
                mapped = None
                for src_pref, dst_pref in slot_map_prefix.items():
                    if s_key == src_pref:
                        mapped = dst_pref
                        break
                    if s_key.startswith(src_pref):
                        mapped = s_key.replace(src_pref, dst_pref, 1)
                        break
                if mapped is not None:
                    _copy_param(self, mapped, tensor)

        if self.slot_distill_freeze_backbone and "head_cam" in self.obs_encoder.key_model_map:
            for p in self.obs_encoder.key_model_map["head_cam"].parameters():
                p.requires_grad = False
        if self.slot_distill_freeze_slot:
            for m in [
                self.slot_token_proj,
                self.slot_attn,
                self.slot_ffn,
                self.slot_norm,
                self.slot_to_latent,
            ]:
                for p in m.parameters():
                    p.requires_grad = False
            self.slot_queries.requires_grad = False
            if not isinstance(self.slot_input_adapter, nn.Identity):
                print(
                    "[SlotDistillLoad] keep slot_input_adapter trainable while freezing slot modules "
                    "(prevents freezing a random dim-bridge)"
                )

        print(
            f"[SlotDistillLoad] ckpt={ckpt_path} loaded={len(loaded)} skipped={len(skipped)} "
            f"load_backbone={self.slot_distill_load_backbone} load_slot={self.slot_distill_load_slot}"
        )
        if len(skipped) > 0:
            print(f"[SlotDistillLoad] skipped examples: {skipped[:8]}")

    def _apply_background_noise(
        self,
        nobs: Dict[str, torch.Tensor],
        task_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Gaussian perturbation to background (non-task) pixels only.
        Expects task_mask as [B, T, 1, H, W] or [B, T, H, W] in {0,1}.
        """
        if "head_cam" not in nobs:
            return nobs
        if self.saliency_noise_std <= 0:
            return nobs

        cam = nobs["head_cam"]
        if task_mask.ndim == 4:
            task_mask = task_mask.unsqueeze(2)
        task_mask = task_mask.to(device=cam.device, dtype=cam.dtype)
        task_mask = (task_mask > 0.5).to(cam.dtype)

        if task_mask.shape[-2:] != cam.shape[-2:]:
            task_mask = F.interpolate(
                task_mask.reshape(-1, 1, task_mask.shape[-2], task_mask.shape[-1]),
                size=cam.shape[-2:],
                mode="nearest",
            ).reshape(cam.shape[0], cam.shape[1], 1, cam.shape[-2], cam.shape[-1])

        bg_mask = 1.0 - task_mask
        noise = torch.randn_like(cam) * self.saliency_noise_std
        if self.saliency_apply_prob < 1.0:
            keep = (torch.rand(cam.shape[0], 1, 1, 1, 1, device=cam.device) < self.saliency_apply_prob).to(cam.dtype)
        else:
            keep = 1.0
        perturbed = torch.clamp(cam + noise * bg_mask * keep, -1.0, 1.0)
        nobs_pert = dict(nobs)
        nobs_pert["head_cam"] = perturbed
        return nobs_pert

    def _add_history_noise(self, history_states: torch.Tensor) -> torch.Tensor:
        if self.training and self.history_noise_std > 0:
            return history_states + torch.randn_like(history_states) * self.history_noise_std
        return history_states

    def _encode_obs_latents_and_tokens(
        self, nobs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = nobs["agent_pos"].shape[0]

        # Encode each frame, then reshape back to temporal tokens.
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )

        obs_features_bt = self.obs_encoder(this_nobs)  # (B*T, D)
        obs_tokens_raw = obs_features_bt.reshape(batch_size, self.n_obs_steps, self.obs_feature_dim)
        slot_inputs = self.slot_input_adapter(obs_tokens_raw)
        obs_tokens_slot = self.slot_token_proj(slot_inputs)

        vis = obs_tokens_raw[..., : self.visual_dim]
        prop = obs_tokens_raw[..., self.visual_dim :]
        # Condition path: normalize visual/proprio branches separately, then fuse.
        if self.cond_split_ln_enable:
            vis = self.cond_visual_ln(vis)
            prop = self.cond_proprio_ln(prop)
            cond_tokens = torch.cat([vis, prop], dim=-1)
        else:
            cond_tokens = obs_tokens_raw
        cond_tokens = self.cond_fuse_mlp(cond_tokens)
        obs_latents = self.obs_projector(cond_tokens.reshape(batch_size, -1))
        if self.cond_parallel_enable:
            vis_latent = self.cond_visual_proj(vis.reshape(batch_size, -1))
            prop_latent = self.cond_proprio_proj(prop.reshape(batch_size, -1))
            vis_latent = self.cond_visual_latent_ln(vis_latent)
            prop_latent = self.cond_proprio_latent_ln(prop_latent)
        else:
            vis_latent = obs_latents
            prop_latent = torch.zeros_like(obs_latents)
        # Extract pure robot joint state (agent_pos) BEFORE encoder fusion — no visual leakage.
        # Shape: [B, T, 9] → [B, 9 * n_obs_steps]
        agent_pos = nobs["agent_pos"][:, : self.n_obs_steps, ...].reshape(batch_size, -1)
        return obs_latents, obs_tokens_slot, vis_latent, prop_latent, agent_pos

    def _extract_headcam_spatial_tokens(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor | None:
        """
        Extract per-frame spatial tokens from head_cam using ResNet layer4 feature map.
        Returns shape [B, T, HW, C] or None if unavailable.
        """
        if "head_cam" not in nobs:
            return None
        if "head_cam" not in self.obs_encoder.key_model_map:
            return None
        if "head_cam" not in self.obs_encoder.key_transform_map:
            return None

        model = self.obs_encoder.key_model_map["head_cam"]
        # This path is for torchvision ResNet-style backbones used in this project.
        required = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
        if not all(hasattr(model, name) for name in required):
            return None

        cam = nobs["head_cam"][:, : self.n_obs_steps, ...]  # [B,T,3,H,W]
        bsz, tsz = cam.shape[:2]
        cam_bt = cam.reshape(-1, *cam.shape[2:])  # [B*T,3,H,W]
        cam_bt = self.obs_encoder.key_transform_map["head_cam"](cam_bt)

        x = model.conv1(cam_bt)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)  # [B*T,C,Hf,Wf]

        # Convert to spatial token sequence.
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B*T,HW,C]
        x = x.reshape(bsz, tsz, x.shape[1], x.shape[2])  # [B,T,HW,C]
        return x

    def _compute_mask_supervised_obj_latent(
        self,
        obs_tokens: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Lightweight mask supervision target for slot latent:
        use task-mask area as temporal weights over already-computed obs tokens.
        This avoids any extra heavy visual forward (prevents OOM).
        """
        if task_mask.ndim == 4:
            task_mask = task_mask.unsqueeze(2)  # [B,T,1,H,W]
        task_mask = task_mask[:, : self.n_obs_steps, ...].to(obs_tokens.device).float()
        task_mask = (task_mask > 0.5).float()

        # Temporal weights from per-frame task area.
        # frame_area: [B,T]
        frame_area = task_mask.mean(dim=(2, 3, 4))
        area_sum = frame_area.sum(dim=1, keepdim=True)
        uniform = torch.full_like(frame_area, 1.0 / max(1, frame_area.shape[1]))
        frame_w = torch.where(area_sum > 1e-6, frame_area / area_sum.clamp(min=1e-6), uniform)

        # Weighted temporal pooling of obs tokens -> object prototype [B,C].
        proto = torch.sum(obs_tokens * frame_w.unsqueeze(-1), dim=1)

        if proto.shape[-1] != self.latent_dim:
            proto = self.mask_proto_to_latent(proto)
        return proto

    def _compute_object_latent(
        self,
        obs_tokens: torch.Tensor,
        history_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        # Slot extraction over temporal tokens.
        batch_size = obs_tokens.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)

        slots, _ = self.slot_attn(
            query=queries,
            key=obs_tokens,
            value=obs_tokens,
            need_weights=False,
        )
        slots = self.slot_norm(slots + queries)
        slots = slots + self.slot_ffn(slots)

        if self.slot_selection_mode == "fixed_index":
            idx = self.slot_obj_index
            slot_weights = torch.zeros(
                slots.shape[0], slots.shape[1], device=slots.device, dtype=slots.dtype
            )
            slot_weights[:, idx] = 1.0
            obj_token = slots[:, idx, :]
        else:
            # Task-relevant slot selection with history-conditioned query.
            hist_query = F.normalize(self.hist_query_proj(history_latents), dim=-1)
            slot_keys = F.normalize(slots, dim=-1)
            logits = torch.einsum("bd,bsd->bs", hist_query, slot_keys) / math.sqrt(slots.shape[-1])
            logits = logits / self.slot_selection_temperature
            slot_weights = F.softmax(logits, dim=-1)
            obj_token = torch.einsum("bs,bsd->bd", slot_weights, slots)

        obj_latent = self.slot_to_latent(obj_token)
        return obj_latent, slot_weights, None

    def _get_dino_obj_latent(
        self,
        nobs: Dict[str, torch.Tensor],
        history_latents: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Extract object latent and spatial alpha from DINOv3 slot encoder.
        With two-scale: coarse centroid → crop → fine slot feat + precise centroid.

        Returns:
            obj_latent:      [B, latent_dim]
            slot_weights:    [B, num_slots]
            alpha:           [B, num_slots, N_patches] or None
            global_centroid: [B, 2] or None (fine centroid for start correction)
            where_stats:     [B, 3] or None (log area, std_x, std_y in global frame)
        """
        head_cam = nobs["head_cam"][:, : self.n_obs_steps, ...]  # [B, T, 3, H, W]
        B, T, C, H_img, W_img = head_cam.shape
        device = head_cam.device

        # ── Stage 1: full-image, get mask decoder output for crop ──
        s1_slot_feat, s1_alpha, s1_proj_tokens = self.dino_slot_encoder(
            head_cam, return_attention=True, return_proj_tokens=True,
        )
        s1_slot_feat = s1_slot_feat.mean(dim=1)  # [B, S, D]
        s1_alpha = s1_alpha.mean(dim=1)          # [B, S, N]
        s1_proj_tokens = s1_proj_tokens.mean(dim=1)  # [B, N, 256]
        N = s1_alpha.shape[2]
        H_a = W_a = int(N ** 0.5)

        if not self.dino_two_scale_enable:
            if self.slot_selection_mode == "fixed_index":
                slot_weights = torch.zeros(s1_slot_feat.shape[0], s1_slot_feat.shape[1],
                                            device=device, dtype=s1_slot_feat.dtype)
                slot_weights[:, self.slot_obj_index] = 1.0
            else:
                hist_query = F.normalize(self.hist_query_proj(history_latents), dim=-1)
                slot_keys = F.normalize(s1_slot_feat, dim=-1)
                logits = torch.einsum("bd,bsd->bs", hist_query, slot_keys) / math.sqrt(s1_slot_feat.shape[-1])
                logits = logits / self.slot_selection_temperature
                slot_weights = F.softmax(logits, dim=-1)
            if self.object_condition_source == "mask_pool":
                obj_token = self._pool_tokens_with_alpha(s1_proj_tokens, s1_alpha, slot_weights)
            elif self.slot_selection_mode == "fixed_index":
                obj_token = s1_slot_feat[:, self.slot_obj_index, :]
            else:
                obj_token = torch.einsum("bs,bsd->bd", slot_weights, s1_slot_feat)
            obj_latent = self.dino_slot_proj(obj_token)
            return obj_latent, slot_weights, s1_alpha, None, None

        # ── Two-scale: mask-decoder-guided crop (or attention fallback) ──
        if self.dino_slot_encoder.has_mask_decoder and self.dino_slot_encoder._mask_decoder_mode == "sbd_hr":
            s1_fg = s1_slot_feat[:, self.slot_obj_index, :]  # [B, 256]
            # Decode mask at half resolution to save memory (224×224 vs 448×448),
            # then scale bbox coords back to full resolution.
            mask_hw = (H_img // 2, W_img // 2)  # 224×224 for 448 input
            s1_mask_logits = self.dino_slot_encoder.decode_mask_logits(
                fg_slot=s1_fg, proj_tokens=s1_proj_tokens,
                hw=(H_a, W_a), out_hw=mask_hw,
            )
            s1_mask = (torch.sigmoid(s1_mask_logits) >= 0.5).float()  # [B, H/2, W/2]
            scale = H_img / mask_hw[0]  # 2.0
            # Bbox in half-resolution, then scale up
            row_any = s1_mask.amax(dim=-1)  # [B, H/2]
            col_any = s1_mask.amax(dim=-2)  # [B, W/2]
            x1_px = ((col_any > 0).float().argmax(dim=-1).float() * scale).long()
            x2_px = ((mask_hw[1] - 1 - (col_any > 0).float().flip(-1).argmax(dim=-1)).float() * scale).clamp(max=W_img-1).long()
            y1_px = ((row_any > 0).float().argmax(dim=-1).float() * scale).long()
            y2_px = ((mask_hw[0] - 1 - (row_any > 0).float().flip(-1).argmax(dim=-1)).float() * scale).clamp(max=H_img-1).long()
            margin_px = 30
            x1_px = (x1_px - margin_px).clamp(0)
            y1_px = (y1_px - margin_px).clamp(0)
            x2_px = (x2_px + margin_px).clamp(max=W_img - 1)
            y2_px = (y2_px + margin_px).clamp(max=H_img - 1)
            cw = (x2_px - x1_px).clamp(min=int(0.2 * W_img), max=int(0.7 * W_img)).long()
            ch = (y2_px - y1_px).clamp(min=int(0.2 * H_img), max=int(0.7 * H_img)).long()
            x1_b = x1_px.long()
            y1_b = y1_px.long()
            # Repad to match cw, ch if clamp changed them
            x1_b = torch.min(x1_b, (W_img - cw).long())
            y1_b = torch.min(y1_b, (H_img - ch).long())
            cx_px = (x1_b + cw // 2).long()
            cy_px = (y1_b + ch // 2).long()
        # Coordinate grid at attention resolution (needed for both branches)
        gx = torch.arange(W_a, device=device).float() / max(1, W_a - 1)
        gy = torch.arange(H_a, device=device).float() / max(1, H_a - 1)
        gy_m, gx_m = torch.meshgrid(gy, gx, indexing="ij")

        if hasattr(self.dino_slot_encoder, 'has_mask_decoder') and self.dino_slot_encoder.has_mask_decoder:
            # mask-decoder path: cx_px, cy_px, cw, ch, x1_b, y1_b already set above
            pass
        else:
            # Fallback: attention-alpha centroid, fixed 50% crop
            fg_a = s1_alpha[:, self.slot_obj_index, :]
            fg_n = fg_a / fg_a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            cx_a = (fg_n.reshape(B, H_a, W_a) * gx_m).sum(dim=(-2, -1))
            cy_a = (fg_n.reshape(B, H_a, W_a) * gy_m).sum(dim=(-2, -1))
            cw = torch.full((B,), int(0.5 * W_img), device=device, dtype=torch.long)
            ch = torch.full((B,), int(0.5 * H_img), device=device, dtype=torch.long)
            cx_px = (cx_a * W_img).long()
            cy_px = (cy_a * H_img).long()
            x1_b = torch.clamp(cx_px - cw // 2, 0, W_img - 1)
            y1_b = torch.clamp(cy_px - ch // 2, 0, H_img - 1)

        # Crop per (B,T) sample
        head_flat = head_cam.reshape(B * T, C, H_img, W_img)
        cx_f = cx_px.unsqueeze(1).expand(-1, T).reshape(-1)
        cy_f = cy_px.unsqueeze(1).expand(-1, T).reshape(-1)
        cw_f = cw.unsqueeze(1).expand(-1, T).reshape(-1)
        ch_f = ch.unsqueeze(1).expand(-1, T).reshape(-1)
        crops = []
        for i in range(B * T):
            x1 = max(0, min(cx_f[i].item() - cw_f[i].item() // 2, W_img - cw_f[i].item()))
            y1 = max(0, min(cy_f[i].item() - ch_f[i].item() // 2, H_img - ch_f[i].item()))
            c = head_flat[i:i+1, :, y1:y1+ch_f[i].item(), x1:x1+cw_f[i].item()]
            c = F.interpolate(c, size=(H_img, W_img), mode="bilinear", align_corners=False)
            crops.append(c)
        head_crop = torch.cat(crops, dim=0).reshape(B, T, C, H_img, W_img)

        # Stage 2 forward with fine head
        s2_slot_feat_bt, s2_alpha_bt, s2_proj_tokens_bt = self.dino_slot_encoder(
            head_crop, return_attention=True, return_proj_tokens=True, head="fine"
        )
        s2_slot_feat = s2_slot_feat_bt.mean(dim=1)  # [B, S, D]
        s2_alpha = s2_alpha_bt.mean(dim=1)  # [B, S, N]
        s2_proj_tokens = s2_proj_tokens_bt.mean(dim=1)  # [B, N, D]

        # Fine centroid, mapped to global coordinates
        s2_fg = s2_alpha[:, self.slot_obj_index, :]
        s2_fn = s2_fg / s2_fg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        s2_cx = (s2_fn.reshape(B, H_a, W_a) * gx_m).sum(dim=(-2, -1))
        s2_cy = (s2_fn.reshape(B, H_a, W_a) * gy_m).sum(dim=(-2, -1))

        if self.cond_centroid_ema_enable and s2_alpha_bt.ndim == 4:
            s2_fg_bt = s2_alpha_bt[:, :, self.slot_obj_index, :]  # [B,T,N]
            s2_fn_bt = s2_fg_bt / s2_fg_bt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            s2_cx_bt = (s2_fn_bt.reshape(B, T, H_a, W_a) * gx_m.view(1, 1, H_a, W_a)).sum(dim=(-2, -1))
            s2_cy_bt = (s2_fn_bt.reshape(B, T, H_a, W_a) * gy_m.view(1, 1, H_a, W_a)).sum(dim=(-2, -1))
            global_cx_bt = (x1_b.float().unsqueeze(1) + s2_cx_bt * cw.float().unsqueeze(1)) / W_img
            global_cy_bt = (y1_b.float().unsqueeze(1) + s2_cy_bt * ch.float().unsqueeze(1)) / H_img
            centroid_bt = torch.stack([global_cx_bt, global_cy_bt], dim=-1)  # [B,T,2]
            ema = centroid_bt[:, 0, :]
            for t_idx in range(1, centroid_bt.shape[1]):
                ema = (
                    self.cond_centroid_ema_decay * ema
                    + (1.0 - self.cond_centroid_ema_decay) * centroid_bt[:, t_idx, :]
                )
            global_centroid = ema
        else:
            global_cx = (x1_b.float() + s2_cx * cw.float()) / W_img
            global_cy = (y1_b.float() + s2_cy * ch.float()) / H_img
            global_centroid = torch.stack([global_cx, global_cy], dim=-1)

        # Where statistics in global image frame:
        # [log(area_proxy), std_x, std_y] to stabilize tiny-object conditioning.
        gx_flat = gx_m.reshape(1, -1)
        gy_flat = gy_m.reshape(1, -1)
        var_x_local = (s2_fn * (gx_flat - s2_cx.unsqueeze(-1)).pow(2)).sum(dim=-1)
        var_y_local = (s2_fn * (gy_flat - s2_cy.unsqueeze(-1)).pow(2)).sum(dim=-1)
        std_x_global = var_x_local.clamp(min=0.0).sqrt() * (cw.float() / max(1, W_img))
        std_y_global = var_y_local.clamp(min=0.0).sqrt() * (ch.float() / max(1, H_img))
        n_patch = float(s2_fn.shape[-1])
        area_eff_local = 1.0 / s2_fn.pow(2).sum(dim=-1).clamp(min=1e-8)
        area_local = (area_eff_local / max(1.0, n_patch)).clamp(min=1e-6, max=1.0)
        crop_area = ((cw.float() / max(1, W_img)) * (ch.float() / max(1, H_img))).clamp(min=1e-6, max=1.0)
        area_global = (area_local * crop_area).clamp(min=1e-6, max=1.0)
        where_stats = torch.stack([area_global.log(), std_x_global, std_y_global], dim=-1)

        # FG slot feature for conditioning
        if self.slot_selection_mode == "fixed_index":
            slot_weights = torch.zeros(s2_slot_feat.shape[0], s2_slot_feat.shape[1],
                                        device=device, dtype=s2_slot_feat.dtype)
            slot_weights[:, self.slot_obj_index] = 1.0
        else:
            hist_query = F.normalize(self.hist_query_proj(history_latents), dim=-1)
            slot_keys = F.normalize(s2_slot_feat, dim=-1)
            logits = torch.einsum("bd,bsd->bs", hist_query, slot_keys) / math.sqrt(s2_slot_feat.shape[-1])
            logits = logits / self.slot_selection_temperature
            slot_weights = F.softmax(logits, dim=-1)
        if self.object_condition_source == "mask_pool":
            obj_token = self._pool_tokens_with_alpha(s2_proj_tokens, s2_alpha, slot_weights)
        elif self.slot_selection_mode == "fixed_index":
            obj_token = s2_slot_feat[:, self.slot_obj_index, :]
        else:
            obj_token = torch.einsum("bs,bsd->bd", slot_weights, s2_slot_feat)

        obj_latent = self.dino_slot_proj(obj_token)
        return obj_latent, slot_weights, s2_alpha, global_centroid, where_stats

    def _encode_centroid(self, global_centroid: torch.Tensor) -> torch.Tensor:
        freqs = self.centroid_freqs * (2.0 * math.pi)
        c_proj = global_centroid.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        centroid_enc = torch.cat([torch.sin(c_proj), torch.cos(c_proj)], dim=-1)
        return centroid_enc.reshape(global_centroid.shape[0], -1)

    def _encode_cond_centroid(self, centroid_t: torch.Tensor) -> torch.Tensor:
        """
        Encode normalized centroid (in [-1, 1]) for condition branch.
        Returns [B, latent_dim].
        """
        if self.cond_centroid_low_freq_enable and self.cond_centroid_encoder is not None:
            return self.cond_centroid_encoder(centroid_t)
        freqs_c = self.centroid_freqs * (2.0 * math.pi)
        c_proj = centroid_t.unsqueeze(-1) * freqs_c.unsqueeze(0).unsqueeze(0)
        centroid_enc = torch.cat([torch.sin(c_proj), torch.cos(c_proj)], dim=-1)
        centroid_enc = centroid_enc.reshape(centroid_t.shape[0], -1)  # [B, 4*n_freqs]
        return self.centroid_to_cond(centroid_enc)

    def _compute_centroid_confidence(self, alpha: torch.Tensor | None) -> torch.Tensor | None:
        """
        Estimate object centroid confidence from slot attention peakiness.
        Returns [B, 1] in [0, 1]. Higher means more concentrated attention.
        """
        if alpha is None:
            return None
        if alpha.ndim == 4:
            fg = alpha[:, :, self.slot_obj_index, :]  # [B,T,N]
            fg = fg / fg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            peak = fg.max(dim=-1).values.mean(dim=1)  # [B]
            n_patch = fg.shape[-1]
        elif alpha.ndim == 3:
            fg = alpha[:, self.slot_obj_index, :]  # [B,N]
            fg = fg / fg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            peak = fg.max(dim=-1).values  # [B]
            n_patch = fg.shape[-1]
        else:
            return None

        if n_patch <= 1:
            conf = torch.ones_like(peak)
        else:
            uniform = 1.0 / float(n_patch)
            conf = (peak - uniform) / max(1e-6, (1.0 - uniform))
            conf = conf.clamp(0.0, 1.0)
        return conf.unsqueeze(-1)

    def _predict_joint_offset(
        self,
        history_states: torch.Tensor,
        global_centroid: torch.Tensor | None,
        is_eval: bool,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if (not self.slot_start_enable) or (global_centroid is None):
            return None, None

        bsz = history_states.shape[0]
        hist_flat = history_states.reshape(bsz, -1)
        centroid_enc = self._encode_centroid(global_centroid)
        joint_offset = self.offset_mlp(torch.cat([centroid_enc, hist_flat], dim=-1))

        clamp_val = self.eval_offset_clamp if is_eval else self.train_offset_clamp
        if clamp_val > 0:
            joint_offset = torch.clamp(joint_offset, -clamp_val, clamp_val)

        hist_rms = hist_flat.pow(2).mean(dim=-1).sqrt().clamp(min=1e-4)
        offset_rms = joint_offset.pow(2).mean(dim=-1).sqrt()
        offset_ratio = offset_rms / hist_rms

        ratio_cap = self.eval_offset_ratio_cap if is_eval else self.train_offset_ratio_cap
        if ratio_cap > 0:
            safe_scale = torch.clamp(ratio_cap / (offset_ratio + 1e-6), max=1.0).detach()
            joint_offset = joint_offset * safe_scale.unsqueeze(-1)

        global_scale = self.eval_offset_scale if is_eval else self.train_offset_scale
        if global_scale != 1.0:
            joint_offset = joint_offset * global_scale
        return joint_offset, offset_ratio

    def _apply_start_offset(
        self,
        history_states: torch.Tensor,
        global_centroid: torch.Tensor | None,
        is_eval: bool,
        detach_offset: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        joint_offset, offset_ratio = self._predict_joint_offset(
            history_states=history_states,
            global_centroid=global_centroid,
            is_eval=is_eval,
        )
        if joint_offset is None:
            return history_states, None, offset_ratio

        joint_offset_per_step = joint_offset.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
        if detach_offset:
            joint_offset_per_step = joint_offset_per_step.detach()
        history_states = history_states - joint_offset_per_step
        return history_states, joint_offset, offset_ratio


    def _compose_start_and_cond(
        self,
        history_latents: torch.Tensor,
        obs_latents: torch.Tensor,
        visual_latents: torch.Tensor,
        proprio_latents: torch.Tensor,
        obj_latent: torch.Tensor | None,
        alpha: torch.Tensor | None = None,
        global_centroid: torch.Tensor | None = None,
        centroid_confidence: torch.Tensor | None = None,
        where_stats: torch.Tensor | None = None,
        prop_raw: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.slot_start_enable:
            if obj_latent is None:
                raise ValueError("obj_latent is required when slot_start_enable=True")
            # offset_mlp already corrected history at raw level → encoder → clean z0
            # skip _compute_start_latent (old latent-space correction) to avoid double correction
            start_latent = history_latents
            gate = torch.zeros(history_latents.shape[0], device=history_latents.device)
        else:
            start_latent = history_latents
            gate = None

        cond_gate = torch.zeros(
            1,
            device=history_latents.device,
            dtype=history_latents.dtype,
        )  # placeholder when slot_cond disabled
        self._last_where_gate_mean = 0.0
        if self.slot_cond_enable:
            if self.slot_semantic_enable and obj_latent is None:
                raise ValueError("obj_latent is required when slot_cond_enable=True")
            if self.slot_semantic_enable:
                obj_cond = self.obj_to_cond(obj_latent)
            else:
                obj_cond = torch.zeros_like(history_latents)

            # When two-scale is active, fuse global centroid encoding into condition.
            # This gives the flow matcher a direct spatial anchor alongside fine slot features.
            if (
                self.slot_geometry_enable
                and self.dino_two_scale_enable
                and global_centroid is not None
                and self.cond_centroid_enable
            ):
                centroid_t = global_centroid * 2.0 - 1.0  # [0,1]→[-1,1]
                centroid_cond = self._encode_cond_centroid(centroid_t)
                conf = None
                if centroid_confidence is not None:
                    conf = centroid_confidence.clamp(min=self.cond_centroid_conf_min, max=1.0)
                centroid_scale = self.centroid_cond_scale

                # Hybrid where-injection: geometry controls condition branch via
                # FiLM + residual gate, instead of direct feature overwrite.
                if self.cond_where_enable:
                    where_latent = centroid_cond
                    if (
                        self.cond_where_stats_enable
                        and where_stats is not None
                        and self.cond_where_stats_encoder is not None
                        and self.cond_where_merge is not None
                    ):
                        where_stats_latent = self.cond_where_stats_encoder(where_stats)
                        where_latent = self.cond_where_merge(
                            torch.cat([where_latent, where_stats_latent], dim=-1)
                        )
                    if self.cond_where_conf_gate_enable and conf is not None:
                        where_latent = conf * where_latent

                    if self.cond_where_film_enable and self.cond_where_to_film is not None:
                        film = self.cond_where_to_film(where_latent)
                        gamma, beta = torch.chunk(film, chunks=2, dim=-1)
                        scale = self.cond_where_film_max_scale
                        gamma = 1.0 + scale * torch.tanh(gamma)
                        beta = scale * torch.tanh(beta)
                        where_applied = gamma * obj_cond + beta
                    else:
                        where_applied = obj_cond + centroid_scale * where_latent

                    if self.cond_where_gate is not None:
                        if conf is None:
                            conf_in = torch.ones(
                                where_latent.shape[0],
                                1,
                                device=where_latent.device,
                                dtype=where_latent.dtype,
                            )
                        else:
                            conf_in = conf
                        where_gate = self.cond_where_gate(
                            torch.cat([where_latent, conf_in], dim=-1)
                        )
                        if self.cond_where_gate_min > 0:
                            where_gate = self.cond_where_gate_min + (1.0 - self.cond_where_gate_min) * where_gate
                        obj_cond = obj_cond + where_gate * (where_applied - obj_cond)
                        self._last_where_gate_mean = float(where_gate.mean().detach().item())
                    else:
                        obj_cond = where_applied
                else:
                    if self.cond_centroid_conf_gate_enable and conf is not None:
                        centroid_scale = centroid_scale * conf
                    obj_cond = obj_cond + centroid_scale * centroid_cond
            if self.slot_adapter is not None:
                obj_cond = self.slot_adapter(obj_cond)
            obj_cond = self.cond_slot_latent_ln(obj_cond)
            vis_cond = self.slot_cond_image_dropout_layer(visual_latents)
            prop_cond = proprio_latents

            # Gate slot contribution: sigmoid([vis_cond, obj_cond]) → [0,1]
            # Prevents the model from learning to ignore the slot branch.
            if self.slot_cond_gate is not None:
                cond_gate = self.slot_cond_gate(
                    torch.cat([vis_cond, obj_cond], dim=-1)
                )
                obj_cond = cond_gate * obj_cond

            if self.slot_cond_mode == "replace":
                cond_latent = self.cond_joint_fuse_replace(
                    torch.cat([obj_cond, prop_cond], dim=-1)
                )
            else:
                cond_latent = self.cond_joint_fuse_add(
                    torch.cat([vis_cond, prop_cond, self.cond_obj_weight * obj_cond], dim=-1)
                )
        else:
            cond_latent = obs_latents

        return start_latent, cond_latent, gate, cond_gate

    @staticmethod
    def _quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product for quaternions in [w, x, y, z] order."""
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    @staticmethod
    def _rotvec_to_quat_torch(rotvec: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
        half = 0.5 * angle
        small = angle < 1e-6
        scale = torch.where(
            small,
            0.5 - angle.pow(2) / 48.0,
            torch.sin(half) / angle.clamp_min(1e-8),
        )
        quat = torch.cat([torch.cos(half), rotvec * scale], dim=-1)
        return F.normalize(quat, dim=-1, eps=1e-8)

    @staticmethod
    def _quat_to_rotvec_torch(quat: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(quat, dim=-1, eps=1e-8)
        quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
        w = quat[..., :1]
        xyz = quat[..., 1:]
        sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(sin_half, w)
        scale = torch.where(
            sin_half < 1e-6,
            2.0 + angle.pow(2) / 12.0,
            angle / sin_half.clamp_min(1e-8),
        )
        return xyz * scale

    def _rotvec_delta_torch(self, curr_rotvec: torch.Tensor, target_rotvec: torch.Tensor) -> torch.Tensor:
        """Relative rotvec delta satisfying R_target ~= R_delta * R_curr."""
        curr_quat = self._rotvec_to_quat_torch(curr_rotvec)
        target_quat = self._rotvec_to_quat_torch(target_rotvec)
        curr_inv = curr_quat.clone()
        curr_inv[..., 1:] = -curr_inv[..., 1:]
        delta_quat = self._quat_mul_torch(target_quat, curr_inv)
        return self._quat_to_rotvec_torch(delta_quat)

    def _quat_delta_torch(self, curr_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Relative quaternion delta satisfying q_target ~= q_curr * q_delta."""
        curr_quat = F.normalize(curr_quat, dim=-1, eps=1e-8)
        target_quat = F.normalize(target_quat, dim=-1, eps=1e-8)
        curr_inv = curr_quat.clone()
        curr_inv[..., 1:] = -curr_inv[..., 1:]
        w1, x1, y1, z1 = curr_inv.unbind(dim=-1)
        w2, x2, y2, z2 = target_quat.unbind(dim=-1)
        delta = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )
        return F.normalize(delta, dim=-1, eps=1e-8)

    def _build_delta_history_states(self, raw_agent_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute TCP/EE history into action-like deltas before encoding z0.

        A2A assumes history encoder input and future action target share semantics.
        For delta-EE policies, raw agent_pos is absolute TCP but action is delta TCP;
        this adapter makes the flow start latent live in the same delta-action space.
        """
        hist = raw_agent_pos[:, : self.n_obs_steps, :]
        hist = hist[..., : min(hist.shape[-1], self.action_dim)]
        delta = torch.zeros_like(hist)
        if hist.shape[1] > 1:
            delta[:, 1:, :3] = hist[:, 1:, :3] - hist[:, :-1, :3]
            if hist.shape[-1] >= 8:
                # Sim EE layout: [pos(3), quat_wxyz(4), gripper...].
                # data2zarr exports q_delta = inv(q_curr) * q_next.
                delta[:, 1:, 3:7] = self._quat_delta_torch(hist[:, :-1, 3:7], hist[:, 1:, 3:7])
            elif hist.shape[-1] >= 6:
                # Real TCP layout: [pos(3), rotvec(3), gripper].
                delta[:, 1:, 3:6] = self._rotvec_delta_torch(hist[:, :-1, 3:6], hist[:, 1:, 3:6])
            elif hist.shape[-1] > 3:
                delta[:, 1:, 3:] = hist[:, 1:, 3:] - hist[:, :-1, 3:]
        # Gripper action convention remains absolute target gripper, not delta.
        if hist.shape[-1] >= 8:
            delta[:, :, 7:] = hist[:, :, 7:]
            delta[:, 0, 3] = 1.0
        elif hist.shape[-1] > 6:
            delta[:, :, 6:] = hist[:, :, 6:]
        delta = self.normalizer["action"].normalize(delta)
        return torch.nn.functional.pad(delta, (0, self.action_dim - delta.shape[-1])) if delta.shape[-1] < self.action_dim else delta

    def _get_history_states_for_start(self, raw_obs: Dict[str, torch.Tensor], nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.delta_action_history_enable:
            return self._build_delta_history_states(raw_obs["agent_pos"])
        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        return torch.nn.functional.pad(history_states, (0, self.action_dim - history_states.shape[-1])) if history_states.shape[-1] < self.action_dim else history_states

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        # Pad UR3 actions (7D) to 9D AFTER normalize (normalizer was fit on 7D)
        nactions = torch.nn.functional.pad(nactions, (0, self.action_dim - nactions.shape[-1])) if nactions.shape[-1] < self.action_dim else nactions
        batch_size = nactions.shape[0]

        obs_latents, obs_tokens, visual_latents, proprio_latents, _prop_raw = self._encode_obs_latents_and_tokens(nobs)

        history_states = self._get_history_states_for_start(batch["obs"], nobs)
        B = history_states.shape[0]
        device = history_states.device

        # Physical correlated jitter: same 9-D noise on history AND prop_raw
        _prop_raw_noisy = _prop_raw
        _prop_raw_clean = _prop_raw
        history_states_clean = history_states

        # Object-centric history: compute centroid FIRST, then offset history
        # Modality Dropout: randomly mask centroid to teach offset_mlp "unreliable→output 0"
        _centroid_dropout = False
        if self.training and self.slot_start_enable and self.dino_enable and torch.rand(1).item() < 0.2:
            _centroid_dropout = True

        need_obj_latent = self._needs_object_path()
        if need_obj_latent and self.dino_enable:
            dummy_hist = torch.zeros(B, self.latent_dim, device=device)
            obj_latent, slot_weights, alpha, global_centroid, where_stats = self._get_dino_obj_latent(nobs, dummy_hist)
            centroid_confidence = self._compute_centroid_confidence(alpha)
        elif need_obj_latent:
            temp_hist = self.history_action_encoder(history_states)
            obj_latent, slot_weights, alpha = self._compute_object_latent(obs_tokens, temp_hist)
            global_centroid = None
            centroid_confidence = None
            where_stats = None
        else:
            obj_latent, slot_weights, alpha, global_centroid = None, None, None, None
            centroid_confidence = None
            where_stats = None

        # Cross-pairing: roll centroid by 1 → synthetic mismatch (optional, conservative by default).
        _cross_offset_gt = None
        if self.training and self.slot_start_enable and self.start_cross_pair_prob > 0 and B > 1:
            use_cross = torch.rand(1).item() < self.start_cross_pair_prob
        else:
            use_cross = False

        # Offset history using centroid.
        offset_ratio = None
        offset_centroid = None if _centroid_dropout else global_centroid
        if self.slot_start_enable and offset_centroid is not None:
            history_states, joint_offset, offset_ratio = self._apply_start_offset(
                history_states=history_states,
                global_centroid=offset_centroid,
                is_eval=False,
                detach_offset=self.start_detach_offset,
            )
            history_states_clean, _, _ = self._apply_start_offset(
                history_states=history_states_clean,
                global_centroid=offset_centroid,
                is_eval=False,
                detach_offset=True,
            )

            joint_offset_cross = None
            if use_cross:
                hist_flat = history_states.reshape(B, -1)
                centroid_enc = self._encode_centroid(offset_centroid)
                centroid_enc_x = torch.roll(centroid_enc, shifts=1, dims=0)
                hist_flat_x = torch.roll(hist_flat, shifts=1, dims=0)
                joint_offset_cross = self.offset_mlp(torch.cat([centroid_enc_x, hist_flat], dim=-1))
                _cross_offset_gt = (hist_flat - hist_flat_x).reshape(B, self.n_obs_steps, -1).mean(dim=1)

            if self._debug_step < 5 and joint_offset is not None:
                off_norm = joint_offset.norm(dim=-1).mean().item()
                hist_norm = history_states.norm(dim=(-2, -1)).mean().item()
                ratio_val = 0.0 if hist_norm <= 1e-8 else (off_norm / hist_norm)
                print(
                    f"[OFFSET RATIO] |offset|={off_norm:.4f} |hist|={hist_norm:.4f} ratio={ratio_val:.4f}",
                    flush=True,
                )
                self._debug_step += 1
        else:
            joint_offset = None
            joint_offset_cross = None

        if self.training and self.slot_start_enable and self.history_jitter_prob > 0:
            jit_mask = (torch.rand(B, 1, device=device) < self.history_jitter_prob).float()
            phys_noise = torch.randn_like(history_states) * self.history_jitter_std
            history_states = history_states + phys_noise * jit_mask.unsqueeze(-1)
            
        history_latents = self.history_action_encoder(history_states)
        history_latents_clean = self.history_action_encoder(history_states_clean)
        start_latent, cond_latent, gate, cond_gate = self._compose_start_and_cond(
            history_latents=history_latents,
            obs_latents=obs_latents,
            visual_latents=visual_latents,
            proprio_latents=proprio_latents,
            obj_latent=obj_latent,
            alpha=alpha,
            global_centroid=global_centroid,
            centroid_confidence=centroid_confidence,
            where_stats=where_stats,
            prop_raw=_prop_raw_noisy,
        )
        start_latent = self._select_flow_start(start_latent)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        future_action_latents = self.action_encoder(future_actions)

        # ── Detach start_latent: prevent flow loss from cheating through the start ──
        start_latent_detached = start_latent.detach()
        start_latent_for_sampling = (
            start_latent_detached if self.start_detach_for_sampling_losses else start_latent
        )
        if True:
            flow_loss, metrics = self.flow_matcher.compute_loss(
                self.flow_net,
                target=future_action_latents,
                start=start_latent_detached,
                global_cond=cond_latent,
            )
            loss = flow_loss
            metrics["flow_loss"] = flow_loss.item()
            if joint_offset_cross is not None:
                offset_l2 = joint_offset_cross.pow(2).mean() * self.start_offset_l2_weight
                loss = loss + offset_l2
                metrics["offset_l2"] = offset_l2.item()
            if joint_offset is not None:
                offset_l2 = joint_offset.pow(2).mean() * self.start_offset_l2_weight
                loss = loss + offset_l2
                metrics["offset_l2"] = offset_l2.item()
            if joint_offset_cross is not None and _cross_offset_gt is not None:
                offset_align_loss = F.mse_loss(joint_offset_cross, _cross_offset_gt) * self.start_offset_align_weight
                loss = loss + offset_align_loss
                metrics["offset_align_loss"] = offset_align_loss.item()
        else:
            loss = torch.tensor(0.0, device=start_latent.device)
            metrics = {}
            metrics["flow_loss"] = 0.0

        # ── Explicit start correction loss ──
        # Teaches delta to undo history jitter: MSE(start_latent, clean_history).
        # When history is clean (no jitter, gate≈0), this loss is naturally ≈0.
        # When history is jittered, delta must fix it back to clean_history.
        # correction_loss removed: offset_mlp handles correction at raw level. 
        # Old correction_loss gradient would corrupt history_action_encoder.
        metrics["correction_loss"] = 0.0
        metrics["alignment_loss"] = 0.0

        if slot_weights is not None:
            metrics["slot_entropy"] = (
                -(slot_weights * (slot_weights + 1e-8).log()).sum(dim=-1).mean().item()
            )
        else:
            metrics["slot_entropy"] = 0.0
        if centroid_confidence is not None:
            metrics["centroid_conf"] = float(centroid_confidence.mean().item())
        else:
            metrics["centroid_conf"] = 0.0
        if gate is not None:
            metrics["start_gate_val"] = gate.detach().mean().item()  # batch avg gate

        # Mild L2 penalty on proprio_latents to prevent feature inflation → gate drift
        feat_reg = proprio_latents.pow(2).mean() * 1e-5
        loss = loss + feat_reg
        metrics["feat_reg"] = feat_reg.item()

        if self.enc_contrastive_weight > 0:
            image_features = obs_latents.view(batch_size, -1)
            action_features = future_action_latents.view(batch_size, -1)
            contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
            loss += self.enc_contrastive_weight * contrastive_loss
            metrics["enc_contrastive_loss"] = contrastive_loss.item()

        if self.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=start_latent_for_sampling,
                num_steps=self.num_sampling_steps,
                global_cond=cond_latent,
            )

            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, future_action_latents)
                loss += self.consistency_weight * consistency_loss
                metrics["consistency_loss"] = consistency_loss.item()

            if self.flow_contrastive_weight > 0:
                image_features = obs_latents.view(batch_size, -1)
                action_features = action_latents_pred.view(batch_size, -1)
                contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
                loss += self.flow_contrastive_weight * contrastive_loss
                metrics["flow_contrastive_loss"] = contrastive_loss.item()

            if self.action_ae["flow_recon_weight"] > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                action_recon_loss = F.l1_loss(actions_recon, future_actions)
                metrics["flow_action_recon_loss"] = action_recon_loss.item()
                loss += self.action_ae["flow_recon_weight"] * action_recon_loss

            if (
                self.saliency_reg_enable
                and self.saliency_reg_weight > 0
                and ("task_mask" in batch)
            ):
                nobs_pert = self._apply_background_noise(nobs, batch["task_mask"])
                obs_latents_pert, obs_tokens_pert, visual_latents_pert, proprio_latents_pert, _ = self._encode_obs_latents_and_tokens(nobs_pert)
                if need_obj_latent:
                    if self.dino_enable:
                        obj_latent_pert, _, alpha_pert, _, where_stats_pert = self._get_dino_obj_latent(nobs_pert, history_latents)
                        centroid_conf_pert = self._compute_centroid_confidence(alpha_pert)
                    else:
                        obj_latent_pert, _, alpha_pert = self._compute_object_latent(obs_tokens_pert, history_latents)
                        centroid_conf_pert = None
                        where_stats_pert = None
                else:
                    obj_latent_pert, alpha_pert = None, None
                    centroid_conf_pert = None
                    where_stats_pert = None
                start_latent_pert, cond_latent_pert, _, _ = self._compose_start_and_cond(
                    history_latents=history_latents,
                    obs_latents=obs_latents_pert,
                    visual_latents=visual_latents_pert,
                    proprio_latents=proprio_latents_pert,
                    obj_latent=obj_latent_pert,
                    alpha=alpha_pert,
                    global_centroid=None,
                    centroid_confidence=centroid_conf_pert,
                    where_stats=where_stats_pert,
                    prop_raw=_prop_raw,
                )
                start_latent_pert = self._select_flow_start(start_latent_pert)
                start_latent_pert = (
                    start_latent_pert.detach()
                    if self.start_detach_for_sampling_losses
                    else start_latent_pert
                )

                action_latents_pert = self.flow_matcher.sample(
                    self.flow_net,
                    shape=(batch_size, self.latent_dim),
                    device=obs_latents.device,
                    start=start_latent_pert,
                    num_steps=max(1, int(self.saliency_num_steps)),
                    global_cond=cond_latent_pert,
                )
                saliency_bg_loss = F.l1_loss(action_latents_pert, action_latents_pred.detach())
                loss += self.saliency_reg_weight * saliency_bg_loss
                metrics["saliency_bg_loss"] = saliency_bg_loss.item()
        else:
            action_latents_pred = future_action_latents

        if self.action_ae["enc_recon_weight"] > 0:
            actions_recon = self.action_decoder(future_action_latents)
            action_recon_loss = F.l1_loss(actions_recon, future_actions)
            metrics["enc_action_recon_loss"] = action_recon_loss.item()
            loss += self.action_ae["enc_recon_weight"] * action_recon_loss

        metrics["gate_reg"] = 0.0
        metrics["start_gate_sparsity"] = 0.0

        # Prevent slot condition gate from collapsing to zero
        if self.slot_cond_gate is not None and self.slot_cond_enable:
            cond_gate_mean = cond_gate.mean()
            cond_gate_reg = (1.0 - cond_gate_mean) ** 2
            loss += self.gate_reg_weight * cond_gate_reg
            metrics["cond_gate_mean"] = cond_gate_mean.item()
            metrics["cond_gate_reg"] = cond_gate_reg.item()
        else:
            metrics["cond_gate_mean"] = 0.0
            metrics["cond_gate_reg"] = 0.0
        metrics["where_gate_mean"] = float(self._last_where_gate_mean)

        # OBL-style mask supervision for slot branch:
        # align slot-derived object latent with mask-pooled visual object prototype.
        if (
            self.slot_mask_supervise_enable
            and self.slot_mask_supervise_weight > 0
            and need_obj_latent
            and ("task_mask" in batch)
        ):
            obj_latent_mask = self._compute_mask_supervised_obj_latent(obs_tokens, batch["task_mask"])
            if obj_latent_mask is not None:
                slot_mask_loss = F.l1_loss(obj_latent, obj_latent_mask.detach())
                loss += self.slot_mask_supervise_weight * slot_mask_loss
                metrics["slot_mask_loss"] = slot_mask_loss.item()
            else:
                metrics["slot_mask_loss"] = 0.0
        else:
            metrics["slot_mask_loss"] = 0.0

        metrics["total_loss"] = float(loss.detach().item())
        metrics["offset_l2"] = metrics.get("offset_l2", 0.0)
        metrics["offset_ratio"] = (
            float(offset_ratio.mean().item()) if offset_ratio is not None else 0.0
        )
        if self.training and self._debug_step < 5:
            if joint_offset is not None:
                metrics["offset_max_abs"] = joint_offset.abs().max().item()
                metrics["offset_mean_abs"] = joint_offset.abs().mean().item()
        self._last_loss_metrics = {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
        }
        return loss

    def get_last_loss_metrics(self) -> Dict[str, float]:
        return dict(self._last_loss_metrics)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]

        obs_latents, obs_tokens, visual_latents, proprio_latents, _prop_raw = self._encode_obs_latents_and_tokens(nobs)
        history_states = self._get_history_states_for_start(obs_dict, nobs)
        B = history_states.shape[0]

        need_obj_latent = self._needs_object_path()
        if need_obj_latent and self.dino_enable:
            dummy_hist = torch.zeros(B, self.latent_dim, device=history_states.device)
            obj_latent, _, alpha, global_centroid, where_stats = self._get_dino_obj_latent(nobs, dummy_hist)
            centroid_confidence = self._compute_centroid_confidence(alpha)
        elif need_obj_latent:
            temp_hist = self.history_action_encoder(history_states)
            obj_latent, _, alpha = self._compute_object_latent(obs_tokens, temp_hist)
            global_centroid = None
            centroid_confidence = None
            where_stats = None
        else:
            obj_latent, alpha, global_centroid = None, None, None
            centroid_confidence = None
            where_stats = None

        history_states, _, _ = self._apply_start_offset(
            history_states=history_states,
            global_centroid=global_centroid,
            is_eval=True,
            detach_offset=True,
        )

        history_latents = self.history_action_encoder(history_states)
        start_latent, cond_latent, gate, _ = self._compose_start_and_cond(
            history_latents=history_latents,
            obs_latents=obs_latents,
            visual_latents=visual_latents,
            proprio_latents=proprio_latents,
            obj_latent=obj_latent,
            alpha=alpha,
            global_centroid=global_centroid,
            centroid_confidence=centroid_confidence,
            where_stats=where_stats,
            prop_raw=_prop_raw,
        )
        start_latent = self._select_flow_start(start_latent)

        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(batch_size, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=start_latent,
            global_cond=cond_latent,
            return_traces=False,
        )

        with torch.no_grad():
            action_pred = self.action_decoder(action_latents_pred)

        # Log gate every eval step

        real_action_dim = self.normalizer["action"].params_dict["scale"].shape[0]
        if action_pred.shape[-1] > real_action_dim:
            action_pred = action_pred[..., :real_action_dim]
        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action = action_pred[:, : self.n_action_steps]
        return {"action": action, "action_pred": action_pred}

    @torch.no_grad()
    def get_latents_for_visualization(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        nactions = torch.nn.functional.pad(nactions, (0, self.action_dim - nactions.shape[-1])) if nactions.shape[-1] < self.action_dim else nactions

        obs_latents, obs_tokens, visual_latents, proprio_latents, _prop_raw = self._encode_obs_latents_and_tokens(nobs)
        history_states = self._get_history_states_for_start(batch["obs"], nobs)
        B_viz = history_states.shape[0]
        need_obj_latent = self._needs_object_path()
        if need_obj_latent and self.dino_enable:
            dummy_hist = torch.zeros(B_viz, self.latent_dim, device=history_states.device)
            obj_latent, _, alpha, global_centroid, where_stats = self._get_dino_obj_latent(nobs, dummy_hist)
            centroid_confidence = self._compute_centroid_confidence(alpha)
        elif need_obj_latent:
            temp_hist = self.history_action_encoder(history_states)
            obj_latent, _, alpha = self._compute_object_latent(obs_tokens, temp_hist)
            global_centroid = None
            centroid_confidence = None
            where_stats = None
        else:
            obj_latent, alpha, global_centroid = None, None, None
            centroid_confidence = None
            where_stats = None
        history_states, _, _ = self._apply_start_offset(
            history_states=history_states,
            global_centroid=global_centroid,
            is_eval=True,
            detach_offset=True,
        )
        history_latents = self.history_action_encoder(history_states)
        start_latents, _, _, _ = self._compose_start_and_cond(
            history_latents=history_latents,
            obs_latents=obs_latents,
            visual_latents=visual_latents,
            proprio_latents=proprio_latents,
            obj_latent=obj_latent,
            alpha=alpha,
            global_centroid=global_centroid,
            centroid_confidence=centroid_confidence,
            where_stats=where_stats,
            prop_raw=_prop_raw,
        )
        start_latents = self._select_flow_start(start_latents)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)

        return start_latents, future_latents

    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        if num_steps is None:
            num_steps = self.num_sampling_steps

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        nactions = torch.nn.functional.pad(nactions, (0, self.action_dim - nactions.shape[-1])) if nactions.shape[-1] < self.action_dim else nactions
        batch_size = nactions.shape[0]
        n_samples = min(n_samples, batch_size)

        nobs_small = dict_apply(nobs, lambda x: x[:n_samples])
        raw_obs_small = dict_apply(batch["obs"], lambda x: x[:n_samples])
        obs_latents, obs_tokens, visual_latents, proprio_latents, _prop_raw = self._encode_obs_latents_and_tokens(nobs_small)

        history_states = self._get_history_states_for_start(raw_obs_small, nobs_small)
        history_states_clean_viz = history_states.clone()  # save before offset for baseline
        B_flow = history_states.shape[0]
        need_obj_latent = self._needs_object_path()
        if need_obj_latent and self.dino_enable:
            dummy_hist = torch.zeros(B_flow, self.latent_dim, device=history_states.device)
            obj_latent, _, alpha, global_centroid, where_stats = self._get_dino_obj_latent(nobs_small, dummy_hist)
            centroid_confidence = self._compute_centroid_confidence(alpha)
        elif need_obj_latent:
            temp_hist = self.history_action_encoder(history_states)
            obj_latent, _, alpha = self._compute_object_latent(obs_tokens, temp_hist)
            global_centroid = None
            centroid_confidence = None
            where_stats = None
        else:
            obj_latent, alpha, global_centroid = None, None, None
            centroid_confidence = None
            where_stats = None
        history_states, _, _ = self._apply_start_offset(
            history_states=history_states,
            global_centroid=global_centroid,
            is_eval=True,
            detach_offset=True,
        )
        history_latents = self.history_action_encoder(history_states)
        start_latents, cond_latent, gate_viz, _ = self._compose_start_and_cond(
            history_latents=history_latents,
            obs_latents=obs_latents,
            alpha=alpha,
            visual_latents=visual_latents,
            proprio_latents=proprio_latents,
            obj_latent=obj_latent,
            global_centroid=global_centroid,
            centroid_confidence=centroid_confidence,
            where_stats=where_stats,
            prop_raw=_prop_raw,
        )
        start_latents = self._select_flow_start(start_latents)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:n_samples, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)

        # ── Corrected trajectory (with start correction) ──
        _, (traj_history, _) = self.flow_matcher.sample(
            self.flow_net,
            shape=(n_samples, self.latent_dim),
            device=obs_latents.device,
            num_steps=num_steps,
            start=start_latents,
            global_cond=cond_latent,
            return_traces=True,
        )

        # ── Baseline trajectory (gate=0, start=clean_history) ──
        history_latents_clean_viz = self.history_action_encoder(history_states_clean_viz)
        _, (traj_baseline, _) = self.flow_matcher.sample(
            self.flow_net,
            shape=(n_samples, self.latent_dim),
            device=obs_latents.device,
            num_steps=num_steps,
            start=history_latents_clean_viz,
            global_cond=cond_latent,
            return_traces=True,
        )

        traj_history_cpu = []
        for t in traj_history:
            if hasattr(t, "cpu"):
                traj_history_cpu.append(t.cpu())
            else:
                traj_history_cpu.append(torch.tensor(t))

        traj_stacked = torch.stack(traj_history_cpu, dim=0)
        trajectories_corrected = [traj_stacked[:, i, :].numpy() for i in range(n_samples)]

        # Baseline trajectories
        traj_base_cpu = []
        for t in traj_baseline:
            if hasattr(t, "cpu"):
                traj_base_cpu.append(t.cpu())
            else:
                traj_base_cpu.append(torch.tensor(t))
        traj_base_stacked = torch.stack(traj_base_cpu, dim=0)
        trajectories_baseline = [traj_base_stacked[:, i, :].numpy() for i in range(n_samples)]

                # Log eval gate alongside visualizations
        gate_path = os.path.join(self.output_dir if hasattr(self, 'output_dir') else '.', 'eval_gate.log')
        if gate_viz is not None:
            with open(gate_path, 'a') as gf:
                gf.write(f'{gate_viz.mean().item():.4f}\n')
        future_latents_np = future_latents.cpu().numpy()
        return (trajectories_corrected, trajectories_baseline), future_latents_np
