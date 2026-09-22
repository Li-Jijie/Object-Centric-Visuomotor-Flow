#!/usr/bin/env python3
"""
Train slot head on top of a FROZEN DINO backbone with mask supervision.

Design:
- Frozen DINO backbone (ViT recommended)
- Trainable slot head (token_proj + slot queries + slot attention + slot ffn + out)
- Mask-supervised losses (FG/BG KL + mass + leakage)
- Optional DINO pooled feature reconstruction (FG/BG)
- Optional weak/strong consistency regularization

This script is incremental and does not modify existing training pipelines.
"""

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


@dataclass
class SampleRef:
    rgb_path: str
    mask_path: str
    frame_idx: int


def _resize_mask_nearest(mask_2d: np.ndarray, h: int, w: int) -> np.ndarray:
    src_h, src_w = mask_2d.shape[:2]
    if src_h == h and src_w == w:
        return mask_2d
    ys = np.linspace(0, src_h - 1, h).round().astype(np.int32)
    xs = np.linspace(0, src_w - 1, w).round().astype(np.int32)
    return mask_2d[np.ix_(ys, xs)]


class SlotDistillDataset(Dataset):
    def __init__(self, demo_root: str, frame_stride: int = 1, max_demos: int = -1):
        self.refs: List[SampleRef] = []
        self._rgb_cache = {}
        self._mask_cache = {}
        demo_names = [
            d for d in sorted(os.listdir(demo_root))
            if d.startswith("demo_") and os.path.isdir(os.path.join(demo_root, d))
        ]
        if max_demos > 0:
            demo_names = demo_names[:max_demos]

        for dn in tqdm(demo_names, desc="Index dataset"):
            demo_dir = os.path.join(demo_root, dn)
            rgb_path = os.path.join(demo_dir, "rgb.mp4")
            mask_path = os.path.join(demo_dir, "task_mask.npy")
            if not (os.path.isfile(rgb_path) and os.path.isfile(mask_path)):
                continue
            masks = np.load(mask_path)
            t = len(masks)
            for i in range(0, t, frame_stride):
                self.refs.append(SampleRef(rgb_path, mask_path, i))

        if len(self.refs) == 0:
            raise RuntimeError("No training samples found. Check demo_root.")
        print(f"[DATA] samples={len(self.refs)}")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx: int):
        r = self.refs[idx]
        if r.rgb_path not in self._rgb_cache:
            self._rgb_cache[r.rgb_path] = iio.mimread(r.rgb_path)
        if r.mask_path not in self._mask_cache:
            self._mask_cache[r.mask_path] = np.load(r.mask_path)

        rgbs = self._rgb_cache[r.rgb_path]
        masks = self._mask_cache[r.mask_path]

        rgb = np.asarray(rgbs[r.frame_idx], dtype=np.uint8)
        m = np.asarray(masks[r.frame_idx], dtype=np.uint8)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.shape != rgb.shape[:2]:
            m = _resize_mask_nearest(m, rgb.shape[0], rgb.shape[1])
        return rgb, m


class FrozenDinoSlotModel(nn.Module):
    def __init__(
        self,
        dino_model_path: str,
        num_slots: int,
        slot_dim: int,
        slot_heads: int,
        img_size: int,
        device: torch.device,
        teacher_fp16: bool,
        mask_decoder_mode: str = "upsample",
        dino_layer: int = 12,
        two_scale_enable: bool = False,
    ):
        super().__init__()
        self.device_ref = device
        self.teacher_fp16 = bool(teacher_fp16 and device.type == "cuda")
        self.img_size = int(img_size)
        self.mask_decoder_mode = str(mask_decoder_mode)
        self.dino_layer = int(dino_layer)
        self.two_scale_enable = bool(two_scale_enable)

        self.processor = AutoImageProcessor.from_pretrained(dino_model_path, local_files_only=True)
        self.dino = AutoModel.from_pretrained(dino_model_path, local_files_only=True).to(device)
        self.patch_size = int(getattr(self.dino.config, "patch_size", 16))
        if self.teacher_fp16:
            # Keep parameters/bias in a consistent dtype with fp16 inputs.
            self.dino = self.dino.half()
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.img_size, self.img_size)
            dummy_px = self._processor_pixel_values(dummy)
            patch, _ = self._extract_patch_tokens(dummy_px)
        token_dim = int(patch.shape[-1])

        self.token_proj = nn.Linear(token_dim, slot_dim)
        self.slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=slot_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.slot_ffn = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.slot_norm = nn.LayerNorm(slot_dim)
        self.out = nn.Linear(slot_dim, slot_dim)
        self.mask_head = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, 1),
        )
        hr_hidden = max(64, slot_dim // 2)
        self.mask_head_sbd = nn.Sequential(
            nn.Conv2d(slot_dim * 2 + 2, slot_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(slot_dim, hr_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hr_hidden, 1, kernel_size=1),
        )
        # ── Fine head (two-scale) ──
        if self.two_scale_enable:
            self.fine_token_proj = nn.Linear(token_dim, slot_dim)
            self.fine_slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
            self.fine_slot_attn = nn.MultiheadAttention(
                embed_dim=slot_dim, num_heads=slot_heads, dropout=0.0, batch_first=True,
            )
            self.fine_slot_ffn = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.GELU(),
                nn.Linear(slot_dim, slot_dim),
            )
            self.fine_slot_norm = nn.LayerNorm(slot_dim)
            self.fine_out = nn.Linear(slot_dim, slot_dim)
            self.fine_mask_head_sbd = nn.Sequential(
                nn.Conv2d(slot_dim * 2 + 2, slot_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(slot_dim, hr_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hr_hidden, 1, kernel_size=1),
            )
        self.num_slots = num_slots
        self.token_dim = token_dim

    def _slot_forward(self, tokens, queries, attn, ffn, norm, out_proj):
        """Single-pass slot attention: queries attend to tokens."""
        B, N, D = tokens.shape
        q = queries.unsqueeze(0).expand(B, -1, -1)
        slots, attn_w = attn(query=q, key=tokens, value=tokens, need_weights=True, average_attn_weights=False)
        slots = norm(q + slots)
        slots = slots + ffn(slots)
        attn_w = attn_w.mean(dim=1)
        slot_feat = F.normalize(out_proj(slots), dim=-1)
        return slot_feat, attn_w

    @torch.no_grad()
    def _processor_pixel_values(self, rgb_01: torch.Tensor) -> torch.Tensor:
        # Apply official HF image processor after geometric/photometric augmentation.
        rgb = rgb_01.detach().clamp(0.0, 1.0)
        rgb_u8 = (rgb * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        imgs = [im for im in rgb_u8]
        processed = self.processor(
            images=imgs,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        px = processed["pixel_values"]
        if self.teacher_fp16:
            px = px.to(dtype=torch.float16)
        else:
            px = px.to(dtype=torch.float32)
        return px.to(self.device_ref, non_blocking=True)

    @torch.no_grad()
    def _extract_patch_tokens(self, pixel_values: torch.Tensor):
        x = pixel_values
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.teacher_fp16):
            out = self.dino(pixel_values=x, output_hidden_states=True)
            if self.dino_layer < len(out.hidden_states):
                tokens = out.hidden_states[self.dino_layer]
            else:
                tokens = out.last_hidden_state

        exp_h = max(1, int(x.shape[-2]) // self.patch_size)
        exp_w = max(1, int(x.shape[-1]) // self.patch_size)
        exp_patch = exp_h * exp_w
        n = tokens.shape[1]
        if n >= exp_patch:
            # Keep the final HxW spatial tokens; any leading cls/register tokens are dropped.
            patch = tokens[:, n - exp_patch :, :]
            return patch.detach(), exp_h

        if n > 1:
            n_patch = n - 1
            g = int(math.sqrt(n_patch))
            if g * g == n_patch:
                patch = tokens[:, 1:, :]
            else:
                g = int(math.sqrt(n))
                patch = tokens
        else:
            patch = tokens
            g = 1

        if patch.shape[1] != g * g:
            g = int(math.sqrt(patch.shape[1]))
            patch = patch[:, : g * g, :]

        return patch.detach(), g

    def forward(self, rgb_01: torch.Tensor, head: str = "coarse"):
        rgb = F.interpolate(rgb_01, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        pixel_values = self._processor_pixel_values(rgb)
        patch, g = self._extract_patch_tokens(pixel_values)

        if head == "fine" and self.two_scale_enable and hasattr(self, "fine_token_proj"):
            tokens = self.fine_token_proj(patch)
        else:
            tokens = self.token_proj(patch)

        if head == "fine" and self.two_scale_enable:
            slot_feat, attn_w = self._slot_forward(
                tokens, self.fine_slot_queries, self.fine_slot_attn,
                self.fine_slot_ffn, self.fine_slot_norm, self.fine_out,
            )
        else:
            slot_feat, attn_w = self._slot_forward(
                tokens, self.slot_queries, self.slot_attn,
                self.slot_ffn, self.slot_norm, self.out,
            )

        return slot_feat, attn_w, (g, g), patch, tokens

    def _coord_grid(self, bsz: int, h: int, w: int, device: torch.device, dtype: torch.dtype):
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]
        return grid.expand(bsz, -1, -1, -1)

    def decode_mask_logits(self, fg_slot: torch.Tensor, proj_tokens: torch.Tensor, hw, out_hw, head: str = "coarse"):
        bsz = proj_tokens.shape[0]
        h_tok, w_tok = int(hw[0]), int(hw[1])
        out_h, out_w = int(out_hw[0]), int(out_hw[1])

        if self.mask_decoder_mode == "upsample":
            fg_slot_expand = fg_slot.unsqueeze(1).expand(-1, proj_tokens.shape[1], -1)
            dec_in = torch.cat([proj_tokens, fg_slot_expand], dim=-1)
            # For fine head, use fine_mask_head if available
            mask_head = self.fine_mask_head if (head == "fine" and self.two_scale_enable and hasattr(self, "fine_mask_head")) else self.mask_head
            fg_logits_tok = mask_head(dec_in).squeeze(-1).reshape(bsz, 1, h_tok, w_tok)
            fg_logits_hr = F.interpolate(
                fg_logits_tok,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            return fg_logits_hr

        tok_map = proj_tokens.reshape(bsz, h_tok, w_tok, -1).permute(0, 3, 1, 2).contiguous()
        tok_up = F.interpolate(tok_map, size=(out_h, out_w), mode="bilinear", align_corners=False)
        slot_map = fg_slot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out_h, out_w)
        coord = self._coord_grid(bsz, out_h, out_w, tok_up.device, tok_up.dtype)
        dec_in_hr = torch.cat([tok_up, slot_map, coord], dim=1)
        mask_head_sbd = self.fine_mask_head_sbd if (head == "fine" and self.two_scale_enable) else self.mask_head_sbd
        fg_logits_hr = mask_head_sbd(dec_in_hr).squeeze(1)
        return fg_logits_hr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "close_box_recon_s_v1", "close_box_recon_s_bgaug", "pick_cube_recon_s_bgaug", "push_cube_recon_s_bgaug"],
        help="Use built-in hyper-parameter preset to shorten launch command."
             " close_box_recon_s_bgaug: v1 + background randomization for OOD robustness.",
    )
    p.add_argument("--demo-root", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dino-model-path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--max-demos", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=0, help="Cap total training samples (0=no cap).")

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--dino-layer", type=int, default=12, help="Which DINO block to extract tokens from (9=more spatial, 12=more semantic).")
    p.add_argument("--two-scale-enable", action="store_true", help="Enable two-scale: coarse→crop→fine head.")
    p.add_argument("--two-scale-crop-prob", type=float, default=1.0, help="Probability of using cropped sample for fine head.")
    p.add_argument("--two-scale-crop-ratio", type=float, default=0.5, help="Crop size ratio.")
    p.add_argument("--num-slots", type=int, default=2)
    p.add_argument("--slot-dim", type=int, default=256)
    p.add_argument("--slot-heads", type=int, default=4)
    p.add_argument("--slot-assign-mode", type=str, default="dynamic", choices=["fixed", "dynamic"])
    p.add_argument("--fg-slot-index", type=int, default=0)
    p.add_argument("--bg-slot-index", type=int, default=1)

    p.add_argument("--attn-loss-weight", type=float, default=2.0)
    p.add_argument("--bg-loss-weight", type=float, default=1.0)
    p.add_argument("--mass-loss-weight", type=float, default=1.0)
    p.add_argument("--mask-warmup-epochs", type=int, default=3)
    p.add_argument("--mask-threshold", type=float, default=127.5)
    p.add_argument("--attn-supervision", type=str, default="dice", choices=["dice", "kl", "hybrid"])
    p.add_argument("--attn-hybrid-kl-weight", type=float, default=0.2, help="KL weight in hybrid attn supervision.")
    p.add_argument("--dice-eps", type=float, default=1e-6)
    p.add_argument("--slot-entropy-weight", type=float, default=0.0, help="Optional entropy penalty; >0 encourages sharper slot assignment.")
    p.add_argument("--mask-decoder-enable", action="store_true", help="Enable token+slot mask decoder supervision.")
    p.add_argument(
        "--mask-decoder-mode",
        type=str,
        default="sbd_hr",
        choices=["upsample", "sbd_hr"],
        help="upsample: 14x14 logits upsampled to image; sbd_hr: high-res coordinate-aware decoder.",
    )
    p.add_argument("--mask-decoder-weight", type=float, default=1.0, help="Weight of decoder BCE+Dice loss.")
    p.add_argument("--mask-decoder-dice-weight", type=float, default=1.0, help="Relative Dice weight inside decoder loss.")
    p.add_argument("--feat-recon-enable", action="store_true", help="Enable DINO feature reconstruction.")
    p.add_argument(
        "--feat-recon-mode",
        type=str,
        default="cos_fg_bg",
        choices=["cos_fg_bg", "masked_mse_fg"],
        help="cos_fg_bg: original fg/bg cosine on tokens; masked_mse_fg: MSE((pred*mask),(target*mask)).",
    )
    p.add_argument("--feat-recon-fg-weight", type=float, default=0.5)
    p.add_argument("--feat-recon-bg-weight", type=float, default=0.1)
    p.add_argument("--consistency-enable", action="store_true", help="Enable weak/strong RGB consistency.")
    p.add_argument("--consistency-weight", type=float, default=0.2)

    p.add_argument("--augment-prob", type=float, default=0.9)
    p.add_argument("--aug-rot-deg", type=float, default=12.0)
    p.add_argument("--aug-translate", type=float, default=0.10)
    p.add_argument("--aug-hflip-prob", type=float, default=0.5, help="Horizontal flip probability (joint on image+mask).")
    p.add_argument("--aug-scale-min", type=float, default=0.85)
    p.add_argument("--aug-scale-max", type=float, default=1.15)
    p.add_argument("--aug-brightness", type=float, default=0.5)
    p.add_argument("--aug-contrast", type=float, default=0.5)
    p.add_argument("--aug-saturation", type=float, default=0.4)
    p.add_argument("--aug-gamma", type=float, default=0.4)
    p.add_argument("--aug-noise-std", type=float, default=0.03)
    p.add_argument("--aug-blur-prob", type=float, default=0.3)
    p.add_argument("--aug-blur-kernel", type=int, default=3)

    p.add_argument("--bg-randomize-enable", action="store_true", help="Randomize background color to improve OOD robustness.")
    p.add_argument("--bg-randomize-prob", type=float, default=0.5, help="Probability of applying background randomization.")
    p.add_argument(
        "--bg-randomize-mode", type=str, default="solid",
        choices=["solid", "noise", "checkerboard"],
        help="solid: random uniform color; noise: per-pixel Gaussian; checkerboard: random 2-tone grid.",
    )

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--teacher-fp16", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default="")
    p.add_argument("--csv-path", type=str, default="")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--save-best-by", type=str, default="mass", choices=["mass", "loss"])
    p.add_argument("--no-tb", action="store_true")
    p.add_argument("--preview-enable", action="store_true", help="Save one mask prediction preview image during training.")
    p.add_argument("--preview-split", type=str, default="val", choices=["val", "train"], help="Data split used for preview export.")
    p.add_argument("--preview-every-epoch", type=int, default=1, help="Export preview every N epochs.")
    p.add_argument("--preview-index", type=int, default=0, help="Sample index inside the selected batch for preview.")
    p.add_argument("--preview-quantile", type=float, default=0.85, help="Quantile threshold for attn-based preview when mask decoder is disabled.")
    p.add_argument("--preview-dir", type=str, default="", help="Directory to save preview images. Defaults to <log_dir>/preview.")
    args = p.parse_args()
    _apply_preset(args)
    return args


def _flag_provided(flag_name: str) -> bool:
    # support forms: --flag value, --flag=value, and bool flags --flag
    argv = sys.argv[1:]
    return any(a == flag_name or a.startswith(flag_name + "=") for a in argv)


def _apply_preset(args):
    if args.preset == "none":
        return
    if args.preset not in ("close_box_recon_s_v1", "close_box_recon_s_bgaug", "pick_cube_recon_s_bgaug", "push_cube_recon_s_bgaug"):
        raise ValueError(f"Unknown preset: {args.preset}")

    preset = {
        "epochs": 20,
        "batch_size": 8,
        "num_workers": 0,
        "frame_stride": 2,
        "img_size": 224,
        "num_slots": 2,
        "slot_assign_mode": "fixed",
        "fg_slot_index": 0,
        "bg_slot_index": 1,
        "attn_loss_weight": 2.0,
        "bg_loss_weight": 1.0,
        "mass_loss_weight": 1.0,
        "slot_entropy_weight": 0.01,
        "feat_recon_enable": True,
        "feat_recon_mode": "masked_mse_fg",
        "feat_recon_fg_weight": 0.5,
        "feat_recon_bg_weight": 0.1,
        "consistency_enable": True,
        "consistency_weight": 0.2,
        "mask_decoder_enable": True,
        "mask_decoder_mode": "sbd_hr",
        "mask_decoder_weight": 1.0,
        "mask_decoder_dice_weight": 1.0,
        "augment_prob": 0.9,
        "aug_hflip_prob": 0.5,
        "aug_rot_deg": 12.0,
        "aug_translate": 0.10,
        "aug_scale_min": 0.85,
        "aug_scale_max": 1.15,
        "aug_brightness": 0.5,
        "aug_contrast": 0.5,
        "aug_saturation": 0.4,
        "aug_gamma": 0.4,
        "aug_noise_std": 0.03,
        "aug_blur_prob": 0.3,
        "aug_blur_kernel": 3,
        "fp16": True,
        "teacher_fp16": True,
        "save_best": True,
        "preview_enable": True,
        "preview_split": "val",
        "preview_every_epoch": 1,
        "preview_index": 0,
    }

    if (args.preset == "close_box_recon_s_bgaug") or (args.preset == "pick_cube_recon_s_bgaug") or (args.preset == "push_cube_recon_s_bgaug"):
        preset.update({
            "img_size": 448,
            "epochs": 30,
            "lr": 5e-5,
            "weight_decay": 1e-3,
            "frame_stride": 1,
            "dino_layer": 9,
            "aug_rot_deg": 20,
            "aug_translate": 0.25,
            "aug_scale_min": 0.4,
            "aug_scale_max": 1.6,
            "mask_warmup_epochs": 2,
            "mask_decoder_weight": 1.5,
            "mask_decoder_dice_weight": 1.0,
            "attn_loss_weight": 2.0,
            "bg_loss_weight": 0.4,
            "bg_randomize_enable": True,
            "bg_randomize_prob": 0.5,
            "bg_randomize_mode": "solid",
            "two_scale_enable": True,
            "two_scale_crop_prob": 1.0,
            "max_samples": 5000,
        })

    # pick_cube / push_cube: tiny objects (~0.5% FG) need stronger BCE
    if args.preset in ("pick_cube_recon_s_bgaug", "push_cube_recon_s_bgaug"):
        preset.update({
            "mask_decoder_weight": 2.5,
        })

    arg_to_flag = {
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "num_workers": "--num-workers",
        "frame_stride": "--frame-stride",
        "max_samples": "--max-samples",
        "img_size": "--img-size",
        "dino_layer": "--dino-layer",
        "two_scale_enable": "--two-scale-enable",
        "two_scale_crop_prob": "--two-scale-crop-prob",
        "two_scale_crop_ratio": "--two-scale-crop-ratio",
        "lr": "--lr",
        "weight_decay": "--weight-decay",
        "num_slots": "--num-slots",
        "slot_assign_mode": "--slot-assign-mode",
        "fg_slot_index": "--fg-slot-index",
        "bg_slot_index": "--bg-slot-index",
        "attn_loss_weight": "--attn-loss-weight",
        "bg_loss_weight": "--bg-loss-weight",
        "mass_loss_weight": "--mass-loss-weight",
        "slot_entropy_weight": "--slot-entropy-weight",
        "feat_recon_enable": "--feat-recon-enable",
        "feat_recon_mode": "--feat-recon-mode",
        "feat_recon_fg_weight": "--feat-recon-fg-weight",
        "feat_recon_bg_weight": "--feat-recon-bg-weight",
        "consistency_enable": "--consistency-enable",
        "consistency_weight": "--consistency-weight",
        "mask_decoder_enable": "--mask-decoder-enable",
        "mask_decoder_mode": "--mask-decoder-mode",
        "mask_decoder_weight": "--mask-decoder-weight",
        "mask_decoder_dice_weight": "--mask-decoder-dice-weight",
        "mask_warmup_epochs": "--mask-warmup-epochs",
        "augment_prob": "--augment-prob",
        "aug_hflip_prob": "--aug-hflip-prob",
        "aug_rot_deg": "--aug-rot-deg",
        "aug_translate": "--aug-translate",
        "aug_scale_min": "--aug-scale-min",
        "aug_scale_max": "--aug-scale-max",
        "aug_brightness": "--aug-brightness",
        "aug_contrast": "--aug-contrast",
        "aug_saturation": "--aug-saturation",
        "aug_gamma": "--aug-gamma",
        "aug_noise_std": "--aug-noise-std",
        "aug_blur_prob": "--aug-blur-prob",
        "aug_blur_kernel": "--aug-blur-kernel",
        "bg_randomize_enable": "--bg-randomize-enable",
        "bg_randomize_prob": "--bg-randomize-prob",
        "bg_randomize_mode": "--bg-randomize-mode",
        "fp16": "--fp16",
        "teacher_fp16": "--teacher-fp16",
        "save_best": "--save-best",
        "preview_enable": "--preview-enable",
        "preview_split": "--preview-split",
        "preview_every_epoch": "--preview-every-epoch",
        "preview_index": "--preview-index",
    }

    for key, value in preset.items():
        flag = arg_to_flag[key]
        if not _flag_provided(flag):
            setattr(args, key, value)


def _apply_joint_geom_aug(rgb: torch.Tensor, mask: torch.Tensor, args) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.augment_prob <= 0:
        return rgb, mask
    b, _, h, w = rgb.shape
    out_rgb = rgb.clone()
    out_mask = mask.clone()
    for i in range(b):
        if random.random() > args.augment_prob:
            continue
        if args.aug_hflip_prob > 0 and random.random() < args.aug_hflip_prob:
            out_rgb[i] = torch.flip(out_rgb[i], dims=[2])
            out_mask[i] = torch.flip(out_mask[i], dims=[2])
        angle = (2.0 * random.random() - 1.0) * float(args.aug_rot_deg)
        scale = random.uniform(float(args.aug_scale_min), float(args.aug_scale_max))
        max_tx = float(args.aug_translate) * float(w)
        max_ty = float(args.aug_translate) * float(h)
        tx = (2.0 * random.random() - 1.0) * max_tx
        ty = (2.0 * random.random() - 1.0) * max_ty
        translate = [int(round(tx)), int(round(ty))]

        out_rgb[i] = TVF.affine(
            out_rgb[i], angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR, fill=0.0,
        )
        out_mask[i] = TVF.affine(
            out_mask[i], angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST, fill=0.0,
        )
    return out_rgb.clamp(0.0, 1.0), out_mask.clamp(0.0, 1.0)


def _apply_rgb_aug(rgb: torch.Tensor, args) -> torch.Tensor:
    if args.augment_prob <= 0:
        return rgb
    out = rgb.clone()
    blur_k = max(1, int(args.aug_blur_kernel))
    if blur_k % 2 == 0:
        blur_k += 1

    for i in range(out.shape[0]):
        if random.random() > args.augment_prob:
            continue
        x = out[i : i + 1]
        if args.aug_brightness > 0:
            f = 1.0 + (2.0 * random.random() - 1.0) * args.aug_brightness
            x = x * f
        if args.aug_contrast > 0:
            f = 1.0 + (2.0 * random.random() - 1.0) * args.aug_contrast
            xm = x.mean(dim=(2, 3), keepdim=True)
            x = (x - xm) * f + xm
        if args.aug_saturation > 0:
            f = 1.0 + (2.0 * random.random() - 1.0) * args.aug_saturation
            gray = x.mean(dim=1, keepdim=True)
            x = gray + (x - gray) * f
        if args.aug_gamma > 0:
            g = 2.0 ** ((2.0 * random.random() - 1.0) * args.aug_gamma)
            x = torch.clamp(x, 1e-6, 1.0) ** g
        if args.aug_blur_prob > 0 and random.random() < args.aug_blur_prob and blur_k > 1:
            x = F.avg_pool2d(x, kernel_size=blur_k, stride=1, padding=blur_k // 2)
        if args.aug_noise_std > 0:
            x = x + torch.randn_like(x) * args.aug_noise_std
        out[i : i + 1] = x.clamp(0.0, 1.0)
    return out


def _crop_around_centroid(x, cx, cy, crop_guide=None, crop_ratio=0.5):
    """Fixed-ratio crop around centroid.

    Args:
        x: [B, C, H, W] input tensor
        cx, cy: [B] normalized centroids in [0,1]
        crop_guide: [B, N] optional attention map (unused, kept for API compat)
        crop_ratio: float, ratio of image to crop
    Returns:
        cropped and resized tensor [B, C, H, W] (same as input dims)
    """
    B, C, H, W = x.shape
    ch = cw = int(H * crop_ratio)
    cx_px = cx * W
    cy_px = cy * H
    x1 = (cx_px - cw / 2).long().clamp(0, W - cw)
    y1 = (cy_px - ch / 2).long().clamp(0, H - ch)
    crops = []
    for i in range(B):
        crop = x[i:i+1, :, y1[i]:y1[i]+ch, x1[i]:x1[i]+cw]
        crop = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
        crops.append(crop)
    return torch.cat(crops, dim=0)


def _apply_background_randomize(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    prob: float,
    mode: str,
) -> torch.Tensor:
    """
    Randomize background pixels to improve OOD robustness.

    Args:
        rgb:  [B, 3, H, W] in [0, 1]
        mask: [B, 1, H, W] in {0, 1}
        prob: per-sample application probability
        mode: "solid" | "noise" | "checkerboard"
    Returns:
        rgb with background replaced
    """
    if prob <= 0:
        return rgb
    out = rgb.clone()
    bg = 1.0 - mask  # [B, 1, H, W]
    for i in range(out.shape[0]):
        if random.random() > prob:
            continue
        if mode == "solid":
            color = torch.rand(3, 1, 1, device=out.device)
            out[i] = out[i] * mask[i] + color * bg[i]
        elif mode == "noise":
            noise = torch.rand_like(out[i])
            out[i] = out[i] * mask[i] + noise * bg[i]
        elif mode == "checkerboard":
            H, W = out.shape[2], out.shape[3]
            c1 = torch.rand(3, 1, 1, device=out.device)
            c2 = torch.rand(3, 1, 1, device=out.device)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=out.device),
                torch.arange(W, device=out.device),
                indexing="ij",
            )
            check = ((grid_y // 16 + grid_x // 16) % 2).float().view(1, H, W)
            color = c1 * check + c2 * (1.0 - check)
            out[i] = out[i] * mask[i] + color * bg[i]
    return out


def _prep_batch(rgb_u8, mask_u8, img_size, device, mask_threshold, train: bool, args):
    rgb = rgb_u8.permute(0, 3, 1, 2).float() / 255.0
    mask = mask_u8.unsqueeze(1).float()
    rgb = F.interpolate(rgb, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask, size=(img_size, img_size), mode="nearest")

    rgb = rgb.to(device)
    mask = (mask.to(device) > mask_threshold).float()

    if train:
        rgb, mask = _apply_joint_geom_aug(rgb, mask, args)
        if args.bg_randomize_enable:
            rgb = _apply_background_randomize(
                rgb, mask, args.bg_randomize_prob, args.bg_randomize_mode,
            )
        rgb = _apply_rgb_aug(rgb, args)

    return rgb, mask


@torch.no_grad()
def _save_preview_image(
    model: FrozenDinoSlotModel,
    rgb_u8,
    mask_u8,
    args,
    device: torch.device,
    out_path: str,
):
    rgb_u8_t = torch.as_tensor(rgb_u8)
    mask_u8_t = torch.as_tensor(mask_u8)
    rgb_01, mask_01 = _prep_batch(
        rgb_u8_t, mask_u8_t, args.img_size, device, args.mask_threshold, train=False, args=args
    )

    with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=args.fp16):
        slot_feat, attn_w, hw, _patch_tokens, proj_tokens = model(rgb_01)
        h, w = hw
        bsz, nslots, _ = attn_w.shape
        idx = int(max(0, min(args.preview_index, bsz - 1)))

        attn_dist_all = attn_w.reshape(bsz, nslots, -1)
        attn_dist_all = attn_dist_all / attn_dist_all.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        mask_ds = F.interpolate(mask_01, size=(h, w), mode="bilinear", align_corners=False).squeeze(1).clamp(0.0, 1.0)
        bg_mask_ds = (1.0 - mask_ds).clamp(0.0, 1.0)
        mask_flat = mask_ds.reshape(bsz, -1)
        bg_flat = bg_mask_ds.reshape(bsz, -1)
        mass_fg_all = (attn_dist_all * mask_flat.unsqueeze(1)).sum(dim=-1)
        mass_bg_all = (attn_dist_all * bg_flat.unsqueeze(1)).sum(dim=-1)

        if args.slot_assign_mode == "dynamic":
            fg_idx = mass_fg_all.argmax(dim=1)
            if args.num_slots == 2:
                bg_idx = 1 - fg_idx
            else:
                bg_score = mass_bg_all.clone()
                bg_score.scatter_(1, fg_idx.unsqueeze(1), -1e9)
                bg_idx = bg_score.argmax(dim=1)
        else:
            fg_idx = torch.full((bsz,), args.fg_slot_index, device=device, dtype=torch.long)
            bg_idx = torch.full((bsz,), args.bg_slot_index, device=device, dtype=torch.long)

        if args.mask_decoder_enable:
            fg_slot = slot_feat[torch.arange(bsz, device=device), fg_idx, :]
            fg_logits_hr = model.decode_mask_logits(
                fg_slot=fg_slot,
                proj_tokens=proj_tokens,
                hw=(h, w),
                out_hw=(mask_01.shape[-2], mask_01.shape[-1]),
            )
            pred_prob = torch.sigmoid(fg_logits_hr)
            pred_bin = (pred_prob >= 0.5).float()
        else:
            attn_fg = attn_dist_all[torch.arange(bsz, device=device), fg_idx, :].reshape(bsz, 1, h, w)
            attn_up = F.interpolate(attn_fg, size=(mask_01.shape[-2], mask_01.shape[-1]), mode="bilinear", align_corners=False).squeeze(1)
            attn_flat = attn_up.reshape(bsz, -1)
            q = float(np.clip(args.preview_quantile, 0.0, 1.0))
            thr = torch.quantile(attn_flat, q=q, dim=1, keepdim=True)
            pred_prob = attn_up
            pred_bin = (attn_flat >= thr).float().reshape_as(attn_up)

    rgb = (rgb_01[idx].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    gt = (mask_01[idx, 0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    pm = (pred_bin[idx].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    heat = (pred_prob[idx].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)

    gt3 = np.repeat(gt[..., None], 3, axis=2)
    pm3 = np.repeat(pm[..., None], 3, axis=2)
    heat3 = np.zeros_like(rgb)
    heat3[..., 0] = heat
    overlay = (0.65 * rgb.astype(np.float32) + 0.35 * heat3.astype(np.float32)).clip(0, 255).astype(np.uint8)
    coarse_panels = [rgb, gt3, pm3, overlay]
    fine_panels = None

    # ── Two-scale: fine head preview ──
    if args.two_scale_enable and pred_bin[idx].sum() > 0:
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=args.fp16):
            mask_t = torch.as_tensor(pred_bin[idx:idx+1]).float().unsqueeze(0).to(device)
            H_img, W_img = rgb_01.shape[2], rgb_01.shape[3]
            attn_g = args.img_size // 16
            m_ds = F.interpolate(mask_t, size=(attn_g, attn_g), mode="bilinear").squeeze(1)
            total_m = m_ds.sum().clamp(min=1e-6)
            gx = torch.arange(attn_g, device=device).float()
            gy = torch.arange(attn_g, device=device).float()
            gy_m, gx_m = torch.meshgrid(gy, gx, indexing="ij")
            cx = (m_ds * gx_m).sum() / total_m / attn_g
            cy = (m_ds * gy_m).sum() / total_m / attn_g
            ch = cw = int(args.img_size * args.two_scale_crop_ratio)
            x1 = max(0, min(int(cx.item() * W_img) - cw // 2, W_img - cw))
            y1 = max(0, min(int(cy.item() * H_img) - ch // 2, H_img - ch))
            crop = rgb_01[idx:idx+1, :, y1:y1+ch, x1:x1+cw]
            crop = F.interpolate(crop, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
            # Crop GT mask the same way
            crop_gt = mask_01[idx:idx+1, :, y1:y1+ch, x1:x1+cw]
            crop_gt = F.interpolate(crop_gt, size=(args.img_size, args.img_size), mode="nearest")
            fine_out = model(crop, head="fine")
            fine_slot_feat, fine_attn_w = fine_out[0], fine_out[1]
            fine_proj_tokens = fine_out[4]
            fine_h, fine_w = fine_out[2]

            # Fine mask
            fg_slot_f = fine_slot_feat[:, int(args.fg_slot_index), :]
            fg_logits_f = model.decode_mask_logits(
                fg_slot_f, fine_proj_tokens, (fine_h, fine_w),
                (args.img_size, args.img_size), head="fine",
            )
            fine_prob = torch.sigmoid(fg_logits_f).squeeze().cpu().numpy()
            fine_bin = (fine_prob >= 0.5).astype(np.uint8)
            fine_heat = (np.clip(fine_prob, 0, 1) * 255).astype(np.uint8)

            # Crop display
            crop_np = crop.squeeze(0).permute(1, 2, 0).cpu().numpy()
            crop_rgb = (np.clip(crop_np, 0, 1) * 255).astype(np.uint8)
            # Crop GT
            crop_gt_np = crop_gt[idx, 0].cpu().numpy()
            crop_gt3 = np.repeat((crop_gt_np * 255).astype(np.uint8)[..., None], 3, axis=2)
            # Fine pred 3-ch
            fine_bin3 = np.repeat((fine_bin * 255).astype(np.uint8)[..., None], 3, axis=2)
            # Fine overlay
            fine_heat3 = np.zeros_like(crop_rgb)
            fine_heat3[..., 0] = fine_heat
            fine_overlay = (0.65 * crop_rgb.astype(np.float32) + 0.35 * fine_heat3.astype(np.float32)).clip(0, 255).astype(np.uint8)
            fine_panels = [crop_rgb, crop_gt3, fine_bin3, fine_overlay]

    canvas = np.concatenate(coarse_panels, axis=1)
    if fine_panels is not None:
        fine_row = np.concatenate(fine_panels, axis=1)
        canvas = np.concatenate([canvas, fine_row], axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    iio.imwrite(out_path, canvas)


def _dice_loss(prob_flat: torch.Tensor, target_flat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (prob_flat * target_flat).sum(dim=-1)
    denom = prob_flat.sum(dim=-1) + target_flat.sum(dim=-1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def main():
    args = parse_args()
    if args.num_slots < 2:
        raise ValueError("This FG/BG version requires --num-slots >= 2.")
    if args.fg_slot_index == args.bg_slot_index:
        raise ValueError("--fg-slot-index and --bg-slot-index must be different.")
    if max(args.fg_slot_index, args.bg_slot_index) >= args.num_slots:
        raise ValueError("Slot index out of range for --num-slots.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INIT] device={device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(
            "slot_distill/logs",
            os.path.splitext(os.path.basename(args.output))[0],
        )
    os.makedirs(log_dir, exist_ok=True)

    csv_path = args.csv_path if args.csv_path else os.path.join(log_dir, "metrics.csv")
    write_csv_header = not os.path.isfile(csv_path)
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if write_csv_header:
        csv_w.writerow([
            "global_step", "epoch", "split", "loss", "attn_loss", "bg_loss",
            "mass_fg_in_mask", "mass_bg_in_bg", "fg_slot_flip_rate", "slot_entropy",
            "feat_fg_loss", "feat_bg_loss", "cons_loss",
            "mask_dec_bce", "mask_dec_dice", "mask_dec_loss",
            "attn_w", "bg_w",
        ])
        csv_f.flush()

    tb_writer = None
    if (not args.no_tb) and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=log_dir)
    elif (not args.no_tb) and SummaryWriter is None:
        print("[WARN] TensorBoard not available, continue without TB logging.")

    ds = SlotDistillDataset(args.demo_root, args.frame_stride, args.max_demos)
    n_total = len(ds)
    n_val = int(round(n_total * args.val_ratio))
    n_val = max(1, n_val) if (args.val_ratio > 0 and n_total > 1) else 0
    n_train = n_total - n_val

    indices = list(range(n_total))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    if args.max_samples > 0 and len(indices) > args.max_samples:
        indices = indices[:args.max_samples]
        print(f"[DATA] capped samples from {n_total} to {args.max_samples}")
        n_val = int(args.max_samples * args.val_ratio)
        n_train = args.max_samples - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:] if n_val > 0 else []

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if n_val > 0 else None
    print(f"[DATA] split train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False,
        )
    preview_dir = args.preview_dir if args.preview_dir else os.path.join(log_dir, "preview")

    model = FrozenDinoSlotModel(
        dino_model_path=args.dino_model_path,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        slot_heads=args.slot_heads,
        img_size=args.img_size,
        device=device,
        teacher_fp16=args.teacher_fp16,
        mask_decoder_mode=args.mask_decoder_mode,
        dino_layer=args.dino_layer,
        two_scale_enable=args.two_scale_enable,
    ).to(device)

    # ensure frozen backbone
    model.dino.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in trainable_params))
    n_total_params = int(sum(p.numel() for p in model.parameters()))
    print(f"[INIT] backbone=frozen_dino token_dim={model.token_dim} trainable_params={n_trainable}/{n_total_params} ({100.0*n_trainable/max(1,n_total_params):.2f}%)")

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0
    best_loss = float("inf")
    best_mass = float("-inf")
    best_main = float("-inf") if args.save_best_by == "mass" else float("inf")

    def run_one_batch(rgb_u8, mask_u8, train: bool, attn_w_scale: float, bg_w_scale: float):
        rgb_u8_t = torch.as_tensor(rgb_u8)
        mask_u8_t = torch.as_tensor(mask_u8)

        rgb_01, mask_01 = _prep_batch(
            rgb_u8_t, mask_u8_t, args.img_size, device, args.mask_threshold, train=train, args=args,
        )

        if train:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            slot_feat, attn_w, hw, patch_tokens, proj_tokens = model(rgb_01)
            h, w = hw
            bsz, nslots, _ = attn_w.shape

            # ── Two-scale: fine head on GT-centroid crops ──
            fine_mask_dec_loss = torch.zeros((), device=device)
            fine_mask_bce_val = torch.zeros((), device=device)
            fine_mask_dice_val = torch.zeros((), device=device)
            if args.two_scale_enable and train:
                attn_g = args.img_size // 16
                fine_indices = [i for i in range(bsz) if torch.rand(1).item() < args.two_scale_crop_prob]
                if len(fine_indices) > 0:
                    rgb_fine = rgb_01[fine_indices].clone()
                    mask_fine = mask_01[fine_indices].clone()
                    for j, i in enumerate(fine_indices):
                        m_ds = F.interpolate(mask_01[i:i+1], size=(attn_g, attn_g),
                                             mode="bilinear", align_corners=False).squeeze(1)
                        total_m = m_ds.sum().clamp(min=1e-6)
                        gx = torch.arange(attn_g, device=device).float()
                        gy = torch.arange(attn_g, device=device).float()
                        gy_m, gx_m = torch.meshgrid(gy, gx, indexing="ij")
                        cx_i = (m_ds * gx_m).sum() / total_m / attn_g
                        cy_i = (m_ds * gy_m).sum() / total_m / attn_g
                        jitter = (torch.rand(2, device=device) * 2 - 1) * 0.1
                        cx_i = (cx_i + jitter[0]).clamp(0.0, 1.0)
                        cy_i = (cy_i + jitter[1]).clamp(0.0, 1.0)
                        crop_guide = m_ds.reshape(1, -1)
                        rgb_fine[j:j+1] = _crop_around_centroid(
                            rgb_fine[j:j+1], cx_i.unsqueeze(0), cy_i.unsqueeze(0), crop_guide)
                        mask_fine[j:j+1] = _crop_around_centroid(
                            mask_fine[j:j+1], cx_i.unsqueeze(0), cy_i.unsqueeze(0), crop_guide)
                    fine_slot_feat, fine_attn_w, fine_hw, _, fine_proj_tokens = model(rgb_fine, head="fine")
                    fine_h, fine_w = fine_hw
                    fg_slot_f = fine_slot_feat[:, int(args.fg_slot_index), :]
                    fg_logits_f = model.decode_mask_logits(
                        fg_slot_f, fine_proj_tokens, (fine_h, fine_w),
                        (mask_fine.shape[-2], mask_fine.shape[-1]),
                        head="fine",
                    )
                    fg_prob_f = torch.sigmoid(fg_logits_f)
                    mask_target_f = mask_fine.squeeze(1)
                    fine_bce = F.binary_cross_entropy_with_logits(fg_logits_f, mask_target_f)
                    fine_dice = _dice_loss(
                        fg_prob_f.reshape(len(fine_indices), -1),
                        mask_target_f.reshape(len(fine_indices), -1),
                        eps=float(args.dice_eps),
                    )
                    fine_mask_dec_loss = fine_bce + float(args.mask_decoder_dice_weight) * fine_dice
                    fine_mask_bce_val = fine_bce.detach()
                    fine_mask_dice_val = fine_dice.detach()

            attn_dist_all = attn_w.reshape(bsz, nslots, -1)
            attn_dist_all = attn_dist_all / attn_dist_all.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            mask_ds = F.interpolate(mask_01, size=(h, w), mode="bilinear", align_corners=False).squeeze(1).clamp(0.0, 1.0)
            bg_mask_ds = (1.0 - mask_ds).clamp(0.0, 1.0)
            mask_dist = mask_ds.reshape(mask_ds.shape[0], -1)
            mask_dist = mask_dist / mask_dist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            bg_dist = bg_mask_ds.reshape(bg_mask_ds.shape[0], -1)
            bg_dist = bg_dist / bg_dist.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            mask_flat = mask_ds.reshape(mask_ds.shape[0], -1)
            bg_flat = bg_mask_ds.reshape(bg_mask_ds.shape[0], -1)
            mass_fg_all = (attn_dist_all * mask_flat.unsqueeze(1)).sum(dim=-1)
            mass_bg_all = (attn_dist_all * bg_flat.unsqueeze(1)).sum(dim=-1)

            if args.slot_assign_mode == "dynamic":
                fg_idx = mass_fg_all.argmax(dim=1)
                if args.num_slots == 2:
                    bg_idx = 1 - fg_idx
                else:
                    bg_score = mass_bg_all.clone()
                    bg_score.scatter_(1, fg_idx.unsqueeze(1), -1e9)
                    bg_idx = bg_score.argmax(dim=1)
            else:
                fg_idx = torch.full((bsz,), args.fg_slot_index, device=device, dtype=torch.long)
                bg_idx = torch.full((bsz,), args.bg_slot_index, device=device, dtype=torch.long)

            bidx = torch.arange(bsz, device=device, dtype=torch.long)
            fg_slot = slot_feat[bidx, fg_idx, :]
            bg_slot = slot_feat[bidx, bg_idx, :]
            attn_fg_dist = attn_dist_all[bidx, fg_idx, :]
            attn_bg_dist = attn_dist_all[bidx, bg_idx, :]

            attn_fg_kl = F.kl_div(torch.log(attn_fg_dist.clamp(min=1e-8)), mask_dist, reduction="batchmean")
            attn_bg_kl = F.kl_div(torch.log(attn_bg_dist.clamp(min=1e-8)), bg_dist, reduction="batchmean")
            # Dice on attention maps encourages region overlap instead of a sharp single hotspot.
            attn_fg_prob = attn_fg_dist / attn_fg_dist.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            attn_bg_prob = attn_bg_dist / attn_bg_dist.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            attn_fg_dice = _dice_loss(attn_fg_prob, mask_flat, eps=float(args.dice_eps))
            attn_bg_dice = _dice_loss(attn_bg_prob, bg_flat, eps=float(args.dice_eps))
            if args.attn_supervision == "kl":
                attn_fg_loss, attn_bg_loss = attn_fg_kl, attn_bg_kl
            elif args.attn_supervision == "hybrid":
                w_kl = float(max(0.0, min(1.0, args.attn_hybrid_kl_weight)))
                w_dice = 1.0 - w_kl
                attn_fg_loss = w_dice * attn_fg_dice + w_kl * attn_fg_kl
                attn_bg_loss = w_dice * attn_bg_dice + w_kl * attn_bg_kl
            else:
                attn_fg_loss, attn_bg_loss = attn_fg_dice, attn_bg_dice
            attn_loss = 0.5 * (attn_fg_loss + attn_bg_loss)

            mass_fg_in_mask = (attn_fg_dist * mask_flat).sum(dim=-1).clamp(min=0.0, max=1.0)
            mass_bg_in_bg = (attn_bg_dist * bg_flat).sum(dim=-1).clamp(min=0.0, max=1.0)
            leakage_fg_to_bg = (attn_fg_dist * bg_flat).sum(dim=-1)
            leakage_bg_to_fg = (attn_bg_dist * mask_flat).sum(dim=-1)
            bg_loss = 0.5 * (leakage_fg_to_bg.mean() + leakage_bg_to_fg.mean())
            mass_loss = 0.5 * ((1.0 - mass_fg_in_mask).mean() + (1.0 - mass_bg_in_bg).mean())

            entropy = -(attn_dist_all * torch.log(attn_dist_all.clamp(min=1e-8))).sum(dim=-1).mean()
            flip_rate = (fg_idx != int(args.fg_slot_index)).float().mean()
            feat_fg_loss = torch.zeros((), device=device)
            feat_bg_loss = torch.zeros((), device=device)
            cons_loss = torch.zeros((), device=device)
            mask_dec_bce = torch.zeros((), device=device)
            mask_dec_dice = torch.zeros((), device=device)
            mask_dec_loss = torch.zeros((), device=device)

            if args.mask_decoder_enable:
                # Decode directly at image resolution (SBD-style) or legacy upsample mode.
                fg_logits_hr = model.decode_mask_logits(
                    fg_slot=fg_slot,
                    proj_tokens=proj_tokens,
                    hw=(h, w),
                    out_hw=(mask_01.shape[-2], mask_01.shape[-1]),
                )  # [B, H_img, W_img]
                fg_prob_hr = torch.sigmoid(fg_logits_hr)
                mask_target_hr = mask_01.squeeze(1)  # [B, H_img, W_img], binary
                mask_dec_bce = F.binary_cross_entropy_with_logits(fg_logits_hr, mask_target_hr)
                mask_dec_dice = _dice_loss(
                    fg_prob_hr.reshape(bsz, -1),
                    mask_target_hr.reshape(bsz, -1),
                    eps=float(args.dice_eps),
                )
                mask_dec_loss = mask_dec_bce + float(args.mask_decoder_dice_weight) * mask_dec_dice

            if args.feat_recon_enable:
                # Token-level reconstruction (spatially-aware):
                # use fg/bg slot to reconstruct each token embedding, then match DINO-projected tokens.
                target_tokens_raw = proj_tokens.detach()  # [B,HW,D]
                fg_slot_exp = fg_slot.unsqueeze(1).expand(-1, target_tokens_raw.shape[1], -1)
                bg_slot_exp = bg_slot.unsqueeze(1).expand(-1, target_tokens_raw.shape[1], -1)
                w_fg = attn_fg_dist.unsqueeze(-1)
                w_bg = attn_bg_dist.unsqueeze(-1)
                pred_tokens_raw = w_fg * fg_slot_exp + w_bg * bg_slot_exp

                if args.feat_recon_mode == "masked_mse_fg":
                    # Core masked reconstruction:
                    # Loss = MSE((Recon_Tokens * Mask), (DINO_Tokens * Mask))
                    m = mask_flat.unsqueeze(-1)
                    feat_fg_loss = F.mse_loss(pred_tokens_raw * m, target_tokens_raw * m)
                    feat_bg_loss = torch.zeros((), device=device)
                else:
                    target_tokens = F.normalize(target_tokens_raw, dim=-1)
                    pred_tokens = F.normalize(pred_tokens_raw, dim=-1)
                    token_cos = 1.0 - F.cosine_similarity(pred_tokens, target_tokens, dim=-1)  # [B,HW]
                    fg_norm = mask_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    bg_norm = bg_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    feat_fg_loss = ((token_cos * mask_flat).sum(dim=-1, keepdim=True) / fg_norm).mean()
                    feat_bg_loss = ((token_cos * bg_flat).sum(dim=-1, keepdim=True) / bg_norm).mean()

            if args.consistency_enable and train:
                rgb_strong = _apply_rgb_aug(rgb_01.clone(), args)
                slot_feat_s, attn_w_s, _hw_s, _patch_s, _proj_s = model(rgb_strong)
                attn_dist_s = attn_w_s.reshape(bsz, nslots, -1)
                attn_dist_s = attn_dist_s / attn_dist_s.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                fg_slot_s = slot_feat_s[bidx, fg_idx, :]
                attn_fg_s = attn_dist_s[bidx, fg_idx, :]
                cons_feat = F.l1_loss(fg_slot_s, fg_slot.detach())
                cons_attn = F.kl_div(
                    torch.log(attn_fg_s.clamp(min=1e-8)),
                    attn_fg_dist.detach(),
                    reduction="batchmean",
                )
                cons_loss = cons_feat + cons_attn

            loss = (
                attn_w_scale * attn_loss
                + bg_w_scale * bg_loss
                + args.mass_loss_weight * mass_loss
                + args.slot_entropy_weight * entropy
                + args.feat_recon_fg_weight * feat_fg_loss
                + args.feat_recon_bg_weight * feat_bg_loss
                + args.consistency_weight * cons_loss
                + args.mask_decoder_weight * mask_dec_loss
                + args.mask_decoder_weight * fine_mask_dec_loss
            )

        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        return (
            float(loss.item()),
            float(attn_loss.item()),
            float(bg_loss.item()),
            float(mass_fg_in_mask.mean().item()),
            float(mass_bg_in_bg.mean().item()),
            float(flip_rate.item()),
            float(entropy.item()),
            float(feat_fg_loss.item()),
            float(feat_bg_loss.item()),
            float(cons_loss.item()),
            float(mask_dec_bce.item()),
            float(mask_dec_dice.item()),
            float(mask_dec_loss.item()),
            int(rgb_01.shape[0]),
        )

    for ep in range(1, args.epochs + 1):
        model.train()
        stats = {
            "loss": 0.0, "attn": 0.0, "bg": 0.0, "mfg": 0.0, "mbg": 0.0, "flip": 0.0,
            "entropy": 0.0, "feat_fg": 0.0, "feat_bg": 0.0, "cons": 0.0,
            "mask_bce": 0.0, "mask_dice": 0.0, "mask_loss": 0.0, "n": 0
        }

        warm = min(1.0, float(ep) / float(args.mask_warmup_epochs)) if args.mask_warmup_epochs > 0 else 1.0
        attn_w_scale = args.attn_loss_weight * warm
        bg_w_scale = args.bg_loss_weight * warm

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for rgb_u8, mask_u8 in pbar:
            loss_v, attn_v, bg_v, mfg_v, mbg_v, flip_v, entropy_v, feat_fg_v, feat_bg_v, cons_v, mask_bce_v, mask_dice_v, mask_loss_v, b = run_one_batch(
                rgb_u8, mask_u8, train=True, attn_w_scale=attn_w_scale, bg_w_scale=bg_w_scale
            )
            stats["loss"] += loss_v * b
            stats["attn"] += attn_v * b
            stats["bg"] += bg_v * b
            stats["mfg"] += mfg_v * b
            stats["mbg"] += mbg_v * b
            stats["flip"] += flip_v * b
            stats["entropy"] += entropy_v * b
            stats["feat_fg"] += feat_fg_v * b
            stats["feat_bg"] += feat_bg_v * b
            stats["cons"] += cons_v * b
            stats["mask_bce"] += mask_bce_v * b
            stats["mask_dice"] += mask_dice_v * b
            stats["mask_loss"] += mask_loss_v * b
            stats["n"] += b
            global_step += 1

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_step", loss_v, global_step)
                tb_writer.add_scalar("train/attn_step", attn_v, global_step)
                tb_writer.add_scalar("train/bg_step", bg_v, global_step)
                tb_writer.add_scalar("train/mass_fg_step", mfg_v, global_step)
                tb_writer.add_scalar("train/mass_bg_step", mbg_v, global_step)
                tb_writer.add_scalar("train/flip_step", flip_v, global_step)
                tb_writer.add_scalar("train/entropy_step", entropy_v, global_step)
                tb_writer.add_scalar("train/feat_fg_step", feat_fg_v, global_step)
                tb_writer.add_scalar("train/feat_bg_step", feat_bg_v, global_step)
                tb_writer.add_scalar("train/cons_step", cons_v, global_step)
                tb_writer.add_scalar("train/mask_bce_step", mask_bce_v, global_step)
                tb_writer.add_scalar("train/mask_dice_step", mask_dice_v, global_step)
                tb_writer.add_scalar("train/mask_loss_step", mask_loss_v, global_step)

            if global_step % max(1, args.log_interval) == 0:
                csv_w.writerow([
                    global_step, ep, "train_step", loss_v, attn_v, bg_v,
                    mfg_v, mbg_v, flip_v, entropy_v, feat_fg_v, feat_bg_v, cons_v, mask_bce_v, mask_dice_v, mask_loss_v, attn_w_scale, bg_w_scale,
                ])
                csv_f.flush()

            pbar.set_postfix(
                loss=f"{stats['loss']/max(1,stats['n']):.4f}",
                attn=f"{stats['attn']/max(1,stats['n']):.4f}",
                bg=f"{stats['bg']/max(1,stats['n']):.4f}",
                mfg=f"{stats['mfg']/max(1,stats['n']):.4f}",
                mbg=f"{stats['mbg']/max(1,stats['n']):.4f}",
                flip=f"{stats['flip']/max(1,stats['n']):.3f}",
                ent=f"{stats['entropy']/max(1,stats['n']):.3f}",
            )

        tr_loss = stats["loss"] / max(1, stats["n"])
        tr_attn = stats["attn"] / max(1, stats["n"])
        tr_bg = stats["bg"] / max(1, stats["n"])
        tr_mfg = stats["mfg"] / max(1, stats["n"])
        tr_mbg = stats["mbg"] / max(1, stats["n"])
        tr_flip = stats["flip"] / max(1, stats["n"])
        tr_entropy = stats["entropy"] / max(1, stats["n"])
        tr_feat_fg = stats["feat_fg"] / max(1, stats["n"])
        tr_feat_bg = stats["feat_bg"] / max(1, stats["n"])
        tr_cons = stats["cons"] / max(1, stats["n"])
        tr_mask_bce = stats["mask_bce"] / max(1, stats["n"])
        tr_mask_dice = stats["mask_dice"] / max(1, stats["n"])
        tr_mask_loss = stats["mask_loss"] / max(1, stats["n"])

        val_loss = val_attn = val_bg = val_mfg = val_mbg = val_flip = val_entropy = val_feat_fg = val_feat_bg = val_cons = float("nan")
        val_mask_bce = val_mask_dice = val_mask_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vs = {
                "loss": 0.0, "attn": 0.0, "bg": 0.0, "mfg": 0.0, "mbg": 0.0, "flip": 0.0,
                "entropy": 0.0, "feat_fg": 0.0, "feat_bg": 0.0, "cons": 0.0,
                "mask_bce": 0.0, "mask_dice": 0.0, "mask_loss": 0.0, "n": 0
            }
            with torch.no_grad():
                for rgb_u8, mask_u8 in tqdm(val_loader, desc=f"Val {ep}/{args.epochs}", leave=False):
                    loss_v, attn_v, bg_v, mfg_v, mbg_v, flip_v, entropy_v, feat_fg_v, feat_bg_v, cons_v, mask_bce_v, mask_dice_v, mask_loss_v, b = run_one_batch(
                        rgb_u8, mask_u8, train=False, attn_w_scale=attn_w_scale, bg_w_scale=bg_w_scale
                    )
                    vs["loss"] += loss_v * b
                    vs["attn"] += attn_v * b
                    vs["bg"] += bg_v * b
                    vs["mfg"] += mfg_v * b
                    vs["mbg"] += mbg_v * b
                    vs["flip"] += flip_v * b
                    vs["entropy"] += entropy_v * b
                    vs["feat_fg"] += feat_fg_v * b
                    vs["feat_bg"] += feat_bg_v * b
                    vs["cons"] += cons_v * b
                    vs["mask_bce"] += mask_bce_v * b
                    vs["mask_dice"] += mask_dice_v * b
                    vs["mask_loss"] += mask_loss_v * b
                    vs["n"] += b
            val_loss = vs["loss"] / max(1, vs["n"])
            val_attn = vs["attn"] / max(1, vs["n"])
            val_bg = vs["bg"] / max(1, vs["n"])
            val_mfg = vs["mfg"] / max(1, vs["n"])
            val_mbg = vs["mbg"] / max(1, vs["n"])
            val_flip = vs["flip"] / max(1, vs["n"])
            val_entropy = vs["entropy"] / max(1, vs["n"])
            val_feat_fg = vs["feat_fg"] / max(1, vs["n"])
            val_feat_bg = vs["feat_bg"] / max(1, vs["n"])
            val_cons = vs["cons"] / max(1, vs["n"])
            val_mask_bce = vs["mask_bce"] / max(1, vs["n"])
            val_mask_dice = vs["mask_dice"] / max(1, vs["n"])
            val_mask_loss = vs["mask_loss"] / max(1, vs["n"])

        csv_w.writerow([global_step, ep, "train_epoch", tr_loss, tr_attn, tr_bg, tr_mfg, tr_mbg, tr_flip, tr_entropy, tr_feat_fg, tr_feat_bg, tr_cons, tr_mask_bce, tr_mask_dice, tr_mask_loss, attn_w_scale, bg_w_scale])
        csv_w.writerow([global_step, ep, "val_epoch", val_loss, val_attn, val_bg, val_mfg, val_mbg, val_flip, val_entropy, val_feat_fg, val_feat_bg, val_cons, val_mask_bce, val_mask_dice, val_mask_loss, attn_w_scale, bg_w_scale])
        csv_f.flush()

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_epoch", tr_loss, ep)
            tb_writer.add_scalar("train/attn_epoch", tr_attn, ep)
            tb_writer.add_scalar("train/bg_epoch", tr_bg, ep)
            tb_writer.add_scalar("train/mass_fg_epoch", tr_mfg, ep)
            tb_writer.add_scalar("train/mass_bg_epoch", tr_mbg, ep)
            tb_writer.add_scalar("train/flip_epoch", tr_flip, ep)
            tb_writer.add_scalar("train/entropy_epoch", tr_entropy, ep)
            tb_writer.add_scalar("train/feat_fg_epoch", tr_feat_fg, ep)
            tb_writer.add_scalar("train/feat_bg_epoch", tr_feat_bg, ep)
            tb_writer.add_scalar("train/cons_epoch", tr_cons, ep)
            tb_writer.add_scalar("train/mask_bce_epoch", tr_mask_bce, ep)
            tb_writer.add_scalar("train/mask_dice_epoch", tr_mask_dice, ep)
            tb_writer.add_scalar("train/mask_loss_epoch", tr_mask_loss, ep)
            tb_writer.add_scalar("train/attn_weight", attn_w_scale, ep)
            tb_writer.add_scalar("train/bg_weight", bg_w_scale, ep)
            if not math.isnan(val_loss):
                tb_writer.add_scalar("val/loss_epoch", val_loss, ep)
                tb_writer.add_scalar("val/attn_epoch", val_attn, ep)
                tb_writer.add_scalar("val/bg_epoch", val_bg, ep)
                tb_writer.add_scalar("val/mass_fg_epoch", val_mfg, ep)
                tb_writer.add_scalar("val/mass_bg_epoch", val_mbg, ep)
                tb_writer.add_scalar("val/flip_epoch", val_flip, ep)
                tb_writer.add_scalar("val/entropy_epoch", val_entropy, ep)
                tb_writer.add_scalar("val/feat_fg_epoch", val_feat_fg, ep)
                tb_writer.add_scalar("val/feat_bg_epoch", val_feat_bg, ep)
                tb_writer.add_scalar("val/cons_epoch", val_cons, ep)
                tb_writer.add_scalar("val/mask_bce_epoch", val_mask_bce, ep)
                tb_writer.add_scalar("val/mask_dice_epoch", val_mask_dice, ep)
                tb_writer.add_scalar("val/mask_loss_epoch", val_mask_loss, ep)

        print(
            f"[EPOCH {ep}] train_loss={tr_loss:.4f} train_attn={tr_attn:.4f} train_bg={tr_bg:.4f} "
            f"train_mfg={tr_mfg:.4f} train_mbg={tr_mbg:.4f} train_flip={tr_flip:.3f} train_entropy={tr_entropy:.4f} "
            f"train_feat_fg={tr_feat_fg:.4f} train_feat_bg={tr_feat_bg:.4f} train_cons={tr_cons:.4f} "
            f"train_mask_bce={tr_mask_bce:.4f} train_mask_dice={tr_mask_dice:.4f} "
            f"val_loss={val_loss:.4f} val_attn={val_attn:.4f} val_bg={val_bg:.4f} val_mfg={val_mfg:.4f} val_mbg={val_mbg:.4f} val_flip={val_flip:.3f} val_entropy={val_entropy:.4f} "
            f"val_feat_fg={val_feat_fg:.4f} val_feat_bg={val_feat_bg:.4f} val_cons={val_cons:.4f} "
            f"val_mask_bce={val_mask_bce:.4f} val_mask_dice={val_mask_dice:.4f}"
        )

        if args.preview_enable and (ep % max(1, args.preview_every_epoch) == 0):
            src_loader = val_loader if args.preview_split == "val" else train_loader
            if src_loader is None:
                src_loader = train_loader
            try:
                rgb_u8_prev, mask_u8_prev = next(iter(src_loader))
                model.eval()
                out_preview = os.path.join(preview_dir, f"epoch_{ep:04d}.png")
                _save_preview_image(
                    model=model,
                    rgb_u8=rgb_u8_prev,
                    mask_u8=mask_u8_prev,
                    args=args,
                    device=device,
                    out_path=out_preview,
                )
                print(f"[PREVIEW] saved {out_preview}")
            except Exception as e:
                print(f"[PREVIEW][WARN] failed at epoch {ep}: {e}")

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "num_slots": args.num_slots,
            "slot_dim": args.slot_dim,
            "slot_heads": args.slot_heads,
            "img_size": args.img_size,
            "token_dim": model.token_dim,
            "global_step": global_step,
            "epoch": ep,
            "train_loss_epoch": tr_loss,
            "val_loss_epoch": val_loss,
            "train_mass_fg_epoch": tr_mfg,
            "val_mass_fg_epoch": val_mfg,
            "train_mass_bg_epoch": tr_mbg,
            "val_mass_bg_epoch": val_mbg,
            "train_flip_epoch": tr_flip,
            "val_flip_epoch": val_flip,
            "backbone_type": "dino_frozen",
            "dino_model_path": args.dino_model_path,
            "dino_layer": args.dino_layer,
            "two_scale_enable": args.two_scale_enable,
            "args": vars(args),
        }
        out_ep = args.output.replace(".pt", f".ep{ep}.pt")
        torch.save(ckpt, out_ep)
        print(f"[SAVE] {out_ep}")

        if args.save_best:
            metric_loss = val_loss if not math.isnan(val_loss) else tr_loss
            metric_mass = val_mfg if not math.isnan(val_mfg) else tr_mfg

            if metric_loss < best_loss:
                best_loss = metric_loss
                out_best_loss = args.output.replace(".pt", ".best_loss.pt")
                torch.save(ckpt, out_best_loss)
                print(f"[BEST-LOSS] {out_best_loss} loss={best_loss:.6f}")

            if metric_mass > best_mass:
                best_mass = metric_mass
                out_best_mass = args.output.replace(".pt", ".best_mass.pt")
                torch.save(ckpt, out_best_mass)
                print(f"[BEST-MASS] {out_best_mass} mass={best_mass:.6f}")

            if args.save_best_by == "mass":
                main_metric = metric_mass
                better_main = main_metric > best_main
            else:
                main_metric = metric_loss
                better_main = main_metric < best_main

            if better_main:
                best_main = main_metric
                out_best = args.output.replace(".pt", ".best.pt")
                torch.save(ckpt, out_best)
                print(f"[BEST] {out_best} {args.save_best_by}={best_main:.6f}")

    csv_f.close()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
