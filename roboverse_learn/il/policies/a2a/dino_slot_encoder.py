"""
DINOv3 + Slot Attention encoder for A2A Slot Policy.

Loads the frozen DINOv3 backbone and pre-trained slot attention module
from the slot distillation checkpoint. Produces slot features for policy conditioning.

Architecture (from train_slot_on_frozen_dino.py):
    Image → Frozen DINOv3 ViT-S → patch tokens (384-dim)
        → token_proj (384→256) → slot_attn → slot features (256-dim)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoSlotEncoder(nn.Module):
    """
    Frozen DINOv3 backbone + pre-trained slot attention.

    Processes a batch of images and extracts slot features.
    Supports temporal sequences: input [B, T, 3, H, W] → output [B, T, num_slots, slot_dim].
    """

    def __init__(
        self,
        dino_model_path: str,
        slot_ckpt_path: str,
        num_slots: int = 2,
        slot_dim: int = 256,
        slot_heads: int = 4,
        img_size: int = 224,
        freeze_dino: bool = True,
        freeze_slot: bool = True,
        dino_layer: int = 12,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.img_size = img_size
        self.dino_layer = int(dino_layer)

        # Dual slot heads (coarse + fine for two-scale)
        self.has_fine_head = False

        # ── Load DINOv3 backbone ──
        import json, os
        from transformers import AutoModel
        from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel
        try:
            self.dino = AutoModel.from_pretrained(dino_model_path, local_files_only=True)
        except Exception:
            # Fallback for newer huggingface_hub that rejects local paths
            config_path = os.path.join(dino_model_path, "config.json")
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
            config = DINOv3ViTConfig(**cfg_dict)
            self.dino = DINOv3ViTModel(config)
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(dino_model_path, "model.safetensors"))
            self.dino.load_state_dict(state_dict, strict=False)
        if freeze_dino:
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.hidden_size = self.dino.config.hidden_size  # 384
        patch_size = self.dino.config.patch_size  # 16
        self.num_patches = (img_size // patch_size) ** 2  # 196

        # ── Slot attention modules ──
        self.token_proj = nn.Linear(self.hidden_size, slot_dim)    # 384 → 256
        self.fine_token_proj = nn.Linear(self.hidden_size, slot_dim)  # separate for fine head
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
        self.out = nn.Linear(slot_dim, slot_dim)  # 256 → 256

        # ── Fine slot head (for two-scale crop → fine) ──
        self.fine_slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.fine_slot_attn = nn.MultiheadAttention(
            embed_dim=slot_dim, num_heads=slot_heads, dropout=0.0, batch_first=True)
        self.fine_slot_ffn = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim),
            nn.GELU(), nn.Linear(slot_dim, slot_dim))
        self.fine_slot_norm = nn.LayerNorm(slot_dim)
        self.fine_out = nn.Linear(slot_dim, slot_dim)

        # ── Mask decoder (for two-scale crop guidance) ──
        self.has_mask_decoder = False

        # ── Load DINOv3 image processor for exact preprocessing ──
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(
            dino_model_path, local_files_only=True
        )
        # Extract normalization parameters from the processor config
        # HF formula: pixel = (image * rescale_factor - mean) / std
        # For [0,1] input with rescale_factor=1/255, this is equivalent to
        # (image * 1/255 - mean) / std = (image - mean * 255) / (std * 255)
        proc_cfg = self.processor
        rescale = float(getattr(proc_cfg, "rescale_factor", 1.0 / 255.0))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if hasattr(proc_cfg, "image_mean"):
            mean = [float(v) for v in proc_cfg.image_mean]
        if hasattr(proc_cfg, "image_std"):
            std = [float(v) for v in proc_cfg.image_std]
        # Pre-compute mean and std in [0,1] input space
        self.register_buffer(
            "norm_mean",
            torch.tensor([m / rescale for m in mean], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "norm_std",
            torch.tensor([s / rescale for s in std], dtype=torch.float32).view(1, 3, 1, 1),
        )

        # ── Load slot weights from distill checkpoint ──
        if slot_ckpt_path:
            self._load_slot_weights(slot_ckpt_path)

        # ── Freeze slot modules ──
        if freeze_slot:
            self._set_slot_requires_grad(False)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _load_slot_weights(self, ckpt_path: str):
        payload = torch.load(ckpt_path, map_location="cpu")
        state = payload.get("state_dict", payload)
        if not isinstance(state, dict):
            raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

        # Read training hyperparams that affect inference
        train_args = payload.get("args", {})
        ckpt_dino_layer = train_args.get("dino_layer", None)
        if ckpt_dino_layer is not None and ckpt_dino_layer != self.dino_layer:
            print(f"[DinoSlotEncoder] overriding dino_layer {self.dino_layer}→{ckpt_dino_layer} (from checkpoint)")
            self.dino_layer = int(ckpt_dino_layer)

        slot_map = {
            "token_proj.weight": "token_proj.weight",
            "token_proj.bias": "token_proj.bias",
            "fine_token_proj.weight": "fine_token_proj.weight",
            "fine_token_proj.bias": "fine_token_proj.bias",
            "slot_queries": "slot_queries",
            "slot_attn.in_proj_weight": "slot_attn.in_proj_weight",
            "slot_attn.in_proj_bias": "slot_attn.in_proj_bias",
            "slot_attn.out_proj.weight": "slot_attn.out_proj.weight",
            "slot_attn.out_proj.bias": "slot_attn.out_proj.bias",
            "slot_ffn.0.weight": "slot_ffn.0.weight",
            "slot_ffn.0.bias": "slot_ffn.0.bias",
            "slot_ffn.1.weight": "slot_ffn.1.weight",
            "slot_ffn.1.bias": "slot_ffn.1.bias",
            "slot_ffn.3.weight": "slot_ffn.3.weight",
            "slot_ffn.3.bias": "slot_ffn.3.bias",
            "slot_norm.weight": "slot_norm.weight",
            "slot_norm.bias": "slot_norm.bias",
            "out.weight": "out.weight",
            "out.bias": "out.bias",
            # Fine head
            "fine_slot_queries": "fine_slot_queries",
            "fine_slot_attn.in_proj_weight": "fine_slot_attn.in_proj_weight",
            "fine_slot_attn.in_proj_bias": "fine_slot_attn.in_proj_bias",
            "fine_slot_attn.out_proj.weight": "fine_slot_attn.out_proj.weight",
            "fine_slot_attn.out_proj.bias": "fine_slot_attn.out_proj.bias",
            "fine_slot_ffn.0.weight": "fine_slot_ffn.0.weight",
            "fine_slot_ffn.0.bias": "fine_slot_ffn.0.bias",
            "fine_slot_ffn.1.weight": "fine_slot_ffn.1.weight",
            "fine_slot_ffn.1.bias": "fine_slot_ffn.1.bias",
            "fine_slot_ffn.3.weight": "fine_slot_ffn.3.weight",
            "fine_slot_ffn.3.bias": "fine_slot_ffn.3.bias",
            "fine_slot_norm.weight": "fine_slot_norm.weight",
            "fine_slot_norm.bias": "fine_slot_norm.bias",
            "fine_out.weight": "fine_out.weight",
            "fine_out.bias": "fine_out.bias",
        }
        fine_keys = [k for k in slot_map if k.startswith("fine_")]
        self.has_fine_head = any(k in state for k in fine_keys)

        loaded, skipped = [], []
        for ckpt_key, my_key in slot_map.items():
            if ckpt_key not in state:
                skipped.append(ckpt_key)
                continue
            ckpt_tensor = state[ckpt_key]
            target = self.state_dict()
            if my_key not in target:
                skipped.append(f"{ckpt_key}→{my_key}(missing)")
                continue
            if target[my_key].shape != ckpt_tensor.shape:
                skipped.append(
                    f"{ckpt_key}→{my_key}: "
                    f"{tuple(ckpt_tensor.shape)}→{tuple(target[my_key].shape)}"
                )
                continue
            target[my_key] = ckpt_tensor
            loaded.append(ckpt_key)

        if loaded:
            self.load_state_dict(target, strict=False)

        # ── Load mask decoders from checkpoint (coarse + fine, used for two-scale crop) ──
        coarse_mask_keys = [k for k in state if k.startswith("mask_head")]
        fine_mask_keys = [k for k in state if k.startswith("fine_mask_head")]
        if coarse_mask_keys:
            self._init_mask_decoder(state, coarse_mask_keys, prefix="mask_head", attr="mask_head")
        if fine_mask_keys:
            self._init_mask_decoder(state, fine_mask_keys, prefix="fine_mask_head", attr="fine_mask_head")

        print(
            f"[DinoSlotEncoder] loaded {len(loaded)}/{len(slot_map)} slot params "
            f"from {ckpt_path}"
        )
        if skipped:
            print(f"[DinoSlotEncoder] skipped: {skipped}")
        print(f"[DinoSlotEncoder] mask_decoder={'✓' if self.has_mask_decoder else '✗'} ")

    def _init_mask_decoder(self, state, keys, prefix, attr):
        """Initialize a single mask decoder (coarse or fine) from checkpoint weights."""
        try:
            sbd_keys = [k for k in keys if "sbd" in k.lower() or "conv" in k.lower()]
            if sbd_keys:
                # Match training architecture exactly: Conv2d→GELU→Conv2d→GELU→Conv2d
                head = nn.Sequential(
                    nn.Conv2d(514, 256, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, 1, 1),
                )
                mstate = {k.replace(prefix + "_sbd.", ""): state[k] for k in sbd_keys}
                head.load_state_dict(mstate, strict=True)
                setattr(self, attr, head)
                self.has_mask_decoder = True
                self._mask_decoder_mode = "sbd_hr"
            else:
                head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                )
                mstate = {k.replace(prefix + ".", ""): state[k] for k in keys}
                head.load_state_dict(mstate, strict=False)
                setattr(self, attr, head)
                self.has_mask_decoder = True
                self._mask_decoder_mode = "upsample"
        except Exception as e:
            print(f"[DinoSlotEncoder] failed to load {attr}: {e}")

    def _set_slot_requires_grad(self, requires_grad: bool):
        for m in [
            self.token_proj, self.slot_attn, self.slot_ffn,
            self.slot_norm, self.out,
            self.fine_token_proj, self.fine_slot_attn, self.fine_slot_ffn,
            self.fine_slot_norm, self.fine_out,
        ]:
            for p in m.parameters():
                p.requires_grad = requires_grad
        self.slot_queries.requires_grad = requires_grad
        self.fine_slot_queries.requires_grad = requires_grad

    # ------------------------------------------------------------------
    # Preprocessing (exact DINOv3 processor formula, GPU-only)
    # ------------------------------------------------------------------
    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Normalize RGB images using DINOv3's own processor parameters.

        Matches the HuggingFace DINOv3ViTImageProcessorFast formula:
            pixel = (image * rescale_factor - mean) / std
        Rearranged for [0,1] input:
            pixel = (image - mean/rescale) / (std/rescale)

        Args:
            rgb: [..., 3, H, W] in [0, 1]
        Returns:
            pixel_values: [..., 3, 224, 224] normalized
        """
        # Resize to model input size
        if rgb.shape[-2:] != (self.img_size, self.img_size):
            lead_dims = rgb.shape[:-3]
            flat = rgb.reshape(-1, *rgb.shape[-3:])
            flat = F.interpolate(
                flat, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
            rgb = flat.reshape(*lead_dims, *flat.shape[-3:])
        return (rgb - self.norm_mean) / self.norm_std

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _find_target_layer(self):
        """Find the target transformer block, caching the result."""
        if hasattr(self, "_cached_target_layer"):
            return self._cached_target_layer
        # Walk the model to find ModuleList of transformer layers
        candidates = []
        for name, module in self.dino.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > self.dino_layer:
                # Check if modules inside look like transformer blocks
                first = module[0]
                if hasattr(first, "forward"):
                    candidates.append((len(module), module))
        if not candidates:
            raise RuntimeError("Cannot find transformer layer ModuleList in DINO model")
        # Pick the one with most layers (encoder, not decoder)
        candidates.sort(key=lambda x: -x[0])
        target = candidates[0][1][self.dino_layer]
        self._cached_target_layer = target
        return target

    @torch.no_grad()
    def _extract_patch_tokens(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Extract patch tokens from DINOv3 using a forward hook on the target layer.
        Avoids output_hidden_states=True which stores ALL intermediate layers
        and causes OOM at large batch sizes.
        """
        # Find the target layer (robust across transformers versions)
        target_layer = self._find_target_layer()
        captured = {}

        def hook(module, input, output):
            # output is (hidden_states, ...) tuple from transformer block
            captured["tokens"] = output[0].detach() if isinstance(output, tuple) else output.detach()

        handle = target_layer.register_forward_hook(hook)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = self.dino(pixel_values=pixel_values, output_hidden_states=False)
        finally:
            handle.remove()

        tokens = captured["tokens"].float().detach()

        n = tokens.shape[1]
        # DINOv3 uses prefix tokens ([CLS] plus optional register tokens) before
        # patch tokens. Match the slot-distill training path by explicitly
        # keeping only the final square patch grid.
        expected_g = self.img_size // int(self.dino.config.patch_size)
        expected_n = expected_g * expected_g
        if n >= expected_n:
            prefix_n = n - expected_n
            patch = tokens[:, prefix_n:, :]
            g = expected_g
        else:
            g = int(math.sqrt(n))
            patch = tokens[:, : g * g, :]

        return patch.float().detach(), g

    # ------------------------------------------------------------------
    # Slot forward
    # ------------------------------------------------------------------
    def _run_slot_attention(
        self, tokens: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run slot attention over input tokens.
        Args:
            tokens: [B, N, slot_dim]  (already projected through token_proj)
            return_attention: if True, also return slot→patch attention weights
        Returns:
            slot_feat: [B, num_slots, slot_dim]  (after out projection)
            slots_raw: [B, num_slots, slot_dim]  (before out projection)
            (attn): [B, num_slots, N]  (only if return_attention=True)
        """
        B = tokens.shape[0]
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # [B, S, D]
        slots, attn = self.slot_attn(
            query=q, key=tokens, value=tokens,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        slots = self.slot_norm(slots + q)
        slots = slots + self.slot_ffn(slots)  # [B, S, D]
        slot_feat = self.out(slots)  # [B, S, D]
        if return_attention:
            return slot_feat, slots, attn  # attn: [B, S, N]
        return slot_feat, slots

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        return_slots_raw: bool = False,
        return_attention: bool = False,
        return_proj_tokens: bool = False,
        head: str = "coarse",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, T, 3, H, W] or [B, 3, H, W]
            head: "coarse" (full-image) or "fine" (crop-zoom, dual-head mode)
        Returns:
            slot_feat, extra, (proj_tokens if return_proj_tokens)
            where extra is attn (return_attention=True) or slots_raw (return_slots_raw=True)
        """
        has_temporal = images.dim() == 5
        if has_temporal:
            B, T = images.shape[:2]
            images_flat = images.reshape(B * T, *images.shape[2:])
        else:
            B = images.shape[0]
            T = 1
            images_flat = images

        # Preprocess and extract DINO patch tokens
        pixel_values = self.preprocess(images_flat)
        patch_tokens, g = self._extract_patch_tokens(pixel_values)

        use_fine = (head == "fine" and self.has_fine_head)
        _token_proj = self.fine_token_proj if use_fine else self.token_proj
        tokens = _token_proj(patch_tokens)
        _slot_attn = self.fine_slot_attn if use_fine else self.slot_attn
        _slot_norm = self.fine_slot_norm if use_fine else self.slot_norm
        _slot_ffn = self.fine_slot_ffn if use_fine else self.slot_ffn
        _out = self.fine_out if use_fine else self.out
        _queries = self.fine_slot_queries if use_fine else self.slot_queries

        q = _queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        slots, attn = _slot_attn(query=q, key=tokens, value=tokens, need_weights=return_attention, average_attn_weights=True)
        slots = _slot_norm(slots + q)
        slots = slots + _slot_ffn(slots)
        slot_feat = F.normalize(_out(slots), dim=-1)  # L2 norm, matches training's FrozenDinoSlotModel

        if has_temporal:
            slot_feat = slot_feat.reshape(B, T, self.num_slots, self.slot_dim)
            if attn is not None:
                attn = attn.reshape(B, T, self.num_slots, attn.shape[-1])
            if return_proj_tokens:
                tokens = tokens.reshape(B, T, tokens.shape[-2], self.slot_dim)

        extra = None
        if return_attention and attn is not None:
            extra = attn
        elif return_slots_raw:
            extra = slots
        if return_proj_tokens:
            return slot_feat, extra, tokens
        return slot_feat, extra

    def _coord_grid(self, bsz: int, h: int, w: int, device: torch.device, dtype: torch.dtype):
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1, 2, H, W]
        return grid.expand(bsz, -1, -1, -1)

    def decode_mask_logits(
        self, fg_slot: torch.Tensor, proj_tokens: torch.Tensor,
        hw: Tuple[int, int], out_hw: Tuple[int, int],
        head: str = "coarse",
    ) -> torch.Tensor:
        """
        Decode foreground slot feature + patch tokens → pixel-level mask logits.
        Used for two-scale crop guidance (more accurate than attention alpha).

        Args:
            head: "coarse" uses coarse mask decoder, "fine" uses fine mask decoder.
        """
        if not self.has_mask_decoder:
            raise RuntimeError("Mask decoder not available in this checkpoint.")
        mask_head = getattr(self, "fine_mask_head", None) if head == "fine" else self.mask_head
        if mask_head is None:
            mask_head = self.mask_head
        B = fg_slot.shape[0]
        h, w = hw
        H_out, W_out = out_hw
        if self._mask_decoder_mode == "sbd_hr":
            # Match training FrozenDinoSlotModel.decode_mask_logits exactly:
            # 1. Upsample tokens to output resolution FIRST
            # 2. Broadcast slot + coord at output resolution
            # 3. Apply conv at high resolution
            # Use fp16 autocast to match training memory profile (training wraps
            # this entire path in torch.autocast).
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                tok_map = proj_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
                tok_up = F.interpolate(tok_map, size=(H_out, W_out), mode="bilinear", align_corners=False)
                slot_map = fg_slot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_out, W_out)
                coord = self._coord_grid(B, H_out, W_out, tok_up.device, tok_up.dtype)
                dec_in_hr = torch.cat([tok_up, slot_map, coord], dim=1)
                fg_logits_hr = mask_head(dec_in_hr).squeeze(1)
            return fg_logits_hr.float()
        else:
            # Simple MLP: concat slot + average token per patch → predict
            slot_avg = torch.cat([fg_slot, proj_tokens.mean(dim=1)], dim=-1)  # [B, 2D]
            logit = mask_head(slot_avg)  # [B, 1]
            logits = logit[:, :, None, None].expand(-1, -1, H_out, W_out)
        return logits.squeeze(1)  # [B, H_out, W_out]
