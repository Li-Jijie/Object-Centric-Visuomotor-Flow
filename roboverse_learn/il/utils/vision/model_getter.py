import os
import contextlib

import torch
import torch.nn as nn
import torchvision


def _load_state_dict_flex(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
    return payload


def _fallback_local_path(name, weights):
    # user-provided explicit fallback path
    env_path = os.environ.get("TORCHVISION_RESNET18_WEIGHTS", "")
    if name == "resnet18" and env_path and os.path.isfile(env_path):
        return env_path

    # default local fallback paths for torchvision resnet18 IMAGENET1K_V1
    if name == "resnet18" and str(weights) in {"IMAGENET1K_V1", "ResNet18_Weights.IMAGENET1K_V1", "DEFAULT", "ResNet18_Weights.DEFAULT"}:
        candidates = [
            os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    return ""


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m", null/none, or local file path
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    # normalize null-like weights
    if isinstance(weights, str) and weights.lower() in {"none", "null", ""}:
        weights = None

    func = getattr(torchvision.models, name)

    # local file path directly provided
    if isinstance(weights, str) and os.path.isfile(weights):
        resnet = func(weights=None, **kwargs)
        state = _load_state_dict_flex(weights)
        missing, unexpected = resnet.load_state_dict(state, strict=False)
        print(f"[ResNetLoad] local={weights} missing={len(missing)} unexpected={len(unexpected)}")
        resnet.fc = torch.nn.Identity()
        return resnet

    # standard torchvision weight loading with local fallback on failure
    try:
        resnet = func(weights=weights, **kwargs)
    except Exception as e:
        local_path = _fallback_local_path(name, weights)
        if not local_path:
            raise
        print(f"[ResNetLoad] torchvision weights load failed ({e}); fallback local={local_path}")
        resnet = func(weights=None, **kwargs)
        state = _load_state_dict_flex(local_path)
        missing, unexpected = resnet.load_state_dict(state, strict=False)
        print(f"[ResNetLoad] fallback loaded missing={len(missing)} unexpected={len(unexpected)}")

    resnet.fc = torch.nn.Identity()
    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model


class DinoV3GlobalBackbone(nn.Module):
    """HF DINOv3 wrapper returning a global feature vector per image."""

    def __init__(
        self,
        model_path: str,
        pool: str = "cls",
        local_files_only: bool = True,
        freeze: bool = True,
        adapter_enable: bool = False,
        adapter_hidden_dim: int = 512,
        adapter_out_dim: int = 512,
        adapter_dropout: float = 0.1,
        layer: int = -1,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:
            raise ImportError(
                "transformers is required for get_dinov3_global. "
                "Please install transformers in this environment."
            ) from e

        self.model = AutoModel.from_pretrained(model_path, local_files_only=local_files_only)
        self.pool = str(pool).lower()
        self.freeze = bool(freeze)
        self.adapter_enable = bool(adapter_enable)
        self.layer = int(layer)  # -1 = last layer, >=0 = specific transformer layer
        if self.pool not in {"cls", "mean_patch", "mean_all"}:
            raise ValueError(f"Unsupported pool={pool}, expected one of ['cls','mean_patch','mean_all']")
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        hidden_size = int(getattr(self.model.config, "hidden_size", 384))
        if self.adapter_enable:
            h = int(adapter_hidden_dim)
            o = int(adapter_out_dim)
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, h),
                nn.SiLU(),
                nn.LayerNorm(h),
                nn.Dropout(float(adapter_dropout)),
                nn.Linear(h, o),
            )
            self.out_dim = o
        else:
            self.adapter = nn.Identity()
            self.out_dim = hidden_size

        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_adapter = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(
            f"[DinoV3GlobalBackbone] freeze={self.freeze} pool={self.pool} "
            f"adapter={self.adapter_enable} out_dim={self.out_dim} "
            f"trainable_backbone={n_train}/{n_total} trainable_adapter={n_adapter}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone deterministic/frozen even when parent calls .train()
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            if self.layer >= 0:
                out = self.model(pixel_values=x, output_hidden_states=True)
                tokens = out.hidden_states[self.layer]  # [B,N,D]
            else:
                out = self.model(pixel_values=x)
                tokens = out.last_hidden_state  # [B,N,D]
        feat = None
        if self.pool == "cls":
            if tokens.shape[1] > 1:
                feat = tokens[:, 0, :]
            else:
                feat = tokens.mean(dim=1)
        elif self.pool == "mean_patch":
            if tokens.shape[1] > 1:
                feat = tokens[:, 1:, :].mean(dim=1)
            else:
                feat = tokens.mean(dim=1)
        else:
            feat = tokens.mean(dim=1)
        return self.adapter(feat)


def get_dinov3_global(
    model_path,
    pool="cls",
    local_files_only=True,
    freeze=True,
    adapter_enable=False,
    adapter_hidden_dim=512,
    adapter_out_dim=512,
    adapter_dropout=0.1,
    layer=-1,
    **kwargs,
):
    """
    Build a DINOv3 global-vector backbone compatible with MultiImageObsEncoder.

    Args:
        model_path: local HF model directory
        pool: cls | mean_patch | mean_all
        local_files_only: enforce offline loading
        layer: -1 = last hidden state, >=0 = specific transformer layer
    """
    _ = kwargs  # keep Hydra compatibility for unused keys
    return DinoV3GlobalBackbone(
        model_path=model_path,
        pool=pool,
        local_files_only=bool(local_files_only),
        freeze=bool(freeze),
        adapter_enable=bool(adapter_enable),
        adapter_hidden_dim=int(adapter_hidden_dim),
        adapter_out_dim=int(adapter_out_dim),
        adapter_dropout=float(adapter_dropout),
        layer=int(layer),
    )
