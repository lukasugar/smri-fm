from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from smri_mae.model_mae import MaskedViT


class FakeBackbone(nn.Module):
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(8, embed_dim)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        batch = images.shape[0]
        flat = images.reshape(batch, -1).float()
        base = self.proj(flat)
        return {
            "cls": base[:, None, :],
            "reg": torch.stack([base, base + 1.0], dim=1),
            "patch": torch.stack([base, base + 1.0, base + 2.0], dim=1),
        }


class SmriMaeBackbone(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.model = MaskedViT(**kwargs)
        self.embed_dim = self.model.patch_embed.out_features

    def forward(self, images: Tensor) -> dict[str, Tensor | None]:
        cls, reg, patch = self.model.forward_embedding(images)
        return {"cls": cls, "reg": reg, "patch": patch}


def load_smri_mae_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")


def _build_fake_backbone(cfg: Mapping[str, Any]) -> nn.Module:
    return FakeBackbone(embed_dim=int(cfg.get("embed_dim", 4)))


def _build_smri_mae_backbone(cfg: Mapping[str, Any]) -> nn.Module:
    kwargs = {
        "img_size": cfg["img_size"],
        "patch_size": cfg["patch_size"],
        "in_chans": cfg.get("in_chans", 1),
        **dict(cfg.get("model_kwargs") or {}),
    }
    backbone = SmriMaeBackbone(**kwargs)
    if cfg.get("checkpoint_path"):
        load_smri_mae_checkpoint(backbone.model, cfg["checkpoint_path"])
    return backbone


_BACKBONE_BUILDERS: dict[str, Callable[[Mapping[str, Any]], nn.Module]] = {
    "fake": _build_fake_backbone,
    "smri_mae": _build_smri_mae_backbone,
}


def list_backbones() -> list[str]:
    return sorted(_BACKBONE_BUILDERS)


def build_backbone(cfg: Mapping[str, Any]):
    name = cfg.get("name")
    try:
        builder = _BACKBONE_BUILDERS[name]
    except KeyError:
        available = ", ".join(list_backbones())
        raise ValueError(f"unknown backbone {name!r}. available backbones: {available}") from None
    return builder(cfg)
