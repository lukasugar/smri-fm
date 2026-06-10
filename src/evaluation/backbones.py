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
    def __init__(
        self,
        *,
        use_input_mask: bool = False,
        calculate_mask: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if calculate_mask not in {None, "mean"}:
            raise ValueError(
                f"calculate_mask must be one of None or 'mean', got {calculate_mask!r}"
            )
        if use_input_mask and calculate_mask is not None:
            raise ValueError("use_input_mask and calculate_mask cannot both be enabled")
        self.use_input_mask = bool(use_input_mask)
        self.calculate_mask = calculate_mask
        self.model = MaskedViT(**kwargs)
        self.embed_dim = self.model.patch_embed.out_features

    def _resolve_mask(self, images: Tensor, mask: Tensor | None) -> Tensor | None:
        if self.use_input_mask:
            if mask is None:
                raise ValueError("use_input_mask=True requires a mask input")
            return mask
        if self.calculate_mask == "mean":
            dims = tuple(range(1, images.ndim))
            return images > images.mean(dim=dims, keepdim=True)
        return None

    def forward(self, images: Tensor, mask: Tensor | None = None) -> dict[str, Tensor | None]:
        mask = self._resolve_mask(images, mask)
        cls, reg, patch = self.model.forward_embedding(images, mask=mask)
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
