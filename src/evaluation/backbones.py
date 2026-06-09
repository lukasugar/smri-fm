from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from smri_mae.model_mae import MaskedViT


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
