from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from evaluation.backbones import SmriMaeBackbone, load_smri_mae_checkpoint
from evaluation.core import EvaluationTask, TargetSpec
from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer


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


def build_backbone(cfg: Mapping[str, Any]):
    name = cfg.get("name")
    if name == "fake":
        return FakeBackbone(embed_dim=int(cfg.get("embed_dim", 4)))
    if name != "smri_mae":
        raise ValueError("unknown backbone {!r}. available backbones: fake, smri_mae".format(name))
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


def build_head(cfg: Mapping[str, Any], *, target_spec: TargetSpec, input_dim: int):
    name = cfg.get("name")
    if name == "linear":
        return LinearHead(input_dim=input_dim, output_dim=target_spec.dim, pooling=cfg["pooling"])
    if name == "attn":
        raise NotImplementedError("attention head is configured but not implemented yet")
    raise ValueError("unknown head {!r}. available heads: attn, linear".format(name))


def build_trainer(
    mode_cfg: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    backbone,
    head,
    task: EvaluationTask,
):
    name = mode_cfg.get("name")
    if name == "probe":
        return ProbeTrainer(cfg=dict(cfg), backbone=backbone, head=head, task=task)
    if name == "full":
        raise NotImplementedError("full fine-tuning is configured but not implemented yet")
    raise ValueError("unknown trainer mode {!r}. available modes: full, probe".format(name))
