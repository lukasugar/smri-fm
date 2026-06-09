from collections.abc import Callable, Mapping
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


def _build_linear_head(
    cfg: Mapping[str, Any], *, target_spec: TargetSpec, input_dim: int
) -> nn.Module:
    return LinearHead(input_dim=input_dim, output_dim=target_spec.dim, pooling=cfg["pooling"])


_HEAD_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "linear": _build_linear_head,
}


def list_heads() -> list[str]:
    return sorted(_HEAD_BUILDERS)


def build_head(cfg: Mapping[str, Any], *, target_spec: TargetSpec, input_dim: int):
    name = cfg.get("name")
    try:
        builder = _HEAD_BUILDERS[name]
    except KeyError:
        available = ", ".join(list_heads())
        raise ValueError(f"unknown head {name!r}. available heads: {available}") from None
    return builder(cfg, target_spec=target_spec, input_dim=input_dim)


def _build_probe_trainer(
    mode_cfg: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    backbone,
    head,
    task: EvaluationTask,
) -> Any:
    return ProbeTrainer(cfg=dict(cfg), backbone=backbone, head=head, task=task)


_TRAINER_BUILDERS: dict[str, Callable[..., Any]] = {
    "probe": _build_probe_trainer,
}


def list_trainers() -> list[str]:
    return sorted(_TRAINER_BUILDERS)


def build_trainer(
    mode_cfg: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    backbone,
    head,
    task: EvaluationTask,
):
    name = mode_cfg.get("name")
    try:
        builder = _TRAINER_BUILDERS[name]
    except KeyError:
        available = ", ".join(list_trainers())
        raise ValueError(f"unknown trainer mode {name!r}. available modes: {available}") from None
    return builder(mode_cfg, cfg=cfg, backbone=backbone, head=head, task=task)
