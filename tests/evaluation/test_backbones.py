from pathlib import Path

import torch

from evaluation.backbones import SmriMaeBackbone, load_smri_mae_checkpoint


def tiny_model_kwargs():
    return {
        "img_size": [8, 8, 8],
        "patch_size": 4,
        "in_chans": 1,
        "depth": 1,
        "embed_dim": 8,
        "num_heads": 2,
        "class_token": True,
        "reg_tokens": 2,
    }


def test_smri_mae_backbone_returns_named_representations():
    backbone = SmriMaeBackbone(**tiny_model_kwargs())

    reps = backbone(torch.zeros(2, 1, 8, 8, 8))

    assert backbone.embed_dim == 8
    assert reps["cls"].shape == (2, 1, 8)
    assert reps["reg"].shape == (2, 2, 8)
    assert reps["patch"].shape == (2, 8, 8)


def test_smri_mae_backbone_omits_unavailable_registers():
    kwargs = tiny_model_kwargs()
    kwargs["reg_tokens"] = 0
    backbone = SmriMaeBackbone(**kwargs)

    reps = backbone(torch.zeros(2, 1, 8, 8, 8))

    assert reps["reg"] is None


def test_load_smri_mae_checkpoint_accepts_pretrain_model_key(tmp_path):
    backbone = SmriMaeBackbone(**tiny_model_kwargs())
    checkpoint_path = Path(tmp_path) / "checkpoint.pt"
    torch.save({"model": backbone.model.state_dict()}, checkpoint_path)

    load_smri_mae_checkpoint(backbone.model, checkpoint_path)
