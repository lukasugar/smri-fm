from pathlib import Path

import pytest
import torch

from evaluation.backbones import SmriMaeBackbone, _build_smri_mae_backbone, load_smri_mae_checkpoint


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


def test_load_smri_mae_checkpoint_filters_mae_encoder_keys(tmp_path):
    backbone = SmriMaeBackbone(**tiny_model_kwargs())
    encoder_state = {f"encoder.{key}": value for key, value in backbone.model.state_dict().items()}
    encoder_state["decoder.head.weight"] = torch.zeros(1)
    checkpoint_path = Path(tmp_path) / "checkpoint.pt"
    torch.save({"model": encoder_state}, checkpoint_path)

    load_smri_mae_checkpoint(backbone.model, checkpoint_path)


def test_smri_mae_backbone_can_use_input_mask():
    backbone = SmriMaeBackbone(**tiny_model_kwargs(), use_input_mask=True)
    images = torch.ones(1, 1, 8, 8, 8)
    mask = torch.zeros_like(images, dtype=torch.bool)
    mask[:, :, 0, 0, 0] = True
    mask[:, :, 4:8, 4:8, 4:8] = True

    reps = backbone(images, mask=mask)

    assert reps["patch"].shape == (1, 2, 8)


def test_smri_mae_backbone_requires_input_mask_when_enabled():
    backbone = SmriMaeBackbone(**tiny_model_kwargs(), use_input_mask=True)

    with pytest.raises(ValueError, match="use_input_mask"):
        backbone(torch.ones(1, 1, 8, 8, 8))


def test_smri_mae_backbone_can_calculate_mean_mask():
    backbone = SmriMaeBackbone(**tiny_model_kwargs(), calculate_mask="mean")
    images = torch.zeros(1, 1, 8, 8, 8)
    images[:, :, 4:8, 4:8, 4:8] = 10.0

    reps = backbone(images)

    assert reps["patch"].shape == (1, 1, 8)


def test_smri_mae_backbone_rejects_unknown_calculated_mask():
    with pytest.raises(ValueError, match="calculate_mask"):
        SmriMaeBackbone(**tiny_model_kwargs(), calculate_mask="median")


def test_smri_mae_backbone_rejects_conflicting_mask_sources():
    with pytest.raises(ValueError, match="use_input_mask"):
        SmriMaeBackbone(**tiny_model_kwargs(), use_input_mask=True, calculate_mask="mean")


def test_build_smri_mae_backbone_passes_mask_options():
    cfg = {
        "img_size": [8, 8, 8],
        "patch_size": 4,
        "in_chans": 1,
        "use_input_mask": True,
        "calculate_mask": None,
        "model_kwargs": {
            "depth": 1,
            "embed_dim": 8,
            "num_heads": 2,
            "class_token": True,
            "reg_tokens": 0,
        },
    }

    backbone = _build_smri_mae_backbone(cfg)

    assert backbone.use_input_mask is True
    assert backbone.calculate_mask is None
