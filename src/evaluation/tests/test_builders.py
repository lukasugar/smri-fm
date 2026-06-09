import pytest

from evaluation.builders import (
    build_backbone,
    build_head,
    build_trainer,
    list_backbones,
    list_heads,
    list_trainers,
)
from evaluation.core import TargetSpec
from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer

from .fakes import FakeBackbone, FakeRegressionTask


def test_builder_registries_list_available_components():
    assert list_backbones() == ["fake", "smri_mae"]
    assert list_heads() == ["linear"]
    assert list_trainers() == ["probe"]


def test_build_head_linear():
    head = build_head(
        {"name": "linear", "pooling": "mean"},
        target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
        input_dim=4,
    )

    assert isinstance(head, LinearHead)


def test_build_head_attn_reports_available_heads():
    with pytest.raises(ValueError, match="unknown head 'attn'.*linear"):
        build_head(
            {"name": "attn"},
            target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
            input_dim=4,
        )


def test_build_head_unknown_reports_available_heads():
    with pytest.raises(ValueError, match="unknown head 'missing'.*linear"):
        build_head(
            {"name": "missing"},
            target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
            input_dim=4,
        )


def test_build_trainer_probe():
    trainer = build_trainer(
        {"name": "probe"},
        cfg={"optimization": {}, "evaluation": {}},
        backbone=FakeBackbone(),
        head=LinearHead(4, 1, "first"),
        task=FakeRegressionTask(),
    )

    assert isinstance(trainer, ProbeTrainer)


def test_build_trainer_full_reports_available_trainers():
    with pytest.raises(ValueError, match="unknown trainer mode 'full'.*probe"):
        build_trainer(
            {"name": "full"},
            cfg={},
            backbone=FakeBackbone(),
            head=LinearHead(4, 1, "first"),
            task=FakeRegressionTask(),
        )


def test_build_backbone_unknown_reports_available_backbones():
    with pytest.raises(ValueError, match="unknown backbone 'missing'.*fake, smri_mae"):
        build_backbone({"name": "missing"})


def test_build_trainer_unknown_reports_available_trainers():
    with pytest.raises(ValueError, match="unknown trainer mode 'missing'.*probe"):
        build_trainer(
            {"name": "missing"},
            cfg={},
            backbone=FakeBackbone(),
            head=LinearHead(4, 1, "first"),
            task=FakeRegressionTask(),
        )
