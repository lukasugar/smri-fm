import importlib.util

import pytest

import evaluation.backbones as backbones
import evaluation.heads as heads
import evaluation.trainers as trainers
from evaluation.core import TargetSpec
from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer

from .fakes import FakeBackbone, FakeRegressionTask


def test_builder_registries_list_available_components():
    assert backbones.list_backbones() == ["fake", "smri_mae"]
    assert heads.list_heads() == ["linear"]
    assert trainers.list_trainers() == ["probe"]


def test_builders_module_is_removed():
    assert importlib.util.find_spec("evaluation.builders") is None


def test_build_backbone_fake():
    backbone = backbones.build_backbone({"name": "fake", "embed_dim": 6})

    assert isinstance(backbone, backbones.FakeBackbone)
    assert backbone.embed_dim == 6


def test_build_head_linear():
    head = heads.build_head(
        {"name": "linear", "pooling": "mean"},
        target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
        input_dim=4,
    )

    assert isinstance(head, LinearHead)


def test_build_head_attn_reports_available_heads():
    with pytest.raises(ValueError, match="unknown head 'attn'.*linear"):
        heads.build_head(
            {"name": "attn"},
            target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
            input_dim=4,
        )


def test_build_head_unknown_reports_available_heads():
    with pytest.raises(ValueError, match="unknown head 'missing'.*linear"):
        heads.build_head(
            {"name": "missing"},
            target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
            input_dim=4,
        )


def test_build_trainer_probe():
    trainer = trainers.build_trainer(
        {"name": "probe"},
        cfg={"optimization": {}, "evaluation": {}},
        backbone=FakeBackbone(),
        head=LinearHead(4, 1, "first"),
        task=FakeRegressionTask(),
    )

    assert isinstance(trainer, ProbeTrainer)


def test_build_trainer_full_reports_available_trainers():
    with pytest.raises(ValueError, match="unknown trainer mode 'full'.*probe"):
        trainers.build_trainer(
            {"name": "full"},
            cfg={},
            backbone=FakeBackbone(),
            head=LinearHead(4, 1, "first"),
            task=FakeRegressionTask(),
        )


def test_build_backbone_unknown_reports_available_backbones():
    with pytest.raises(ValueError, match="unknown backbone 'missing'.*fake, smri_mae"):
        backbones.build_backbone({"name": "missing"})


def test_build_trainer_unknown_reports_available_trainers():
    with pytest.raises(ValueError, match="unknown trainer mode 'missing'.*probe"):
        trainers.build_trainer(
            {"name": "missing"},
            cfg={},
            backbone=FakeBackbone(),
            head=LinearHead(4, 1, "first"),
            task=FakeRegressionTask(),
        )
