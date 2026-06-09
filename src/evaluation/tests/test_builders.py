import pytest

from evaluation.builders import build_backbone, build_head, build_trainer
from evaluation.core import TargetSpec
from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer

from .fakes import FakeBackbone, FakeRegressionTask


def test_build_head_linear():
    head = build_head(
        {"name": "linear", "pooling": "mean"},
        target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
        input_dim=4,
    )

    assert isinstance(head, LinearHead)


def test_build_head_attn_not_implemented():
    with pytest.raises(NotImplementedError, match="attention head"):
        build_head(
            {"name": "attn"},
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


def test_build_trainer_full_not_implemented():
    with pytest.raises(NotImplementedError, match="full fine-tuning"):
        build_trainer(
            {"name": "full"},
            cfg={},
            backbone=FakeBackbone(),
            head=LinearHead(4, 1, "first"),
            task=FakeRegressionTask(),
        )
