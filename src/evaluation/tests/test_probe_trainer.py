import json

import pytest
import torch

from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer

from .fakes import FakeBackbone, FakeRegressionTask


class MaskRecordingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1
        self.seen_mask = None

    def forward(self, images, mask=None):
        self.seen_mask = mask
        batch = images.shape[0]
        return {"cls": torch.ones(batch, 1, 1), "reg": None, "patch": None}


def make_cfg(tmp_path):
    return {
        "name": "fake_probe",
        "output_dir": str(tmp_path),
        "task": {"overwrite_data": True},
        "representation": "cls",
        "optimization": {
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_workers": 0,
        },
        "evaluation": {"selection_metric": "mae", "selection_mode": "min"},
        "device": "cpu",
        "seed": 7338,
    }


def test_probe_trainer_runs_and_writes_outputs(tmp_path):
    task = FakeRegressionTask()
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=1, pooling="first")
    trainer = ProbeTrainer(cfg=make_cfg(tmp_path), backbone=backbone, head=head, task=task)

    result = trainer.run()

    run_dir = tmp_path / "fake_probe"
    assert task.prepare_calls == [True]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "head-best.pt").exists()
    assert result["best_epoch"] is not None

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "val" in metrics
    assert "test" in metrics


def test_probe_trainer_rejects_missing_representation(tmp_path):
    task = FakeRegressionTask()
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=1, pooling="first")
    cfg = make_cfg(tmp_path)
    cfg["representation"] = "missing"
    trainer = ProbeTrainer(cfg=cfg, backbone=backbone, head=head, task=task)

    with pytest.raises(ValueError, match="representation 'missing' is unavailable"):
        trainer.run()


def test_probe_trainer_rejects_classification_for_now(tmp_path):
    task = FakeRegressionTask()
    spec_type = type(task.target_spec())
    task.target_spec = lambda: spec_type(
        kind="classification",
        dim=2,
        loss="cross_entropy",
    )
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=2, pooling="first")
    trainer = ProbeTrainer(cfg=make_cfg(tmp_path), backbone=backbone, head=head, task=task)

    with pytest.raises(NotImplementedError, match="classification"):
        trainer.run()


def test_probe_trainer_forwards_optional_batch_mask():
    backbone = MaskRecordingBackbone()
    trainer = object.__new__(ProbeTrainer)
    trainer.backbone = backbone
    images = torch.zeros(2, 1, 2, 2, 2)
    mask = torch.ones_like(images, dtype=torch.bool)

    reps = trainer._forward_backbone({"image": images, "mask": mask}, torch.device("cpu"))

    assert backbone.seen_mask is mask
    assert reps["cls"].shape == (2, 1, 1)
