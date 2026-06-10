import importlib
import json
import sys
from types import SimpleNamespace

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
        "wandb_logging": False,
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
    assert "train_loss" in metrics["history"][0]


def test_probe_trainer_skips_wandb_when_disabled(tmp_path, monkeypatch):
    def fail_if_wandb_imported(name):
        if name == "wandb":
            raise AssertionError("wandb should not be imported")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", fail_if_wandb_imported)
    cfg = make_cfg(tmp_path)
    cfg["wandb_logging"] = False
    task = FakeRegressionTask()
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=1, pooling="first")
    trainer = ProbeTrainer(cfg=cfg, backbone=backbone, head=head, task=task)

    trainer.run()


def test_probe_trainer_logs_to_wandb_by_default(tmp_path, monkeypatch):
    logs = []
    init_kwargs = {}

    def init(**kwargs):
        init_kwargs.update(kwargs)
        return SimpleNamespace()

    fake_wandb = SimpleNamespace(init=init, log=logs.append)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = make_cfg(tmp_path)
    cfg.pop("wandb_logging")
    cfg["model"] = {"checkpoint_path": "backbone.pt"}
    cfg["wandb"] = {"project": "smri-fm-test"}
    task = FakeRegressionTask()
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=1, pooling="first")
    trainer = ProbeTrainer(cfg=cfg, backbone=backbone, head=head, task=task)

    trainer.run()

    assert init_kwargs["name"] == "fake_probe"
    assert init_kwargs["config"] == cfg
    assert init_kwargs["project"] == "smri-fm-test"
    assert any("train/loss" in logged for logged in logs)
    assert any("val/mae" in logged for logged in logs)
    assert any("final/test/mae" in logged for logged in logs)
    assert any(logged.get("paths/head_best_checkpoint") for logged in logs)
    assert any(logged.get("paths/backbone_checkpoint") == "backbone.pt" for logged in logs)


def test_probe_trainer_runs_when_wandb_import_fails(tmp_path, monkeypatch):
    def fail_wandb_import(name):
        if name == "wandb":
            raise ImportError("no wandb")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", fail_wandb_import)
    task = FakeRegressionTask()
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=1, pooling="first")
    cfg = make_cfg(tmp_path)
    cfg["wandb_logging"] = True
    trainer = ProbeTrainer(cfg=cfg, backbone=backbone, head=head, task=task)

    result = trainer.run()

    assert result["best_epoch"] is not None


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
