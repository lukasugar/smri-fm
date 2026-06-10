from pathlib import Path

import pytest
import torch

from evaluation.core import DatasetBundle, TargetSpec
from evaluation.tasks import build_task


DATA_ROOT = Path("data/asparagus/data/REGR002_FOMO26_BrainAge")


def test_fomo_brain_age_gap_task_loads_asparagus_splits_as_is():
    task = build_task({"name": "fomo_brain_age_gap", "data_root": str(DATA_ROOT), "fold": 0})

    bundle = task.datasets()

    assert isinstance(bundle, DatasetBundle)
    assert len(bundle.train) == 8
    assert len(bundle.val) == 1
    assert len(bundle.test) == 1

    sample = bundle.train[0]

    assert set(sample) >= {"image", "target", "id", "meta"}
    assert sample["image"].shape == torch.Size([1, 176, 256, 256])
    assert sample["target"].shape == torch.Size([1])
    assert sample["meta"]["path"].endswith(".pt")


def test_fomo_brain_age_gap_task_reports_regression_target_and_metrics():
    task = build_task({"name": "fomo_brain_age_gap", "data_root": str(DATA_ROOT)})

    assert task.target_spec() == TargetSpec(kind="regression", dim=1, loss="mse")
    assert "mae" in task.metrics(torch.tensor([[2.0]]), torch.tensor([[1.0]]))


def test_fomo_brain_age_gap_task_rejects_missing_data_root(tmp_path):
    task = build_task({"name": "fomo_brain_age_gap", "data_root": str(tmp_path / "missing")})

    with pytest.raises(FileNotFoundError, match="REGR002_FOMO26_BrainAge"):
        task.prepare()


def test_fomo_brain_age_gap_task_can_be_built_from_registry():
    task = build_task({"name": "fomo_brain_age_gap", "data_root": str(DATA_ROOT), "fold": 0})

    assert task.name == "fomo_brain_age_gap"
