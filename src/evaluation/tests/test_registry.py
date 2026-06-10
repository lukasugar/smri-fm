import pytest

from evaluation.tasks import build_task, list_tasks
from evaluation.tasks.fake_regression import FakeRegressionTask


def test_task_registry_builds_registered_task():
    task = build_task({"name": "fake_regression"})

    assert isinstance(task, FakeRegressionTask)


def test_task_registry_lists_available_tasks():
    assert list_tasks() == ["fake_regression", "fomo_brain_age_gap"]


def test_task_registry_reports_available_task_names():
    with pytest.raises(
        ValueError,
        match="unknown task 'missing'.*fake_regression.*fomo_brain_age_gap",
    ):
        build_task({"name": "missing"})
