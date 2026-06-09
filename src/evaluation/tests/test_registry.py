import pytest

from evaluation.registry import Registry
from evaluation.tasks import build_task, list_tasks
from evaluation.tasks.fake_regression import FakeRegressionTask


def test_task_registry_builds_registered_task():
    task = build_task({"name": "fake_regression"})

    assert isinstance(task, FakeRegressionTask)


def test_task_registry_lists_available_tasks():
    assert list_tasks() == ["fake_regression"]


def test_task_registry_reports_available_task_names():
    with pytest.raises(ValueError, match="unknown task 'missing'.*fake_regression"):
        build_task({"name": "missing"})


def test_registry_builds_registered_item():
    registry = Registry("task")
    registry.register("fake", lambda cfg: {"cfg": cfg})

    built = registry.build({"name": "fake", "x": 1})

    assert built == {"cfg": {"name": "fake", "x": 1}}


def test_registry_reports_available_names():
    registry = Registry("head")
    registry.register("linear", lambda cfg: cfg)

    with pytest.raises(ValueError, match="unknown head 'attn'.*linear"):
        registry.build({"name": "attn"})


def test_registry_requires_name():
    registry = Registry("task")

    with pytest.raises(ValueError, match="requires a 'name'"):
        registry.build({})
