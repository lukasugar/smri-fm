from evaluation.core import EvaluationTask
from evaluation.tasks.fake_regression import FakeRegressionTask

_TASK_REGISTRY: dict[str, type[EvaluationTask]] = {
    "fake_regression": FakeRegressionTask,
}


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)


def build_task(cfg):
    name = cfg.get("name")
    try:
        task_cls = _TASK_REGISTRY[name]
    except KeyError:
        available = ", ".join(list_tasks())
        raise ValueError(f"unknown task {name!r}. available tasks: {available}") from None
    return task_cls()
