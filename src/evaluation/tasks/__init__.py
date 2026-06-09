from evaluation.tasks.fake_regression import FakeRegressionTask


def build_task(cfg):
    name = cfg.get("name")
    if name == "fake_regression":
        return FakeRegressionTask()
    raise ValueError("unknown task {!r}. available tasks: fake_regression".format(name))
