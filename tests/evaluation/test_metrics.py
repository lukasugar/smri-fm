import pytest
import torch

from evaluation.metrics import is_better, regression_metrics


def test_regression_metrics():
    preds = torch.tensor([[1.0], [3.0], [5.0]])
    targets = torch.tensor([[1.0], [1.0], [7.0]])

    metrics = regression_metrics(preds, targets)

    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["rmse"] == pytest.approx((8.0 / 3.0) ** 0.5)
    assert metrics["bias"] == pytest.approx(0.0)


def test_is_better_min_and_max():
    assert is_better(1.0, None, "min")
    assert is_better(0.9, 1.0, "min")
    assert not is_better(1.1, 1.0, "min")

    assert is_better(0.8, None, "max")
    assert is_better(0.9, 0.8, "max")
    assert not is_better(0.7, 0.8, "max")


def test_is_better_rejects_unknown_mode():
    with pytest.raises(ValueError, match="selection_mode"):
        is_better(1.0, None, "median")
