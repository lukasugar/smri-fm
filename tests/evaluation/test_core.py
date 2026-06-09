import pytest
import torch
from torch.utils.data import TensorDataset

from evaluation.core import DatasetBundle, TargetSpec, validate_batch


def test_target_spec_accepts_regression():
    spec = TargetSpec(kind="regression", dim=1, loss="mse")

    assert spec.kind == "regression"
    assert spec.dim == 1
    assert spec.loss == "mse"


def test_dataset_bundle_holds_lazy_datasets():
    train = TensorDataset(torch.zeros(2, 1))
    val = TensorDataset(torch.zeros(1, 1))
    test = TensorDataset(torch.zeros(1, 1))

    bundle = DatasetBundle(train=train, val=val, test=test)

    assert len(bundle.train) == 2
    assert len(bundle.val) == 1
    assert len(bundle.test) == 1


def test_validate_batch_requires_image_and_target():
    with pytest.raises(ValueError, match="image"):
        validate_batch({"target": torch.zeros(2, 1)})

    with pytest.raises(ValueError, match="target"):
        validate_batch({"image": torch.zeros(2, 1, 4, 4, 4)})


def test_validate_batch_accepts_extensible_schema():
    batch = {
        "image": torch.zeros(2, 1, 4, 4, 4),
        "target": torch.zeros(2, 1),
        "id": ["a", "b"],
        "mask": torch.ones(2, 1, 4, 4, 4),
    }

    validate_batch(batch)
