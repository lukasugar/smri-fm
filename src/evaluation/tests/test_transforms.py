import pytest
import torch
from torch.utils.data import Dataset

from evaluation.core import DatasetBundle
from evaluation.transforms import (
    PadCenterCrop,
    TransformDataset,
    apply_transforms,
    build_transform,
)


class OneSampleDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def test_pad_center_crop_pads_image_to_target_size():
    image = torch.ones(1, 2, 3, 4)
    target = torch.tensor([1.0])
    sample = {"image": image, "target": target, "id": "sample-1"}

    transformed = PadCenterCrop(size=[4, 5, 6], pad_value=0.0)(sample)

    assert transformed["image"].shape == (1, 4, 5, 6)
    assert transformed["target"] is target
    assert transformed["id"] == "sample-1"
    assert sample["image"].shape == (1, 2, 3, 4)
    assert torch.equal(transformed["image"][:, 1:3, 1:4, 1:5], image)


def test_pad_center_crop_center_crops_image_to_target_size():
    image = torch.arange(1 * 4 * 5 * 6).reshape(1, 4, 5, 6)
    sample = {"image": image, "target": torch.tensor([1.0])}

    transformed = PadCenterCrop(size=[2, 3, 4])(sample)

    assert transformed["image"].shape == (1, 2, 3, 4)
    assert torch.equal(transformed["image"], image[:, 1:3, 1:4, 1:5])


def test_pad_center_crop_can_target_configured_key():
    image = torch.ones(1, 2, 2, 2)
    auxiliary = torch.ones(1, 1, 1, 1)
    sample = {"image": image, "aux": auxiliary, "target": torch.tensor([1.0])}

    transformed = PadCenterCrop(size=[2, 2, 2], key="aux")(sample)

    assert transformed["image"] is image
    assert transformed["aux"].shape == (1, 2, 2, 2)


def test_pad_center_crop_rejects_non_tensor_value():
    transform = PadCenterCrop(size=[2, 2, 2])

    with pytest.raises(TypeError, match="image"):
        transform({"image": "not-a-tensor", "target": torch.tensor([1.0])})


def test_build_transform_returns_none_for_absent_config():
    assert build_transform(None) is None
    assert build_transform({}) is None


def test_build_transform_builds_pad_center_crop():
    transform = build_transform(
        {"name": "pad_center_crop", "size": [2, 2, 2], "pad_value": 3.0}
    )

    sample = {"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([1.0])}
    transformed = transform(sample)

    assert transformed["image"].shape == (1, 2, 2, 2)
    assert transformed["image"][0, 0, 0, 0] == 3.0


def test_build_transform_rejects_unknown_transform():
    with pytest.raises(ValueError, match="unknown transform 'resize'"):
        build_transform({"name": "resize", "size": [2, 2, 2]})


def test_transform_dataset_applies_transform_lazily():
    dataset = OneSampleDataset(
        {"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([1.0])}
    )
    wrapped = TransformDataset(dataset, PadCenterCrop(size=[2, 2, 2]))

    assert len(wrapped) == 1
    assert wrapped[0]["image"].shape == (1, 2, 2, 2)


def test_apply_transforms_wraps_all_splits():
    train = OneSampleDataset(
        {"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([1.0])}
    )
    val = OneSampleDataset(
        {"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([2.0])}
    )
    test = OneSampleDataset(
        {"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([3.0])}
    )
    bundle = DatasetBundle(train=train, val=val, test=test)

    wrapped = apply_transforms(bundle, {"name": "pad_center_crop", "size": [2, 2, 2]})

    assert wrapped.train[0]["image"].shape == (1, 2, 2, 2)
    assert wrapped.val[0]["image"].shape == (1, 2, 2, 2)
    assert wrapped.test[0]["image"].shape == (1, 2, 2, 2)
