import torch

from .fakes import FakeBackbone, FakeRegressionTask


def test_fake_task_prepare_tracks_overwrite():
    task = FakeRegressionTask()

    task.prepare(overwrite_data=True)

    assert task.prepare_calls == [True]


def test_fake_task_datasets_return_required_schema():
    task = FakeRegressionTask()
    sample = task.datasets().train[0]

    assert set(sample) >= {"image", "target", "id", "meta"}
    assert sample["image"].shape == (1, 2, 2, 2)
    assert sample["target"].shape == (1,)


def test_fake_backbone_returns_token_representations():
    backbone = FakeBackbone(embed_dim=4)

    reps = backbone(torch.zeros(2, 1, 2, 2, 2))

    assert reps["cls"].shape == (2, 1, 4)
    assert reps["reg"].shape == (2, 2, 4)
    assert reps["patch"].shape == (2, 3, 4)
