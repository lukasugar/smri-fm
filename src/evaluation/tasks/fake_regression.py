from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from evaluation.core import DatasetBundle, TargetSpec
from evaluation.metrics import regression_metrics


class TinyRegressionDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        image = torch.full((1, 2, 2, 2), float(index))
        target = torch.tensor([float(index)])
        return {"image": image, "target": target, "id": f"sample-{index}", "meta": {"index": index}}


@dataclass
class FakeRegressionTask:
    name: str = "fake_regression"

    def __post_init__(self) -> None:
        self.prepare_calls: list[bool] = []

    def prepare(self, overwrite_data: bool = False) -> None:
        self.prepare_calls.append(overwrite_data)

    def target_spec(self) -> TargetSpec:
        return TargetSpec(kind="regression", dim=1, loss="mse")

    def datasets(self) -> DatasetBundle:
        return DatasetBundle(
            train=TinyRegressionDataset(8),
            val=TinyRegressionDataset(4),
            test=TinyRegressionDataset(4),
        )

    def collate_fn(self):
        return None

    def metrics(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        return regression_metrics(predictions, targets)
