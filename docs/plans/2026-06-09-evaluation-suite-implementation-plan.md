# Evaluation Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the initial `src/evaluation` package with registry-based tasks, backbone adapters, linear probing, config loading, metrics, outputs, and tests.

**Architecture:** The implementation follows the validated design in `docs/plans/2026-06-09-evaluation-suite-design.md`. Tasks own data preparation, lazy datasets, target metadata, optional collation, and metrics. The probe trainer freezes a backbone adapter, selects one named token representation, trains a head, selects the best checkpoint from validation metrics, and writes run artifacts.

**Tech Stack:** Python 3.11, PyTorch, OmegaConf, pytest, NumPy, pandas, nibabel only when real MRI tasks are added. Use `uv run ...` for all Python commands.

---

## Scope

This plan implements the reusable evaluation framework and verifies it with fake
components and CPU smoke tests. It intentionally does not implement a real
DLBS/FOMO downloader in the first pass, because that requires local data and
task-specific decisions. The task API will be ready for those tasks.

## Task 1: Core Types And Batch Validation

**Files:**

- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/core.py`
- Create: `tests/evaluation/test_core.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_core.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_core.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/__init__.py`:

```python
"""Internal evaluation and fine-tuning utilities."""
```

Create `src/evaluation/core.py`:

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TargetSpec:
    kind: Literal["regression", "classification"]
    dim: int
    loss: str


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset


class EvaluationTask(Protocol):
    name: str

    def prepare(self, overwrite_data: bool = False) -> None: ...

    def target_spec(self) -> TargetSpec: ...

    def datasets(self) -> DatasetBundle: ...

    def collate_fn(self) -> Callable | None: ...

    def metrics(self, predictions: Tensor, targets: Tensor) -> dict[str, float]: ...


def validate_batch(batch: dict) -> None:
    missing = [key for key in ("image", "target") if key not in batch]
    if missing:
        raise ValueError(
            "evaluation batches must contain required keys: "
            f"{', '.join(missing)} missing"
        )
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_core.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/__init__.py src/evaluation/core.py tests/evaluation/test_core.py
git commit -m "Add evaluation core types"
```

## Task 2: Explicit Registries

**Files:**

- Create: `src/evaluation/registry.py`
- Create: `tests/evaluation/test_registry.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_registry.py`:

```python
import pytest

from evaluation.registry import Registry


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
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.registry'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/registry.py`:

```python
from collections.abc import Callable, Mapping
from typing import Any


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._builders: dict[str, Callable[[Mapping[str, Any]], Any]] = {}

    def register(self, name: str, builder: Callable[[Mapping[str, Any]], Any]) -> None:
        if name in self._builders:
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._builders[name] = builder

    def build(self, cfg: Mapping[str, Any]) -> Any:
        name = cfg.get("name")
        if not name:
            raise ValueError(f"{self.kind} config requires a 'name'")
        if name not in self._builders:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise ValueError(f"unknown {self.kind} '{name}'. available {self.kind}s: {available}")
        return self._builders[str(name)](cfg)

    def names(self) -> list[str]:
        return sorted(self._builders)
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_registry.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/registry.py tests/evaluation/test_registry.py
git commit -m "Add evaluation registries"
```

## Task 3: Linear Head

**Files:**

- Create: `src/evaluation/heads.py`
- Create: `tests/evaluation/test_heads.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_heads.py`:

```python
import pytest
import torch

from evaluation.heads import LinearHead


def test_linear_head_first_pooling():
    head = LinearHead(input_dim=3, output_dim=1, pooling="first")
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()

    tokens = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])

    assert torch.equal(head(tokens), torch.tensor([[6.0]]))


def test_linear_head_mean_pooling():
    head = LinearHead(input_dim=2, output_dim=1, pooling="mean")
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()

    tokens = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    assert torch.equal(head(tokens), torch.tensor([[8.0]]))


def test_linear_head_requires_token_sequence():
    head = LinearHead(input_dim=2, output_dim=1, pooling="mean")

    with pytest.raises(ValueError, match="\\[B, T, D\\]"):
        head(torch.zeros(2, 2))


def test_linear_head_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="unknown pooling"):
        LinearHead(input_dim=2, output_dim=1, pooling="max")
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_heads.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.heads'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/heads.py`:

```python
import torch.nn as nn
from torch import Tensor


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pooling: str):
        super().__init__()
        if pooling not in {"first", "mean"}:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"LinearHead expects token sequence shaped [B, T, D], got {tokens.shape}")
        if self.pooling == "first":
            features = tokens[:, 0]
        elif self.pooling == "mean":
            features = tokens.mean(dim=1)
        else:
            raise AssertionError(f"unreachable pooling: {self.pooling}")
        return self.linear(features)
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_heads.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/heads.py tests/evaluation/test_heads.py
git commit -m "Add linear evaluation head"
```

## Task 4: Regression Metrics And Selection

**Files:**

- Create: `src/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_metrics.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.metrics'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/metrics.py`:

```python
import torch
from torch import Tensor


def regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    predictions = predictions.detach().float().reshape(targets.shape)
    targets = targets.detach().float()
    residuals = predictions - targets
    mae = residuals.abs().mean()
    rmse = torch.sqrt((residuals.square()).mean())
    bias = residuals.mean()
    total = ((targets - targets.mean()).square()).sum()
    residual = residuals.square().sum()
    r2 = 1.0 - residual / total if total > 0 else torch.tensor(float("nan"))
    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "bias": float(bias.item()),
        "r2": float(r2.item()),
    }


def is_better(value: float, best: float | None, selection_mode: str) -> bool:
    if selection_mode == "min":
        return best is None or value < best
    if selection_mode == "max":
        return best is None or value > best
    raise ValueError(f"selection_mode must be 'min' or 'max', got {selection_mode!r}")
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_metrics.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "Add evaluation metrics"
```

## Task 5: Fake Components For Trainer Tests

**Files:**

- Create: `tests/evaluation/fakes.py`
- Create: `tests/evaluation/test_fakes.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_fakes.py`:

```python
import torch

from tests.evaluation.fakes import FakeBackbone, FakeRegressionTask


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
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_fakes.py -v
```

Expected: FAIL because `tests.evaluation.fakes` does not exist.

**Step 3: Write minimal implementation**

Create `tests/evaluation/__init__.py` if pytest cannot import the package.

Create `tests/evaluation/fakes.py`:

```python
from dataclasses import dataclass

import torch
import torch.nn as nn
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


class FakeBackbone(nn.Module):
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(8, embed_dim)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        batch = images.shape[0]
        flat = images.reshape(batch, -1).float()
        base = self.proj(flat)
        return {
            "cls": base[:, None, :],
            "reg": torch.stack([base, base + 1.0], dim=1),
            "patch": torch.stack([base, base + 1.0, base + 2.0], dim=1),
        }
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_fakes.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/evaluation/__init__.py tests/evaluation/fakes.py tests/evaluation/test_fakes.py
git commit -m "Add fake evaluation components"
```

## Task 6: Probe Trainer Core Loop

**Files:**

- Create: `src/evaluation/trainers.py`
- Create: `tests/evaluation/test_probe_trainer.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_probe_trainer.py`:

```python
import json

import pytest
import torch

from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer
from tests.evaluation.fakes import FakeBackbone, FakeRegressionTask


def make_cfg(tmp_path):
    return {
        "name": "fake_probe",
        "output_dir": str(tmp_path),
        "task": {"overwrite_data": True},
        "representation": "cls",
        "optimization": {
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_workers": 0,
        },
        "evaluation": {"selection_metric": "mae", "selection_mode": "min"},
        "device": "cpu",
        "seed": 7338,
    }


def test_probe_trainer_runs_and_writes_outputs(tmp_path):
    task = FakeRegressionTask()
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=1, pooling="first")
    trainer = ProbeTrainer(cfg=make_cfg(tmp_path), backbone=backbone, head=head, task=task)

    result = trainer.run()

    run_dir = tmp_path / "fake_probe"
    assert task.prepare_calls == [True]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "head-best.pt").exists()
    assert result["best_epoch"] is not None

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "val" in metrics
    assert "test" in metrics


def test_probe_trainer_rejects_missing_representation(tmp_path):
    task = FakeRegressionTask()
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=1, pooling="first")
    cfg = make_cfg(tmp_path)
    cfg["representation"] = "missing"
    trainer = ProbeTrainer(cfg=cfg, backbone=backbone, head=head, task=task)

    with pytest.raises(ValueError, match="representation 'missing' is unavailable"):
        trainer.run()


def test_probe_trainer_rejects_classification_for_now(tmp_path):
    task = FakeRegressionTask()
    task.target_spec = lambda: type(task.target_spec())(
        kind="classification",
        dim=2,
        loss="cross_entropy",
    )
    backbone = FakeBackbone(embed_dim=4)
    head = LinearHead(input_dim=4, output_dim=2, pooling="first")
    trainer = ProbeTrainer(cfg=make_cfg(tmp_path), backbone=backbone, head=head, task=task)

    with pytest.raises(NotImplementedError, match="classification"):
        trainer.run()
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_probe_trainer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.trainers'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/trainers.py` with:

```python
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.core import EvaluationTask, validate_batch
from evaluation.metrics import is_better


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProbeTrainer:
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        backbone: nn.Module,
        head: nn.Module,
        task: EvaluationTask,
    ):
        self.cfg = cfg
        self.backbone = backbone
        self.head = head
        self.task = task

    def run(self) -> dict[str, Any]:
        spec = self.task.target_spec()
        if spec.kind != "regression":
            raise NotImplementedError("classification linear probing is not implemented yet")

        set_seed(int(self.cfg.get("seed", 0)))
        device = torch.device(self.cfg.get("device", "cpu"))
        run_dir = Path(self.cfg["output_dir"]) / self.cfg["name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        task_cfg = self.cfg.get("task", {})
        self.task.prepare(overwrite_data=bool(task_cfg.get("overwrite_data", False)))
        bundle = self.task.datasets()
        collate_fn = self.task.collate_fn()
        opt_cfg = self.cfg["optimization"]
        loaders = {
            "train": DataLoader(
                bundle.train,
                batch_size=int(opt_cfg["batch_size"]),
                shuffle=True,
                num_workers=int(opt_cfg.get("num_workers", 0)),
                collate_fn=collate_fn,
            ),
            "val": DataLoader(
                bundle.val,
                batch_size=int(opt_cfg["batch_size"]),
                shuffle=False,
                num_workers=int(opt_cfg.get("num_workers", 0)),
                collate_fn=collate_fn,
            ),
            "test": DataLoader(
                bundle.test,
                batch_size=int(opt_cfg["batch_size"]),
                shuffle=False,
                num_workers=int(opt_cfg.get("num_workers", 0)),
                collate_fn=collate_fn,
            ),
        }

        self.backbone.to(device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.head.to(device).train()

        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
        loss_fn = nn.MSELoss()

        selection = self.cfg["evaluation"]
        best_score: float | None = None
        best_epoch: int | None = None
        history = []

        for epoch in range(int(opt_cfg["epochs"])):
            self._train_one_epoch(loaders["train"], optimizer, loss_fn, device)
            val_metrics, _, _ = self._evaluate(loaders["val"], device)
            score = val_metrics[selection["selection_metric"]]
            if is_better(float(score), best_score, selection["selection_mode"]):
                best_score = float(score)
                best_epoch = epoch
                torch.save(self.head.state_dict(), run_dir / "head-best.pt")
            history.append({"epoch": epoch, "val": val_metrics})

        self.head.load_state_dict(torch.load(run_dir / "head-best.pt", map_location=device))
        val_metrics, val_preds, val_targets = self._evaluate(loaders["val"], device)
        test_metrics, test_preds, test_targets = self._evaluate(loaders["test"], device)

        metrics = {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "history": history,
            "val": val_metrics,
            "test": test_metrics,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        self._write_predictions(run_dir / "predictions.csv", test_preds, test_targets)

        return metrics

    def _train_one_epoch(self, loader, optimizer, loss_fn, device: torch.device) -> None:
        self.head.train()
        for batch in loader:
            validate_batch(batch)
            images = batch["image"].to(device)
            targets = batch["target"].to(device).float()
            with torch.no_grad():
                tokens = self._select_tokens(self.backbone(images))
            predictions = self.head(tokens).reshape(targets.shape)
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _evaluate(self, loader, device: torch.device) -> tuple[dict[str, float], Tensor, Tensor]:
        self.head.eval()
        predictions = []
        targets = []
        for batch in loader:
            validate_batch(batch)
            images = batch["image"].to(device)
            target = batch["target"].to(device).float()
            tokens = self._select_tokens(self.backbone(images))
            prediction = self.head(tokens).reshape(target.shape)
            predictions.append(prediction.cpu())
            targets.append(target.cpu())
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)
        return self.task.metrics(predictions_tensor, targets_tensor), predictions_tensor, targets_tensor

    def _select_tokens(self, reps: dict[str, Tensor]) -> Tensor:
        representation = self.cfg["representation"]
        if representation not in reps or reps[representation] is None:
            available = sorted(key for key, value in reps.items() if value is not None)
            raise ValueError(
                f"representation '{representation}' is unavailable; available: {available}"
            )
        return reps[representation]

    def _write_predictions(self, path: Path, predictions: Tensor, targets: Tensor) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "prediction", "target"])
            writer.writeheader()
            for index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
                writer.writerow(
                    {
                        "index": index,
                        "prediction": float(prediction.reshape(-1)[0]),
                        "target": float(target.reshape(-1)[0]),
                    }
                )
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_probe_trainer.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/trainers.py tests/evaluation/test_probe_trainer.py
git commit -m "Add probe trainer"
```

## Task 7: sMRI MAE Backbone Adapter

**Files:**

- Create: `src/evaluation/backbones.py`
- Create: `tests/evaluation/test_backbones.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_backbones.py`:

```python
from pathlib import Path

import torch

from evaluation.backbones import SmriMaeBackbone, load_smri_mae_checkpoint


def tiny_model_kwargs():
    return {
        "img_size": [8, 8, 8],
        "patch_size": 4,
        "in_chans": 1,
        "depth": 1,
        "embed_dim": 8,
        "num_heads": 2,
        "class_token": True,
        "reg_tokens": 2,
    }


def test_smri_mae_backbone_returns_named_representations():
    backbone = SmriMaeBackbone(**tiny_model_kwargs())

    reps = backbone(torch.zeros(2, 1, 8, 8, 8))

    assert backbone.embed_dim == 8
    assert reps["cls"].shape == (2, 1, 8)
    assert reps["reg"].shape == (2, 2, 8)
    assert reps["patch"].shape == (2, 8, 8)


def test_smri_mae_backbone_omits_unavailable_registers():
    kwargs = tiny_model_kwargs()
    kwargs["reg_tokens"] = 0
    backbone = SmriMaeBackbone(**kwargs)

    reps = backbone(torch.zeros(2, 1, 8, 8, 8))

    assert reps["reg"] is None


def test_load_smri_mae_checkpoint_accepts_pretrain_model_key(tmp_path):
    backbone = SmriMaeBackbone(**tiny_model_kwargs())
    checkpoint_path = Path(tmp_path) / "checkpoint.pt"
    torch.save({"model": backbone.model.state_dict()}, checkpoint_path)

    load_smri_mae_checkpoint(backbone.model, checkpoint_path)
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_backbones.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.backbones'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/backbones.py`:

```python
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from smri_mae.model_mae import MaskedViT


class SmriMaeBackbone(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.model = MaskedViT(**kwargs)
        self.embed_dim = self.model.patch_embed.out_features

    def forward(self, images: Tensor) -> dict[str, Tensor | None]:
        cls, reg, patch = self.model.forward_embedding(images)
        return {"cls": cls, "reg": reg, "patch": patch}


def load_smri_mae_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")
```

If real checkpoints contain extra prefixes, extend `load_smri_mae_checkpoint`
with tested key normalization rather than adding ad hoc logic inside the trainer.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_backbones.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/backbones.py tests/evaluation/test_backbones.py
git commit -m "Add sMRI MAE evaluation backbone"
```

## Task 8: Builders And Unsupported Future Options

**Files:**

- Create: `src/evaluation/builders.py`
- Create: `tests/evaluation/test_builders.py`
- Modify: `src/evaluation/backbones.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_builders.py`:

```python
import pytest

from evaluation.builders import build_backbone, build_head, build_trainer
from evaluation.core import TargetSpec
from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer
from tests.evaluation.fakes import FakeBackbone, FakeRegressionTask


def test_build_head_linear():
    head = build_head(
        {"name": "linear", "pooling": "mean"},
        target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
        input_dim=4,
    )

    assert isinstance(head, LinearHead)


def test_build_head_attn_not_implemented():
    with pytest.raises(NotImplementedError, match="attention head"):
        build_head(
            {"name": "attn"},
            target_spec=TargetSpec(kind="regression", dim=1, loss="mse"),
            input_dim=4,
        )


def test_build_trainer_probe():
    trainer = build_trainer(
        {"name": "probe"},
        cfg={"optimization": {}, "evaluation": {}},
        backbone=FakeBackbone(),
        head=LinearHead(4, 1, "first"),
        task=FakeRegressionTask(),
    )

    assert isinstance(trainer, ProbeTrainer)


def test_build_trainer_full_not_implemented():
    with pytest.raises(NotImplementedError, match="full fine-tuning"):
        build_trainer(
            {"name": "full"},
            cfg={},
            backbone=FakeBackbone(),
            head=LinearHead(4, 1, "first"),
            task=FakeRegressionTask(),
        )
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_builders.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.builders'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/builders.py`:

```python
from collections.abc import Mapping
from typing import Any

from evaluation.backbones import SmriMaeBackbone, load_smri_mae_checkpoint
from evaluation.core import EvaluationTask, TargetSpec
from evaluation.heads import LinearHead
from evaluation.trainers import ProbeTrainer


def build_backbone(cfg: Mapping[str, Any]):
    name = cfg.get("name")
    if name != "smri_mae":
        raise ValueError("unknown backbone {!r}. available backbones: smri_mae".format(name))
    kwargs = {
        "img_size": cfg["img_size"],
        "patch_size": cfg["patch_size"],
        "in_chans": cfg.get("in_chans", 1),
        **dict(cfg.get("model_kwargs") or {}),
    }
    backbone = SmriMaeBackbone(**kwargs)
    if cfg.get("checkpoint_path"):
        load_smri_mae_checkpoint(backbone.model, cfg["checkpoint_path"])
    return backbone


def build_head(cfg: Mapping[str, Any], *, target_spec: TargetSpec, input_dim: int):
    name = cfg.get("name")
    if name == "linear":
        return LinearHead(input_dim=input_dim, output_dim=target_spec.dim, pooling=cfg["pooling"])
    if name == "attn":
        raise NotImplementedError("attention head is configured but not implemented yet")
    raise ValueError("unknown head {!r}. available heads: attn, linear".format(name))


def build_trainer(
    mode_cfg: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    backbone,
    head,
    task: EvaluationTask,
):
    name = mode_cfg.get("name")
    if name == "probe":
        return ProbeTrainer(cfg=dict(cfg), backbone=backbone, head=head, task=task)
    if name == "full":
        raise NotImplementedError("full fine-tuning is configured but not implemented yet")
    raise ValueError("unknown trainer mode {!r}. available modes: full, probe".format(name))
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_builders.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/builders.py tests/evaluation/test_builders.py
git commit -m "Add evaluation builders"
```

## Task 9: Config Loader And CLI

**Files:**

- Create: `src/evaluation/main.py`
- Create: `src/evaluation/config/default_probe.yaml`
- Create: `tests/evaluation/test_main.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_main.py`:

```python
from pathlib import Path

from omegaconf import OmegaConf

from evaluation.main import load_config


def test_load_config_merges_overrides(tmp_path):
    cfg_path = Path(tmp_path) / "config.yaml"
    cfg_path.write_text(
        """
name: base
output_dir: runs/evaluation
optimization:
  epochs: 1
""".strip()
    )

    cfg = load_config(cfg_path, ["name=override", "optimization.epochs=2"])

    assert cfg.name == "override"
    assert cfg.optimization.epochs == 2
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_main.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.main'`.

**Step 3: Write minimal implementation**

Create `src/evaluation/config/default_probe.yaml` from the design doc, with
small CPU-friendly defaults:

```yaml
name: eval_probe
output_dir: runs/evaluation

task:
  name: null
  overwrite_data: false

model:
  name: smri_mae
  checkpoint_path: null
  img_size: [208, 240, 208]
  patch_size: 8
  in_chans: 1
  model_kwargs:
    class_token: true
    reg_tokens: 0

mode:
  name: probe

representation: cls

head:
  name: linear
  pooling: first

optimization:
  epochs: 50
  batch_size: 8
  lr: 1e-3
  weight_decay: 0.0
  num_workers: 0

evaluation:
  selection_metric: mae
  selection_mode: min

device: cuda
seed: 7338
```

Create `src/evaluation/main.py`:

```python
import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    raise NotImplementedError("wire task registry and end-to-end CLI in the next task")


if __name__ == "__main__":
    cli()
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_main.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation/main.py src/evaluation/config/default_probe.yaml tests/evaluation/test_main.py
git commit -m "Add evaluation config loader"
```

## Task 10: End-To-End Test Task And CLI Smoke Test

**Files:**

- Create: `src/evaluation/tasks/__init__.py`
- Create: `src/evaluation/tasks/fake_regression.py`
- Modify: `src/evaluation/main.py`
- Create: `tests/evaluation/test_cli_smoke.py`

**Step 1: Write the failing tests**

Create `tests/evaluation/test_cli_smoke.py`:

```python
from pathlib import Path

from evaluation.main import main


def test_main_runs_fake_probe(tmp_path):
    cfg_path = Path(tmp_path) / "config.yaml"
    cfg_path.write_text(
        f"""
name: fake_probe
output_dir: {tmp_path}
task:
  name: fake_regression
  overwrite_data: false
model:
  name: fake
  embed_dim: 4
mode:
  name: probe
representation: cls
head:
  name: linear
  pooling: first
optimization:
  epochs: 1
  batch_size: 2
  lr: 1e-2
  weight_decay: 0.0
  num_workers: 0
evaluation:
  selection_metric: mae
  selection_mode: min
device: cpu
seed: 7338
""".strip()
    )

    main(cfg_path, [])

    assert (tmp_path / "fake_probe" / "metrics.json").exists()
    assert (tmp_path / "fake_probe" / "predictions.csv").exists()
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/evaluation/test_cli_smoke.py -v
```

Expected: FAIL because `evaluation.main.main` and the fake task/model registry
path are not wired.

**Step 3: Write minimal implementation**

Create `src/evaluation/tasks/fake_regression.py` by moving the fake task/dataset
implementation from `tests/evaluation/fakes.py` into production code. Keep the
test helper importing or subclassing from production code to avoid duplication.

Create `src/evaluation/tasks/__init__.py`:

```python
from evaluation.tasks.fake_regression import FakeRegressionTask


def build_task(cfg):
    name = cfg.get("name")
    if name == "fake_regression":
        return FakeRegressionTask()
    raise ValueError("unknown task {!r}. available tasks: fake_regression".format(name))
```

Modify `src/evaluation/builders.py` to support a fake backbone for smoke tests:

```python
class FakeBackbone(nn.Module):
    ...
```

Use the same fake backbone behavior from `tests/evaluation/fakes.py`, or import
the production fake into the tests. This fake backbone is intentionally useful
only for smoke tests and examples.

Modify `src/evaluation/main.py`:

```python
from pathlib import Path

from omegaconf import OmegaConf

from evaluation.builders import build_backbone, build_head, build_trainer
from evaluation.tasks import build_task


def main(config_path: str | Path, overrides: list[str] | None = None):
    cfg = load_config(config_path, overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task = build_task(cfg_dict["task"])
    target_spec = task.target_spec()

    backbone = build_backbone(cfg_dict["model"])
    head = build_head(cfg_dict["head"], target_spec=target_spec, input_dim=backbone.embed_dim)
    trainer = build_trainer(
        cfg_dict["mode"],
        cfg=cfg_dict,
        backbone=backbone,
        head=head,
        task=task,
    )
    return trainer.run()
```

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/evaluation/test_cli_smoke.py -v
```

Expected: PASS.

**Step 5: Run the full evaluation test suite**

Run:

```bash
uv run pytest tests/evaluation -v
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/evaluation tests/evaluation
git commit -m "Wire evaluation CLI smoke path"
```

## Task 11: Documentation And Example Config

**Files:**

- Create: `src/evaluation/README.md`
- Create: `src/evaluation/config/fake_probe.yaml`
- Modify: `README.md` if a short pointer is useful

**Step 1: Write documentation**

Create `src/evaluation/README.md`:

````markdown
# evaluation

Internal evaluation and fine-tuning suite.

## Run a smoke probe

```bash
uv run python -m evaluation.main --config src/evaluation/config/fake_probe.yaml
```

Outputs are written under `runs/evaluation/<name>/`:

- `metrics.json`
- `predictions.csv`
- `head-best.pt`

## Add a task

Add a task class implementing `EvaluationTask`, register it in
`evaluation.tasks.build_task`, and make its datasets return at least:

- `image`
- `target`

Optional keys like `id`, `meta`, `mask`, or `covariates` are allowed.

## Add a backbone

Add a backbone adapter that returns named token sequences such as `cls`, `reg`,
or `patch`. The adapter should not pool tokens; heads own pooling.
````

Create `src/evaluation/config/fake_probe.yaml` using the config from Task 10.

**Step 2: Run docs smoke command**

Run:

```bash
uv run python -m evaluation.main --config src/evaluation/config/fake_probe.yaml
```

Expected: command exits successfully and writes `runs/evaluation/fake_probe/metrics.json`.

**Step 3: Run tests**

Run:

```bash
uv run pytest tests/evaluation -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add src/evaluation/README.md src/evaluation/config/fake_probe.yaml README.md
git commit -m "Document evaluation smoke workflow"
```

## Task 12: Final Verification

**Files:**

- No new files expected.

**Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/evaluation -v
```

Expected: PASS.

**Step 2: Run existing tests**

Run:

```bash
uv run pytest src/asparagus_bridge/test_smri_mae_segmentation.py -v
```

Expected: PASS or a clearly documented environment/data dependency. If it fails
because of an unrelated dependency issue, record the exact failure in the final
implementation notes.

**Step 3: Run all non-third-party tests**

Run:

```bash
uv run pytest -v
```

Expected: PASS.

**Step 4: Inspect git diff**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected: only intended evaluation files and docs changed since the last commit.

**Step 5: Final commit if needed**

If any verification fixes or docs edits remain:

```bash
git add <changed-files>
git commit -m "Finish evaluation suite implementation"
```

## Follow-Up Work

Add real tasks after this framework lands:

- `dlbs_brain_age`: local data preparation, manifest creation, lazy NIfTI loading,
  regression metrics, split handling.
- FOMO brain-age task if needed.
- classification linear probing once a classification task is selected.
- attention head once token-sequence modeling is ready.
- full fine-tuning mode after the probe path is stable.
