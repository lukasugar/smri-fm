# Evaluation Sample Transforms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional sample-level evaluation transforms, starting with image-only pad-and-center-crop to a configured spatial size.

**Architecture:** Keep transforms task-agnostic by wrapping datasets returned from `EvaluationTask.datasets()`. Each transform receives the full sample dict and returns the full sample dict, touching only the configured key by default. If transform config is absent, the dataset bundle is returned unchanged.

**Tech Stack:** Python 3.11, PyTorch tensors and `torch.nn.functional.pad`, `torch.utils.data.Dataset`, pytest, OmegaConf-loaded YAML configs.

---

### Task 1: Add Transform Tests

**Files:**
- Create: `src/evaluation/tests/test_transforms.py`

**Step 1: Write failing tests for pad-center-crop**

Create `src/evaluation/tests/test_transforms.py` with these tests:

```python
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
    train = OneSampleDataset({"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([1.0])})
    val = OneSampleDataset({"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([2.0])})
    test = OneSampleDataset({"image": torch.ones(1, 1, 1, 1), "target": torch.tensor([3.0])})
    bundle = DatasetBundle(train=train, val=val, test=test)

    wrapped = apply_transforms(bundle, {"name": "pad_center_crop", "size": [2, 2, 2]})

    assert wrapped.train[0]["image"].shape == (1, 2, 2, 2)
    assert wrapped.val[0]["image"].shape == (1, 2, 2, 2)
    assert wrapped.test[0]["image"].shape == (1, 2, 2, 2)
```

**Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest src/evaluation/tests/test_transforms.py -q
```

Expected: FAIL because `evaluation.transforms` does not exist.

---

### Task 2: Implement Transform Module

**Files:**
- Create: `src/evaluation/transforms.py`
- Test: `src/evaluation/tests/test_transforms.py`

**Step 1: Add the minimal transform implementation**

Create `src/evaluation/transforms.py`:

```python
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from evaluation.core import DatasetBundle


SampleTransform = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class PadCenterCrop:
    size: Sequence[int]
    key: str = "image"
    pad_value: float = 0.0

    def __post_init__(self) -> None:
        if len(self.size) not in {2, 3}:
            raise ValueError("size must contain 2 or 3 spatial dimensions")
        if any(int(dim) <= 0 for dim in self.size):
            raise ValueError("size dimensions must be positive")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        value = sample[self.key]
        if not isinstance(value, Tensor):
            raise TypeError(f"{self.key!r} must be a torch.Tensor")
        transformed = dict(sample)
        transformed[self.key] = self._pad_center_crop(value)
        return transformed

    def _pad_center_crop(self, tensor: Tensor) -> Tensor:
        spatial_size = tuple(int(dim) for dim in self.size)
        if tensor.ndim < len(spatial_size):
            raise ValueError(
                f"{self.key!r} has shape {tuple(tensor.shape)}, "
                f"but size has {len(spatial_size)} spatial dimensions"
            )

        leading_shape = tensor.shape[: tensor.ndim - len(spatial_size)]
        spatial_shape = tensor.shape[-len(spatial_size) :]
        if not leading_shape:
            raise ValueError(f"{self.key!r} must include at least one leading dimension")

        pad_args: list[int] = []
        crop_slices: list[slice] = []
        for current, target in zip(reversed(spatial_shape), reversed(spatial_size), strict=True):
            pad_total = max(0, target - current)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_args.extend([pad_before, pad_after])

        padded = F.pad(tensor, pad_args, mode="constant", value=float(self.pad_value))
        padded_spatial = padded.shape[-len(spatial_size) :]
        for current, target in zip(padded_spatial, spatial_size, strict=True):
            start = max(0, (current - target) // 2)
            crop_slices.append(slice(start, start + target))

        return padded[(..., *crop_slices)]


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: SampleTransform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(self.dataset[index])


def build_transform(cfg: Mapping[str, Any] | None) -> SampleTransform | None:
    if not cfg:
        return None
    name = cfg.get("name")
    if name == "pad_center_crop":
        return PadCenterCrop(
            size=cfg["size"],
            key=str(cfg.get("key", "image")),
            pad_value=float(cfg.get("pad_value", 0.0)),
        )
    raise ValueError("unknown transform {!r}. available transforms: pad_center_crop".format(name))


def apply_transforms(bundle: DatasetBundle, cfg: Mapping[str, Any] | None) -> DatasetBundle:
    transform = build_transform(cfg)
    if transform is None:
        return bundle
    return DatasetBundle(
        train=TransformDataset(bundle.train, transform),
        val=TransformDataset(bundle.val, transform),
        test=TransformDataset(bundle.test, transform),
    )
```

**Step 2: Run transform tests**

Run:

```bash
uv run pytest src/evaluation/tests/test_transforms.py -q
```

Expected: PASS.

**Step 3: Fix only issues revealed by the focused tests**

If a test fails, keep edits limited to `src/evaluation/transforms.py` or the test expectation if the test contradicts the intended sample-in/sample-out contract.

---

### Task 3: Wire Transforms Into ProbeTrainer

**Files:**
- Modify: `src/evaluation/trainers.py`
- Modify: `src/evaluation/tests/test_probe_trainer.py`

**Step 1: Write failing trainer integration test**

Append this test to `src/evaluation/tests/test_probe_trainer.py`:

```python
class ShapeRecordingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1
        self.seen_shapes = []

    def forward(self, images):
        self.seen_shapes.append(tuple(images.shape))
        batch = images.shape[0]
        return {"cls": torch.ones(batch, 1, 1), "reg": None, "patch": None}


def test_probe_trainer_applies_configured_transforms_before_backbone(tmp_path):
    task = FakeRegressionTask()
    backbone = ShapeRecordingBackbone()
    head = LinearHead(input_dim=1, output_dim=1, pooling="first")
    cfg = make_cfg(tmp_path)
    cfg["transforms"] = {"name": "pad_center_crop", "size": [4, 4, 4]}
    trainer = ProbeTrainer(cfg=cfg, backbone=backbone, head=head, task=task)

    trainer.run()

    assert backbone.seen_shapes
    assert all(shape[-3:] == (4, 4, 4) for shape in backbone.seen_shapes)
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest src/evaluation/tests/test_probe_trainer.py::test_probe_trainer_applies_configured_transforms_before_backbone -q
```

Expected: FAIL because `ProbeTrainer` does not apply transforms.

**Step 3: Wire bundle wrapping into trainer**

In `src/evaluation/trainers.py`, add:

```python
from evaluation.transforms import apply_transforms
```

Then update `ProbeTrainer.run()` after dataset construction:

```python
bundle = self.task.datasets()
bundle = apply_transforms(bundle, self.cfg.get("transforms"))
collate_fn = self.task.collate_fn()
```

Keep this before DataLoader creation so train, val, and test splits all see the same configured transform.

**Step 4: Run trainer integration test**

Run:

```bash
uv run pytest src/evaluation/tests/test_probe_trainer.py::test_probe_trainer_applies_configured_transforms_before_backbone -q
```

Expected: PASS.

---

### Task 4: Document Config Shape

**Files:**
- Modify: `src/evaluation/config/default_probe.yaml`
- Modify: `src/evaluation/README.md`

**Step 1: Add disabled default config section**

In `src/evaluation/config/default_probe.yaml`, add this top-level block after `task` or after `model`:

```yaml
# Optional sample-level transforms applied after task datasets are built and
# before batches are passed to the model. Omit or set to null to pass samples
# through unchanged.
transforms: null

# Example:
# transforms:
#   name: pad_center_crop
#   key: image
#   size: [208, 240, 208]
#   pad_value: 0.0
```

**Step 2: Update README config shape**

In `src/evaluation/README.md`, add `transforms: null` to the config shape example and add a short subsection under `## Batch Contract`:

```markdown
## Transforms

Top-level `transforms` is optional. When omitted or set to `null`, task samples
are passed through unchanged. The first supported transform is:

```yaml
transforms:
  name: pad_center_crop
  key: image
  size: [208, 240, 208]
  pad_value: 0.0
```

Transforms receive the full sample dict and return the full sample dict. By
default `pad_center_crop` only modifies `sample["image"]`; all other keys such
as `target`, `id`, and `meta` are preserved.
```

**Step 3: Run docs/config smoke tests**

Run:

```bash
uv run pytest src/evaluation/tests/test_main.py src/evaluation/tests/test_cli_smoke.py -q
```

Expected: PASS.

---

### Task 5: Full Verification

**Files:**
- Test all evaluation tests.

**Step 1: Run focused evaluation test suite**

Run:

```bash
uv run pytest src/evaluation/tests -q
```

Expected: PASS.

**Step 2: Run smoke CLI with no transforms**

Run:

```bash
uv run python -m evaluation.main --config src/evaluation/config/fake_probe.yaml
```

Expected: command exits successfully and behavior is unchanged because no transform is configured.

**Step 3: Run smoke CLI with transform override**

Run:

```bash
uv run python -m evaluation.main \
  --config src/evaluation/config/fake_probe.yaml \
  'transforms.name=pad_center_crop' \
  'transforms.size=[2,2,2]' \
  transforms.key=image \
  transforms.pad_value=0.0
```

Expected: command exits successfully. The fake task images are already `[1, 2, 2, 2]`, so this verifies the config path without changing shape.

**Step 4: Inspect git diff**

Run:

```bash
git diff -- src/evaluation docs/plans/2026-06-10-evaluation-sample-transforms.md
```

Expected: diff only contains the transform module, tests, trainer wiring, docs/config updates, and this plan.

**Step 5: Commit**

Run:

```bash
git add \
  docs/plans/2026-06-10-evaluation-sample-transforms.md \
  src/evaluation/transforms.py \
  src/evaluation/trainers.py \
  src/evaluation/tests/test_transforms.py \
  src/evaluation/tests/test_probe_trainer.py \
  src/evaluation/config/default_probe.yaml \
  src/evaluation/README.md
git commit -m "feat: add evaluation sample transforms"
```

Expected: commit succeeds.
