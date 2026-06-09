# Evaluation Suite Design

## Goal

Build a small internal evaluation and fine-tuning suite under `src/evaluation`.
The first useful path is linear probing for regression tasks, while keeping the
module boundaries clear enough to add classification, full fine-tuning, new
heads, and new tasks later.

The suite should not depend on Asparagus. It can use small wrappers around
existing models, especially `smri_mae.model_mae.MaskedViT`, whose
`forward_embedding()` method already returns the backbone representations needed
for evaluation.

## Initial Scope

Implement the generic task registry and config structure from the start. The
first trainer should support probe mode with a linear head for regression.

The following options should be representable in config but may fail with clear
`NotImplementedError` messages until implemented:

- full fine-tuning
- attention head
- classification tasks

## Task API

Tasks own task-specific data preparation, dataset construction, target metadata,
and metrics.

```python
class EvaluationTask(Protocol):
    name: str

    def prepare(self, overwrite_data: bool = False) -> None: ...

    def target_spec(self) -> TargetSpec: ...

    def datasets(self) -> DatasetBundle: ...

    def collate_fn(self) -> Callable | None:
        return None

    def metrics(self, predictions: Tensor, targets: Tensor) -> dict[str, float]: ...
```

`prepare()` is intentionally broad for now. A task may download, index, split,
preprocess, cache tensors, or no-op. The only required behavior is idempotency:
when `overwrite_data=False`, existing prepared data should be reused. When
`overwrite_data=True`, the task may rebuild or download again.

`datasets()` returns lazy PyTorch dataset objects, not loaded images. Dataset
constructors should keep only lightweight state in memory, such as paths,
labels, split IDs, metadata rows, or transforms. NIfTI files and image tensors
should be loaded in `__getitem__()`.

```python
@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
```

## Sample Schema

Datasets should return strict required keys with extensible optional fields.

Required:

- `image`: a single image tensor
- `target`: a target tensor

Optional common fields:

- `id`: string identifier for prediction outputs
- `meta`: metadata dictionary

Optional task- or trainer-specific fields may also be returned, such as `mask`,
`covariates`, or `affine`. The default probe trainer should only depend on
`image` and `target`. Future trainers can document additional required keys.

Runtime validation should inspect an early batch and fail clearly if required
keys are missing.

## Target Metadata

The task describes prediction semantics with a target spec.

```python
@dataclass
class TargetSpec:
    kind: Literal["regression", "classification"]
    dim: int
    loss: str
    primary_metric: str
```

Examples:

```python
TargetSpec(kind="regression", dim=1, loss="mse", primary_metric="mae")
TargetSpec(kind="classification", dim=2, loss="cross_entropy", primary_metric="auroc")
```

The same linear probe trainer can eventually support regression and
classification. Initially, classification can be rejected explicitly while the
API remains ready for it.

## Registry Structure

Use small explicit registries rather than dynamic import discovery.

```python
TASKS = {
    "dlbs_brain_age": DLBSBrainAgeTask,
}

BACKBONES = {
    "smri_mae": build_smri_mae_backbone,
}

HEADS = {
    "linear": LinearHead,
    "attn": NotImplementedAttentionHead,
}

TRAINERS = {
    "probe": ProbeTrainer,
    "full": NotImplementedFullTrainer,
}
```

This keeps extension points easy to grep and easy to test. Adding a task, head,
backbone, or trainer should require adding the implementation and registering it
in one obvious place.

## Backbone, Representation, And Head Boundary

The backbone adapter exposes named token tensors exactly as produced by the
model. It should not pool unless the underlying model representation is already
pooled.

For `smri_mae`, the adapter can call `MaskedViT.forward_embedding()` and return:

```python
{
    "cls": cls_tokens,      # [B, 1, D], if available
    "reg": reg_tokens,      # [B, R, D], if available
    "patch": patch_tokens,  # [B, N, D]
}
```

The config field `representation` selects one key from that dictionary. It does
not decide pooling.

The head owns processing of token sequences. A linear head can pool tokens and
then apply a linear layer. A future attention head can consume the full token
sequence directly.

```python
reps = backbone(images)
tokens = reps[cfg.representation]
predictions = head(tokens)
```

Example linear head behavior:

```python
class LinearHead(nn.Module):
    def forward(self, tokens):
        if self.pooling == "first":
            features = tokens[:, 0]
        elif self.pooling == "mean":
            features = tokens.mean(dim=1)
        else:
            raise ValueError(f"unknown pooling: {self.pooling}")
        return self.linear(features)
```

Suggested defaults:

- `representation: cls`, `head.pooling: first`
- `representation: reg`, `head.pooling: mean`
- `representation: patch`, `head.pooling: mean`

## Config Shape

The YAML should mirror the major pieces directly.

```yaml
name: dlbs_brain_age_cls_probe
output_dir: runs/evaluation

task:
  name: dlbs_brain_age
  overwrite_data: false
  data_root: data/evaluation/dlbs_brain_age

model:
  name: smri_mae
  checkpoint_path: checkpoints/pretrain/checkpoint-last.pth
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

device: cuda
seed: 7338
```

Dispatch should be explicit:

```python
task = build_task(cfg.task)
backbone = build_backbone(cfg.model)
head = build_head(cfg.head, target_spec=task.target_spec())
trainer = build_trainer(cfg.mode, backbone, head, task, cfg)
```

## Probe Trainer Flow

The initial probe trainer should:

1. Set seed and device.
2. Build the task and call `task.prepare(overwrite_data=cfg.task.overwrite_data)`.
3. Build lazy train, validation, and test datasets from `task.datasets()`.
4. Build DataLoaders with `task.collate_fn()`.
5. Build the backbone adapter and freeze it.
6. Build the head after determining feature dimension from model metadata or a
   dry-run batch.
7. Train only the head.
8. Evaluate validation and test splits with task metrics.
9. Write the resolved config, logs, metrics, predictions, and head checkpoint.

The core batch path is:

```python
images = batch["image"]
targets = batch["target"]

reps = backbone(images)
tokens = reps[cfg.representation]
predictions = head(tokens)
```

For regression, prediction and target tensors should be normalized to shape
`[B, dim]`. Classification can be added later with logits shaped
`[B, num_classes]` and integer class targets.

## Validation And Errors

Fail early and explicitly for invalid names:

- unknown task
- unknown backbone
- unknown trainer mode
- unknown head

Validate an early batch for required keys and basic shapes. The backbone adapter
should fail clearly if a requested representation is unavailable, for example
asking for `reg` when the model was built with `reg_tokens=0`.

Unsupported future options should remain visible in config but raise clear
errors:

```python
raise NotImplementedError("full fine-tuning is configured but not implemented yet")
raise NotImplementedError("attention head is configured but not implemented yet")
raise NotImplementedError("classification linear probing is not implemented yet")
```

## Testing Strategy

Start with fake, small components so tests do not require large MRI data or real
checkpoints.

Initial tests:

- registry dispatch for tasks, backbones, heads, and trainers
- `TargetSpec` handling for regression and unsupported classification
- `LinearHead` pooling with `first` and `mean`
- missing batch key validation
- missing representation validation
- regression metrics
- a CPU smoke test that trains for one or two epochs on a fake regression task
  and writes metrics and predictions

Real task tests can come later using tiny fixtures or optional data-dependent
tests.
