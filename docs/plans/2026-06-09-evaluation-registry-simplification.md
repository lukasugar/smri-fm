# Evaluation Registry Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace evaluation `if/elif` factories with simple explicit registries only where they reduce construction boilerplate.

**Architecture:** Tasks use a class registry because current task constructors are simple. Backbones, heads, and trainers use builder-function registries because they need config-dependent construction, extra runtime dependencies, or both. Public `build_*` functions stay in place so `evaluation.main` and callers do not change.

**Tech Stack:** Python 3.11, PyTorch, OmegaConf, pytest. Use `uv run ...` for all Python commands.

---

### Task 1: Task Class Registry

**Files:**
- Modify: `src/evaluation/tasks/__init__.py`
- Test: `src/evaluation/tests/test_registry.py`

**Step 1: Write the failing test**

Add tests that verify task registry behavior through the public API:

```python
from evaluation.tasks import build_task, list_tasks
from evaluation.tasks.fake_regression import FakeRegressionTask


def test_task_registry_builds_registered_task():
    task = build_task({"name": "fake_regression"})

    assert isinstance(task, FakeRegressionTask)


def test_task_registry_lists_available_tasks():
    assert list_tasks() == ["fake_regression"]
```

Also add an unknown-task assertion that expects the available names in the error.

**Step 2: Run test to verify it fails**

Run: `uv run pytest src/evaluation/tests/test_registry.py -v`

Expected: FAIL because `list_tasks` does not exist.

**Step 3: Implement the minimal registry**

Replace the task `if` branch with:

```python
_TASK_REGISTRY: dict[str, type[EvaluationTask]] = {
    "fake_regression": FakeRegressionTask,
}
```

Implement `list_tasks()` and `build_task(cfg)` using the registry. Keep constructor kwargs out for now to preserve current behavior and avoid adding unused config semantics.

**Step 4: Run tests**

Run: `uv run pytest src/evaluation/tests/test_registry.py src/evaluation/tests/test_cli_smoke.py -v`

Expected: PASS.

### Task 2: Builder-Function Registries

**Files:**
- Modify: `src/evaluation/builders.py`
- Test: `src/evaluation/tests/test_builders.py`

**Step 1: Write the failing tests**

Add tests for registry-backed public helpers:

```python
from evaluation.builders import list_backbones, list_heads, list_trainers


def test_builder_registries_list_available_components():
    assert list_backbones() == ["fake", "smri_mae"]
    assert list_heads() == ["linear"]
    assert list_trainers() == ["probe"]
```

Update unknown-component tests to assert the available names come from the registry.

**Step 2: Run test to verify it fails**

Run: `uv run pytest src/evaluation/tests/test_builders.py -v`

Expected: FAIL because the `list_*` helpers do not exist.

**Step 3: Implement minimal builder registries**

In `builders.py`, create private dictionaries:

```python
_BACKBONE_BUILDERS: dict[str, Callable[[Mapping[str, Any]], nn.Module]]
_HEAD_BUILDERS: dict[str, Callable[..., nn.Module]]
_TRAINER_BUILDERS: dict[str, Callable[..., Any]]
```

Use builder functions for each concrete component. Keep `build_backbone`, `build_head`, and `build_trainer` as the public API and have them delegate to the dictionaries.

**Step 4: Run tests**

Run: `uv run pytest src/evaluation/tests/test_builders.py src/evaluation/tests/test_cli_smoke.py -v`

Expected: PASS.

### Task 3: Simplicity Check

**Files:**
- Modify: `src/evaluation/builders.py`
- Modify: `src/evaluation/tasks/__init__.py`

**Step 1: Review complexity**

Confirm the result has fewer or equally clear branches than the current code. If the builder registries require awkward adapters or obscure control flow, revert the builder registry part and keep only the task registry.

**Step 2: Run full evaluation tests**

Run: `uv run pytest src/evaluation/tests -v`

Expected: PASS.

**Step 3: Commit**

```bash
git add docs/plans/2026-06-09-evaluation-registry-simplification.md src/evaluation src/evaluation/tests
git commit -m "Simplify evaluation component registries"
```
