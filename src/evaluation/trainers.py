import csv
import importlib
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.core import EvaluationTask, validate_batch
from evaluation.metrics import is_better
from evaluation.transforms import apply_transforms

# Utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _WandbLogger:
    def __init__(self, wandb: Any | None = None):
        self._wandb = wandb

    @classmethod
    def start(cls, cfg: dict[str, Any]):
        if cfg.get("wandb_logging", True) is False:
            return cls()
        try:
            wandb = importlib.import_module("wandb")
            init_kwargs = {"config": cfg}
            wandb_cfg = cfg.get("wandb", {}) or {}
            project = wandb_cfg.get("project")
            if project:
                init_kwargs["project"] = project
            wandb.init(**init_kwargs)
        except Exception:
            return cls()
        return cls(wandb)

    def log(self, values: dict[str, Any]) -> None:
        if self._wandb is None:
            return
        try:
            self._wandb.log(values)
        except Exception:
            self._wandb = None

# Implementations of custom trainers
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
        wandb_logger = _WandbLogger.start(self.cfg)

        task_cfg = self.cfg.get("task", {})
        self.task.prepare(overwrite_data=bool(task_cfg.get("overwrite_data", False)))
        bundle = self.task.datasets()
        bundle = apply_transforms(bundle, self.cfg.get("transforms"))
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
            train_loss = self._train_one_epoch(loaders["train"], optimizer, loss_fn, device)
            val_metrics, val_loss, _, _ = self._evaluate(loaders["val"], loss_fn, device)
            score = val_metrics[selection["selection_metric"]]
            if is_better(float(score), best_score, selection["selection_mode"]):
                best_score = float(score)
                best_epoch = epoch
                torch.save(self.head.state_dict(), run_dir / "head-best.pt")
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val": val_metrics}
            )
            print(self._format_epoch_log(epoch, train_loss, val_loss, val_metrics), flush=True)
            wandb_logger.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    **{f"val/{name}": value for name, value in val_metrics.items()},
                }
            )

        self.head.load_state_dict(torch.load(run_dir / "head-best.pt", map_location=device))
        val_metrics, val_loss, _, _ = self._evaluate(loaders["val"], loss_fn, device)
        test_metrics, test_loss, test_preds, test_targets = self._evaluate(
            loaders["test"], loss_fn, device
        )

        metrics = {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "history": history,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val": val_metrics,
            "test": test_metrics,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        self._write_predictions(run_dir / "predictions.csv", test_preds, test_targets)
        final_log = {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "paths/run_dir": str(run_dir),
            "paths/head_best_checkpoint": str(run_dir / "head-best.pt"),
            "final/val/loss": val_loss,
            "final/test/loss": test_loss,
            **{f"final/val/{name}": value for name, value in val_metrics.items()},
            **{f"final/test/{name}": value for name, value in test_metrics.items()},
        }
        backbone_checkpoint = (self.cfg.get("model", {}) or {}).get("checkpoint_path")
        if backbone_checkpoint:
            final_log["paths/backbone_checkpoint"] = backbone_checkpoint
        wandb_logger.log(final_log)

        return metrics

    def _train_one_epoch(self, loader, optimizer, loss_fn, device: torch.device) -> float:
        self.head.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            validate_batch(batch)
            targets = batch["target"].to(device).float()
            with torch.no_grad():
                tokens = self._select_tokens(self._forward_backbone(batch, device))
            predictions = self.head(tokens).reshape(targets.shape)
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
        if total_examples == 0:
            return float("nan")
        return total_loss / total_examples

    def _forward_backbone(
        self,
        batch: dict[str, Tensor],
        device: torch.device,
    ) -> dict[str, Tensor | None]:
        images = batch["image"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
            return self.backbone(images, mask=mask)
        return self.backbone(images)

    @torch.no_grad()
    def _evaluate(
        self,
        loader,
        loss_fn,
        device: torch.device,
    ) -> tuple[dict[str, float], float, Tensor, Tensor]:
        self.head.eval()
        predictions = []
        targets = []
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            validate_batch(batch)
            target = batch["target"].to(device).float()
            tokens = self._select_tokens(self._forward_backbone(batch, device))
            prediction = self.head(tokens).reshape(target.shape)
            loss = loss_fn(prediction, target)
            batch_size = target.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            predictions.append(prediction.cpu())
            targets.append(target.cpu())
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)
        loss = total_loss / total_examples if total_examples else float("nan")
        return self.task.metrics(predictions_tensor, targets_tensor), loss, predictions_tensor, targets_tensor

    def _format_epoch_log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict[str, float],
    ) -> str:
        values = [
            f"epoch={epoch}",
            f"train_loss={train_loss:.6g}",
            f"val_loss={val_loss:.6g}",
            *(f"val/{name}={value:.6g}" for name, value in sorted(val_metrics.items())),
        ]
        return " ".join(values)

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



# Factory methods
def _build_probe_trainer(
    mode_cfg: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    backbone,
    head,
    task: EvaluationTask,
) -> Any:
    return ProbeTrainer(cfg=dict(cfg), backbone=backbone, head=head, task=task)


_TRAINER_BUILDERS: dict[str, Callable[..., Any]] = {
    "probe": _build_probe_trainer,
}


def list_trainers() -> list[str]:
    return sorted(_TRAINER_BUILDERS)


def build_trainer(
    mode_cfg: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    backbone,
    head,
    task: EvaluationTask,
):
    name = mode_cfg.get("name")
    try:
        builder = _TRAINER_BUILDERS[name]
    except KeyError:
        available = ", ".join(list_trainers())
        raise ValueError(f"unknown trainer mode {name!r}. available modes: {available}") from None
    return builder(mode_cfg, cfg=cfg, backbone=backbone, head=head, task=task)
