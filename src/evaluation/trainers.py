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
        val_metrics, _, _ = self._evaluate(loaders["val"], device)
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
