import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.training.metrics import classification_metrics


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        checkpoint_dir=None,
        log_dir=None,
        monitor="f1_macro",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_dir = Path(log_dir) if log_dir else None

        self.monitor = monitor

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = []
        self.best_metric = float("-inf")
        self.start_epoch = 1

    def train_one_epoch(self, loader, max_batches=None):
        self.model.train()

        total_loss = 0.0
        all_targets = []
        all_predictions = []

        num_batches = len(loader)

        if max_batches is not None:
            num_batches = min(num_batches, max_batches)

        progress = tqdm(
            enumerate(loader, start=1),
            total=num_batches,
            desc="Train",
        )

        for batch_idx, (X, y) in progress:
            if batch_idx > num_batches:
                break

            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(X)

            loss = self.criterion(logits, y)

            loss.backward()

            self.optimizer.step()

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()

            all_targets.extend(y.detach().cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = total_loss / num_batches

        metrics = classification_metrics(
            all_targets,
            all_predictions,
        )

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, loader, max_batches=None):
        self.model.eval()

        total_loss = 0.0
        all_targets = []
        all_predictions = []

        num_batches = len(loader)

        if max_batches is not None:
            num_batches = min(num_batches, max_batches)

        progress = tqdm(
            enumerate(loader, start=1),
            total=num_batches,
            desc="Val",
        )

        for batch_idx, (X, y) in progress:
            if batch_idx > num_batches:
                break

            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(X)

            loss = self.criterion(logits, y)

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()

            all_targets.extend(y.detach().cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())

        avg_loss = total_loss / num_batches

        metrics = classification_metrics(
            all_targets,
            all_predictions,
        )

        return avg_loss, metrics

    def save_checkpoint(self, epoch, filename, config=None):
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )

        if config is not None:
            checkpoint["config"] = config

        path = self.checkpoint_dir / filename

        torch.save(checkpoint, path)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        max_train_batches=None,
        max_val_batches=None,
        config=None,
    ):
        for epoch in range(
            self.start_epoch,
            epochs + 1,
        ):
            epoch_start = time.time()

            print()
            print("=" * 70)
            print(f"Epoch {epoch}/{epochs}")
            print("=" * 70)

            train_loss, train_metrics = self.train_one_epoch(
                train_loader,
                max_batches=max_train_batches,
            )

            val_loss, val_metrics = self.validate(
                val_loader,
                max_batches=max_val_batches,
            )

            if self.scheduler is not None:
                self.scheduler.step(
                    val_metrics["f1_macro"]
                )

            epoch_time = time.time() - epoch_start

            current_lr = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_f1_macro": train_metrics["f1_macro"],
                "train_f1_weighted": train_metrics["f1_weighted"],
                "train_precision_macro": train_metrics[
                    "precision_macro"
                ],
                "train_recall_macro": train_metrics[
                    "recall_macro"
                ],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_precision_macro": val_metrics[
                    "precision_macro"
                ],
                "val_recall_macro": val_metrics[
                    "recall_macro"
                ],
                "epoch_time_sec": epoch_time,
            }

            self.history.append(row)

            print()
            print(
                f"Train Loss      : {train_loss:.6f}"
            )
            print(
                f"Train Accuracy  : {train_metrics['accuracy']:.6f}"
            )
            print(
                f"Train Macro-F1   : {train_metrics['f1_macro']:.6f}"
            )

            print(
                f"Val Loss        : {val_loss:.6f}"
            )
            print(
                f"Val Accuracy    : {val_metrics['accuracy']:.6f}"
            )
            print(
                f"Val Macro-F1     : {val_metrics['f1_macro']:.6f}"
            )

            print(
                f"Learning Rate   : {current_lr:.8f}"
            )
            print(
                f"Epoch Time      : {epoch_time:.2f} sec"
            )

            current_metric = val_metrics[self.monitor]

            if current_metric > self.best_metric:
                self.best_metric = current_metric

                self.save_checkpoint(
                    epoch,
                    "best.pt",
                    config=config,
                )

                print(
                    f"Best checkpoint updated: "
                    f"{self.best_metric:.6f}"
                )

            self.save_checkpoint(
                epoch,
                "last.pt",
                config=config,
            )

            self.save_history()

        return self.history

    def save_history(self):
        if self.log_dir is None:
            return

        path = self.log_dir / "training_history.csv"

        if not self.history:
            return

        fieldnames = list(self.history[0].keys())

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(self.history)