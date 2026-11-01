import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def evaluate_model(
    model,
    loader,
    device,
    class_names,
):
    model.eval()

    all_targets = []
    all_predictions = []

    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Test"):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)

            loss = criterion(logits, y)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_loss += loss.item()

            all_targets.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    return metrics, report, cm, y_true, y_pred


def save_evaluation_results(
    metrics,
    report,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir / "test_metrics.json"
    )

    report_path = (
        output_dir / "classification_report.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    return metrics_path, report_path