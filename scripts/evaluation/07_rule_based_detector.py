from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "results/detector_features"


def metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc_score": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    parser = argparse.ArgumentParser(description="Rule-based SHAP detector")
    parser.add_argument("--detector-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--k", type=int, choices=[5, 10], default=5)
    args = parser.parse_args()

    detector_dir = Path(args.detector_dir)
    if not detector_dir.is_absolute():
        detector_dir = PROJECT_ROOT / detector_dir

    X = np.load(detector_dir / "pair_X.npy")
    y = np.load(detector_dir / "pair_y_prediction_changed.npy")
    train_ids = np.load(detector_dir / "pair_train_indices.npy")
    test_ids = np.load(detector_dir / "pair_test_indices.npy")

    valid = y >= 0
    train_ids = train_ids[valid[train_ids]]
    test_ids = test_ids[valid[test_ids]]
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError("No valid pair labels available in train/test split.")

    drift_col = 0
    overlap_col = 4 if args.k == 5 else 5
    train_X, train_y = X[train_ids], y[train_ids]
    test_X, test_y = X[test_ids], y[test_ids]

    # Grid calibration is performed only on the training split.
    drift_grid = np.unique(np.quantile(train_X[:, drift_col], np.linspace(0.05, 0.95, 19)))
    overlap_grid = np.unique(np.quantile(train_X[:, overlap_col], np.linspace(0.05, 0.95, 19)))
    best = None
    for drift_threshold in drift_grid:
        for overlap_threshold in overlap_grid:
            score = np.maximum(
                train_X[:, drift_col] / max(float(drift_threshold), 1e-12),
                (float(overlap_threshold) - train_X[:, overlap_col]) / max(float(overlap_threshold), 1e-12),
            )
            prediction = (
                (train_X[:, drift_col] > drift_threshold)
                | (train_X[:, overlap_col] < overlap_threshold)
            ).astype(np.int64)
            value = f1_score(train_y, prediction, zero_division=0)
            if best is None or value > best["train_f1"]:
                best = {
                    "drift_threshold": float(drift_threshold),
                    "overlap_threshold": float(overlap_threshold),
                    "train_f1": float(value),
                    "train_score": score,
                }

    drift_threshold = best["drift_threshold"]
    overlap_threshold = best["overlap_threshold"]
    test_score = np.maximum(
        test_X[:, drift_col] / max(drift_threshold, 1e-12),
        (overlap_threshold - test_X[:, overlap_col]) / max(overlap_threshold, 1e-12),
    )
    test_prediction = (
        (test_X[:, drift_col] > drift_threshold)
        | (test_X[:, overlap_col] < overlap_threshold)
    ).astype(np.int64)

    result = {
        "detector": "rule_based",
        "k": args.k,
        "num_train": int(len(train_ids)),
        "num_test": int(len(test_ids)),
        "thresholds": {
            "drift": drift_threshold,
            f"top_{args.k}_overlap": overlap_threshold,
        },
        "train_f1": best["train_f1"],
        "test": metrics(test_y, test_prediction, test_score),
    }

    output_dir = Path(args.output_dir) if args.output_dir else detector_dir / "rule_based"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(output_dir / "test_predictions.npy", test_prediction)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
