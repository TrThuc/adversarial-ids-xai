from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    parser = argparse.ArgumentParser(description="Learned SHAP explanation detector")
    parser.add_argument("--detector-dir", required=True)
    parser.add_argument("--mode", choices=["pair", "state"], default="pair")
    parser.add_argument("--classifier", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    detector_dir = Path(args.detector_dir)
    if not detector_dir.is_absolute():
        detector_dir = PROJECT_ROOT / detector_dir

    if args.mode == "pair":
        X = np.load(detector_dir / "pair_X.npy")
        y = np.load(detector_dir / "pair_y_prediction_changed.npy")
        train_ids = np.load(detector_dir / "pair_train_indices.npy")
        val_ids = np.load(detector_dir / "pair_val_indices.npy")
        test_ids = np.load(detector_dir / "pair_test_indices.npy")
        valid = y >= 0
        train_ids, val_ids, test_ids = [ids[valid[ids]] for ids in (train_ids, val_ids, test_ids)]
    else:
        X = np.load(detector_dir / "state_X.npy")
        y = np.load(detector_dir / "state_y.npy")
        train_ids = np.load(detector_dir / "state_train_indices.npy")
        val_ids = np.load(detector_dir / "state_val_indices.npy")
        test_ids = np.load(detector_dir / "state_test_indices.npy")

    if len(np.unique(y[train_ids])) < 2:
        raise ValueError("Training split must contain both detector classes.")

    if args.classifier == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X[train_ids], y[train_ids])
    validation_score = model.predict_proba(X[val_ids])[:, 1]
    validation_prediction = (validation_score >= 0.5).astype(np.int64)
    test_score = model.predict_proba(X[test_ids])[:, 1]
    test_prediction = (test_score >= 0.5).astype(np.int64)

    result = {
        "detector": "learned",
        "mode": args.mode,
        "classifier": args.classifier,
        "num_train": int(len(train_ids)),
        "num_validation": int(len(val_ids)),
        "num_test": int(len(test_ids)),
        "validation": evaluate(y[val_ids], validation_prediction, validation_score),
        "test": evaluate(y[test_ids], test_prediction, test_score),
    }

    output_dir = Path(args.output_dir) if args.output_dir else detector_dir / f"learned_{args.mode}_{args.classifier}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "detector.joblib")
    np.save(output_dir / "test_predictions.npy", test_prediction)
    np.save(output_dir / "test_scores.npy", test_score)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
