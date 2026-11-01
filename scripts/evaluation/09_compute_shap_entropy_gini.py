from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_shap(run_dir: Path, filename: str):
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    values = np.load(path)
    if values.ndim != 2:
        raise ValueError(f"Expected 2-D SHAP feature array, got {values.shape}")
    return values.astype(np.float64)


def normalize_attributions(values: np.ndarray):
    absolute = np.abs(values)
    return absolute / np.maximum(absolute.sum(axis=1, keepdims=True), 1e-12)


def normalized_entropy(probabilities: np.ndarray):
    feature_count = probabilities.shape[1]
    entropy = -np.sum(
        probabilities * np.log2(np.maximum(probabilities, 1e-12)),
        axis=1,
    )
    return entropy / np.log2(feature_count)


def gini_impurity(probabilities: np.ndarray):
    return 1.0 - np.sum(probabilities ** 2, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP entropy and Gini metrics")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    clean_shap = load_shap(run_dir, "clean_feature_shap.npy")
    adversarial_shap = load_shap(run_dir, "adversarial_feature_shap.npy")
    if clean_shap.shape != adversarial_shap.shape:
        raise ValueError(
            f"Clean/adversarial shapes differ: {clean_shap.shape} vs {adversarial_shap.shape}"
        )

    clean_prob = normalize_attributions(clean_shap)
    adversarial_prob = normalize_attributions(adversarial_shap)

    entropy_clean = normalized_entropy(clean_prob)
    entropy_adversarial = normalized_entropy(adversarial_prob)
    gini_clean = gini_impurity(clean_prob)
    gini_adversarial = gini_impurity(adversarial_prob)

    metrics = np.column_stack(
        [
            entropy_clean,
            entropy_adversarial,
            entropy_adversarial - entropy_clean,
            np.abs(entropy_adversarial - entropy_clean),
            gini_clean,
            gini_adversarial,
            gini_adversarial - gini_clean,
            np.abs(gini_adversarial - gini_clean),
        ]
    ).astype(np.float32)

    metric_names = [
        "entropy_clean",
        "entropy_adversarial",
        "entropy_delta",
        "entropy_abs_delta",
        "gini_clean",
        "gini_adversarial",
        "gini_delta",
        "gini_abs_delta",
    ]

    # A compact feature matrix for a learned pair detector.
    detector_features = metrics[:, [2, 3, 6, 7]]
    detector_feature_names = [
        "entropy_delta",
        "entropy_abs_delta",
        "gini_delta",
        "gini_abs_delta",
    ]

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "entropy_gini"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "entropy_gini_per_sample.npy", metrics)
    np.save(output_dir / "entropy_gini_detector_features.npy", detector_features)

    with open(output_dir / "entropy_gini_per_sample.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index"] + metric_names)
        for index, row in enumerate(metrics):
            writer.writerow([index] + [float(value) for value in row])

    summary = {
        "source_run_dir": str(run_dir.resolve()),
        "num_samples": int(len(metrics)),
        "num_shap_features": int(clean_shap.shape[1]),
        "entropy_definition": "normalized Shannon entropy of normalized absolute SHAP values",
        "gini_definition": "Gini impurity = 1 - sum(p_j^2)",
        "metric_names": metric_names,
        "detector_feature_names": detector_feature_names,
        "summary": {
            name: {
                "mean": float(metrics[:, column].mean()),
                "median": float(np.median(metrics[:, column])),
                "std": float(metrics[:, column].std()),
                "min": float(metrics[:, column].min()),
                "max": float(metrics[:, column].max()),
            }
            for column, name in enumerate(metric_names)
        },
    }
    (output_dir / "entropy_gini_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "detector_feature_names.json").write_text(
        json.dumps(detector_feature_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Entropy/Gini results written to: {output_dir}")
    print(f"Per-sample metrics: {metrics.shape}")
    print(f"Detector features: {detector_features.shape}")
    print(f"Mean entropy clean/adversarial: {entropy_clean.mean():.6f} / {entropy_adversarial.mean():.6f}")
    print(f"Mean Gini clean/adversarial: {gini_clean.mean():.6f} / {gini_adversarial.mean():.6f}")


if __name__ == "__main__":
    main()
