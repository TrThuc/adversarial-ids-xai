from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "logs/xai/shap/lstm_baseline_a2/fgsm_epsilon_0.10/gradient/nsamples_2000"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/detector_features"


def load_array(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return np.load(path)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.abs(values.astype(np.float64))
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def split_indices(n: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)
    return (
        order[:train_end].astype(np.int64),
        order[train_end:validation_end].astype(np.int64),
        order[validation_end:].astype(np.int64),
    )


def make_pair_features(clean: np.ndarray, adversarial: np.ndarray):
    clean_norm = normalize_rows(clean)
    adversarial_norm = normalize_rows(adversarial)
    shift = np.abs(clean_norm - adversarial_norm)

    clean_top5 = np.argpartition(-clean_norm, 5 - 1, axis=1)[:, :5]
    adv_top5 = np.argpartition(-adversarial_norm, 5 - 1, axis=1)[:, :5]
    clean_top10 = np.argpartition(-clean_norm, 10 - 1, axis=1)[:, :10]
    adv_top10 = np.argpartition(-adversarial_norm, 10 - 1, axis=1)[:, :10]

    def overlap(left, right):
        return np.asarray(
            [len(set(a.tolist()) & set(b.tolist())) / len(a) for a, b in zip(left, right)],
            dtype=np.float32,
        )

    feature_names = [
        "drift_mean",
        "shift_std",
        "shift_max",
        "shift_l2",
        "top5_overlap",
        "top10_overlap",
        "large_shift_fraction",
    ]
    features = np.column_stack(
        [
            shift.mean(axis=1),
            shift.std(axis=1),
            shift.max(axis=1),
            np.linalg.norm(shift, axis=1),
            overlap(clean_top5, adv_top5),
            overlap(clean_top10, adv_top10),
            (shift > 0.01).mean(axis=1),
        ]
    ).astype(np.float32)
    return features, feature_names, clean_norm, adversarial_norm


def make_state_features(attributions: np.ndarray):
    normalized = normalize_rows(attributions)
    sorted_values = np.sort(normalized, axis=1)[:, ::-1]
    entropy = -np.sum(
        normalized * np.log2(np.maximum(normalized, 1e-12)),
        axis=1,
    )
    features = np.column_stack(
        [
            normalized,
            entropy,
            sorted_values[:, :5].sum(axis=1),
            sorted_values[:, :10].sum(axis=1),
            sorted_values[:, 0],
        ]
    ).astype(np.float32)
    feature_names = (
        [f"normalized_shap_{i:02d}" for i in range(normalized.shape[1])]
        + ["shap_entropy", "top5_mass", "top10_mass", "max_mass"]
    )
    return features, feature_names


def main():
    parser = argparse.ArgumentParser(description="Build detector feature datasets from SHAP artifacts")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--attack-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    clean = load_array(run_dir / "clean_feature_shap.npy", "clean SHAP features")
    adversarial = load_array(run_dir / "adversarial_feature_shap.npy", "adversarial SHAP features")
    if clean.ndim != 2 or clean.shape != adversarial.shape:
        raise ValueError(f"Expected matching 2-D SHAP arrays, got {clean.shape} and {adversarial.shape}")

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    if args.attack_dir:
        attack_dir = Path(args.attack_dir)
        if not attack_dir.is_absolute():
            attack_dir = PROJECT_ROOT / attack_dir
    else:
        attack_dir = Path(metrics.get("artifact_dir", ""))

    selected_indices = load_array(run_dir / "selected_attack_indices.npy", "selected attack indices").astype(np.int64)
    if attack_dir and attack_dir.exists():
        clean_predictions = load_array(attack_dir / "clean_predictions.npy", "clean predictions")
        adversarial_predictions = load_array(attack_dir / "adversarial_predictions.npy", "adversarial predictions")
        if selected_indices.max(initial=-1) >= len(clean_predictions):
            raise ValueError("selected_attack_indices exceed attack prediction arrays")
        prediction_changed = (
            clean_predictions[selected_indices] != adversarial_predictions[selected_indices]
        ).astype(np.int64)
    else:
        prediction_changed = np.full(len(clean), -1, dtype=np.int64)

    pair_features, pair_names, clean_norm, adversarial_norm = make_pair_features(clean, adversarial)
    if len(selected_indices) != len(pair_features):
        raise ValueError("SHAP samples and selected attack indices have different lengths")

    run_tag = run_dir.relative_to(PROJECT_ROOT).as_posix().replace("/", "__")
    output_dir = Path(args.output_root)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_train, pair_val, pair_test = split_indices(len(pair_features), args.seed)
    np.save(output_dir / "pair_X.npy", pair_features)
    np.save(output_dir / "pair_y_prediction_changed.npy", prediction_changed)
    np.save(output_dir / "pair_train_indices.npy", pair_train)
    np.save(output_dir / "pair_val_indices.npy", pair_val)
    np.save(output_dir / "pair_test_indices.npy", pair_test)

    state_features, state_names = make_state_features(np.concatenate([clean, adversarial], axis=0))
    state_labels = np.concatenate([
        np.zeros(len(clean), dtype=np.int64),
        np.ones(len(adversarial), dtype=np.int64),
    ])
    # Keep the clean/adversarial members of each pair in the same split.
    state_train_pairs, state_val_pairs, state_test_pairs = split_indices(len(clean), args.seed)
    state_train = np.concatenate([state_train_pairs, state_train_pairs + len(clean)])
    state_val = np.concatenate([state_val_pairs, state_val_pairs + len(clean)])
    state_test = np.concatenate([state_test_pairs, state_test_pairs + len(clean)])
    np.save(output_dir / "state_X.npy", state_features)
    np.save(output_dir / "state_y.npy", state_labels)
    np.save(output_dir / "state_train_indices.npy", state_train)
    np.save(output_dir / "state_val_indices.npy", state_val)
    np.save(output_dir / "state_test_indices.npy", state_test)

    metadata = {
        "source_run_dir": str(run_dir.resolve()),
        "source_attack_dir": str(attack_dir.resolve()) if attack_dir and attack_dir.exists() else None,
        "num_pairs": int(len(pair_features)),
        "num_shap_features": int(clean.shape[1]),
        "pair_features": pair_names,
        "state_features": state_names,
        "pair_label": "prediction_changed; -1 if attack predictions were unavailable",
        "state_label": {"0": "clean", "1": "adversarial"},
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": args.seed},
        "large_shift_threshold": 0.01,
    }
    (output_dir / "detector_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Detector features written to: {output_dir}")
    print(f"Pair mode:  X={pair_features.shape}, y={prediction_changed.shape}")
    print(f"State mode: X={state_features.shape}, y={state_labels.shape}")
    if np.any(prediction_changed >= 0):
        print(f"Prediction-changed pairs: {int(prediction_changed.sum()):,}/{len(prediction_changed):,}")


if __name__ == "__main__":
    main()
