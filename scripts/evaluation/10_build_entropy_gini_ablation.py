from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(description="Build entropy/Gini ablation feature sets")
    parser.add_argument("--detector-dir", required=True)
    parser.add_argument("--entropy-gini-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    detector_dir = Path(args.detector_dir)
    if not detector_dir.is_absolute():
        detector_dir = PROJECT_ROOT / detector_dir

    if not args.entropy_gini_dir:
        raise ValueError("Pass --entropy-gini-dir pointing to the entropy_gini output directory.")
    entropy_dir = Path(args.entropy_gini_dir)
    if not entropy_dir.is_absolute():
        entropy_dir = PROJECT_ROOT / entropy_dir

    # The entropy/Gini script normally writes this directory inside the SHAP run.
    if not (entropy_dir / "entropy_gini_detector_features.npy").exists():
        raise FileNotFoundError(
            "Entropy/Gini feature file not found. Pass --entropy-gini-dir explicitly."
        )

    baseline = load(detector_dir / "pair_X.npy").astype(np.float32)
    entropy_gini = load(entropy_dir / "entropy_gini_detector_features.npy").astype(np.float32)
    labels = load(detector_dir / "pair_y_prediction_changed.npy").astype(np.int64)

    if len(baseline) != len(entropy_gini) or len(baseline) != len(labels):
        raise ValueError(
            f"Mismatched sample counts: baseline={len(baseline)}, "
            f"entropy_gini={len(entropy_gini)}, labels={len(labels)}"
        )

    baseline_names = [
        "drift_mean", "shift_std", "shift_max", "shift_l2",
        "top5_overlap", "top10_overlap", "large_shift_fraction",
    ]
    entropy_gini_names = [
        "entropy_delta", "entropy_abs_delta",
        "gini_delta", "gini_abs_delta",
    ]

    variants = {
        "baseline": (baseline, baseline_names),
        "entropy_only": (
            entropy_gini[:, :2], entropy_gini_names[:2],
        ),
        "gini_only": (
            entropy_gini[:, 2:], entropy_gini_names[2:],
        ),
        "entropy_gini_only": (entropy_gini, entropy_gini_names),
        "baseline_plus_entropy": (
            np.concatenate([baseline, entropy_gini[:, :2]], axis=1),
            baseline_names + entropy_gini_names[:2],
        ),
        "baseline_plus_gini": (
            np.concatenate([baseline, entropy_gini[:, 2:]], axis=1),
            baseline_names + entropy_gini_names[2:],
        ),
        "extended": (
            np.concatenate([baseline, entropy_gini], axis=1),
            baseline_names + entropy_gini_names,
        ),
    }

    output_dir = Path(args.output_dir) if args.output_dir else detector_dir / "ablation"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = [
        "pair_train_indices.npy",
        "pair_val_indices.npy",
        "pair_test_indices.npy",
    ]
    split_indices = {name: load(detector_dir / name) for name in split_files}

    metadata = {
        "source_detector_dir": str(detector_dir.resolve()),
        "source_entropy_gini_dir": str(entropy_dir.resolve()),
        "label_file": "pair_y_prediction_changed.npy",
        "variants": {},
    }

    for name, (features, feature_names) in variants.items():
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        np.save(variant_dir / "X.npy", features.astype(np.float32))
        np.save(variant_dir / "y.npy", labels)
        # Also expose the names expected by 08_train_learned_detector.py.
        np.save(variant_dir / "pair_X.npy", features.astype(np.float32))
        np.save(variant_dir / "pair_y_prediction_changed.npy", labels)
        np.save(variant_dir / "train_indices.npy", split_indices["pair_train_indices.npy"])
        np.save(variant_dir / "val_indices.npy", split_indices["pair_val_indices.npy"])
        np.save(variant_dir / "test_indices.npy", split_indices["pair_test_indices.npy"])
        np.save(variant_dir / "pair_train_indices.npy", split_indices["pair_train_indices.npy"])
        np.save(variant_dir / "pair_val_indices.npy", split_indices["pair_val_indices.npy"])
        np.save(variant_dir / "pair_test_indices.npy", split_indices["pair_test_indices.npy"])
        (variant_dir / "feature_names.json").write_text(
            json.dumps(feature_names, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metadata["variants"][name] = {
            "shape": list(features.shape),
            "features": feature_names,
        }

    (output_dir / "ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Ablation datasets written to: {output_dir}")
    for name, (features, _) in variants.items():
        print(f"{name}: {features.shape}")


if __name__ == "__main__":
    main()
