"""
Kernel SHAP for LSTM/GRU IDS under FGSM/PGD adversarial attacks.

Run from the project root:
    python scripts/xai/04_shap_kernel_all.py --config configs/shap_kernel_config.json
Optional overrides:
    --model lstm --attack fgsm
    --model gru --attack pgd --epsilon 0.05 --alpha 0.01 --steps 10
    --checkpoint models/checkpoints/gru/gru_baseline_a2/0.0003/best.pt
    --explanation-size 0  # explain every paired sample in the attack artifact
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_path(project_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def first_existing(paths: list[Path | None], description: str) -> Path:
    for p in paths:
        if p is not None and p.exists():
            return p
    tried = "\n".join(f"  - {p}" for p in paths if p is not None)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n{tried}")


def load_model(project_root: Path, model_name: str, checkpoint_path: Path,
               num_features: int, num_classes: int, hidden_size: int,
               num_layers: int, dropout: float, device: torch.device):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if model_name == "lstm":
        from src.models.lstm_ids import LSTMIDS
        model_cls = LSTMIDS
    else:
        from src.models.gru_ids import GRUIDS
        model_cls = GRUIDS
    model = model_cls(input_size=num_features, hidden_size=hidden_size,
                      num_layers=num_layers, dropout=dropout, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def extract_class_shap(raw_values: Any, classes: np.ndarray, n_samples: int,
                       n_features: int) -> np.ndarray:
    if isinstance(raw_values, list):
        out = np.stack([np.asarray(raw_values[int(c)])[i] for i, c in enumerate(classes)])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 2:
            out = values
        elif values.ndim == 3:
            if values.shape[0] == n_samples and values.shape[1] == n_features:
                out = np.stack([values[i, :, int(c)] for i, c in enumerate(classes)])
            elif values.shape[0] == n_samples and values.shape[2] == n_features:
                out = np.stack([values[i, int(c), :] for i, c in enumerate(classes)])
            else:
                raise ValueError(f"Unrecognized 3-D SHAP output shape: {values.shape}")
        else:
            raise ValueError(f"Unrecognized SHAP output shape: {values.shape}")
    out = np.asarray(out, dtype=np.float64)
    if out.shape != (n_samples, n_features):
        raise ValueError(f"Selected SHAP shape {out.shape}; expected {(n_samples, n_features)}")
    return out


def top_k_overlap(clean_values: np.ndarray, adv_values: np.ndarray, k: int) -> np.ndarray:
    clean_top = np.argpartition(np.abs(clean_values), -k, axis=1)[:, -k:]
    adv_top = np.argpartition(np.abs(adv_values), -k, axis=1)[:, -k:]
    return np.asarray([
        len(set(a.tolist()) & set(b.tolist())) / k
        for a, b in zip(clean_top, adv_top)
    ], dtype=np.float64)


def find_artifact_dir(project_root: Path, model_name: str, attack_name: str,
                      epsilon: float, alpha: float, steps: int) -> Path:
    base = project_root / "logs" / "attacks" / attack_name / f"{model_name}_baseline_a2"
    if attack_name == "fgsm":
        candidates = [base / f"epsilon_{epsilon:.2f}", base / f"epsilon_{epsilon:g}"]
    else:
        candidates = [
            base / f"epsilon_{epsilon:.2f}_alpha_{alpha:.2f}_steps_{steps}",
            base / f"epsilon_{epsilon:g}_alpha_{alpha:g}_steps_{steps}",
        ]
    return first_existing(candidates, f"{model_name.upper()} {attack_name.upper()} artifacts")


def find_preprocessed_file(project_root: Path, configured: str | None,
                           candidates: list[str], description: str) -> Path:
    paths = [resolve_path(project_root, configured)]
    paths.extend(project_root / c for c in candidates)
    return first_existing(paths, description)


def get_preprocessing_paths(cfg: dict, project_root: Path):
    pp = cfg.get("preprocessing", {})
    features = find_preprocessed_file(project_root, pp.get("features_file"), [
        "data/processed/features.npy", "data/processed/preprocessed/features.npy"
    ], "features.npy")
    sequences = find_preprocessed_file(project_root, pp.get("sequence_indices_file"), [
        "data/processed/sequence_indices.npy", "data/processed/preprocessed/sequence_indices.npy"
    ], "sequence_indices.npy")
    train_ids = find_preprocessed_file(project_root, pp.get("train_indices_file"), [
        "data/processed/train_indices.npy",
        "data/processed/preprocessed/train_sequence_indices.npy",
        "data/processed/preprocessed/train_indices.npy"
    ], "training sequence IDs")
    test_ids = find_preprocessed_file(project_root, pp.get("test_indices_file"), [
        "data/processed/test_indices.npy",
        "data/processed/preprocessed/test_sequence_indices.npy",
        "data/processed/preprocessed/test_indices.npy"
    ], "test sequence IDs")
    metadata = find_preprocessed_file(project_root, pp.get("metadata_file"), [
        "data/processed/preprocessing_metadata.json",
        "data/processed/artifacts/preprocessing_metadata.json",
        "data/processed/preprocessed/preprocessing_metadata.json"
    ], "preprocessing metadata JSON")
    return features, sequences, train_ids, test_ids, metadata


def load_artifacts(artifact_dir: Path, cfg: dict, project_root: Path,
                   window_size: int, num_features: int) -> dict:
    def maybe(name):
        p = artifact_dir / name
        return np.load(p, allow_pickle=False) if p.exists() else None

    clean_x = maybe("clean_X.npy")
    adv_x = maybe("adversarial_X.npy")
    attack_indices = maybe("attack_indices.npy")
    success_mask = maybe("attack_success_mask.npy")
    clean_pred = maybe("clean_predictions.npy")
    adv_pred = maybe("adversarial_predictions.npy")
    clean_correct = maybe("clean_correct_mask.npy")

    if attack_indices is None:
        if clean_correct is None:
            raise FileNotFoundError(f"{artifact_dir} needs attack_indices.npy or clean_correct_mask.npy")
        attack_indices = np.flatnonzero(clean_correct)

    if clean_x is None:
        features_p, seq_p, _, test_ids_p, _ = get_preprocessing_paths(cfg, project_root)
        features = np.load(features_p)
        sequence_indices = np.load(seq_p)
        test_ids = np.load(test_ids_p)
        all_test_x = features[sequence_indices[test_ids]]
        clean_x = all_test_x[attack_indices]

    if adv_x is None:
        raise FileNotFoundError(f"Missing adversarial_X.npy in {artifact_dir}")

    clean_x = np.asarray(clean_x, dtype=np.float32)
    adv_x = np.asarray(adv_x, dtype=np.float32)
    attack_indices = np.asarray(attack_indices, dtype=np.int64).reshape(-1)
    if clean_x.shape != adv_x.shape:
        raise ValueError(f"clean_X {clean_x.shape} != adversarial_X {adv_x.shape}")
    if clean_x.ndim != 3 or clean_x.shape[1:] != (window_size, num_features):
        raise ValueError(f"Expected X shape [N,{window_size},{num_features}], got {clean_x.shape}")
    if len(attack_indices) != len(clean_x):
        raise ValueError("attack_indices length does not match attack X arrays")

    if clean_pred is not None and len(clean_pred) != len(clean_x):
        clean_pred = clean_pred[attack_indices]
    if adv_pred is not None and len(adv_pred) != len(adv_x):
        adv_pred = adv_pred[attack_indices]

    if success_mask is None:
        if clean_pred is None or adv_pred is None:
            raise FileNotFoundError("Cannot derive success mask without prediction arrays")
        success_mask = clean_pred != adv_pred
    success_mask = np.asarray(success_mask).reshape(-1).astype(bool)
    if len(success_mask) != len(clean_x) and len(success_mask) > int(attack_indices.max(initial=-1)):
        success_mask = success_mask[attack_indices]
    if len(success_mask) != len(clean_x):
        raise ValueError(f"Success mask length {len(success_mask)} != attack sample count {len(clean_x)}")
    if clean_pred is not None and len(clean_pred) != len(clean_x):
        raise ValueError("Could not align clean predictions")
    if adv_pred is not None and len(adv_pred) != len(adv_x):
        raise ValueError("Could not align adversarial predictions")
    return {"clean_X": clean_x, "adversarial_X": adv_x, "attack_indices": attack_indices,
            "success_mask": success_mask, "clean_predictions": clean_pred,
            "adversarial_predictions": adv_pred}


def load_background(cfg: dict, project_root: Path, size: int, seed: int,
                    window_size: int) -> np.ndarray:
    features_p, seq_p, train_ids_p, _, _ = get_preprocessing_paths(cfg, project_root)
    features = np.load(features_p)
    sequence_indices = np.load(seq_p)
    train_ids = np.load(train_ids_p)
    if len(train_ids) == 0:
        raise ValueError("Training sequence ID array is empty")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(train_ids, size=min(size, len(train_ids)), replace=False)
    background = features[sequence_indices[chosen]].astype(np.float32)
    if background.ndim != 3 or background.shape[1] != window_size:
        raise ValueError(f"Unexpected background shape: {background.shape}")
    return background


def main():
    parser = argparse.ArgumentParser(description="Kernel SHAP for LSTM/GRU × FGSM/PGD")
    parser.add_argument("--config", default="configs/shap_kernel_config.json")
    parser.add_argument("--model", choices=["lstm", "gru"])
    parser.add_argument("--attack", choices=["fgsm", "pgd"])
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint", help="Checkpoint override; relative to project root or absolute")
    parser.add_argument("--artifact-dir", help="Attack artifact directory override")
    parser.add_argument("--explanation-size", type=int)
    parser.add_argument("--background-size", type=int)
    parser.add_argument("--nsamples", type=int)
    parser.add_argument("--shap-batch-size", type=int, default=25,
                        help="Number of samples explained per SHAP call to limit CPU/RAM use (default: 25).")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = (Path.cwd() / cfg.get("project_root", ".")).resolve()
    model_name = args.model or cfg.get("model", "lstm")
    attack_name = args.attack or cfg.get("attack", "pgd")
    epsilon = args.epsilon if args.epsilon is not None else float(cfg.get("epsilon", 0.05))
    alpha = args.alpha if args.alpha is not None else float(cfg.get("alpha", 0.01))
    steps = args.steps if args.steps is not None else int(cfg.get("steps", 10))
    bg_size = args.background_size or int(cfg.get("background_size", 50))
    exp_size = (args.explanation_size if args.explanation_size is not None
                else int(cfg.get("explanation_size", 100)))
    if exp_size < 0:
        raise ValueError("--explanation-size must be 0 (all samples) or a positive integer.")
    if args.shap_batch_size < 1:
        raise ValueError("--shap-batch-size must be at least 1.")
    nsamples = args.nsamples or int(cfg.get("nsamples", 100))
    seed = int(cfg.get("seed", 42))
    window_size = int(cfg.get("window_size", 5))
    num_features = int(cfg.get("num_features", 42))
    num_classes = int(cfg.get("num_classes", 10))
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set device to cpu.")

    checkpoint_cfg = cfg.get("checkpoints", {}).get(model_name)
    checkpoint_path = resolve_path(project_root, args.checkpoint or checkpoint_cfg)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Pass --checkpoint with the actual path.")
    if args.artifact_dir:
        artifact_dir = resolve_path(project_root, args.artifact_dir)
    else:
        artifact_dir = find_artifact_dir(project_root, model_name, attack_name, epsilon, alpha, steps)
    if artifact_dir is None or not artifact_dir.exists():
        raise FileNotFoundError(f"Attack artifacts directory not found: {artifact_dir}")

    _, _, _, _, metadata_path = get_preprocessing_paths(cfg, project_root)
    metadata = load_json(metadata_path)
    feature_names = metadata.get("feature_columns", [])
    if len(feature_names) != num_features:
        feature_names = [f"feature_{i:02d}" for i in range(num_features)]

    output_pattern = cfg.get("output_dir", "logs/xai/shap/{model}_baseline_a2/{attack}_epsilon_{epsilon}/kernel")
    output_dir = resolve_path(project_root, output_pattern.format(
        model=model_name, attack=attack_name, epsilon=f"{epsilon:.2f}",
        alpha=f"{alpha:.2f}", steps=steps
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"KERNEL SHAP | {model_name.upper()} + {attack_name.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Attack artifacts: {artifact_dir}")
    print(f"Output: {output_dir}")
    print(f"SHAP {shap.__version__} | torch {torch.__version__} | device {device}")
    exp_label = "ALL" if exp_size == 0 else str(exp_size)
    print(f"background={bg_size}, explanation_size={exp_label}, nsamples={nsamples}, shap_batch_size={args.shap_batch_size}")

    data = load_artifacts(artifact_dir, cfg, project_root, window_size, num_features)
    success_positions = np.flatnonzero(data["success_mask"])
    if exp_size == 0:
        # Explain every paired clean/adversarial sample available in this attack artifact,
        # not only samples whose predictions changed. This avoids selecting only successful attacks.
        selected = np.arange(len(data["clean_X"]), dtype=np.int64)
    else:
        count_requested = min(exp_size, len(data["clean_X"]))
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(data["clean_X"]), size=count_requested, replace=False))
    count = len(selected)
    if count == 0:
        raise ValueError("No attack samples found in the artifact directory.")
    clean_x = data["clean_X"][selected]
    adv_x = data["adversarial_X"][selected]
    attack_indices = data["attack_indices"][selected]

    model = load_model(project_root, model_name, checkpoint_path, num_features, num_classes,
                       int(cfg.get("hidden_size", 64)), int(cfg.get("num_layers", 1)),
                       float(cfg.get("dropout", 0.3)), device)
    flat_features = window_size * num_features

    def predict_flat(flat):
        x = np.asarray(flat, dtype=np.float32).reshape(
            -1, window_size, num_features
        )

        with torch.no_grad():
            logits = model(torch.from_numpy(x).to(device))
            probabilities = torch.softmax(logits, dim=1)

        result = probabilities.detach().cpu().numpy()

        if not np.isfinite(result).all():
            raise ValueError(
                "Model produced NaN or Inf probabilities. "
                "Check clean/adversarial inputs and model checkpoint."
            )

        return result

    clean_flat = clean_x.reshape(count, flat_features).astype(np.float32)
    adv_flat = adv_x.reshape(count, flat_features).astype(np.float32)
    for name, values in [
        ("clean_flat", clean_flat),
        ("adv_flat", adv_flat),
    ]:
        print(
            f"{name}: "
            f"min={values.min():.6g}, "
            f"max={values.max():.6g}, "
            f"mean={values.mean():.6g}, "
            f"finite={np.isfinite(values).all()}"
        )

        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf.")
        
    clean_preds = predict_flat(clean_flat).argmax(axis=1)
    adv_preds = predict_flat(adv_flat).argmax(axis=1)
    comparison_classes = clean_preds.copy()
    print(f"Prediction-changing examples in artifact: {len(success_positions):,}/{len(data['clean_X']):,}")
    print(f"Pairs selected for SHAP: {count:,}; prediction changes among selected pairs: {int((clean_preds != adv_preds).sum()):,}/{count:,}")

    background = load_background(cfg, project_root, bg_size, seed, window_size)
    background_flat = background.reshape(len(background), flat_features)
    explainer = shap.KernelExplainer(predict_flat, background_flat)

    def explain_in_batches(values: np.ndarray, classes: np.ndarray, label: str) -> np.ndarray:
        parts = []
        for start in range(0, len(values), args.shap_batch_size):
            end = min(start + args.shap_batch_size, len(values))
            print(f"Computing {label} SHAP: samples {start + 1:,}-{end:,}/{len(values):,}")
            raw = explainer.shap_values(values[start:end], nsamples=nsamples)
            parts.append(extract_class_shap(raw, classes[start:end], end - start, flat_features))
        return np.concatenate(parts, axis=0)

    clean_shap = explain_in_batches(clean_flat, comparison_classes, "clean")
    adv_shap = explain_in_batches(adv_flat, comparison_classes, "adversarial")
    # Aggregate absolute SHAP magnitude across the five timesteps into 42 features.
    clean_feat = np.abs(clean_shap.reshape(count, window_size, num_features)).sum(axis=1)
    adv_feat = np.abs(adv_shap.reshape(count, window_size, num_features)).sum(axis=1)
    clean_norm = clean_feat / np.maximum(clean_feat.sum(axis=1, keepdims=True), 1e-12)
    adv_norm = adv_feat / np.maximum(adv_feat.sum(axis=1, keepdims=True), 1e-12)
    drift = np.mean(np.abs(clean_norm - adv_norm), axis=1)

    overlaps = {str(k): top_k_overlap(clean_feat, adv_feat, int(k)) for k in cfg.get("top_k_values", [5, 10])}
    np.save(output_dir / "clean_feature_shap.npy", clean_feat)
    np.save(output_dir / "adversarial_feature_shap.npy", adv_feat)
    np.save(output_dir / "comparison_classes.npy", comparison_classes)
    np.save(output_dir / "selected_attack_positions.npy", selected)
    np.save(output_dir / "selected_attack_indices.npy", attack_indices)
    np.save(output_dir / "drift_per_sample.npy", drift)
    for k, values in overlaps.items():
        np.save(output_dir / f"top_{k}_overlap_per_sample.npy", values)

    clean_global = clean_feat.mean(axis=0)
    adv_global = adv_feat.mean(axis=0)
    summary = [{
        "feature": str(feature_names[i]),
        "clean_mean_abs_shap": float(clean_global[i]),
        "adversarial_mean_abs_shap": float(adv_global[i]),
        "absolute_change": float(abs(adv_global[i] - clean_global[i]))
    } for i in range(num_features)]
    summary.sort(key=lambda x: x["adversarial_mean_abs_shap"], reverse=True)
    with (output_dir / "feature_importance_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "clean_mean_abs_shap", "adversarial_mean_abs_shap", "absolute_change"])
        writer.writeheader()
        writer.writerows(summary)

    metrics = {
        "model": model_name, "attack": attack_name, "epsilon": epsilon,
        "alpha": alpha if attack_name == "pgd" else None,
        "steps": steps if attack_name == "pgd" else None,
        "checkpoint": str(checkpoint_path), "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir), "shap_version": shap.__version__,
        "torch_version": torch.__version__, "device": str(device), "seed": seed,
        "background_size_actual": int(len(background)), "explanation_size_actual": int(count),
        "nsamples": nsamples, "successful_attack_count": int(len(success_positions)),
        "selected_prediction_changes": int((clean_preds != adv_preds).sum()),
        "drift_mean": float(drift.mean()), "drift_median": float(np.median(drift)),
        "drift_min": float(drift.min()), "drift_max": float(drift.max()),
        "top_k_overlap": {k: {"mean": float(v.mean()), "median": float(np.median(v)),
                              "min": float(v.min()), "max": float(v.max())} for k, v in overlaps.items()},
        "top_adversarial_features": summary[:10],
        "notes": [
            "Kernel SHAP is exploratory when nsamples is small relative to 210 flattened inputs.",
            "Explanation samples include every paired sample in the attack artifact when explanation_size is 0; this is not necessarily the entire original test dataset.",
            "SHAP is computed in batches to reduce peak memory use; runtime may still be very long for all samples.",
            "Feature-level attribution sums absolute SHAP magnitude across temporal positions.",
            "Drift is mean absolute difference between normalized feature-importance vectors.",
            "The same clean-model predicted class is compared for clean and adversarial inputs."
        ]
    }
    save_json(output_dir / "metrics.json", metrics)
    print("\nFinished.")
    print(f"Drift mean/median: {metrics['drift_mean']:.6f} / {metrics['drift_median']:.6f}")
    for k, vals in overlaps.items():
        print(f"Top-{k} overlap mean/median: {vals.mean():.4f} / {np.median(vals):.4f}")
    print(f"Results saved: {output_dir}")


if __name__ == "__main__":
    main()
