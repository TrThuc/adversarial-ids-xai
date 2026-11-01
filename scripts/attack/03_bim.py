

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import NetworkSequenceDataset
from src.models.gru_ids import GRUIDS
from src.models.lstm_ids import LSTMIDS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def bim_attack(model, x, y, criterion, epsilon, alpha, steps):
    """Untargeted BIM with an L-infinity projection around the clean input."""
    x_original = x.detach().clone()
    x_adversarial = x_original.clone()

    for _ in range(steps):
        x_adversarial.requires_grad_(True)
        logits = model(x_adversarial)
        loss = criterion(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        gradient = x_adversarial.grad.detach()
        x_adversarial = x_adversarial.detach() + alpha * gradient.sign()
        delta = torch.clamp(
            x_adversarial - x_original,
            min=-epsilon,
            max=epsilon,
        )
        x_adversarial = (x_original + delta).detach()

    return x_adversarial


def build_model(model_name, config):
    model_class = LSTMIDS if model_name == "lstm" else GRUIDS
    return model_class(
        input_size=config["data"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        num_classes=config["data"]["num_classes"],
    )


def main():
    parser = argparse.ArgumentParser(description="BIM attack for LSTM/GRU IDS")
    parser.add_argument("--model", choices=["lstm", "gru"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    if args.epsilon <= 0 or args.alpha <= 0 or args.steps <= 0:
        raise ValueError("epsilon, alpha and steps must be positive.")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    config = load_config(config_path)
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    dataset = NetworkSequenceDataset(
        features_path=processed_dir / "features.npy",
        sequence_indices_path=processed_dir / "sequence_indices.npy",
        labels_path=processed_dir / "y.npy",
        split_indices_path=processed_dir / "test_indices.npy",
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model, config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_samples = clean_correct = attackable = attack_successes = 0
    adversarial_correct = 0
    total_linf = total_l2 = 0.0
    clean_predictions = []
    adversarial_predictions = []
    labels = []
    clean_correct_mask = []
    clean_examples = []
    adversarial_examples = []
    adversarial_labels = []
    attack_indices = []
    attack_success_mask = []
    global_index = 0
    batch_count = 0
    total_batches = len(loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)

    print(f"Total batches : {total_batches:,}")

    for x, y in loader:
        if args.max_batches is not None and batch_count >= args.max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            clean_logits = model(x)
            clean_pred = clean_logits.argmax(dim=1)

        x_adv = bim_attack(
            model, x, y, criterion,
            args.epsilon, args.alpha, args.steps,
        )
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1)

        clean_mask = clean_pred.eq(y)
        success_mask = clean_mask & adv_pred.ne(y)
        perturbation = x_adv - x

        batch_size = len(y)
        total_samples += batch_size
        clean_correct += int(clean_mask.sum().item())
        attackable += int(clean_mask.sum().item())
        attack_successes += int(success_mask.sum().item())
        adversarial_correct += int(adv_pred.eq(y).sum().item())
        total_linf += float(perturbation.detach().abs().amax(dim=(1, 2)).sum().item())
        total_l2 += float(perturbation.detach().flatten(1).norm(p=2, dim=1).sum().item())

        clean_predictions.extend(clean_pred.cpu().numpy().tolist())
        adversarial_predictions.extend(adv_pred.cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())
        clean_correct_mask.extend(clean_mask.cpu().numpy().tolist())

        selected = clean_mask.detach().cpu().numpy().astype(bool)
        if selected.any():
            clean_examples.append(x.detach().cpu().numpy()[selected])
            adversarial_examples.append(x_adv.detach().cpu().numpy()[selected])
            adversarial_labels.append(y.detach().cpu().numpy()[selected])
            attack_indices.extend((global_index + np.flatnonzero(selected)).tolist())
            attack_success_mask.extend(success_mask.detach().cpu().numpy()[selected].tolist())

        global_index += batch_size
        batch_count += 1

        if batch_count % 50 == 0 or batch_count == total_batches:
            progress = 100.0 * batch_count / total_batches
            print(
                f"Progress : {progress:6.2f}% | "
                f"Processed batches : {batch_count:,}/{total_batches:,} | "
                f"Samples : {total_samples:,}"
            )

    if total_samples == 0:
        raise RuntimeError("No test samples were processed.")

    empty_shape = (0, 5, config["data"]["input_size"])
    clean_x = np.concatenate(clean_examples).astype(np.float32) if clean_examples else np.empty(empty_shape, np.float32)
    adversarial_x = np.concatenate(adversarial_examples).astype(np.float32) if adversarial_examples else np.empty(empty_shape, np.float32)
    attack_y = np.concatenate(adversarial_labels).astype(np.int64) if adversarial_labels else np.empty((0,), np.int64)
    attack_indices = np.asarray(attack_indices, dtype=np.int64)
    attack_success_mask = np.asarray(attack_success_mask, dtype=bool)

    output_dir = (
        PROJECT_ROOT / "logs" / "attacks" / "bim"
        / f"{args.model}_baseline_a2"
        / f"epsilon_{args.epsilon:g}_alpha_{args.alpha:g}_steps_{args.steps}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "clean_X.npy", clean_x)
    np.save(output_dir / "adversarial_X.npy", adversarial_x)
    np.save(output_dir / "adversarial_y.npy", attack_y)
    np.save(output_dir / "attack_indices.npy", attack_indices)
    np.save(output_dir / "attack_success_mask.npy", attack_success_mask)
    np.save(output_dir / "clean_predictions.npy", np.asarray(clean_predictions, dtype=np.int64))
    np.save(output_dir / "adversarial_predictions.npy", np.asarray(adversarial_predictions, dtype=np.int64))
    np.save(output_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(output_dir / "clean_correct_mask.npy", np.asarray(clean_correct_mask, dtype=bool))

    metrics = {
        "attack": "BIM",
        "model": args.model.upper(),
        "experiment": config["experiment"]["name"],
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "steps": args.steps,
        "total_test_samples": total_samples,
        "clean_correct_samples": clean_correct,
        "attackable_samples": attackable,
        "attack_successes": attack_successes,
        "clean_accuracy": clean_correct / total_samples,
        "adversarial_accuracy": adversarial_correct / total_samples,
        "attack_success_rate": attack_successes / attackable if attackable else 0.0,
        "mean_linf": total_linf / total_samples,
        "mean_l2": total_l2 / total_samples,
        "batch_count": batch_count,
        "clean_X_shape": list(clean_x.shape),
        "adversarial_X_shape": list(adversarial_x.shape),
        "successful_attacks_aligned": int(attack_success_mask.sum()),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("BIM attack completed.")
    print(f"Output: {output_dir}")
    print(f"Clean accuracy: {metrics['clean_accuracy']:.6f}")
    print(f"Adversarial accuracy: {metrics['adversarial_accuracy']:.6f}")
    print(f"Attack success rate: {metrics['attack_success_rate']:.6f}")


if __name__ == "__main__":
    main()
