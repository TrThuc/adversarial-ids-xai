import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NetworkSequenceDataset
from src.models.lstm_ids import LSTMIDS


CLASS_NAMES = [
    "backdoor",
    "ddos",
    "dos",
    "injection",
    "mitm",
    "normal",
    "password",
    "ransomware",
    "scanning",
    "xss",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fgsm_attack(model, x, y, criterion, epsilon):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True
    # Forward pass
    logits = model(x_adv)
    # Compute loss
    loss = criterion(logits, y)
    # Clear previous gradients
    model.zero_grad()

    # Gradient with respect to input
    loss.backward()

    gradient = x_adv.grad.detach()

    # FGSM perturbation
    perturbation = epsilon * gradient.sign()

    x_adv = x_adv.detach() + perturbation

    return x_adv.detach()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to LSTM configuration JSON",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained LSTM checkpoint",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="FGSM epsilon in preprocessed feature space",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit for debugging. Use all batches by default.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    config = load_config(config_path)

    seed = config["experiment"]["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("FGSM ATTACK - LSTM")
    print("=" * 70)

    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Epsilon    : {args.epsilon}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    processed_dir = PROJECT_ROOT / "data" / "processed"

    test_dataset = NetworkSequenceDataset(
        features_path=processed_dir / "features.npy",
        sequence_indices_path=processed_dir / "sequence_indices.npy",
        labels_path=processed_dir / "y.npy",
        split_indices_path=processed_dir / "test_indices.npy",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test samples : {len(test_dataset):,}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = LSTMIDS(
        input_size=config["data"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        num_classes=config["data"]["num_classes"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(f"Checkpoint epoch : {checkpoint['epoch']}")

    criterion = torch.nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Attack
    # ---------------------------------------------------------

    total_samples = 0

    clean_correct = 0
    attacked_samples = 0

    adversarial_correct = 0
    attack_successes = 0

    total_linf = 0.0
    total_l2 = 0.0

    all_clean_predictions = []
    all_adv_predictions = []
    all_labels = []

    all_clean_correct_mask = []

    adversarial_examples = []
    adversarial_labels = []

    batch_count = 0

    for x, y in test_loader:

        if (
            args.max_batches is not None
            and batch_count >= args.max_batches
        ):
            break

        batch_count += 1

        x = x.to(device)
        y = y.to(device)

        # -----------------------------------------------------
        # Clean prediction
        # -----------------------------------------------------

        with torch.no_grad():

            clean_logits = model(x)

            clean_predictions = torch.argmax(
                clean_logits,
                dim=1,
            )

        clean_correct_mask = (
            clean_predictions == y
        )

        num_clean_correct = (
            clean_correct_mask.sum().item()
        )

        # -----------------------------------------------------
        # FGSM
        # -----------------------------------------------------

        x_adv = fgsm_attack(
            model=model,
            x=x,
            y=y,
            criterion=criterion,
            epsilon=args.epsilon,
        )

        # -----------------------------------------------------
        # Adversarial prediction
        # -----------------------------------------------------

        with torch.no_grad():

            adv_logits = model(x_adv)

            adv_predictions = torch.argmax(
                adv_logits,
                dim=1,
            )

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        batch_size = y.size(0)

        total_samples += batch_size
        clean_correct += num_clean_correct

        adversarial_correct += (
            adv_predictions == y
        ).sum().item()

        # Attack success:
        #
        # clean prediction correct
        # AND
        # adversarial prediction incorrect
        #

        attack_success_mask = (
            clean_correct_mask
            & (adv_predictions != y)
        )

        batch_attack_successes = (
            attack_success_mask.sum().item()
        )

        attack_successes += batch_attack_successes
        attacked_samples += num_clean_correct

        # -----------------------------------------------------
        # Perturbation magnitude
        # -----------------------------------------------------

        delta = x_adv - x

        batch_linf = (
            delta.abs()
            .view(delta.size(0), -1)
            .max(dim=1)
            .values
        )

        batch_l2 = (
            torch.norm(
                delta.view(delta.size(0), -1),
                p=2,
                dim=1,
            )
        )

        total_linf += batch_linf.sum().item()
        total_l2 += batch_l2.sum().item()

        # -----------------------------------------------------
        # Store predictions
        # -----------------------------------------------------

        all_clean_predictions.extend(
            clean_predictions.cpu().numpy().tolist()
        )

        all_adv_predictions.extend(
            adv_predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            y.cpu().numpy().tolist()
        )

        all_clean_correct_mask.extend(
            clean_correct_mask.cpu().numpy().tolist()
        )

        # -----------------------------------------------------
        # Store adversarial examples
        #
        # Only store examples that were originally
        # classified correctly.
        # -----------------------------------------------------

        correct_indices = (
            clean_correct_mask
            .nonzero(as_tuple=False)
            .flatten()
        )

        if len(correct_indices) > 0:

            adversarial_examples.append(
                x_adv[
                    correct_indices
                ].cpu().numpy()
            )

            adversarial_labels.append(
                y[
                    correct_indices
                ].cpu().numpy()
            )

        if batch_count % 50 == 0:

            print(
                f"Processed batches : {batch_count:,} | "
                f"Samples : {total_samples:,}"
            )

    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

    clean_accuracy = (
        clean_correct / total_samples
        if total_samples > 0
        else 0.0
    )

    adversarial_accuracy = (
        adversarial_correct / total_samples
        if total_samples > 0
        else 0.0
    )

    attack_success_rate = (
        attack_successes / attacked_samples
        if attacked_samples > 0
        else 0.0
    )

    mean_linf = (
        total_linf / total_samples
        if total_samples > 0
        else 0.0
    )

    mean_l2 = (
        total_l2 / total_samples
        if total_samples > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "logs"
        / "attacks"
        / "fgsm"
        / "lstm_baseline_a2"
        / (
                    f"epsilon_{args.epsilon:g}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Concatenate adversarial samples
    if adversarial_examples:

        adversarial_examples = np.concatenate(
            adversarial_examples,
            axis=0,
        )

        adversarial_labels = np.concatenate(
            adversarial_labels,
            axis=0,
        )

    else:

        adversarial_examples = np.empty(
            (0, 5, config["data"]["input_size"]),
            dtype=np.float32,
        )

        adversarial_labels = np.empty(
            (0,),
            dtype=np.int64,
        )

    np.save(
        output_dir / "adversarial_X.npy",
        adversarial_examples.astype(np.float32),
    )

    np.save(
        output_dir / "adversarial_y.npy",
        adversarial_labels.astype(np.int64),
    )

    np.save(
        output_dir / "clean_predictions.npy",
        np.asarray(
            all_clean_predictions,
            dtype=np.int64,
        ),
    )

    np.save(
        output_dir / "adversarial_predictions.npy",
        np.asarray(
            all_adv_predictions,
            dtype=np.int64,
        ),
    )

    np.save(
        output_dir / "labels.npy",
        np.asarray(
            all_labels,
            dtype=np.int64,
        ),
    )

    np.save(
        output_dir / "clean_correct_mask.npy",
        np.asarray(
            all_clean_correct_mask,
            dtype=bool,
        ),
    )

    metrics = {
        "attack": "FGSM",
        "model": "LSTM",
        "experiment": config["experiment"]["name"],
        "epsilon": args.epsilon,
        "total_test_samples": total_samples,
        "clean_correct_samples": clean_correct,
        "attackable_samples": attacked_samples,
        "attack_successes": attack_successes,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "attack_success_rate": attack_success_rate,
        "mean_linf": mean_linf,
        "mean_l2": mean_l2,
        "batch_count": batch_count,
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FGSM RESULTS")
    print("=" * 70)

    print(
        f"Total test samples      : "
        f"{total_samples:,}"
    )

    print(
        f"Clean correct samples   : "
        f"{clean_correct:,}"
    )

    print(
        f"Attackable samples      : "
        f"{attacked_samples:,}"
    )

    print(
        f"Attack successes        : "
        f"{attack_successes:,}"
    )

    print(
        f"Clean Accuracy          : "
        f"{clean_accuracy:.6f}"
    )

    print(
        f"Adversarial Accuracy    : "
        f"{adversarial_accuracy:.6f}"
    )

    print(
        f"Attack Success Rate     : "
        f"{attack_success_rate:.6f}"
    )

    print(
        f"Mean L-inf              : "
        f"{mean_linf:.6f}"
    )

    print(
        f"Mean L2                 : "
        f"{mean_l2:.6f}"
    )

    print()
    print(
        f"Adversarial X shape     : "
        f"{adversarial_examples.shape}"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()