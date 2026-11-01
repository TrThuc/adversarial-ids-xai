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


def pgd_attack(
    model,
    x,
    y,
    criterion,
    epsilon,
    alpha,
    steps,
):
    """
    Untargeted PGD attack.

    x shape:
        [B, T, F]

    epsilon:
        Maximum L-infinity perturbation.

    alpha:
        Step size per iteration.

    steps:
        Number of PGD iterations.
    """

    # ---------------------------------------------------------
    # Start from clean input
    # ---------------------------------------------------------

    x_orig = x.detach().clone()

    x_adv = x_orig.clone()

    # ---------------------------------------------------------
    # Iterative PGD
    # ---------------------------------------------------------

    for _ in range(steps):

        x_adv.requires_grad_(True)

        logits = model(x_adv)

        loss = criterion(logits, y)

        model.zero_grad()

        loss.backward()

        gradient = x_adv.grad.detach()

        # Gradient ascent on loss
        x_adv = (
            x_adv.detach()
            + alpha * gradient.sign()
        )

        # -----------------------------------------------------
        # Projection onto epsilon L-infinity ball
        # -----------------------------------------------------

        perturbation = x_adv - x_orig

        perturbation = torch.clamp(
            perturbation,
            min=-epsilon,
            max=epsilon,
        )

        x_adv = (
            x_orig + perturbation
        ).detach()

    return x_adv


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    config = load_config(config_path)

    set_seed(
        config["experiment"]["seed"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("PGD ATTACK - LSTM")
    print("=" * 70)

    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Epsilon    : {args.epsilon}")
    print(f"Alpha      : {args.alpha}")
    print(f"Steps      : {args.steps}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    test_dataset = NetworkSequenceDataset(
        features_path=(
            processed_dir
            / "features.npy"
        ),
        sequence_indices_path=(
            processed_dir
            / "sequence_indices.npy"
        ),
        labels_path=(
            processed_dir
            / "y.npy"
        ),
        split_indices_path=(
            processed_dir
            / "test_indices.npy"
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Test samples : "
        f"{len(test_dataset):,}"
    )

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

    print(
        f"Checkpoint epoch : "
        f"{checkpoint['epoch']}"
    )

    criterion = torch.nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    total_samples = 0

    clean_correct = 0

    attackable_samples = 0

    attack_successes = 0

    adversarial_correct = 0

    total_linf = 0.0
    total_l2 = 0.0

    all_clean_predictions = []
    all_adv_predictions = []
    all_labels = []
    all_clean_correct_mask = []

    # ---------------------------------------------------------
    # Samples for XAI
    #
    # These arrays contain ONLY clean-correct samples.
    # Therefore:
    #
    # clean_X[i] <-> adversarial_X[i]
    #
    # is guaranteed to be a 1:1 pair.
    # ---------------------------------------------------------

    clean_examples = []
    adversarial_examples = []
    adversarial_labels = []

    attack_indices = []
    attack_success_mask_aligned = []

    global_test_index = 0

    batch_count = 0

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

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
        # PGD
        # -----------------------------------------------------

        x_adv = pgd_attack(
            model=model,
            x=x,
            y=y,
            criterion=criterion,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
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
        # Basic metrics
        # -----------------------------------------------------

        batch_size = y.size(0)

        total_samples += batch_size

        clean_correct += (
            num_clean_correct
        )

        adversarial_correct += (
            (adv_predictions == y)
            .sum()
            .item()
        )

        # -----------------------------------------------------
        # Attack success
        #
        # clean was correct
        # adversarial becomes incorrect
        # -----------------------------------------------------

        attack_success_mask = (
            clean_correct_mask
            & (adv_predictions != y)
        )

        batch_attack_successes = (
            attack_success_mask
            .sum()
            .item()
        )

        attack_successes += (
            batch_attack_successes
        )

        attackable_samples += (
            num_clean_correct
        )

        # -----------------------------------------------------
        # Perturbation
        # -----------------------------------------------------

        delta = x_adv - x

        flat_delta = (
            delta
            .view(delta.size(0), -1)
        )

        batch_linf = (
            flat_delta.abs()
            .max(dim=1)
            .values
        )

        batch_l2 = torch.norm(
            flat_delta,
            p=2,
            dim=1,
        )

        total_linf += (
            batch_linf.sum().item()
        )

        total_l2 += (
            batch_l2.sum().item()
        )

        # -----------------------------------------------------
        # Store predictions
        # -----------------------------------------------------

        all_clean_predictions.extend(
            clean_predictions
            .cpu()
            .numpy()
            .tolist()
        )

        all_adv_predictions.extend(
            adv_predictions
            .cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            y.cpu()
            .numpy()
            .tolist()
        )

        all_clean_correct_mask.extend(
            clean_correct_mask
            .cpu()
            .numpy()
            .tolist()
        )

        # -----------------------------------------------------
        # Save clean/adversarial pairs
        #
        # ONLY clean-correct samples are stored.
        #
        # clean_examples[i]
        #      <-> adversarial_examples[i]
        # -----------------------------------------------------

        correct_indices = (
            clean_correct_mask
            .nonzero(
                as_tuple=False
            )
            .flatten()
        )

        if len(correct_indices) > 0:

            correct_indices_cpu = (
                correct_indices
                .cpu()
                .numpy()
            )

            # Clean samples
            clean_examples.append(
                x[
                    correct_indices
                ]
                .detach()
                .cpu()
                .numpy()
            )

            # Adversarial samples
            adversarial_examples.append(
                x_adv[
                    correct_indices
                ]
                .detach()
                .cpu()
                .numpy()
            )

            # Labels
            adversarial_labels.append(
                y[
                    correct_indices
                ]
                .cpu()
                .numpy()
            )

            # Global indices in the test set
            attack_indices.extend(
                (
                    global_test_index
                    + correct_indices_cpu
                )
                .tolist()
            )

            # Attack success aligned with
            # clean_X / adversarial_X
            attack_success_mask_aligned.extend(
                attack_success_mask[
                    correct_indices
                ]
                .cpu()
                .numpy()
                .tolist()
            )

        global_test_index += batch_size

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if batch_count % 50 == 0:

            print(
                f"Processed batches : "
                f"{batch_count:,} | "
                f"Samples : "
                f"{total_samples:,}"
            )

    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

    clean_accuracy = (
        clean_correct / total_samples
    )

    adversarial_accuracy = (
        adversarial_correct
        / total_samples
    )

    attack_success_rate = (
        attack_successes
        / attackable_samples
    )

    mean_linf = (
        total_linf
        / total_samples
    )

    mean_l2 = (
        total_l2
        / total_samples
    )

    # ---------------------------------------------------------
    # Combine clean/adversarial samples
    # ---------------------------------------------------------

    if clean_examples:

        clean_examples = np.concatenate(
            clean_examples,
            axis=0,
        ).astype(np.float32)

        adversarial_examples = np.concatenate(
            adversarial_examples,
            axis=0,
        ).astype(np.float32)

        adversarial_labels = np.concatenate(
            adversarial_labels,
            axis=0,
        ).astype(np.int64)

        attack_indices = np.asarray(
            attack_indices,
            dtype=np.int64,
        )

        attack_success_mask_aligned = np.asarray(
            attack_success_mask_aligned,
            dtype=bool,
        )

    else:

        empty_shape = (
            0,
            5,
            config["data"]["input_size"],
        )

        clean_examples = np.empty(
            empty_shape,
            dtype=np.float32,
        )

        adversarial_examples = np.empty(
            empty_shape,
            dtype=np.float32,
        )

        adversarial_labels = np.empty(
            (0,),
            dtype=np.int64,
        )

        attack_indices = np.empty(
            (0,),
            dtype=np.int64,
        )

        attack_success_mask_aligned = np.empty(
            (0,),
            dtype=bool,
        )

    # ---------------------------------------------------------
    # Consistency checks
    # ---------------------------------------------------------

    assert (
        len(clean_examples)
        == len(adversarial_examples)
    )

    assert (
        len(clean_examples)
        == len(adversarial_labels)
    )

    assert (
        len(clean_examples)
        == len(attack_indices)
    )

    assert (
        len(clean_examples)
        == len(attack_success_mask_aligned)
    )

    assert (
        len(clean_examples)
        == clean_correct
    )

    assert (
        attack_success_mask_aligned.sum()
        == attack_successes
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "logs"
        / "attacks"
        / "pgd"
        / "lstm_baseline_a2"
        / (
            f"epsilon_{args.epsilon:g}"
            f"_alpha_{args.alpha:g}"
            f"_steps_{args.steps}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Save clean/adversarial arrays
    # ---------------------------------------------------------

    np.save(
        output_dir / "clean_X.npy",
        clean_examples,
    )

    np.save(
        output_dir / "adversarial_X.npy",
        adversarial_examples,
    )

    np.save(
        output_dir / "adversarial_y.npy",
        adversarial_labels,
    )

    # ---------------------------------------------------------
    # Save attack alignment information
    # ---------------------------------------------------------

    np.save(
        output_dir / "attack_indices.npy",
        attack_indices,
    )

    np.save(
        output_dir / "attack_success_mask.npy",
        attack_success_mask_aligned,
    )

    # ---------------------------------------------------------
    # Save full-test predictions
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics = {
        "attack": "PGD",
        "model": "LSTM",
        "experiment": config["experiment"]["name"],
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "steps": args.steps,
        "total_test_samples": total_samples,
        "clean_correct_samples": clean_correct,
        "attackable_samples": attackable_samples,
        "attack_successes": attack_successes,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "attack_success_rate": attack_success_rate,
        "mean_linf": mean_linf,
        "mean_l2": mean_l2,
        "batch_count": batch_count,
        "clean_X_shape": list(clean_examples.shape),
        "adversarial_X_shape": list(
            adversarial_examples.shape
        ),
        "successful_attacks_aligned": int(
            attack_success_mask_aligned.sum()
        ),
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
    # Print
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("PGD RESULTS")
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
        f"{attackable_samples:,}"
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
        f"Clean X shape           : "
        f"{clean_examples.shape}"
    )

    print(
        f"Adversarial X shape     : "
        f"{adversarial_examples.shape}"
    )

    print(
        f"Successful attacks     : "
        f"{attack_success_mask_aligned.sum():,}"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()

