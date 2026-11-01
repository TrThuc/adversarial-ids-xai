import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import torch
from torch.utils.data import DataLoader

from src.data.dataset import NetworkSequenceDataset
from src.models.lstm_ids import LSTMIDS
from src.evaluation.evaluate import (
    evaluate_model,
    save_evaluation_results,
)
from src.evaluation.confusion_matrix import (
    save_confusion_matrix,
)


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


def load_config(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT / config_path
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            PROJECT_ROOT / checkpoint_path
        )

    config = load_config(
        config_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("LSTM CLEAN TEST EVALUATION")
    print("=" * 70)

    print(f"Device     : {device}")
    print(
        f"Checkpoint : {checkpoint_path}"
    )

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    test_dataset = (
        NetworkSequenceDataset(
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config[
            "training"
        ]["batch_size"],
        shuffle=False,
        num_workers=config[
            "training"
        ]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Test samples : "
        f"{len(test_dataset):,}"
    )

    model = LSTMIDS(
        input_size=config[
            "data"
        ]["input_size"],
        hidden_size=config[
            "model"
        ]["hidden_size"],
        num_layers=config[
            "model"
        ]["num_layers"],
        dropout=config[
            "model"
        ]["dropout"],
        num_classes=config[
            "data"
        ]["num_classes"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    print(
        f"Checkpoint epoch : "
        f"{checkpoint['epoch']}"
    )

    metrics, report, cm, _, _ = (
        evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            class_names=CLASS_NAMES,
        )
    )

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Test Loss       : "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Macro-F1        : "
        f"{metrics['f1_macro']:.6f}"
    )

    print(
        f"Weighted-F1     : "
        f"{metrics['f1_weighted']:.6f}"
    )

    print(
        f"Macro Precision : "
        f"{metrics['precision_macro']:.6f}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['recall_macro']:.6f}"
    )

    experiment_name = (
        config[
            "experiment"
        ]["name"]
    )

    output_dir = (
        PROJECT_ROOT
        / "logs"
        / "lstm"
        / experiment_name
        / "test_results"
    )

    save_evaluation_results(
        metrics=metrics,
        report=report,
        output_dir=output_dir,
    )

    save_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        output_path=(
            output_dir
            / "confusion_matrix.png"
        ),
    )

    print()
    print(
        f"Results saved to:"
        f"\n{output_dir}"
    )


if __name__ == "__main__":
    main()