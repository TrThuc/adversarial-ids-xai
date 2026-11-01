import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NetworkSequenceDataset
from src.models.gru_ids import GRUIDS
from src.training.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config",
    )

    args = parser.parse_args()

    project_root = PROJECT_ROOT

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_config(config_path)

    seed = config["experiment"]["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("GRU TRAINING")
    print("=" * 70)

    print(f"Device : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU    : {torch.cuda.get_device_name(0)}"
        )

    processed_dir = (
        project_root / "data" / "processed"
    )

    features_path = (
        processed_dir / "features.npy"
    )

    sequence_indices_path = (
        processed_dir / "sequence_indices.npy"
    )

    labels_path = (
        processed_dir / "y.npy"
    )

    train_indices_path = (
        processed_dir / "train_indices.npy"
    )

    val_indices_path = (
        processed_dir / "val_indices.npy"
    )

    test_indices_path = (
        processed_dir / "test_indices.npy"
    )

    train_dataset = NetworkSequenceDataset(
        features_path=features_path,
        sequence_indices_path=sequence_indices_path,
        labels_path=labels_path,
        split_indices_path=train_indices_path,
    )

    val_dataset = NetworkSequenceDataset(
        features_path=features_path,
        sequence_indices_path=sequence_indices_path,
        labels_path=labels_path,
        split_indices_path=val_indices_path,
    )

    test_dataset = NetworkSequenceDataset(
        features_path=features_path,
        sequence_indices_path=sequence_indices_path,
        labels_path=labels_path,
        split_indices_path=test_indices_path,
    )

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print(f"Train samples : {len(train_dataset):,}")
    print(f"Val samples   : {len(val_dataset):,}")
    print(f"Test samples  : {len(test_dataset):,}")

    model = GRUIDS(
        input_size=config["model"]["input_size"]
        if "input_size" in config["model"]
        else config["data"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        num_classes=config["data"]["num_classes"],
    )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler_config = config["training"].get(
        "scheduler"
    )

    scheduler = None

    if scheduler_config:
        scheduler_type = scheduler_config["type"]

        if scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config["mode"],
                factor=scheduler_config["factor"],
                patience=scheduler_config["patience"],
            )

    experiment_name = config["experiment"]["name"]

    checkpoint_dir = (
        project_root
        / "models"
        / "checkpoints"
        / "gru"
        / experiment_name
    )

    log_dir = (
        project_root
        / "logs"
        / "gru"
        / experiment_name
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        monitor=config["checkpoint"]["monitor"],
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        max_train_batches=config["training"].get(
            "max_train_batches"
        ),
        max_val_batches=config["training"].get(
            "max_val_batches"
        ),
        config=config,
    )

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best validation "
        f"{trainer.monitor}: "
        f"{trainer.best_metric:.6f}"
    )

    print(
        f"Checkpoint directory:"
        f"\n{checkpoint_dir}"
    )

    print(
        f"Log directory:"
        f"\n{log_dir}"
    )


if __name__ == "__main__":
    main()