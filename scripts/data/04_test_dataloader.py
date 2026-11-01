from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FEATURES_PATH = (
    PROCESSED_DIR
    / "features.npy"
)

SEQUENCE_PATH = (
    PROCESSED_DIR
    / "sequence_indices.npy"
)

LABEL_PATH = (
    PROCESSED_DIR
    / "y.npy"
)

TRAIN_PATH = (
    PROCESSED_DIR
    / "train_indices.npy"
)

VAL_PATH = (
    PROCESSED_DIR
    / "val_indices.npy"
)

TEST_PATH = (
    PROCESSED_DIR
    / "test_indices.npy"
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
NUM_WORKERS = 0


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PHASE A4 - DATASET / DATALOADER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        FEATURES_PATH,
        SEQUENCE_PATH,
        LABEL_PATH,
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Import Dataset
    # --------------------------------------------------------

    import sys

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

    from src.data.dataset import NetworkSequenceDataset

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\n[1] Creating datasets...")

    train_dataset = NetworkSequenceDataset(
        features_path=FEATURES_PATH,
        sequence_indices_path=SEQUENCE_PATH,
        labels_path=LABEL_PATH,
        split_indices_path=TRAIN_PATH,
    )

    val_dataset = NetworkSequenceDataset(
        features_path=FEATURES_PATH,
        sequence_indices_path=SEQUENCE_PATH,
        labels_path=LABEL_PATH,
        split_indices_path=VAL_PATH,
    )

    test_dataset = NetworkSequenceDataset(
        features_path=FEATURES_PATH,
        sequence_indices_path=SEQUENCE_PATH,
        labels_path=LABEL_PATH,
        split_indices_path=TEST_PATH,
    )

    print(
        f"Train samples      : {len(train_dataset):,}"
    )

    print(
        f"Validation samples : {len(val_dataset):,}"
    )

    print(
        f"Test samples       : {len(test_dataset):,}"
    )

    print(
        f"Window size        : {train_dataset.window_size}"
    )

    print(
        f"Features            : {train_dataset.num_features}"
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    print("\n[2] Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print(
        f"Train batches      : {len(train_loader):,}"
    )

    print(
        f"Validation batches : {len(val_loader):,}"
    )

    print(
        f"Test batches       : {len(test_loader):,}"
    )

    # --------------------------------------------------------
    # Test one batch
    # --------------------------------------------------------

    print("\n[3] Testing one batch...")

    X, y = next(iter(train_loader))

    print(
        f"X shape : {X.shape}"
    )

    print(
        f"X dtype : {X.dtype}"
    )

    print(
        f"y shape : {y.shape}"
    )

    print(
        f"y dtype : {y.dtype}"
    )

    print(
        f"X min   : {X.min().item():.6f}"
    )

    print(
        f"X max   : {X.max().item():.6f}"
    )

    print(
        f"X mean  : {X.mean().item():.6f}"
    )

    print(
        f"X std   : {X.std().item():.6f}"
    )

    print(
        f"NaN     : {torch.isnan(X).any().item()}"
    )

    print(
        f"Inf     : {torch.isinf(X).any().item()}"
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    assert X.shape == (
        BATCH_SIZE,
        train_dataset.window_size,
        train_dataset.num_features,
    )

    assert y.shape == (
        BATCH_SIZE,
    )

    assert X.dtype == torch.float32
    assert y.dtype == torch.long

    # --------------------------------------------------------
    # Speed benchmark
    # --------------------------------------------------------

    print("\n[4] DataLoader benchmark...")

    num_batches_to_test = min(
        500,
        len(train_loader),
    )

    start_time = time.perf_counter()

    for batch_idx, (X, y) in enumerate(
        train_loader
    ):

        if batch_idx + 1 >= num_batches_to_test:
            break

    elapsed = (
        time.perf_counter()
        - start_time
    )

    batch_per_second = (
        num_batches_to_test / elapsed
    )

    print(
        f"Test batches : {num_batches_to_test}"
    )

    print(
        f"Time         : {elapsed:.4f} seconds"
    )

    print(
        f"Batch/s      : {batch_per_second:.2f}"
    )

    print("\n" + "=" * 70)
    print("PHASE A4 TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()