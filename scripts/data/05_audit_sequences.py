import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Train_Test_datasets"
    / "Train_Test_Network_dataset"
    / "train_test_network.csv"
)

SEQUENCE_INDICES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_indices.npy"
)

Y_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "y.npy"
)


def main():
    print("=" * 70)
    print("SEQUENCE AUDIT")
    print("=" * 70)

    print("\nLoading dataset...")
    df = pd.read_csv(DATASET)

    sequence_indices = np.load(
        SEQUENCE_INDICES,
        mmap_mode="r",
    )

    y = np.load(
        Y_PATH,
        mmap_mode="r",
    )

    print(f"Records   : {len(df):,}")
    print(f"Sequences : {len(sequence_indices):,}")

    print("\nClass distribution:")
    labels = df["type"]

    print(
        labels.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSequence label distribution:")
    unique, counts = np.unique(
        y,
        return_counts=True,
    )

    for label, count in zip(unique, counts):
        print(
            f"{label}: {count:,}"
        )

    print("\nChecking sequence label consistency...")

    mixed_count = 0
    pure_count = 0

    examples = []

    for seq_id in range(len(sequence_indices)):
        record_ids = sequence_indices[seq_id]

        seq_labels = df.iloc[
            record_ids
        ]["type"].to_numpy()

        unique_labels = np.unique(seq_labels)

        if len(unique_labels) == 1:
            pure_count += 1
        else:
            mixed_count += 1

            if len(examples) < 10:
                examples.append(
                    (
                        seq_id,
                        unique_labels.tolist(),
                        seq_labels.tolist(),
                    )
                )

    total = pure_count + mixed_count

    print(f"\nPure sequences  : {pure_count:,}")
    print(f"Mixed sequences : {mixed_count:,}")

    if total:
        print(
            f"Mixed ratio     : "
            f"{mixed_count / total:.4%}"
        )

    if examples:
        print("\nExample mixed sequences:")

        for seq_id, unique_labels, seq_labels in examples:
            print("-" * 70)
            print(f"Sequence ID : {seq_id}")
            print(f"Unique      : {unique_labels}")
            print(f"Labels      : {seq_labels}")


if __name__ == "__main__":
    main()