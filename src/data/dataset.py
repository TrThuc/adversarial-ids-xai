from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NetworkSequenceDataset(Dataset):
    """
    Dataset for preprocessed ToN-IoT network sequences.

    Data flow:

        features.npy
             +
        sequence_indices.npy
             +
        split_indices.npy
             ↓
        X: [window_size, num_features]
    """

    def __init__(
        self,
        features_path,
        sequence_indices_path,
        labels_path,
        split_indices_path,
    ):
        self.features = np.load(
            features_path,
            mmap_mode="r",
        )

        self.sequence_indices = np.load(
            sequence_indices_path,
            mmap_mode="r",
        )

        self.labels = np.load(
            labels_path,
            mmap_mode="r",
        )

        self.split_indices = np.load(
            split_indices_path,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if self.sequence_indices.ndim != 2:
            raise ValueError(
                "sequence_indices must be 2-dimensional."
            )

        if self.labels.ndim != 1:
            raise ValueError(
                "labels must be 1-dimensional."
            )

        if len(self.sequence_indices) != len(self.labels):
            raise ValueError(
                "sequence_indices and labels have different lengths."
            )

        if self.features.ndim != 2:
            raise ValueError(
                "features must be 2-dimensional."
            )

        if self.split_indices.ndim != 1:
            raise ValueError(
                "split_indices must be 1-dimensional."
            )

        if np.any(
            self.split_indices < 0
        ) or np.any(
            self.split_indices >= len(self.sequence_indices)
        ):
            raise IndexError(
                "split_indices contains invalid sequence IDs."
            )

        self.window_size = self.sequence_indices.shape[1]
        self.num_features = self.features.shape[1]

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx):

        # ----------------------------------------------------
        # Step 1:
        # Dataset index -> sequence ID
        # ----------------------------------------------------

        sequence_id = int(
            self.split_indices[idx]
        )

        # ----------------------------------------------------
        # Step 2:
        # Sequence ID -> 5 record indices
        # ----------------------------------------------------

        record_indices = self.sequence_indices[
            sequence_id
        ]

        # ----------------------------------------------------
        # Step 3:
        # Record indices -> actual feature matrix
        # ----------------------------------------------------

        x = self.features[
            record_indices
        ]

        # ----------------------------------------------------
        # Step 4:
        # Sequence label
        # ----------------------------------------------------

        y = self.labels[
            sequence_id
        ]

        # ----------------------------------------------------
        # Step 5:
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        x = torch.from_numpy(
            np.asarray(x, dtype=np.float32)
        )

        y = torch.tensor(
            int(y),
            dtype=torch.long,
        )

        return x, y