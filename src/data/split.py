from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def create_splits(
    y: np.ndarray,
    test_size: float = 0.20,
    validation_ratio: float = 0.10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if y.ndim != 1:
        raise ValueError(
            f"Expected y to be 1-dimensional, got {y.shape}"
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    all_indices = np.arange(
        len(y),
        dtype=np.int64
    )

    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=y[train_val_indices],
    )

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(val_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )