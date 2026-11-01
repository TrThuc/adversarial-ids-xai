from pathlib import Path
import numpy as np
import pandas as pd


class SequenceBuilder:
    """
    Build sliding-window sequences for the ToN-IoT network dataset.

    Each sequence contains:
        window_size records
        num_features features

    Example:
        window_size = 5
        stride = 1

        [0,1,2,3,4] -> label of record 4
        [1,2,3,4,5] -> label of record 5
        [2,3,4,5,6] -> label of record 6
    """

    def __init__(
        self,
        window_size: int = 5,
        stride: int = 1,
        group_columns=None,
        time_column: str = "ts",
        target_column: str = "type",
    ):
        self.window_size = window_size
        self.stride = stride
        self.group_columns = group_columns or ["src_ip", "dst_ip"]
        self.time_column = time_column
        self.target_column = target_column

    def build(
        self,
        df: pd.DataFrame,
        feature_row_ids: np.ndarray,
    ):
        """
        Build sequence indices.

        Parameters
        ----------
        df:
            Original dataframe containing ts, src_ip, dst_ip and target.
        feature_row_ids:
            Mapping from dataframe row -> row index in features.npy.

        Returns
        -------
        sequence_indices:
            Shape: (num_sequences, window_size)
        labels:
            Shape: (num_sequences,)
        """

        required_columns = (
            self.group_columns
            + [self.time_column, self.target_column]
        )

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if len(df) != len(feature_row_ids):
            raise ValueError(
                "DataFrame length and feature_row_ids length "
                "must be identical."
            )

        # Work on a copy so the original dataframe is untouched.
        work_df = df[
            self.group_columns
            + [self.time_column, self.target_column]
        ].copy()

        # Keep the original row index explicitly.
        work_df["_feature_row_id"] = feature_row_ids

        # Sort exactly according to:
        # source IP -> destination IP -> timestamp
        sort_columns = (
            self.group_columns
            + [self.time_column]
        )

        work_df = work_df.sort_values(
            by=sort_columns,
            kind="mergesort"
        ).reset_index(drop=True)

        sequence_indices = []
        labels = []

        # Group by source/destination pair.
        grouped = work_df.groupby(
            self.group_columns,
            sort=False,
            dropna=False,
        )

        for _, group in grouped:
            group = group.reset_index(drop=True)

            n_records = len(group)

            if n_records < self.window_size:
                continue

            for start in range(
                0,
                n_records - self.window_size + 1,
                self.stride,
            ):
                end = start + self.window_size

                window = group.iloc[start:end]

                # Store indices pointing into features.npy.
                indices = window["_feature_row_id"].to_numpy(
                    dtype=np.int32
                )

                # Sequence label = final record in window.
                target = window[self.target_column].iloc[-1]

                sequence_indices.append(indices)
                labels.append(target)

        if not sequence_indices:
            raise RuntimeError(
                "No sequences were created. "
                "Check window_size, grouping columns and dataset."
            )

        sequence_indices = np.asarray(
            sequence_indices,
            dtype=np.int32
        )

        labels = np.asarray(
            labels,
            dtype=object
        )

        return sequence_indices, labels