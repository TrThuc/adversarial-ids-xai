from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.sequence import SequenceBuilder
from src.data.split import create_splits




DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Train_Test_datasets"
    / "Train_Test_Network_dataset"
    / "train_test_network.csv"
)

A2_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ARTIFACT_DIR = A2_DIR / "artifacts"


NUMERICAL_COLUMNS = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code",
]

CATEGORICAL_COLUMNS = [
    "src_ip",
    "dst_ip",
    "proto",
    "service",
    "conn_state",
    "dns_query",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "ssl_established",
    "ssl_subject",
    "ssl_issuer",
    "http_trans_depth",
    "http_method",
    "http_uri",
    "http_version",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_addl",
    "weird_notice",
]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

TIME_COLUMN = "ts"
TARGET_COLUMN = "type"
WINDOW_SIZE = 5
STRIDE = 1
RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("A2 - TRAIN-ONLY PREPROCESSING")
    print("=" * 70)

    A2_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    print(f"Records: {len(df):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    missing = [
        column
        for column in [TIME_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    feature_row_ids = np.arange(
        len(df),
        dtype=np.int32,
    )

    sequence_builder = SequenceBuilder(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        group_columns=["src_ip", "dst_ip"],
        time_column=TIME_COLUMN,
        target_column=TARGET_COLUMN,
    )

    sequence_indices, sequence_labels = sequence_builder.build(
        df=df,
        feature_row_ids=feature_row_ids,
    )

    classes = sorted(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    y = np.asarray(
        [
            class_to_index[label]
            for label in sequence_labels
        ],
        dtype=np.int64,
    )

    train_indices, val_indices, test_indices = create_splits(
        y=y,
        test_size=0.20,
        validation_ratio=0.10,
        random_state=RANDOM_STATE,
    )

    train_sequence_rows = sequence_indices[train_indices]
    train_row_ids = np.unique(
        train_sequence_rows.reshape(-1)
    )

    print(f"Sequences: {len(sequence_indices):,}")
    print(f"Train sequences: {len(train_indices):,}")
    print(f"Validation sequences: {len(val_indices):,}")
    print(f"Test sequences: {len(test_indices):,}")
    print(f"Rows used to fit preprocessing: {len(train_row_ids):,}")

    X_num_all = df[NUMERICAL_COLUMNS].copy()
    X_cat_all = df[CATEGORICAL_COLUMNS].copy()

    X_num_train = X_num_all.iloc[train_row_ids]
    X_cat_train = X_cat_all.iloc[train_row_ids]

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    numeric_scaler = StandardScaler()
    final_scaler = StandardScaler()

    numeric_imputer.fit(X_num_train)
    numeric_scaler.fit(
        numeric_imputer.transform(X_num_train)
    )

    categorical_imputer.fit(X_cat_train)
    categorical_encoder.fit(
        categorical_imputer.transform(X_cat_train)
    )

    X_num = numeric_imputer.transform(X_num_all)
    X_num = numeric_scaler.transform(X_num)

    X_cat = categorical_imputer.transform(X_cat_all)
    X_cat = categorical_encoder.transform(X_cat)

    X = np.concatenate(
        [X_num, X_cat],
        axis=1,
    ).astype(np.float32)

    final_scaler.fit(X[train_row_ids])
    X = final_scaler.transform(X).astype(np.float32)

    if np.isnan(X).any():
        raise ValueError("NaN detected.")

    if np.isinf(X).any():
        raise ValueError("Inf detected.")

    np.save(A2_DIR / "features.npy", X)
    np.save(A2_DIR / "sequence_indices.npy", sequence_indices)
    np.save(A2_DIR / "y.npy", y)
    np.save(A2_DIR / "train_indices.npy", train_indices)
    np.save(A2_DIR / "val_indices.npy", val_indices)
    np.save(A2_DIR / "test_indices.npy", test_indices)

    joblib.dump(
        numeric_imputer,
        ARTIFACT_DIR / "numeric_imputer.joblib",
    )

    joblib.dump(
        numeric_scaler,
        ARTIFACT_DIR / "numeric_scaler.joblib",
    )

    joblib.dump(
        categorical_imputer,
        ARTIFACT_DIR / "categorical_imputer.joblib",
    )

    joblib.dump(
        categorical_encoder,
        ARTIFACT_DIR / "categorical_encoder.joblib",
    )

    joblib.dump(
        final_scaler,
        ARTIFACT_DIR / "final_scaler.joblib",
    )

    metadata = {
        "experiment": "paper_lstm_42_train_scaled",
        "dataset": str(DATASET_PATH.resolve()),
        "num_records": int(len(df)),
        "num_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": [
            "label",
            "type",
            "ts",
        ],
        "sequence": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "group_columns": ["src_ip", "dst_ip"],
            "target_column": TARGET_COLUMN,
            "num_sequences": int(len(sequence_indices)),
        },
        "split": {
            "test_size": 0.20,
            "validation_ratio": 0.10,
            "random_state": RANDOM_STATE,
            "num_train": int(len(train_indices)),
            "num_validation": int(len(val_indices)),
            "num_test": int(len(test_indices)),
        },
        "preprocessing": {
            "fit_scope": "train_sequence_rows_only",
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "categorical_encoder": "OrdinalEncoder",
            "numeric_scaler": "StandardScaler",
            "final_scaler": "StandardScaler",
            "validation_and_test_operation": "transform_only",
            "train_row_count_for_fit": int(len(train_row_ids)),
        },
    }

    with open(
        ARTIFACT_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("A2 preprocessing completed.")
    print(f"Output directory: {A2_DIR}")
    print(f"Feature matrix: {X.shape}")
    print("All preprocessing artifacts were fit on train rows only.")


if __name__ == "__main__":
    main()