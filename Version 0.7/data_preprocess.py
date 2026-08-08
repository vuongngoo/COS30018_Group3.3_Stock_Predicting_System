import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

DATA_DIR = "data_cache"


def create_sliding_windows(
    features: np.ndarray,
    targets: np.ndarray,
    seq_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:

    # Store the generated feature sequences and corresponding targets
    X_list, y_list = [], []

    # Create a sequence using the previous seq_length trading days
    # and assign the following day's target to that sequence
    for i in range(len(features) - seq_length):
        X_list.append(features[i : i + seq_length])
        y_list.append(targets[i + seq_length])

    # Convert the lists into NumPy arrays
    # X shape: (number of samples, sequence length, number of features)
    # y shape: (number of samples,)
    return np.array(X_list), np.array(y_list)


def preprocess_all_datasets(
    seq_length: int = 60,       # 60 trading days lookback window
    train_ratio: float = 0.8,   # 80% train split
    val_ratio: float = 0.1,     # 10% validation split
    target_horizon: int = 1     # Predict Day +1 direction
) -> Dict[str, np.ndarray]:

    # Find baseline feature CSV files
    # Sentiment feature datasets are excluded from this preprocessing
    feature_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "features_*.csv"))
        if "features_sentiment_" not in os.path.basename(f)
    ]

    # Stop the process if no baseline feature datasets are found
    if not feature_files:
        raise FileNotFoundError(
            f"No baseline 'features_*.csv' files found in '{DATA_DIR}/'. "
            "Run feature_engineering.py first!"
        )

    print(f"=== Starting Preprocessing (BINARY CLASSIFICATION) ===")
    print(
        f"Lookback Window: {seq_length} Days | "
        f"Target Horizon: Day +{target_horizon}"
    )

    # Lists used to store the processed sequences from all tickers
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    # Store the original continuous returns for test evaluation
    y_test_raw_list = []

    # Process each ticker independently
    for file_path in feature_files:

        # Extract the ticker name from the filename
        ticker_label = (
            os.path.basename(file_path)
            .replace("features_", "")
            .replace(".csv", "")
        )

        print(
            f"\n[Preprocessing] Processing ticker: "
            f"{ticker_label}..."
        )

        # Load the feature dataset using Date as the index
        df = pd.read_csv(
            file_path,
            index_col='Date',
            parse_dates=True
        )

        # 1. Guarantee chronological order
        # This is important because the data represents a time series
        df = df.sort_index()

        # 2. Identify the target column
        target_col = f"target_{target_horizon}d_return"

        # Use the first available target column if the expected
        # target column is not found
        if target_col not in df.columns:

            # Find columns containing target values
            target_cols = [
                c for c in df.columns
                if c.startswith('target_')
            ]

            # Stop if no target column exists
            if not target_cols:
                raise KeyError(
                    f"No target column found in {file_path}"
                )

            # Use the first available target column
            target_col = target_cols[0]

        # 3. Remove infinite and missing values
        # to ensure that all inputs contain valid numerical values
        df = (
            df
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        # 4. Convert the continuous return target into
        # a binary classification target
        #
        # 1 = UP: future return is greater than zero
        # 0 = DOWN: future return is zero or negative
        binary_target = (
            df[target_col] > 0
        ).astype(np.float32).values

        # Keep the original continuous returns for later evaluation
        raw_returns = df[target_col].values

        # Select only the engineered features
        # and exclude the target columns
        feature_cols = [
            c for c in df.columns
            if c != target_col
            and not c.startswith('target_')
        ]

        # 5. Chronological Train / Validation / Test Splits
        n = len(df)

        # Calculate the end index of the training dataset
        train_end = int(n * train_ratio)

        # Calculate the end index of the validation dataset
        # The remaining observations are used for testing
        val_end = int(
            n * (train_ratio + val_ratio)
        )

        # Create the feature scaler
        scaler = StandardScaler()

        # Fit the scaler ONLY using the training data
        # to prevent information leakage from validation or test data
        scaler.fit(
            df.iloc[:train_end][feature_cols]
        )

        # Apply the training scaling parameters
        # to all feature observations
        scaled_features = scaler.transform(
            df[feature_cols]
        )

        # 6. Extract training features and binary targets
        train_feat = scaled_features[:train_end]
        train_targ = binary_target[:train_end]

        # Include previous seq_length observations before the
        # validation boundary to provide historical context
        val_feat = scaled_features[
            max(0, train_end - seq_length):val_end
        ]
        val_targ = binary_target[
            max(0, train_end - seq_length):val_end
        ]

        # Include previous seq_length observations before the
        # test boundary to provide historical context
        test_feat = scaled_features[
            max(0, val_end - seq_length):
        ]
        test_targ = binary_target[
            max(0, val_end - seq_length):
        ]

        # Keep the original continuous returns for the test set
        # so predictions can also be evaluated against actual returns
        test_raw_ret = raw_returns[
            max(0, val_end - seq_length):
        ]

        # 7. Create 3D sliding-window tensors
        # Each sample contains 60 days of historical features
        X_tr, y_tr = create_sliding_windows(
            train_feat,
            train_targ,
            seq_length
        )

        X_v, y_v = create_sliding_windows(
            val_feat,
            val_targ,
            seq_length
        )

        X_te, y_te = create_sliding_windows(
            test_feat,
            test_targ,
            seq_length
        )

        # Create corresponding raw continuous targets for test samples
        _, y_te_raw = create_sliding_windows(
            test_feat,
            test_raw_ret,
            seq_length
        )

        # Store the generated sequences for this ticker
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)

        X_val_list.append(X_v)
        y_val_list.append(y_v)

        X_test_list.append(X_te)
        y_test_list.append(y_te)

        # Store the original test returns
        y_test_raw_list.append(y_te_raw)

        # Display the number of sequences generated
        print(
            f"  {ticker_label} Sequences -> "
            f"Train: {X_tr.shape[0]}, "
            f"Val: {X_v.shape[0]}, "
            f"Test: {X_te.shape[0]}"
        )

    # 8. Concatenate sequences from all tickers
    # to create unified training, validation, and test datasets
    X_train = np.concatenate(
        X_train_list,
        axis=0
    )

    y_train = np.concatenate(
        y_train_list,
        axis=0
    )

    X_val = np.concatenate(
        X_val_list,
        axis=0
    )

    y_val = np.concatenate(
        y_val_list,
        axis=0
    )

    X_test = np.concatenate(
        X_test_list,
        axis=0
    )

    y_test = np.concatenate(
        y_test_list,
        axis=0
    )

    # Combine the original continuous test returns
    y_test_raw = np.concatenate(
        y_test_raw_list,
        axis=0
    )

    # Calculate the percentage of positive training samples
    # to check the balance between UP and DOWN classes
    pos_pct = np.mean(
        y_train == 1.0
    ) * 100

    print(
        f"\nTraining Class Balance -> "
        f"Positive (UP): {pos_pct:.2f}% | "
        f"Negative (DOWN): {100 - pos_pct:.2f}%"
    )

    # 9. Save the preprocessed tensors to a compressed NumPy file
    output_file = os.path.join(
        DATA_DIR,
        "preprocessed_tensors.npz"
    )

    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        y_test_raw=y_test_raw
    )

    # Display the preprocessing results
    print("\n=== Classification Preprocessing Complete ===")

    print(
        f"X_train Shape: {X_train.shape} "
        f"(Samples, {seq_length}-Day Lookback, "
        f"{X_train.shape[2]} Features)"
    )

    print(
        f"Saved binary tensors to: "
        f"'{output_file}'"
    )

    # Return the processed datasets
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }


# Run the preprocessing pipeline only when this file
# is executed directly
if __name__ == "__main__":

    # Use 60 trading days of historical features
    # to predict whether the following day's return is UP or DOWN
    preprocess_all_datasets(
        seq_length=60,
        train_ratio=0.8,
        val_ratio=0.1,
        target_horizon=1
    )