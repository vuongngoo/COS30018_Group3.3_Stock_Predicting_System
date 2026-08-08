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

    X_list, y_list = [], []
    for i in range(len(features) - seq_length):
        X_list.append(features[i : i + seq_length])
        y_list.append(targets[i + seq_length])
    return np.array(X_list), np.array(y_list)


def preprocess_sentiment_datasets(
    seq_length: int = 60,       # 60 trading days sequence window
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    target_horizon: int = 1     # Day +1 direction
) -> Dict[str, np.ndarray]:

    feature_files = glob.glob(os.path.join(DATA_DIR, "features_sentiment_*.csv"))

    if not feature_files:
        raise FileNotFoundError(
            f"No 'features_sentiment_*.csv' files found in '{DATA_DIR}/'. "
            "Run sentiment_merge.py first!"
        )

    print(f"=== Starting Sentiment Preprocessing (BINARY CLASSIFICATION) ===")
    print(f"Lookback Window: {seq_length} Days | Target Horizon: Day +{target_horizon}")

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    y_test_raw_list = []

    for file_path in feature_files:
        ticker_label = os.path.basename(file_path).replace("features_sentiment_", "").replace(".csv", "")
        print(f"\n[Preprocessing] Processing ticker: {ticker_label}...")

        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

        # 1. Guarantee Chronological Order
        df = df.sort_index()

        # 2. Select Target Column
        target_col = f'target_{target_horizon}d_return'
        if target_col not in df.columns:
            target_cols = [c for c in df.columns if c.startswith('target_')]
            if not target_cols:
                raise KeyError(f"No target column found in {file_path}")
            target_col = target_cols[0]

        # 3. Clean infinite or NaN values prior to scaling
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # 4. Convert Continuous Return to Binary Classification Direction (1 = UP, 0 = DOWN)
        binary_target = (df[target_col] > 0).astype(np.float32).values
        raw_returns = df[target_col].values

        feature_cols = [c for c in df.columns if not c.startswith('target_')]

        # 5. Chronological Split Indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Fit StandardScaler ONLY on Training Set to prevent data leakage
        scaler = StandardScaler()
        scaler.fit(df.iloc[:train_end][feature_cols])

        # Transform feature data using training parameters
        scaled_features = scaler.transform(df[feature_cols])

        # 6. Extract Train, Val, Test with lookback sequence padding
        train_feat = scaled_features[:train_end]
        train_targ = binary_target[:train_end]

        val_feat = scaled_features[max(0, train_end - seq_length):val_end]
        val_targ = binary_target[max(0, train_end - seq_length):val_end]

        test_feat = scaled_features[max(0, val_end - seq_length):]
        test_targ = binary_target[max(0, val_end - seq_length):]
        test_raw_ret = raw_returns[max(0, val_end - seq_length):]

        # 7. Create 3D Sliding Window Tensors
        X_tr, y_tr = create_sliding_windows(train_feat, train_targ, seq_length)
        X_v, y_v = create_sliding_windows(val_feat, val_targ, seq_length)
        X_te, y_te = create_sliding_windows(test_feat, test_targ, seq_length)
        _, y_te_raw = create_sliding_windows(test_feat, test_raw_ret, seq_length)

        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_val_list.append(X_v)
        y_val_list.append(y_v)
        X_test_list.append(X_te)
        y_test_list.append(y_te)
        y_test_raw_list.append(y_te_raw)

        print(f"  {ticker_label} Sequences -> Train: {X_tr.shape[0]}, Val: {X_v.shape[0]}, Test: {X_te.shape[0]}")

    # 8. Concatenate All Ticker Sequences into Master Datasets
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    y_test_raw = np.concatenate(y_test_raw_list, axis=0)

    # Class balance check
    pos_pct = np.mean(y_train == 1.0) * 100
    print(f"\nTraining Class Balance -> Positive (UP): {pos_pct:.2f}% | Negative (DOWN): {100 - pos_pct:.2f}%")

    # 9. Save Arrays to Sentiment Tensor File
    output_file = os.path.join(DATA_DIR, "preprocessed_tensors_sentiment.npz")
    np.savez_compressed(
        output_file,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        y_test_raw=y_test_raw
    )

    print("\n=== Sentiment Classification Preprocessing Complete ===")
    print(f"X_train Shape: {X_train.shape} (Samples, {seq_length}-Day Lookback, {X_train.shape[2]} Features)")
    print(f"Saved dataset tensors to: '{output_file}'")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }


if __name__ == "__main__":
    preprocess_sentiment_datasets(seq_length=60, train_ratio=0.8, val_ratio=0.1, target_horizon=1)