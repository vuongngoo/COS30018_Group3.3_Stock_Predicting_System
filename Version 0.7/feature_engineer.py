import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data_cache"


def extract_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the original dataset
    data = df.copy()

    # 1. Calculate price momentum and returns over different periods
    data['return_1d'] = data['Close'].pct_change(1)
    data['return_5d'] = data['Close'].pct_change(5)
    data['return_20d'] = data['Close'].pct_change(20)

    # 2. Calculate intraday price movement using candlestick information
    data['high_low_ratio'] = (data['High'] - data['Low']) / (data['Close'] + 1e-8)
    data['close_open_ratio'] = (data['Close'] - data['Open']) / (data['Open'] + 1e-8)

    # 3. Calculate the distance between the current price and
    # Simple Moving Averages over 20, 50, and 200 days
    sma_20 = data['Close'].rolling(20).mean()
    sma_50 = data['Close'].rolling(50).mean()
    sma_200 = data['Close'].rolling(200).mean()

    data['dist_sma_20'] = (data['Close'] - sma_20) / (sma_20 + 1e-8)
    data['dist_sma_50'] = (data['Close'] - sma_50) / (sma_50 + 1e-8)
    data['dist_sma_200'] = (data['Close'] - sma_200) / (sma_200 + 1e-8)

    # 4. Calculate the 14-day Relative Strength Index (RSI)
    # using Wilder's Exponential Weighted Moving Average
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    loss = (-delta.where(delta < 0, 0.0)).ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # Calculate the Relative Strength value
    rs = gain / (loss + 1e-8)

    # Convert Relative Strength into RSI
    data['rsi_14'] = 100 - (100 / (1 + rs))

    # 5. Calculate the Moving Average Convergence Divergence (MACD)
    # using 12-day and 26-day exponential moving averages
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()

    # Normalize MACD by the current closing price
    macd = (ema_12 - ema_26) / (data['Close'] + 1e-8)

    # Calculate the MACD signal line
    signal = macd.ewm(span=9, adjust=False).mean()

    # Store the MACD components as features
    data['macd_line'] = macd
    data['macd_signal'] = signal
    data['macd_hist'] = macd - signal

    # 6. Calculate Bollinger Bands for volatility measurement
    bb_sma = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()

    # Calculate the upper and lower Bollinger Bands
    upper_bb = bb_sma + (bb_std * 2)
    lower_bb = bb_sma - (bb_std * 2)

    # Calculate the normalized Bollinger Band width
    data['bb_width'] = (upper_bb - lower_bb) / (bb_sma + 1e-8)

    # Calculate the current price position within the Bollinger Bands
    data['bb_pct'] = (
        (data['Close'] - lower_bb) /
        (upper_bb - lower_bb + 1e-8)
    )

    # Calculate True Range components for ATR
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()

    # Calculate True Range using the largest price movement
    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    # Calculate the 14-day Average True Range
    atr = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # Normalize ATR by the current closing price
    data['norm_atr'] = atr / (data['Close'] + 1e-8)

    # 7. Calculate volume-based features
    # Calculate the 20-day average trading volume
    volume_sma_20 = data['Volume'].rolling(20).mean()

    # Compare current volume with the 20-day average volume
    data['volume_ratio'] = data['Volume'] / (volume_sma_20 + 1e-8)

    # Calculate the daily percentage change in trading volume
    data['volume_change_1d'] = data['Volume'].pct_change(1)

    # Calculate On-Balance Volume (OBV)
    obv = (
        np.sign(data['Close'].diff()) *
        data['Volume']
    ).fillna(0).cumsum()

    # Calculate the percentage change in OBV over 5 days
    data['obv_change_5d'] = obv.pct_change(5)

    # Return the dataset with the generated technical features
    return data


def process_all_feature_datasets(target_horizon: int = 5):

    # Find all raw stock CSV files in the data directory
    raw_files = glob.glob(
        os.path.join(DATA_DIR, "raw_*.csv")
    )

    # Process each raw dataset individually
    for file_path in raw_files:

        # Extract the filename from the full file path
        filename = os.path.basename(file_path)

        # Extract the ticker name from the filename
        ticker_label = (
            filename
            .replace("raw_", "")
            .replace(".csv", "")
        )

        print(
            f"Processing: {ticker_label}"
        )

        # Load the raw dataset and use Date as the time-series index
        df = pd.read_csv(
            file_path,
            index_col='Date',
            parse_dates=True
        )

        # 1. Generate the technical features
        df_featured = extract_stationary_features(df)

        # 2. Define the prediction target
        # The target represents the future return after the specified
        # number of trading days
        df_featured[
            f'target_{target_horizon}d_return'
        ] = (
            df_featured['Close'].shift(-target_horizon)
            - df_featured['Close']
        ) / (df_featured['Close'] + 1e-8)

        # 3. Remove the original OHLCV price levels
        # because the model uses the engineered features instead
        drop_cols = [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]

        df_dataset = df_featured.drop(
            columns=drop_cols
        )

        # 4. Replace infinite values caused by division operations
        # with NaN and remove rows containing invalid values
        df_dataset = (
            df_dataset
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        # Create the output path for the feature dataset
        output_path = os.path.join(
            DATA_DIR,
            f"features_{ticker_label}.csv"
        )

        # Save the final feature matrix as a CSV file
        df_dataset.to_csv(output_path)

        # Display the size of the generated dataset
        print(
            f" Saved clean feature matrix to: "
            f"'{output_path}' "
            f"({len(df_dataset)} rows x "
            f"{df_dataset.shape[1]} cols)"
        )



if __name__ == "__main__":
    process_all_feature_datasets(
        target_horizon=5
    )