import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data_cache"
DECAY_FACTOR = 0.70  # Carry forward 70% of previous day's sentiment on non-news days


def merge_sentiment_to_new_datasets():
    # Find all original engineered feature files
    feature_files = glob.glob(os.path.join(DATA_DIR, "features_*.csv"))
    
    # Filter out any files that are already merged sentiment outputs
    feature_files = [f for f in feature_files if "features_sentiment_" not in os.path.basename(f)]

    if not feature_files:
        print(f"No original feature files found in '{DATA_DIR}'. Run feature_engineering.py first!")
        return



    for file_path in feature_files:
        filename = os.path.basename(file_path)
        ticker_label = filename.replace("features_", "").replace(".csv", "")

        print(f"\nProcessing: {ticker_label}")

        # Load original technical features & guarantee ascending date order
        df_features = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df_features = df_features.sort_index()

        # Load daily sentiment
        sentiment_file = os.path.join(DATA_DIR, f"daily_sentiment_{ticker_label}.csv")

        if os.path.exists(sentiment_file):
            df_sent = pd.read_csv(sentiment_file, parse_dates=['Date']).set_index('Date')
            df_sent = df_sent.sort_index()

            # Create a full 7-day calendar date range covering start to end
            full_date_range = pd.date_range(
                start=min(df_features.index.min(), df_sent.index.min()),
                end=max(df_features.index.max(), df_sent.index.max()),
                freq='D'
            )
            df_sent_full = df_sent.reindex(full_date_range)

            raw_sentiment = df_sent_full['Daily_Sentiment'].values if 'Daily_Sentiment' in df_sent_full.columns else np.full(len(df_sent_full), np.nan)
            decayed_sentiment = np.zeros(len(df_sent_full))
            current_score = 0.0

            for i in range(len(raw_sentiment)):
                val = raw_sentiment[i]
                # If valid non-NaN news score exists
                if not np.isnan(val) and val != 0.0:
                    current_score = val
                else:
                    current_score = current_score * DECAY_FACTOR
                decayed_sentiment[i] = current_score

            df_sent_full['sentiment_score'] = decayed_sentiment
            df_sent_full['sentiment_7d_sma'] = df_sent_full['sentiment_score'].rolling(window=7, min_periods=1).mean()

            df_merged = df_features.join(
                df_sent_full[['sentiment_score', 'sentiment_7d_sma', 'News_Count']], 
                how='left'
            )

            df_merged['News_Count'] = df_merged['News_Count'].fillna(0).astype(int)
            df_merged['has_news'] = (df_merged['News_Count'] > 0).astype(int)
            df_merged = df_merged.drop(columns=['News_Count'])

        else:
            print(f"  Warning: No sentiment file found for {ticker_label}. Defaulting sentiment features to 0.0.")
            df_merged = df_features.copy()
            df_merged['has_news'] = 0
            df_merged['sentiment_score'] = 0.0
            df_merged['sentiment_7d_sma'] = 0.0

        target_cols = [c for c in df_merged.columns if c.startswith('target_')]
        feature_cols = [c for c in df_merged.columns if not c.startswith('target_')]
        
        # Structure dataframe: Features first, Targets strictly at the end
        df_merged = df_merged[feature_cols + target_cols]

        # Save to NEW output file
        output_filename = f"features_sentiment_{ticker_label}.csv"
        output_path = os.path.join(DATA_DIR, output_filename)
        df_merged.to_csv(output_path)

        print(
            f"  Created NEW dataset -> '{output_filename}' "
            f"({len(df_merged)} rows x {df_merged.shape[1]} cols)"
        )

    print("\nAll sentiment feature datasets successfully generated ")



if __name__ == "__main__":
    merge_sentiment_to_new_datasets()