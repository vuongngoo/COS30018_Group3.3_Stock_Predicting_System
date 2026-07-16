# COS30018 - Intelligent Systems - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

# Task C.6:
# This file is used to generate the dataset for Transformer training (Ensemble Approach 2)

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm

from data_preprocessing import load_scale_and_split_data
from predict_functions import predict_multivariate_multistep

COMPANY = 'CBA.AX'
START_DATE = '2020-01-01'
END_DATE = '2024-07-02'
PRICE_VALUE = "Close"
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
PREDICTION_DAYS = 60
K_DAYS_FUTURE = 7
SPLIT_DATE = '2023-08-02'

# Returns both the scaled datasets (scaled_train, scaled_test) and the raw unscaled equivalents (train_df, test_df)
def create_meta_dataset():
    scaled_train, scaled_test, scalers, train_df, test_df = load_scale_and_split_data(
        ticker=COMPANY, start=START_DATE, end=END_DATE, features_list=FEATURES,
        target_col=PRICE_VALUE, handle_nan="ffill", split_method="date", split_date=SPLIT_DATE
    )

    # Extracts the specific MinMaxScaler instance used for the Close price
    # This scaler is necessary to convert scaled predictions back into raw dollar amounts later
    # Next, Finds the column index of 'Close'
    # This tells the multi-step predictor which column represents the target to overwrite during sequential loops
    target_scaler = scalers[PRICE_VALUE]
    target_col_idx = FEATURES.index(PRICE_VALUE)

    # Load the models
    models = {
        'lstm': tf.keras.models.load_model("models/lstm_model.keras"),
        'rnn': tf.keras.models.load_model("models/rnn_model.keras"),
        'gru': tf.keras.models.load_model("models/gru_model.keras")
    }

    # Grabs the last 60 days (PREDICTION_DAYS = 60) of the training set
    # Then, Stitches the 60-day context to the top of the test features,
    #   converting the resulting pandas DataFrame into a raw NumPy array
    # This guarantees that every test sample has a full sliding window
    prior_context = scaled_train[FEATURES].iloc[-PREDICTION_DAYS:]
    total_matrix = pd.concat((prior_context, scaled_test[FEATURES]), axis=0).values
    
    # Extracts the raw, unscaled closing prices of the test set
    # This is important because our normalization anchor and final targets must be calculated using real-world dollar scales
    test_actuals_close = test_df[PRICE_VALUE].values 

    X_meta = []
    Y_meta = []

    # Calculates the total number of sliding window iterations
    total_samples = len(scaled_test) - K_DAYS_FUTURE

    for idx in tqdm(range(total_samples)):
        
        # Extracts a 60-day historical sliding window containing all 5 features, which serves as the input sequence for the models
        current_window = total_matrix[idx : idx + PREDICTION_DAYS, :]
        
        # Today's actual unscaled close price (Day 60)
        # This is the anchor point for normalization
        anchor_price = test_actuals_close[idx + PREDICTION_DAYS - 1] if (idx + PREDICTION_DAYS - 1) < len(test_actuals_close) else test_actuals_close[-1]

        # Generates a 7-day-ahead future price trajectory for each model
        # The predictions are scale-restored back to actual dollar amounts
        # .flatten() transforms the outputs into 1D arrays of size (7)
        lstm_preds = predict_multivariate_multistep(models['lstm'], current_window, K_DAYS_FUTURE, target_scaler, target_col_idx).flatten()
        rnn_preds  = predict_multivariate_multistep(models['rnn'], current_window, K_DAYS_FUTURE, target_scaler, target_col_idx).flatten()
        gru_preds  = predict_multivariate_multistep(models['gru'], current_window, K_DAYS_FUTURE, target_scaler, target_col_idx).flatten()
        
        # Subtracts the anchor price from the raw predictions
        # Instead of predicting absolute future prices,
        #   the models are now represented by how much they expect the price to drift relative to today
        lstm_diff = lstm_preds - anchor_price
        rnn_diff  = rnn_preds - anchor_price
        gru_diff  = gru_preds - anchor_price
        
        # Combines the three difference arrays into a 2D matrix of shape (k, 3)
        #   Each row represents a future day (t+1 to t+k)
        #   Each column represents the relative drift predicted by a specific model ([LSTM, RNN, GRU]).
        sample_features = np.column_stack((lstm_diff, rnn_diff, gru_diff))
        
        # Extracts the actual unscaled target closing prices for the upcoming 7 days
        # Subtracts the anchor price from the actual prices
        # This turns the targets into relative drifts, aligning them with the format of the input features
        actual_targets = test_actuals_close[idx + PREDICTION_DAYS : idx + PREDICTION_DAYS + K_DAYS_FUTURE]
        target_diffs = actual_targets - anchor_price
        
        # Appends the (7, 3) input matrix to the meta-features collection
        X_meta.append(sample_features)

        # Reshapes the target array from (7) to (7, 1) using expand_dims and adds it to the labels collection
        Y_meta.append(np.expand_dims(target_diffs, axis=-1))

    # Save the dataset
    X_meta = np.array(X_meta)
    Y_meta = np.array(Y_meta)
    os.makedirs('data_meta', exist_ok=True)
    np.save('data_meta/X_meta_train.npy', X_meta)
    np.save('data_meta/Y_meta_train.npy', Y_meta)
    print(f"Shapes: X={X_meta.shape}, Y={Y_meta.shape}")

if __name__ == "__main__":
    create_meta_dataset()