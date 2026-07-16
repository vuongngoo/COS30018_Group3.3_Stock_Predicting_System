# COS30018 - Intelligent Systems - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

# Run this file to execute the prediction of all architectures
# (including the 3 baseline models and 2 ensemble approaches)

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from data_preprocessing import load_scale_and_split_data
from predict_functions import predict_multivariate_multistep

COMPANY = 'CBA.AX'
START_DATE = '2020-01-01'
END_DATE = '2024-07-02'
PRICE_VALUE = "Close"
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
PREDICTION_DAYS = 60
K_DAYS_FUTURE = 7

# Reload the data frames and processing scales
scaled_train, scaled_test, scalers, train_df, test_df = load_scale_and_split_data(
    ticker=COMPANY, start=START_DATE, end=END_DATE, features_list=FEATURES,
    target_col=PRICE_VALUE, handle_nan="ffill", split_method="date", split_date='2023-08-02'
)

target_scaler = scalers[PRICE_VALUE]
target_col_idx = FEATURES.index(PRICE_VALUE)

# Extract Ground Truth for the next K days
actuals = test_df[PRICE_VALUE].values[:K_DAYS_FUTURE]

# Build the historic sliding context window matrix
prior_context = scaled_train[FEATURES].iloc[-PREDICTION_DAYS:]
total_test_matrix = pd.concat((prior_context, scaled_test[FEATURES]), axis=0).values
last_known_window = total_test_matrix[-PREDICTION_DAYS:, :]

# Function to print formatted metric strings
def print_metrics(name_str, pred_array, actual_array):
    mae = mean_absolute_error(actual_array, pred_array)
    rmse = np.sqrt(mean_squared_error(actual_array, pred_array))
    mape = mean_absolute_percentage_error(actual_array, pred_array) * 100
    print(f"{name_str:<38} | MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")

# Function to output detailed progress
def print_detailed_evaluation(name_str, pred_array, actual_array):
    print(f" {'Step (Day)':<12} | {'Actual Price':<14} | {'Predicted Price':<17} | {'Absolute Error':<16} | {'Error (%)':<10}")
    
    for i in range(len(actual_array)):
        act_val = actual_array[i]
        pred_val = pred_array[i]
        abs_err = abs(act_val - pred_val)
        pct_err = (abs_err / act_val) * 100
        print(f" Day {i+1:<8} | ${act_val:<12.2f} | ${pred_val:<15.2f} | ${abs_err:<14.2f} | {pct_err:.2f}%")

# Base Models
dl_architectures = ['lstm', 'rnn', 'gru']
dl_predictions = {}

for arch in dl_architectures:
    model = tf.keras.models.load_model(f"models/{arch}_model.keras")
    pred_seq = predict_multivariate_multistep(model, last_known_window, K_DAYS_FUTURE, target_scaler, target_col_idx)
    dl_predictions[arch] = pred_seq.flatten()
    print_metrics(f"{arch.upper()} (Base Model)", dl_predictions[arch], actuals)



# Ensemble Approach 1: Residual-Based Error Correction

# Calculate past 60 days of LSTM mistakes
past_actual_closes = scaled_test[PRICE_VALUE].values[:PREDICTION_DAYS]
past_lstm_predictions = []
lstm_model = tf.keras.models.load_model("models/lstm_model.keras")
for idx in range(PREDICTION_DAYS):
    sub_window = total_test_matrix[idx : idx + PREDICTION_DAYS, :]
    pred = predict_multivariate_multistep(lstm_model, sub_window, 1, target_scaler, target_col_idx).flatten()[0]
    past_lstm_predictions.append(pred)
scale_restored_actuals = test_df[PRICE_VALUE].values[:PREDICTION_DAYS]
historical_errors = scale_restored_actuals - np.array(past_lstm_predictions)

# Fit ARIMA dynamically to the computed past errors
try:
    with open('models/arima_model.pkl', 'rb') as f:
        arima_m = pickle.load(f)

    arima_fit = arima_m.fit(historical_errors)
    predicted_future_errors = arima_fit.predict(n_periods=K_DAYS_FUTURE)

    approach_1_predictions = dl_predictions['lstm'] + predicted_future_errors
    print_metrics("APPROACH 1 (LSTM + ARIMA Error Adjust)", approach_1_predictions, actuals)
except Exception as e:
    print(f"[ARIMA Error Correction Execution Failed]: {e}")
    approach_1_predictions = None



# Ensemble Approach 2: Transformer Stacking Meta-Learner

if os.path.exists("models/transformer_meta_model.keras"):
    anchor_price = test_df[PRICE_VALUE].iloc[PREDICTION_DAYS - 1]

    lstm_diff = dl_predictions['lstm'] - anchor_price
    rnn_diff  = dl_predictions['rnn'] - anchor_price
    gru_diff  = dl_predictions['gru'] - anchor_price
    
    transformer_input = np.column_stack((lstm_diff, rnn_diff, gru_diff))
    transformer_input = np.expand_dims(transformer_input, axis=0)

    transformer_meta = tf.keras.models.load_model("models/transformer_meta_model.keras")
    predicted_diffs = transformer_meta.predict(transformer_input, verbose=0).flatten()

    meta_predictions = predicted_diffs + anchor_price
    
    print_metrics("APPROACH 2 (Transformer Meta Stack)", meta_predictions, actuals)
else:
    print("[Error]: Trained Transformer meta model not found in 'models/' directory.")
    meta_predictions = None



# Evaluation

# Print detailed evaluations for the base models
for arch in dl_architectures:
    print_detailed_evaluation(f"{arch.upper()} (Base Model)", dl_predictions[arch], actuals)

# Print detailed evaluation for Approach 1
if approach_1_predictions is not None:
    print_detailed_evaluation("Approach 1 (LSTM + ARIMA Error Adjust)", approach_1_predictions, actuals)

# Print detailed evaluation for Approach 2
if meta_predictions is not None:
    print_detailed_evaluation("Approach 2 (Transformer Meta Stack)", meta_predictions, actuals)