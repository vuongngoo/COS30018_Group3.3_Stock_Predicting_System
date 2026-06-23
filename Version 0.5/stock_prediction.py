# COS30018 - Intelligent Systems - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Import the tasks
from data_preprocessing import load_scale_and_split_data
from visualization import display_candlestick_chart, display_moving_boxplot
from model_training_setup import construct_model
from prediction import (
    predict_multistep_close,
    predict_multivariate_single_step,
    predict_multivariate_multistep
)

COMPANY = 'CBA.AX'
START_DATE = '2020-01-01'
END_DATE = '2024-07-02'

PRICE_VALUE = "Close"
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
PREDICTION_DAYS = 60

SPLIT_METHOD = "date"
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-08-02'

K_DAYS_FUTURE = 7

# Task C.2 function call
scaled_train, scaled_test, scalers, train_df, test_df = load_scale_and_split_data(
    ticker=COMPANY,
    start=START_DATE,
    end=END_DATE,
    features_list=FEATURES,
    target_col=PRICE_VALUE,
    handle_nan="ffill",
    split_method=SPLIT_METHOD,
    split_ratio=SPLIT_RATIO,
    split_date=SPLIT_DATE
)



# Task C.3 function call

# 1. Verify Candlestick aggregation by grouping data into 5-day (weekly) blocks
display_candlestick_chart(train_df, ticker=COMPANY, n=5)

# 2. Verify Variance Spread by passing a 20-day moving window configuration
display_moving_boxplot(train_df, ticker=COMPANY, target_col=PRICE_VALUE, window_size=20)



x_train = []
y_train = []

train_values = scaled_train[FEATURES].values
target_col_idx = FEATURES.index(PRICE_VALUE)

for x in range(PREDICTION_DAYS, len(train_values)):
    x_train.append(train_values[x - PREDICTION_DAYS:x, :])
    y_train.append(train_values[x, target_col_idx])

x_train, y_train = np.array(x_train), np.array(y_train)



# Task C.4 function call
model = construct_model(
    layer_type='LSTM',
    layer_sizes=[50, 50, 50],
    input_shape=(x_train.shape[1], x_train.shape[2]),
    dropout_rate=0.2,
    learning_rate=0.001
)

model.fit(x_train, y_train, epochs=25, batch_size=32)

prior_context = scaled_train[FEATURES].iloc[-PREDICTION_DAYS:]
total_test_matrix = pd.concat((prior_context, scaled_test[FEATURES]), axis=0).values

x_test = []
actual_prices = test_df[PRICE_VALUE].values

for x in range(PREDICTION_DAYS, len(total_test_matrix)):
    x_test.append(total_test_matrix[x - PREDICTION_DAYS:x, :])

x_test = np.array(x_test)
predicted_prices = model.predict(x_test)

target_scaler = scalers[PRICE_VALUE]
predicted_prices = target_scaler.inverse_transform(predicted_prices)



plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Stock Predicting System")

plt.xlabel("Time Steps")
plt.ylabel(f"{COMPANY} Currency Valuation ($)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()



# Task C.5 function call

last_known_window = total_test_matrix[-PREDICTION_DAYS:, :]

# Single-step Multivariate Prediction
single_step_val = predict_multivariate_single_step(model, last_known_window, target_scaler)
print(f"Single-step Multivariate Prediction: ${single_step_val:.2f}")

# Multistep Univariate Prediction
multistep_sequence = predict_multistep_close(model, last_known_window, K_DAYS_FUTURE, target_scaler, target_col_idx)
print(f"Multistep Univariate Prediction with ({K_DAYS_FUTURE} days ahead): {np.round(multistep_sequence, 2)}")

# Multistep Multivariate Prediction
combined_sequence = predict_multivariate_multistep(model, last_known_window, K_DAYS_FUTURE, target_scaler, target_col_idx)
print(f"Multistep Multivariate Prediction with ({K_DAYS_FUTURE} days ahead): {np.round(combined_sequence, 2)}")