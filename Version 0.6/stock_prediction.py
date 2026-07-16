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
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import layers

# Import task C.2, C.3 and C.4
from data_preprocessing import load_scale_and_split_data
from visualization import display_candlestick_chart, display_moving_boxplot
from model_training_setup import construct_model

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

# Task C.4 function call
x_train = []
y_train = []

train_values = scaled_train[FEATURES].values
target_col_idx = FEATURES.index(PRICE_VALUE)

for x in range(PREDICTION_DAYS, len(train_values)):
    x_train.append(train_values[x - PREDICTION_DAYS:x, :])
    y_train.append(train_values[x, target_col_idx])

x_train, y_train = np.array(x_train), np.array(y_train)
input_shape = (x_train.shape[1], x_train.shape[2])

# Train and Save Deep Learning Models (LSTM, SimpleRNN, GRU)
dl_architectures = ['LSTM', 'RNN', 'GRU']

for arch in dl_architectures:
    print(f"\n Training: {arch}")
    dl_model = construct_model(
        layer_type=arch,
        layer_sizes=[50, 50, 50],
        input_shape=input_shape,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    dl_model.fit(x_train, y_train, epochs=25, batch_size=32)
    dl_model.save(f"models/{arch.lower()}_model.keras")