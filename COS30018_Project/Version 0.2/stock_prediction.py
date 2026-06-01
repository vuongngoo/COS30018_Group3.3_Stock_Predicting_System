# COS30018 - Intelligent System - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

# stock_prediction.py - Version 0.2
# Updates:
# 1/ Custom Scope and Multi-Feature Processing
# 2/ Data Cleaning (NaN Resolution)
# 3/ Flexible Dataset Splitting
# 4/ Local Caching (Storage Engine)
# 5/ Isolated Feature Scaling

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# CONFIGURATION PARAMETERS

COMPANY = 'CBA.AX'
START_DATE = '2020-01-01'
END_DATE = '2024-07-02'

PRICE_VALUE = "Close"
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
PREDICTION_DAYS = 60

SPLIT_METHOD = "date"
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-08-02'

def load_scale_and_split_data(
    ticker, start, end, features_list, target_col,
    handle_nan="ffill", 
    split_method="date", 
    split_ratio=0.8, 
    split_date=None, 
    cache_dir="data_cache"
):



    # 1 - Local Data Storage and Caching Mechanism

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")
    
    if os.path.exists(cache_file):
        print(f"Loading baseline dataset locally from: {cache_file}")
        raw_data = pd.read_csv(cache_file, index_col=0, parse_dates=True, header=[0, 1]) # Ensure it does not read the first 2 rows
        
        # Check if column headers are currently built as a multi-layered index structure
        # If YES: Tells Pandas to isolate only the names found on Level 0 (the top deck: Open, High, Low, Close, Volume)
        #         and completely discard Level 1 (CBA.AX)
        # If NO: Continue
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
    else:
        print(f"Local cache missing. Fetching data remotely via yfinance...")
        raw_data = yf.download(ticker, start=start, end=end)
        if raw_data.empty:
            raise ValueError(f"Download failed. No operational records returned for ticker: {ticker}")
        raw_data.to_csv(cache_file)
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

    df = raw_data[features_list].copy()
    


    # 2 - Deal with the NaN Issue

    if handle_nan == "ffill":
        df = df.ffill()
    # ffill = Foward Fill
    # Example use:
    # Before ffill: [ 62.5,  62.8,  NaN,  NaN,  63.1 ]
    # After ffill:  [ 62.5,  62.8,  62.8,  62.8,  63.1 ]

    elif handle_nan == "bfill":
        df = df.bfill()
    # bfill = Backward Fill
    # Example use:
    # Before bfill: [ 62.5,  62.8,  NaN,  NaN,  63.1 ]
    # After bfill:  [ 62.5,  62.8,  63.1,  63.1,  63.1 ]

    elif handle_nan == "drop":
        df = df.dropna()
    # Instead of guessing or copying values, dropna() completely
    # destroys the entire row containing an $NaN$ entry
        
    if df.isnull().values.any():
        df = df.dropna()



    # 3 - Advanced Data Splitting Strategy

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    if split_method == "ratio":
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    # This method splits the data based on a pure mathematical proportion
    # (like 80% training and 20% testing)
    # 1/ Calculates how many rows represent the training chunk
    #    If there are 1,000 trading days and a ratio of 0.8, it calculates 800
    #    Then converts that number into a clean index integer (split_idx = 800)
    # 2/ Uses integer-location slicing (iloc) to grab all rows from row 0 up to row 799 for the training set
    # 3/ Grabs everything from row 800 to the very end for the testing set
        
    elif split_method == "date":
        if not split_date:
            raise ValueError("A valid 'split_date' must be specified when using split_method='date'.")
        train_df = df.loc[df.index < split_date].copy()
        test_df = df.loc[df.index >= split_date].copy()
    # Instead of using a row count, this method uses an explicit calendar date
    # as a hard boundary wall to separate the past from the future
    # 1/ Checks if passed a valid date string (like '2023-08-02')
    #    If NO, throw a ValueError crash to prevent errors
    # 2/ Uses label-location slicing (loc) to scan the DataFrame's index (which contains the Date timestamps)
    #    and groups any trading day happening before that date into train_df
    # 3/ Groups any trading day landing on or after that cutoff date into test_df
        
    elif split_method == "random":
        train_df, test_df = train_test_split(df, train_size=split_ratio, shuffle=True, random_state=42)
    # 1/ Calls Scikit-Learn's built-in train_test_split() helper function
    # 2/ shuffle=True tells the function to shuffle the entire timeline randomly
    # 3/ train_size=split_ratio ensures that despite the random shuffling,
    #    train_df still gets 80% of the total available rows and test_df gets the remaining 20%
    # 4/ random_state=42 is a pseudo-random seed. It ensures that the "random" shuffle happens
    #    exactly the same way every time running, making the results reproducible



    # 4 - Multi-Feature Isolated Scaling & Scaler Retention

    # 1/ Instead of scaling the whole table at once, the code loops through the features
    #    (Open, High, Low, Close, Volume) one column at a time.
    #    This is vital because price columns sit around double digits (like $63.00),
    #    while Volume sits in the millions (like 1,416,232).
    #    Slicing them column-by-column allows normalizing them independently
    # 2/ scaler = MinMaxScaler(feature_range=(0, 1)) scales the data
    #    into the range between 0.0 and 1.0
    # 3/ Train:
    #    - fit: The scaler scans the training column to find its
    #           absolute minimum (X_min) and maximum (X_max) values.
    #    - transform: It applies the math formula to every row in that training column,
    #                 turning the raw dollar values into decimals between 0 and 1
    #    - reshape(-1, 1) and flatten(): Scikit-Learn's scaler strictly demands a 2D matrix array as an input,
    #                                    so reshape(-1,1) bumps a flat list into a vertical column matrix.
    #                                    Once the math is done, .flatten() squashes it back into a simple 1D list
    #                                    so it can neatly fit back into the Pandas DataFrame
    # 4/ Test:
    #    Same steps as Train, but by calling only .transform(), the code forces the testing data to be scaled
    #    using the exact same minimum and maximum boundaries learned from the training set
    #

    scalers = {}
    scaled_train = pd.DataFrame(index=train_df.index)
    scaled_test = pd.DataFrame(index=test_df.index)

    for col in features_list:
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Train
        train_matrix = train_df[col].values.reshape(-1, 1)
        scaled_train[col] = scaler.fit_transform(train_matrix).flatten()

        # Test
        test_matrix = test_df[col].values.reshape(-1, 1)
        scaled_test[col] = scaler.transform(test_matrix).flatten()
        
        scalers[col] = scaler

    return scaled_train, scaled_test, scalers, train_df, test_df



# The rest of the code (Not related to Task C.2)

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

x_train = []
y_train = []

train_values = scaled_train[FEATURES].values
target_col_idx = FEATURES.index(PRICE_VALUE)

for x in range(PREDICTION_DAYS, len(train_values)):

    x_train.append(train_values[x - PREDICTION_DAYS:x, :])
    y_train.append(train_values[x, target_col_idx])

x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
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
plt.ylabel(f"{COMPANY} Currency Valuation")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()



last_window = total_test_matrix[-PREDICTION_DAYS:, :]
real_data = np.array([last_window])

next_day_prediction = model.predict(real_data)
next_day_unscaled = target_scaler.inverse_transform(next_day_prediction)
print(f"\n[SYSTEM EVALUATION] Predicted Next-Day '{PRICE_VALUE}' Valuation: ${next_day_unscaled[0][0]:.2f}")