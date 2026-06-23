# COS30018 - Intelligent Systems - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

# Task C.5:
# Implement singlestep-multivariate, multistep-singlevariate and multistep-multivariate predictions:
# 1 - Predict Multistep (k steps) with Univariate (Close)
# 2 - Predict Singlestep (1 step) with Multivariate (Open, Close, High, Low, Volume)
# 3 - Predict Multistep (k steps) with Multivariate (Open, Close, High, Low, Volume)

import numpy as np

# model: LSTM / GRU / RNN
# last_window: The final 60-day window of scaled data from the test set
# k_days: The number of days into the future
# target_scaler: The MinMaxScaler object used for the Close price
# target_idx: The column index position of the Close price
def predict_multistep_close(model, last_window, k_days, target_scaler, target_idx):

    # First, create an independent copy of the starting 60-day historical window to avoid modifying the original
    # Then, check if the array is 2-dimensional (because TensorFlow's LSTM layers require a 3-dimensional input)
    # If 2-dimensional, transfer into 3-dimensinal (for example, from (60,5) into (1,60,5))
    current_window = np.copy(last_window)
    if len(current_window.shape) == 2:
        current_window = np.expand_dims(current_window, axis=0)
    
    # An empty list to store the outputs
    future_predictions = []

    for i in range(k_days):

        # Passes the current 60-day matrix through the LSTM model to generate a prediction for the next day
        # Then extract the raw numerical scalar prediction value from the output array and saves it
        # Checking index [0,0] unwraps the raw float value (strips away the outer matrix brackets [] )
        next_pred = model.predict(current_window)
        future_predictions.append(next_pred[0, 0])
        
        # Extracts a copy of the very last row (the most recent day) from the 60-day matrix sequence
        # (Index 0 enters the batch dimension, -1 grabs the most recent time-step, and : grabs all features)
        # Then overwrite the closing price in the template row with the most recent prediction
        #   Since this is a close price prediction module, we preserve our non-target values (like volume or high/low metrics) 
        #   but overwrite the target cell with the most recent prediction
        next_timestep = np.copy(current_window[0, -1, :])
        next_timestep[target_idx] = next_pred[0, 0]
        
        # Shift the 60-day window forward by one day
        # current_window[:, 1:, :]: Drop the oldest day in the matrix (index 0), reducing the 
        #                           lookback window temporarily from 60 days to 59 days
        # np.append(..., [[next_timestep]], axis=1): Attach the newly synthesized day onto the end of the timeline 
        #                                            along the time-axis (axis=1). This brings the window back to exactly 60 days
        current_window = np.append(current_window[:, 1:, :], [[next_timestep]], axis=1)
    
    # Converts the gathered list of predictions back into a structured NumPy array and reshapes it into a single column
    # (Scikit-Learn scalers require data inputs to be shaped (N, 1) to perform mathematical inverse operations)
    # Then reverses the scaling normalization to convert the data back into raw currency values,
    # collapses the array back down into a clean 1D list, and returns it
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return target_scaler.inverse_transform(future_predictions).flatten()



# model: LSTM / GRU / RNN
# input_matrix: A 60-day window containing all 5 features
#               (['Open', 'High', 'Low', 'Close', 'Volume'])
# target_scaler: The MinMaxScaler object used for the Close price
def predict_multivariate_single_step(model, input_matrix, target_scaler):

    # Same as predict_multistep_close, convert 2D to 3D
    if len(input_matrix.shape) == 2:
        input_matrix = np.expand_dims(input_matrix, axis=0)
        
    # Perform prediction (with all 5 variables)
    # Then reverse the normalization scaling, extracts the raw float number, and returns it
    prediction = model.predict(input_matrix)
    return target_scaler.inverse_transform(prediction)[0][0]



# model: LSTM / GRU / RNN
# last_window: The final 60-day window of scaled data from the test set
# k_days: The number of days into the future
# target_scaler: The MinMaxScaler object used for the Close price
# target_idx: The column index position of the Close price
def predict_multivariate_multistep(model, last_window, k_days, target_scaler, target_idx):

    # Create an independent copy
    current_window = np.copy(last_window)

    # Same as predict_multistep_close and predict_multivariate_single_step, convert 2D to 3D
    if len(current_window.shape) == 2:
        current_window = np.expand_dims(current_window, axis=0)
    
    # An empty list to store the outputs
    predictions = []
    
    # Same as predict_multistep_close
    for i in range(k_days):
        pred_scaled = model.predict(current_window)
        predictions.append(pred_scaled[0, 0])
        
        next_row = np.copy(current_window[0, -1, :])
        next_row[target_idx] = pred_scaled[0, 0] 

        current_window = np.concatenate((current_window[:, 1:, :], [[next_row]]), axis=1)

    # Convert into shape (N,1) then reverses the scaling normalization to convert the data back into raw currency values
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()