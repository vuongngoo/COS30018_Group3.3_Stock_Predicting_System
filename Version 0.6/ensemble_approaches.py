# COS30018 - Intelligent Systems - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

# Task C.6:
# This file is used to train the models:
# 1. ARIMA used for Approach 1
# 2. Transformer used for Approach 2

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import pickle
import os
import pmdarima as pm

# Import task C.2
from data_preprocessing import load_scale_and_split_data

COMPANY = 'CBA.AX'
START_DATE = '2020-01-01'
END_DATE = '2024-07-02'
PRICE_VALUE = "Close"
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
SPLIT_DATE = '2023-08-02'

os.makedirs('models', exist_ok=True)

# Loads the preprocessed data, returning only the unscaled training dataframe (train_df)
#   and throwing away the rest using Python's discard placeholder (_)
_, _, _, train_df, _ = load_scale_and_split_data(
    ticker=COMPANY,
    start=START_DATE,
    end=END_DATE,
    features_list=FEATURES,
    target_col=PRICE_VALUE,
    handle_nan="ffill",
    split_method="date",
    split_date=SPLIT_DATE
)

# Extracts only the raw, unscaled closing prices of CBA.AX to use as the base fitting series for ARIMA
y_train_raw = train_df[PRICE_VALUE]

# Searche for the optimal mathematical parameters (p, d, q) for the ARIMA mode
try:
    arima_m = pm.auto_arima(
        y_train_raw, 
        seasonal=False, 
        error_action='ignore', 
        suppress_warnings=True,
        stepwise=True
    )
    print(f"Optimal ARIMA order selected: {arima_m.order}")

    # Save the base ARIMA model structure
    with open('models/arima_model.pkl', 'wb') as f:
        pickle.dump(arima_m, f)

except Exception as e:
    print(f"[Error training ARIMA]: {e}")

# Load the dataset used for Transformer training
X_path = 'data_meta/X_meta_train.npy'
Y_path = 'data_meta/Y_meta_train.npy'
X_meta = np.load(X_path)
Y_meta = np.load(Y_path)

# Defines a helper function to compile a custom Multi-Head Attention neural network
# Defines the entry layer. Its shape will be (7, 3) (7 days of future horizon x 3 baseline models)
def build_transformer_meta_learner(input_shape, head_size=64, num_heads=2, ff_dim=64, dropout=0.2):
    inputs = layers.Input(shape=input_shape)
        
    # Computes the self-attention weights by using the relative predictions of the baseline models
    #   as both the Query and the Key/Value tensors (inputs, inputs)
    # It learns how much to weigh each model's forecast relative to the others for each future step
    attention_out = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)

    # Temporarily deactivates random neuron connections (20%) to prevent the model from memorizing (overfitting) the data
    attention_out = layers.Dropout(dropout)(attention_out)

    # Implements a Residual Addition (Skip Connection) by adding the original input features back to the attention output,
    # then normalizes the distribution. This helps avoid gradient decay during training
    attention_res = layers.LayerNormalization(epsilon=1e-6)(attention_out + inputs)
        
    # Feed-Forward Network
    # A classic fully connected neural layer that projects the features into 64 nodes
    #   with a Non-Linear Rectified Linear Unit (relu) activation function
    ff_out = layers.Dense(ff_dim, activation="relu")(attention_res)
    ff_out = layers.Dropout(dropout)(ff_out)

    # Shrinks the dimensionality back down to 3 columns to match the skip connection
    ff_out = layers.Dense(input_shape[-1])(ff_out)

    # Adds the second skip connection (combining the feed-forward output with the attention output) and normalizes again
    transformer_out = layers.LayerNormalization(epsilon=1e-6)(ff_out + attention_res)
        
    # Uses a linear dense projection to map the final features to a single output value
    # (1 predicted relative close price drift) for each of the 7 days
    # This outputs a matrix of shape (7, 1)
    outputs = layers.Dense(1)(transformer_out) # Predicts 1 target value (Close) for each step
    
    # Packages the defined layers into an executable Keras Model class and returns it
    model = tf.keras.Model(inputs, outputs)
    return model

# Instantiate and Compile the Meta-Learner
meta_input_shape = (X_meta.shape[1], X_meta.shape[2]) # (K_DAYS_FUTURE, 3)
transformer_meta = build_transformer_meta_learner(meta_input_shape)
    
transformer_meta.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
    
print("\nTransformer Meta-Learner Architecture Summary:")
transformer_meta.summary()
    
# Train the Meta-Learner
transformer_meta.fit(
    X_meta, Y_meta,
    validation_split=0.1,
    epochs=30,
    batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)
    
# Save the Meta-Learner Model
transformer_meta.save("models/transformer_meta_model.keras")
print("\nTransformer Meta-Learner trained and saved")