# COS30018 - Intelligent Systems - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

# Task C.4:
# Implement Construction of Model (NOT training yet):
# 1 - Initialize Model container and calculate layer depth
# 2 - Check the order and pick the Machine cell
# 3 - Stacke the Layers and Dropouts
# 4 - Add the final top layer and compile

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN

# layer_type: A string text (LSTM / GRU / RNN)
# layer_sizes: A list of numbers like [50, 50, 50]
# input_shape: The size of the incoming data grid (60 days x 5 features)
# dropout_rate: Default setting
# learning_rate: Default setting
def construct_model(layer_type, layer_sizes, input_shape, dropout_rate=0.2, learning_rate=0.001):

    # Initialize an empty Keras model container
    # Sequential: The network layers will sit in a straight, linear stack
    # (data goes in the bottom, moves up layer by layer and spits a guess out the top)
    model = Sequential()

    # Counts the items in the size list
    # For example, [50, 50, 50] will be num_layers = 3
    num_layers = len(layer_sizes)
    
    layer_mapping = {
        'LSTM': LSTM,
        'GRU': GRU,
        'RNN': SimpleRNN
    }    
    cell_object = layer_mapping[layer_type] # For example: select LSTM then cell-object = LSTM
    
    # idx: The layer number index (Layer 0, then Layer 1, then Layer 2)
    # units: Number of neural cells belong in that specific layer (like 50)
    for idx, units in enumerate(layer_sizes):

        # Used to store customization settings for the current layer being built
        # In C.4, it just notes how many hidden units to use
        layer_kwargs = {'units': units}
        
        # This handles the entry gate rule
        # Only the absolute first layer needs to know the shape of the raw data table
        # Next layers do not get this setting because they automatically adjust
        # to whatever the layer below them outputs
        if idx == 0:
            layer_kwargs['input_shape'] = input_shape
            
        # When stacking multiple neural layers, they have to communicate differently
        # depending on where they sit in the stack
        # Intermediate Layers (return_sequences = True):
        #   If there are more recurrent layers waiting above the current one,
        #   this layer cannot just pass a single summary guess, it must hand over
        #   its full 3D timeline history so that the next layer up can analyze it
        # The Final Layer (return_sequences = False):
        #   When the loop reaches the absolute top recurrent layer, it does not have
        #   another recurrent layer following it, so it drops the 3D timeline history
        #   and compresses all its findings into a 2D vector
        if idx < num_layers - 1:
            layer_kwargs['return_sequences'] = True
        else:
            layer_kwargs['return_sequences'] = False
            
        # Burst open the layer_kwargs and feed its contents straight into the layer
        model.add(cell_object(**layer_kwargs))
        
        # Set the dropout_rate for the layer
        # As mentioned in C.1 report, if dropout_rate = 0.2, the system will randomly
        # shut off 20% of the layer's neural brains on every single step.
        # This stops the model from memorizing the exact training charts and forces
        # it to learn flexible, general market trends instead
        model.add(Dropout(dropout_rate))
        
    # After having done compliling the layers, it puts a final node right onto the top
    # This node reads the flat pattern summary vector sent by the last recurrent layer
    # and calculates a value: The next day's final predicted stock price
    model.add(Dense(units=1))
    
    # Apply Adam optimizer to the model before training
    # More detail about the Adam optimizer will be included in the C.4 Report
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model