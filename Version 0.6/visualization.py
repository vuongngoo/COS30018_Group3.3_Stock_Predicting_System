# COS30018 - Intelligent System - Swinburne University
# Student Name: Ngo Sy Vuong
# Student ID: 105551480
# Tutor: Nguyen Manh Toan
# Tutorial Session: Wednesday 8:00 - 12:00
# Semester: May - July 2026

# Task C.3:
# Implementation of visualization for data preprocessing
# 1 - Candlestick Chart
# 2 - Boxplot Chart

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns

def display_candlestick_chart(dataframe, ticker="Asset", n=1):

    # Using copy() to generate a deep copy of the dataframe in an isolated memory location.
    # Reason: If using the original dataframe, the downstream calculation mutate the row indices,
    #         which would cause strutural array error when the data reaches the LSTM model.
    plot_df = dataframe.copy()

    # Define strict financial aggregation rules to preserve OHLC properties
    aggregation_rules = {
        'Open': 'first',    # First opening price in the n-day window
        'High': 'max',      # Absolute highest peak across all n days
        'Low': 'min',       # Absolute lowest bottom across all n days
        'Close': 'last',    # Final closing price of the n-day window
        'Volume': 'sum'     # Total accumulated trading volume over the period
    }

    # Using resample(f'{n}B') to group the rows by specific time frequency.
    # 'B' stands for Business days
    # Using 'B' instead of 'D' (Calendar days) to ignore weekends and market holidays,
    # eliminating empty rows
    plot_df = plot_df.resample(f'{n}B').agg(aggregation_rules).dropna()
        
    mpf.plot(
        plot_df, 
        type='candle', # Geometric candlestick bodies and wicks
        style='charles', # standard green and red financial color theme
        title=f"\n{ticker} - {n}-Day Aggregated Candlestick Chart",
        ylabel='Stock Price ($)', 
        ylabel_lower='Shares Traded',
        volume=True, # The lower panel displaying volume bars 
    )



def display_moving_boxplot(dataframe, ticker="Asset", target_col="Close", window_size=20):

    # Extract the raw data of the 'close' column
    raw_prices = dataframe[target_col].values

    # The absolute integer size of the dataset array
    total_days = len(raw_prices)
        
    window_data = []
    window_labels = []
    
    # Instead of sliding foward by 1 single day (which generates too many overlapping plots),
    # it moves 10 days at a time to keep the chart well distributed
    stride = max(1, window_size // 2) 
    
    # The loop start at index 0
    # Steps forward by the stride value
    # total_days - window_size + 1 ensures that the loop stops before running out
    #                              of rows at the end of the array
    for start_idx in range(0, total_days - window_size + 1, stride):
        # The integer marker for the end of the current sliding window
        end_idx = start_idx + window_size
        
        # [start_idx:end_idx] is the Python array slicing syntax
        # Python slicing is exclusive of the upper bound
        # For example, if it is [0;20], then it will take from 0 to 19 for chunk 1
        chunk = raw_prices[start_idx:end_idx]
        # By the time the loop finishes, window_data becomes a list of arrays
        window_data.append(chunk)
        
        # Jump out of the raw NumPy numbers array and look back at the original Pandas dataframe.index
        # at the position matching start_idx
        # '%Y-%m-%d' stands for String Format Time, this strips away the hours, minutes and seconds
        # For example, 2026-06-06 12:00:00 will become 2026-06-06
        start_date = dataframe.index[start_idx].strftime('%Y-%m-%d')

        # Work exactly the same as start_data, but [end_idx - 1] is because index 19 is the 20th day  
        end_date = dataframe.index[end_idx - 1].strftime('%Y-%m-%d')
        window_labels.append(f"{start_date} to {end_date}")
    
    plt.figure(figsize=(14, 7)) # width = 14 inches, height = 7 inches

    # data=window_data: This passes the 2D matrix list.
    # Seaborn will loop through each chunk to calculate the median, upper/lower quartiles,
    # whiskers and outlier dots for each window
    sns.boxplot(data=window_data, color="#2ecc71")
    
    plt.title(f"{ticker} Price Distribution Spread Over Rolling {window_size}-Day Windows", fontsize=14, fontweight='bold')
    plt.xlabel("Sliding Temporal Windows (n Consecutive Trading Days)", fontsize=12)
    plt.ylabel(f"Nominal '{target_col}' Price Valuation ($)", fontsize=12)
    
    # Formats the text labels along the horizontal x-axis
    # ticks=range(): Maps integer locations across the horizontal array plane
    # labels=window_labels: Apply the (f"{start_date} to {end_date}") to those positions
    plt.xticks(ticks=range(len(window_labels)), labels=window_labels, rotation=45, ha='right', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()