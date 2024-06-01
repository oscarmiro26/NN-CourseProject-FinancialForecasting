# flake8: noqa
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

def exponential_moving_average(prices, period, weighting_factor=0.5):
    ema = np.zeros(len(prices))
    sma = np.mean(prices[:period])
    ema[period - 1] = sma
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    return ema

def preprocess_data_ema(series, period, weighting_factor):
    series = series.dropna().astype(float).values

    # Compute the exponentially smoothed moving average (EMA) as the trend
    ema = exponential_moving_average(series, period, weighting_factor)

    # Subtract the trend from the original time-series
    detrended_series = series - ema
    # Remove the initial NaN values from EMA calculation
    detrended_series = detrended_series[period-1:]


    # Seasonal decomposition
    result = seasonal_decompose(detrended_series, model='additive', period=period)
    seasonal = result.seasonal
    residual = detrended_series - seasonal

    return series, ema, detrended_series, seasonal, residual



def preprocess_data(series, window):
    series = series.dropna().astype(float)

    # Compute the moving average (trend)
    trend = series.rolling(window=window).mean() # try other methods - ash

    # Subtract the trend from the original time-series
    detrended_series = series - trend
    detrended_series.dropna(inplace=True)

    # Seasonal decomposition
    result = seasonal_decompose(detrended_series, model='additive', period=window) # try other methods
    seasonal = result.seasonal
    residual = detrended_series - seasonal

    return series, trend, detrended_series, seasonal, residual


def prepare_data(data, window=12):
    original_series_list = []
    trend_list = []
    detrended_series_list = []
    seasonal_list = []
    residual_list = []

    # For each row
    for i in range(data.shape[0]):
        # Extract the time-series from the row
        series = data.iloc[i, 6:]

        # Preprocess the time-series
        preprocessed_data = preprocess_data(series, window)
        original_series, trend, detrended_series, seasonal, residual = preprocessed_data

        original_series_list.append(original_series)
        trend_list.append(trend)
        detrended_series_list.append(detrended_series)
        seasonal_list.append(seasonal)
        residual_list.append(residual)

    return original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list


def plot_lists(original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list):
    # Plotting the first time series as an example
    plt.figure(figsize=(12, 6))
    plt.plot(original_series_list[0], label='Original Time Series')
    plt.plot(trend_list[0], label='Trend (Moving Average)', color='red')
    plt.title('Time Series with Trend')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(detrended_series_list[0], label='Detrended Time Series', color='green')
    plt.title('Detrended Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(seasonal_list[0], label='Seasonal Component', color='purple')
    plt.title('Seasonal Component')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(residual_list[0], label='Deseasonalized (Residual) Time Series', color='orange')
    plt.title('Deseasonalized (Residual) Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def main():
    # Loading the data
    data_file = 'M3C_Monthly.csv'
    data = pd.read_csv(data_file)

    window = 6

    preprocessed_data = prepare_data(data, window)
    plot_lists(*preprocessed_data)


if __name__ == '__main__':
    main()