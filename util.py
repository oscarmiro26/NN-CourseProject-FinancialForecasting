import time
import numpy as np
import torch
import matplotlib.pyplot as plt


def detrend(ts, window=12):
    rolling_mean = ts.rolling(window=window).mean()
    detrended = ts - rolling_mean
    return detrended.dropna()


# Create the dataset with a look-back period
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def smape_loss(y_true, y_pred):
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    diff = torch.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Avoid division by zero
    return torch.mean(diff) * 100

def verify_preprocessing(time_series, trend_list, seasonal_list, residual_list):
    for i in range(len(time_series)):
        original_series = time_series[i]
        trend = trend_list[i]
        seasonal = seasonal_list[i]
        residual = residual_list[i]

        # Reconstruct the series
        reconstructed_series = trend + seasonal + residual

        # Align the original series with the reconstructed series
        aligned_original_series = original_series[original_series.index.isin(reconstructed_series.index)]

        # Plot the original vs reconstructed series
        plt.figure(figsize=(12, 6))
        plt.plot(aligned_original_series.index, aligned_original_series.values, label='Original Series')
        plt.plot(reconstructed_series.index, reconstructed_series.values, label='Reconstructed Series', linestyle='--')
        plt.title(f'Time Series {i+1}: Original vs Reconstructed')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

def reconstruct_series(trend_list, seasonal_list, predicted_residuals, length):

    reconstructed_series = []
    
    for trend, seasonal, residuals in zip(trend_list, seasonal_list, predicted_residuals):
        # Determine the last known trend value and extend it
        last_trend = trend.iloc[-1]
        extended_trend = [last_trend] * length
        
        # Repeat the seasonal pattern from the end of the known data
        seasonal_cycle = seasonal.iloc[-len(seasonal):]  # Get one full cycle
        repeated_seasonal = (seasonal_cycle.tolist() * (length // len(seasonal_cycle) + 1))[:length]
        
        # Add the predicted residuals to the extended trend and repeated seasonality
        future_series = [t + s + r for t, s, r in zip(extended_trend, repeated_seasonal, residuals)]
        
        # Append the reconstructed future series
        reconstructed_series.append(future_series)
    
    return reconstructed_series


def plot_actual_vs_predicted(original_series_list, reconstructed_new_data, length=18):
    num_series = len(original_series_list)
    for i in range(num_series):
        # Get the last 'length' actual points
        actual_data = original_series_list[i].iloc[-length:]
        predicted_data = reconstructed_new_data[i]

        # Ensure predicted data is of the right length
        if len(predicted_data) != length:
            raise ValueError(f"Length of predicted data ({len(predicted_data)}) does not match the required length ({length}).")

        # Create time index for the predicted data
        predicted_index = actual_data.index

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(predicted_index, actual_data, label='Actual Data')
        plt.plot(predicted_index, predicted_data, label='Predicted Data', linestyle='--')
        plt.title(f'Time Series {i+1}: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

def plot_prediction_errors(original_series_list, reconstructed_new_data, length=18):
    """
    Plot prediction error values for all sequences on one plot.

    Args:
        original_series_list (list): List of original series.
        reconstructed_new_data (list): List of predicted series.
        length (int): Number of points to consider for error calculation.

    Returns:
        None
    """
    errors = []

    # Calculate prediction errors for each series
    for i in range(len(original_series_list)):
        # Get the last 'length' actual points
        actual_data = original_series_list[i].iloc[-length:]
        predicted_data = reconstructed_new_data[i]

        # Ensure predicted data is of the right length
        if len(predicted_data) != length:
            raise ValueError(f"Length of predicted data ({len(predicted_data)}) does not match the required length ({length}).")

        # Calculate the error (Mean Squared Error)
        error = np.mean((actual_data.values - predicted_data) ** 2)
        errors.append(error)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label='Prediction Errors', marker='o')
    plt.title('Prediction Errors for All Sequences')
    plt.xlabel('Sequence Index')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

