

"""
Utility Functions

This script contains various utility functions used for time series forecasting.
These functions are designed to assist with data preparation, model evaluation,
and result visualization. The utilities provided in this script include dataset
creation, statistical loss functions, data normalization, and plotting functions.

"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# this file includes 
def create_dataset(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def smape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

def calculate_median_smape(original_series, reconstructed_series, prediction_size):
    smapes = []
    for orig, recon in zip(original_series, reconstructed_series):
        orig_overlap = orig[-prediction_size:]
        recon_overlap = recon[:prediction_size]
        smapes.append(smape(orig_overlap, recon_overlap))
    return np.median(smapes)


def smape_loss(y_true, y_pred):
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    diff = torch.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Avoid division by zero
    return torch.mean(diff) * 100

def smape_tensors(y_true, y_pred):
    y_true = torch.tensor(y_true.values) if isinstance(y_true, pd.Series) else torch.tensor(y_true)
    y_pred = torch.tensor(y_pred.values) if isinstance(y_pred, pd.Series) else torch.tensor(y_pred)
    
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

        # Print the lengths of the first sequence in each list
        if i == 0:
            print(f"Lengths of the first sequence in each list:")
            print(f"Original Series Length: {len(original_series)}")
            print(f"Trend Length: {len(trend)}")
            print(f"Seasonal Length: {len(seasonal)}")
            print(f"Residual Length: {len(residual)}")
        
        # Align all series to the shortest length
        min_length = min(len(original_series), len(trend), len(seasonal), len(residual))
        aligned_original_series = original_series.iloc[-min_length:]
        aligned_trend = trend.iloc[-min_length:]
        aligned_seasonal = seasonal.iloc[-min_length:]
        aligned_residual = residual.iloc[-min_length:]

        # Reconstruct the series
        reconstructed_series = aligned_trend + aligned_seasonal + aligned_residual

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


def denormalize_predictions(predictions, scalers):
    denormalized_predictions = []
    for pred, scaler in zip(predictions, scalers):
        pred_array = np.array(pred).reshape(-1, 1)
        denormalized_pred = scaler.inverse_transform(pred_array)
        denormalized_predictions.append(denormalized_pred.flatten().tolist())
    return denormalized_predictions


def plot_predictions(actual_full_list, predicted_list, naive_predictions, num_points, extra_context_points=30):
    for i, (actual_full, predicted, naive_pred) in enumerate(zip(actual_full_list, predicted_list, naive_predictions)):
        if len(actual_full) < num_points + extra_context_points:
            print(f"Skipping series {i+1} due to insufficient data points.")
            continue

        plt.figure(figsize=(10, 5))

        # Define the range for the actual values and the prediction period
        actual_range = list(range(len(actual_full) - num_points - extra_context_points, len(actual_full)))
        prediction_start = len(actual_full) - num_points - 1
        prediction_range = list(range(prediction_start, prediction_start + num_points + 1))

        # Get the y-value at the prediction start point
        y_value_at_red_line = actual_full.iloc[prediction_start]
        print(f"Series {i+1}: Y value at red line: {y_value_at_red_line}")

        # Print the values being plotted for debugging
        actual_values_for_plot = actual_full.iloc[-num_points - extra_context_points:].values
        print(f"Series {i+1}: Actual values for plotting: {actual_values_for_plot}")
        print(f"Series {i+1}: Naive predictions: {naive_pred}")
        print(f"Series {i+1}: Model predictions: {predicted}")

        # Plot the actual values for the last points including extra context points
        plt.plot(actual_range, actual_values_for_plot, label='Actual', color='blue')

        # Adjust predictions to start from the last known actual residual
        model_pred_values = [y_value_at_red_line] + predicted
        naive_pred_values = [y_value_at_red_line] + naive_pred

        # Plot the model predictions
        plt.plot(prediction_range, model_pred_values, label='Model Predicted', linestyle='--', color='orange')

        # Plot the naive predictions directly
        plt.plot(prediction_range, naive_pred_values, label='Naive Predicted', linestyle='--', color='green')

        # Add a vertical red line to indicate the start of the prediction
        plt.axvline(x=prediction_start, color='red', linestyle='--', label='Prediction Start')

        plt.legend()
        plt.title(f'Actual vs Predicted for Series {i+1} for last {num_points} points')
        plt.xlabel('Data Point Index')
        plt.ylabel('Value')
        plt.xticks(actual_range)  # Ensure x-axis ticks are the actual data point indices
        plt.xlim(min(actual_range), max(actual_range))  # Set x-axis limits to the range of actual data point indices
        plt.show()


def naive_predictor(actual_residuals, prediction_size):
    naive_predictions = []
    for residuals in actual_residuals:
        # Use the value at the prediction start point (19th point from the end)
        if len(residuals) > prediction_size:
            last_value = residuals.iloc[-prediction_size - 1]
        else:
            last_value = residuals.iloc[-1]  # Fallback to the last point if not enough points
        naive_predictions.append([last_value] * prediction_size)
    return naive_predictions




def plot_all_lists_after_preprocessing(original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list):
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

def evaluate_predictions(actual_list, model_predicted_list, naive_predicted_list):
    model_mse_list = []
    model_mae_list = []
    model_r2_list = []
    model_smape_list = []

    naive_mse_list = []
    naive_mae_list = []
    naive_r2_list = []
    naive_smape_list = []

    for actual, model_pred, naive_pred in zip(actual_list, model_predicted_list, naive_predicted_list):
        # Evaluate model predictions
        model_mse = mean_squared_error(actual, model_pred)
        model_mae = mean_absolute_error(actual, model_pred)
        model_r2 = r2_score(actual, model_pred)
        model_smape = smape_tensors(actual, model_pred)
        
        model_mse_list.append(model_mse)
        model_mae_list.append(model_mae)
        model_r2_list.append(model_r2)
        model_smape_list.append(model_smape)

        # Evaluate naive predictions
        naive_mse = mean_squared_error(actual, naive_pred)
        naive_mae = mean_absolute_error(actual, naive_pred)
        naive_r2 = r2_score(actual, naive_pred)
        naive_smape = smape_tensors(actual, naive_pred)
        
        naive_mse_list.append(naive_mse)
        naive_mae_list.append(naive_mae)
        naive_r2_list.append(naive_r2)
        naive_smape_list.append(naive_smape)

    return {
        "model": (model_mse_list, model_mae_list, model_r2_list, model_smape_list),
        "naive": (naive_mse_list, naive_mae_list, naive_r2_list, naive_smape_list)
    }
