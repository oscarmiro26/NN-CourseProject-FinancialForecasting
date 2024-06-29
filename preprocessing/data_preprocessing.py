
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import torch
from preprocessing.config_preprocessing import *
from util.util import verify_preprocessing, create_dataset, plot_all_lists_after_preprocessing
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data(series, span):
    series = series.dropna().astype(float)

    # Compute the exponentially weighted moving average (EWMA) as the trend
    trend = series.ewm(span=span, adjust=False).mean()

    # Subtract the trend from the original time-series
    detrended_series = series - trend

    # Seasonal decomposition
    result = seasonal_decompose(detrended_series, model='additive', period=span)
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

def start_preprocess_data(data, window):
    print('Preprocessing data...')
    original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = prepare_data(data, window)

    if VERIFY_PREPROCESSING:
        print('  Verifying preprocessing...')
        verify_preprocessing(original_series_list, trend_list, seasonal_list, residual_list)
        exit()

    return original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list




def split_data(residual_list, val_split, eval_size, split_save_path='input_data/data_splits.pkl'):
    if os.path.exists(split_save_path):
        print('Loading existing data splits...')
        with open(split_save_path, 'rb') as f:
            train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test = pickle.load(f)
    else:
        print('Creating new data splits...')
        # Hold out the last `eval_size` values from each series in residual_list for final evaluation
        test_residuals_list = [res[-eval_size:] for res in residual_list]
        all_residuals_except_test = [res[:-eval_size] for res in residual_list]
        val_size = int(len(all_residuals_except_test[0]) * val_split)
        train_residuals_list = [res[:-val_size] for res in all_residuals_except_test]
        val_residuals_list = [res[-val_size:] for res in all_residuals_except_test]

        # Save the splits to a file
        with open(split_save_path, 'wb') as f:
            pickle.dump((train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test), f)
    
    return train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test


# Modify the normalize_data functions accordingly
def normalize_data(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test):
    print('Normalizing data...')
    scaled_train_residuals_list = []
    scaled_val_residuals_list = []
    scaled_test_residuals_list = []
    scaled_all_residuals_list = [] 
    train_scalers = []
    val_scalers = []
    test_scalers = []

    for i, (train_res, val_res, test_res, all_res) in enumerate(zip(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test)):
        train_res_np = train_res.values.reshape(-1, 1)
        val_res_np = val_res.values.reshape(-1, 1)
        test_res_np = test_res.values.reshape(-1, 1)
        all_res_np = all_res.values.reshape(-1, 1)

        # Fit the scaler on the training data and transform it
        train_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_train_res = train_scaler.fit_transform(train_res_np)
        scaled_train_residuals_list.append(scaled_train_res)
        train_scalers.append(train_scaler)
        
        # Fit the scaler on the validation data and transform it
        val_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_val_res = val_scaler.fit_transform(val_res_np)
        scaled_val_residuals_list.append(scaled_val_res)
        val_scalers.append(val_scaler)

        # Fit the scaler on the test data and transform it
        test_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_test_res = test_scaler.fit_transform(test_res_np)
        scaled_test_residuals_list.append(scaled_test_res)
        test_scalers.append(test_scaler)

        # Transform all residuals using the scaler fitted on the training data
        scaled_all_res = train_scaler.transform(all_res_np)
        scaled_all_residuals_list.append(scaled_all_res)

    # Concatenate the scaled data
    scaled_train_data = np.concatenate(scaled_train_residuals_list)
    scaled_val_data = np.concatenate(scaled_val_residuals_list)
    scaled_test_data = np.concatenate(scaled_test_residuals_list)

    return scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_residuals_list, train_scalers, val_scalers, test_scalers



def create_datasets(look_back):
    # Load the dataset from the specified CSV file
    print('Loading data...')
    data = pd.read_csv(DATA_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preprocess the data
    original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = start_preprocess_data(data, look_back)
    
    
    # Split the data
    train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test = split_data(residual_list, VALIDATION_SPLIT, PREDICTION_SIZE)
    
    # Debug statements for data splitting
    for i in range(len(train_residuals_list)):
        print(f"Split lengths for sequence {i}: train={len(train_residuals_list[i])}, val={len(val_residuals_list[i])}, test={len(test_residuals_list[i])}")

    # Normalize the data
    scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_residuals_list, train_scalers, val_scalers, test_scalers = normalize_data(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test)


    # Create datasets
    X_train, Y_train = create_dataset(scaled_train_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print("X_train shape:", X_train.shape)

    X_val, Y_val = create_dataset(scaled_val_data, look_back)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    print("X_val shape:", X_val.shape)

    X_test, Y_test = create_dataset(scaled_test_data, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print("X_test shape:", X_test.shape)
    
    # Convert to torch tensors
    X_train = torch.Tensor(X_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    X_test = torch.Tensor(X_test).to(device)
    Y_train = torch.Tensor(Y_train).to(device)
    Y_val = torch.Tensor(Y_val).to(device)
    Y_test = torch.Tensor(Y_test).to(device)

    # Return the created datasets and additional information
    return (
        X_train,       # Training features
        Y_train,       # Training targets
        X_val,         # Validation features
        Y_val,         # Validation targets
        X_test,        # Test features
        Y_test,        # Test targets
        trend_list,    # List of trend components for each series
        seasonal_list, # List of seasonal components for each series
        test_residuals_list, # List of residuals used for testing
        original_series_list, # Original series for all data
        residual_list,  # Full residual list for all series
        train_scalers,  # Scalers for training data
        val_scalers,    # Scalers for validation data
        test_scalers,   # Scalers for test data
        scaled_all_residuals_list # Scaled residuals for all data
    )






def main():
    # Loading the data
    data_file = 'M3C_Monthly.csv'
    data = pd.read_csv(data_file)

    window = 6

    preprocessed_data = prepare_data(data, window)
    plot_all_lists_after_preprocessing(*preprocessed_data)


if __name__ == '__main__':
    main()