
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import torch
from preprocessing.config_preprocessing import *
from util.util import verify_preprocessing, create_dataset, plot_all_lists_after_preprocessing
from sklearn.preprocessing import StandardScaler

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




def split_data(residual_list, val_split, eval_size):
    print('  Creating data splits...')
    
    # Hold out the last `eval_size` values from each series in residual_list for final evaluation
    # These values will be completely unseen during training and validation, providing an unbiased final test list.
    test_residuals_list = [res[-eval_size:] for res in residual_list]
    
    # The remaining data after holding out the evaluation set (we remove the last `eval_size` points)
    # This data will be used for creating the training and validation sets.
    all_residuals_except_test = [res[:-eval_size] for res in residual_list]

    # Calculate the size of the validation set based on the percentage provided (val_split)
    val_size = int(len(all_residuals_except_test[0]) * val_split)

    # Split the remaining data into training set
    # The training set is used to train the model.
    train_residuals_list = [res[:-val_size] for res in all_residuals_except_test]

    # Split the remaining data into validation set
    # The validation set is used to tune hyperparameters and to avoid overfitting.
    val_residuals_list = [res[-val_size:] for res in all_residuals_except_test]
    
    return train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test

# Modify the normalize_data functions accordingly
def normalize_data(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test):
    print('  Normalizing data...')
    scaled_train_residuals_list = []
    scaled_val_residuals_list = []
    scaled_test_residuals_list = []
    scaled_all_residuals_list = [] 
    scalers = []

    for train_res, val_res, test_res, all_res in zip(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test):
        train_res_np = train_res.values.reshape(-1, 1)
        val_res_np = val_res.values.reshape(-1, 1)
        test_res_np = test_res.values.reshape(-1, 1)
        all_res_np = all_res.values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaled_train_res = scaler.fit_transform(train_res_np)
        scaled_val_res = scaler.transform(val_res_np)
        scaled_test_res = scaler.transform(test_res_np)
        scaled_all_res = scaler.transform(all_res_np)

        scaled_train_residuals_list.append(scaled_train_res)
        scaled_val_residuals_list.append(scaled_val_res)
        scaled_test_residuals_list.append(scaled_test_res)
        scaled_all_residuals_list.append(scaled_all_res)

        scalers.append(scaler)

    scaled_train_data = np.concatenate(scaled_train_residuals_list)
    scaled_val_data = np.concatenate(scaled_val_residuals_list)
    scaled_test_data = np.concatenate(scaled_test_residuals_list)

    return scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_residuals_list, scalers



#normalization if we want to give more attention to outliers and peaks. scaling data to have a mean of 0 and a standard deviation of 1.
def normalize_data_outliers(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_eval):
    print('  Normalizing data...')
    scaled_train_residuals_list = []
    scaled_val_residuals_list = []
    scaled_test_residuals_list = []
    scaled_all_residuals_list = [] 
    scalers = []

    for train_res, val_res, test_res, all_res in zip(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_eval):
        train_res_np = train_res.values.reshape(-1, 1)
        val_res_np = val_res.values.reshape(-1, 1)
        test_res_np = test_res.values.reshape(-1, 1)
        all_res_np = all_res.values.reshape(-1, 1)

        scaler = StandardScaler()
        
        scaled_train_res = scaler.fit_transform(train_res_np)
        scaled_val_res = scaler.transform(val_res_np)
        scaled_test_res = scaler.transform(test_res_np)
        scaled_all_res = scaler.transform(all_res_np)

        scaled_train_residuals_list.append(scaled_train_res)
        scaled_val_residuals_list.append(scaled_val_res)
        scaled_test_residuals_list.append(scaled_test_res)
        scaled_all_residuals_list.append(scaled_all_res)

        scalers.append(scaler)

    scaled_train_data = np.concatenate(scaled_train_residuals_list)
    scaled_val_data = np.concatenate(scaled_val_residuals_list)
    scaled_test_data = np.concatenate(scaled_test_residuals_list)

    return scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_residuals_list, scalers


def create_datasets(look_back):
    # Load the dataset from the specified CSV file
    print('Loading data...')
    data = pd.read_csv(DATA_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preprocess the data
    original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = start_preprocess_data(data, look_back)
    
    # Debug statements for preprocessing
    for i, (original, trend, detrended, seasonal, residual) in enumerate(zip(original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list)):
        print(f"Length of original_series_list[{i}]: {len(original)}")
        print(f"Length of trend_list[{i}]: {len(trend)}")
        print(f"Length of detrended_series_list[{i}]: {len(detrended)}")
        print(f"Length of seasonal_list[{i}]: {len(seasonal)}")
        print(f"Length of residual_list[{i}]: {len(residual)}")
        break  # Only print the first series lengths

    # Split the data
    train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test = split_data(residual_list, VALIDATION_SPLIT, PREDICTION_SIZE)
    
    # Debug statements for data splitting
    print("Length of train_residuals_list[0]:", len(train_residuals_list[0]))
    print("Length of val_residuals_list[0]:", len(val_residuals_list[0]))
    print("Length of test_residuals_list[0]:", len(test_residuals_list[0]))

    # Normalize the data - change between normal or the one giving more attention to outliers
    scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_data, scalers = normalize_data(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test)

    # Debug statements for normalization
    print("Length of scaled_train_data:", len(scaled_train_data))
    print("Length of scaled_val_data:", len(scaled_val_data))
    print("Length of scaled_test_data:", len(scaled_test_data))

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
        scalers,       # Scalers used for normalization
        trend_list,    # List of trend components for each series
        seasonal_list, # List of seasonal components for each series
        test_residuals_list, # List of residuals used for testing
        scaled_all_data,     # Scaled data for all series
        original_series_list,# Original series for all data
        residual_list  # Full residual list for all series
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