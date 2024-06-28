import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from preprocessing.data_preprocessing import prepare_data, normalize_data, split_data

def cluster_data(data_file, window, n_clusters=3, val_split=0.2, eval_size=12):
    # Define the base path as the directory containing this configuration file
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the input_data directory relative to the base path
    input_data_path = os.path.join(base_path, '..', 'input_data')

    # Define the full path to the data file relative to the input_data directory
    data_file_path = os.path.join(input_data_path, data_file)

    # Read the data from the CSV file
    data = pd.read_csv(data_file_path)

    # Preprocess the data to get residuals
    original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = prepare_data(data, window)

    # Split the data
    train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test = split_data(residual_list, val_split, eval_size)

    # Normalize the data
    scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_residuals_list, scalers = normalize_data(
        train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_test
    )

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(scaled_train_data)

    # Plotting the clusters
    for i in range(n_clusters):
        plt.figure(figsize=(10, 6))
        for idx, series in enumerate(scaled_all_residuals_list):
            if kmeans.labels_[idx] == i:
                plt.plot(series, label=f'Series {idx}')
        plt.title(f'Cluster {i}')
        plt.legend()
        plt.show()

def main():
    # Parameters
    data_file = 'M3C_Monthly_FINANCE.csv'
    window = 12  # Adjust as needed
    n_clusters = 3
    val_split = 0.2
    eval_size = 12

    # Run the clustering
    cluster_data(data_file, window, n_clusters, val_split, eval_size)

if __name__ == '__main__':
    main()
