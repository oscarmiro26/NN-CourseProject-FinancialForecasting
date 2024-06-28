import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# Define the base path as the directory containing this configuration file
base_path = os.path.dirname(os.path.abspath(__file__))

# Define the path to the input_data directory relative to the base path
input_data_path = os.path.join(base_path, '..', 'input_data')

# Define the full path to the data file relative to the input_data directory
df = os.path.join(input_data_path, 'M3C_Monthly_FINANCE.csv')

# Transpose the dataframe to have time series in columns
df = df.T

# Drop the first row which contains series identifiers
df = df.iloc[1:]

# Ensure all values are numeric and handle non-numeric values
df = df.apply(pd.to_numeric, errors='coerce')

# Extract features from each time series
features = []
for column in df.columns:
    series = df[column].dropna()
    if len(series) > 0:
        result = seasonal_decompose(series, model='additive', period=12)
        trend = result.trend.dropna()
        features.append([
            series.mean(), series.std(),
            trend.mean(), trend.std()
        ])

# Convert features to a DataFrame
features_df = pd.DataFrame(features, columns=['mean', 'std', 'trend_mean', 'trend_std'])

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features_df)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(normalized_features)

# Add the cluster labels to the features DataFrame for reference
features_df['cluster'] = kmeans.labels_

# Convert df.index to strings for plotting
df.index = df.index.map(str)

# Plot the clusters
for i in range(3):
    plt.figure(figsize=(10, 6))
    for idx, column in enumerate(df.columns):
        if kmeans.labels_[idx] == i:
            plt.plot(df.index, df[column], label=column)
    plt.title(f'Cluster {i}')
    plt.legend()
    plt.show()