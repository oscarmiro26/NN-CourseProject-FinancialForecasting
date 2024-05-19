import numpy as np
import torch


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
