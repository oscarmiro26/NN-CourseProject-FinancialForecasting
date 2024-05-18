import pandas as pd


def detrend(ts, window=12):
    rolling_mean = ts.rolling(window=window).mean()
    detrended = ts - rolling_mean
    return detrended.dropna()


def create_lagged_features(series, lag=12):
    df = pd.DataFrame(series)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    df = pd.concat(columns, axis=1)
    df.dropna(inplace=True)
    return df
