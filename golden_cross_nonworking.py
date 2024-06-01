import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The golden cross is quite extra, my idea is to see if the crosses exist in
# the predicted ones, and make a bump for the future from there
def calculate_ema(df, column_name, span):
    return df[column_name].ewm(span=span, adjust=False).mean()

def find_golden_cross(df):
    golden_crosses = (df['ema50'] > df['ema200']) & (df['ema50'].shift(1) <= df['ema200'].shift(1))
    return df.index[golden_crosses]

def main():
    # Load your data
    data_file = 'M3C_Monthly_FINANCE.csv'
    df = pd.read_csv(data_file)

    # Assuming 'Close' is the column with closing prices, update if necessary
    df['ema50'] = calculate_ema(df,'1',50)
    df['ema200'] = calculate_ema(df, '1', 200)

    # Find Golden Cross points
    golden_cross_points = find_golden_cross(df)

    # Plotting
    plt.figure(figsize=(16, 8))
    plt.plot(df[''], label='1')
    plt.plot(df['ema50'], label='50-day EMA', color='green')
    plt.plot(df['ema200'], label='200-day EMA', color='red')

    # Plot Golden Cross points
    plt.scatter(golden_cross_points, df.loc[golden_cross_points, 'Close'], marker='^', color='gold', label='Golden Cross', s=100)

    plt.title('Golden Cross Indicator')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()