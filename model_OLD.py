import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

# File path
file_path = 'M3C.xls'

# Load the required rows (N2522 - 2666, monthly data)
df = pd.read_excel(file_path, sheet_name='M3Month', skiprows=range(1, 1121),
                   nrows=145, header=0)

# Display the first 5 rows of the frame
print(df.head())

# Selecting the very first series for analysis - columns starting from the 7th column (index 6) up to, but not including, the last column (indicated by -1)
series_data = df.iloc[0, 6:-1].dropna()  # Dropping NaN values with dropna()

# Creating a date range for your series
dates = pd.date_range(
    start=f"{df.iloc[0]['Starting Year']}-{df.iloc[0]['Starting Month']}",
    periods=len(series_data), freq='M')
series = pd.Series(data=series_data.values, index=dates)

# Plotting the original series
series.plot(title='Original Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# decomposing
decomposition = seasonal_decompose(series, model='additive')  # useful to try 'multiplicative' if it fits better
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print(decomposition)

# plotting components
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
trend.plot(ax=axes[0], title='Trend')
seasonal.plot(ax=axes[1], title='Seasonality')
residual.plot(ax=axes[2], title='Residuals')
plt.tight_layout()

# the detrending and deseasonalization of the data for improving training stability
series_detrended = series - trend # we substract the trend from the series itself
series_deseasonalized = series_detrended - seasonal # we then remove the seasonal component from the detrended series

# check for stationarity of the deseasonalized data - it is about having constant mean and variance basically in the data
result = adfuller(series_deseasonalized.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
if result[1] < 0.05:
    print("The time series is stationary. We are good.")
else:
    print("The time series is not stationary.")

# display deseasonalized data
series_deseasonalized.plot(title='Detrended and Deseasonalized Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Set up device for PyTorch (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to create sequences for training
def create_sequences(input_data, n_steps):
    X, y = [], []
    for i in range(len(input_data) - n_steps):
        X.append(input_data[i:i + n_steps])
        y.append(input_data[i + n_steps])
    return np.array(X), np.array(y)


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series_deseasonalized.values.reshape(-1, 1))

exit()

# Create sequences
n_steps = 12  # Number of time steps for the GRU input
X, y = create_sequences(series_scaled.flatten(), n_steps)

# Split data into train and test sets (last 18 points for test set)
X_train, y_train = X[:-18], y[:-18]
X_test, y_test = X[-18:], y[-18:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train).float().to(device).reshape(-1, n_steps,
                                                                  1)
y_train_tensor = torch.tensor(y_train).float().to(device).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test).float().to(device).reshape(-1, n_steps, 1)
y_test_tensor = torch.tensor(y_test).float().to(device).reshape(-1, 1)

# Create DataLoader for batch processing
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                          batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                         batch_size=16, shuffle=False)


# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out.view(-1)


# Custom SMAPE loss function
def smape_loss(output, target):
    denominator = (torch.abs(target) + torch.abs(output)) / 2.0
    diff = torch.abs(output - target) / denominator
    diff[denominator == 0] = 0  # Avoid division by zero
    return 100 * torch.mean(diff)


# SMAPE evaluation function
def smape(actuals, predictions):
    denominator = (np.abs(actuals) + np.abs(predictions)) / 2.0
    diff = np.abs(predictions - actuals) / denominator
    diff[denominator == 0] = 0  # Avoid division by zero
    return 100 * np.mean(diff)


# Hyperparameter tuning
learning_rates = [0.001, 0.01]
hidden_dims = [32, 64]
num_layers_list = [1, 2]
batch_sizes = [16, 32]

best_smape = float('inf')
best_params = {}
best_model_architecture = {}

for lr in learning_rates:
    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            for batch_size in batch_sizes:
                print(
                    f"Testing with lr={lr}, hidden_dim={hidden_dim}, num_layers={num_layers}, batch_size={batch_size}")

                # Model initialization
                model = GRUModel(input_dim=1, hidden_dim=hidden_dim,
                                 output_dim=1, num_layers=num_layers).to(
                    device)
                optimizer = Adam(model.parameters(), lr=lr)
                loss_fn = smape_loss

                # Update DataLoader with new batch size
                train_loader = DataLoader(
                    TensorDataset(X_train_tensor, y_train_tensor),
                    batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(
                    TensorDataset(X_test_tensor, y_test_tensor),
                    batch_size=batch_size, shuffle=False)


                # Train the model
                def train_model(model, train_loader, optimizer, loss_fn,
                                epochs=20):
                    model.train()
                    for epoch in range(epochs):
                        epoch_loss = 0
                        for inputs, targets in train_loader:
                            inputs, targets = inputs.to(device), targets.to(
                                device).view(-1)
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = loss_fn(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                        print(
                            f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}')


                train_model(model, train_loader, optimizer, loss_fn, epochs=500)


                # Evaluate the model
                def evaluate_model(model, test_loader):
                    model.eval()
                    actuals, predictions = [], []
                    with torch.no_grad():
                        for inputs, targets in test_loader:
                            inputs = inputs.to(device)
                            outputs = model(inputs)
                            actuals.extend(targets.cpu().numpy())
                            predictions.extend(outputs.cpu().numpy())
                    return actuals, predictions


                actuals, predictions = evaluate_model(model, test_loader)
                actuals_rescaled = scaler.inverse_transform(
                    np.array(actuals).reshape(-1, 1))
                predictions_rescaled = scaler.inverse_transform(
                    np.array(predictions).reshape(-1, 1))

                # Calculate SMAPE
                smape_value = smape(actuals_rescaled, predictions_rescaled)
                print(f"SMAPE: {smape_value:.2f}%")

                # Save best model and parameters
                if smape_value < best_smape:
                    best_smape = smape_value
                    best_params = {
                        'learning_rate': lr,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'batch_size': batch_size
                    }
                    best_model_architecture = {
                        'input_dim': 1,
                        'hidden_dim': hidden_dim,
                        'output_dim': 1,
                        'num_layers': num_layers
                    }
                    torch.save(model.state_dict(), 'best_model.pt')

print(f"Best SMAPE: {best_smape:.2f}% with parameters: {best_params}")

# Load the best model for final evaluation with the correct architecture
best_model = GRUModel(**best_model_architecture).to(device)
best_model.load_state_dict(torch.load('best_model.pt'))

# Evaluate the best model
actuals, predictions = evaluate_model(best_model, test_loader)
actuals_rescaled = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
predictions_rescaled = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(actuals_rescaled, label='Real')
plt.plot(predictions_rescaled, label='Predicted')
plt.title('GRU Predictions vs Actual')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
