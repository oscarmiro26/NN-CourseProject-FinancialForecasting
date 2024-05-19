# train.py

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from models.gru_model import get_model
from data_preprocessing import prepare_data
from util import create_dataset, smape_loss

# Configurations
TRAIN_MODEL = False  # Set to False to skip training and load the saved model

# Load the data
print('Loading data...')
data_file = 'M3C_Monthly.csv'
data = pd.read_csv(data_file)

# Preprocess the data
print('Preprocessing data...')
window = 12
original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = prepare_data(data, window)

# Normalize and combine all residuals
all_residuals = np.concatenate([res.values.reshape(-1, 1) for res in residual_list])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(all_residuals)

look_back = 12
X, Y = create_dataset(scaled_data, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
print('Creating data splits...')
test_size = 18  # Consider making this configurable
X_train, X_test = X[:-test_size], X[-test_size:]
Y_train, Y_test = Y[:-test_size], Y[-test_size:]

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
Y_train = torch.Tensor(Y_train).to(device)
Y_test = torch.Tensor(Y_test).to(device)

# Create DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Get the model
print('Creating model...')
input_size = 1
hidden_size = 50
num_layers = 1
output_size = 1

lr = 0.001

model = get_model(input_size, hidden_size, num_layers, output_size).to(device)

if TRAIN_MODEL:
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    print('Training model...')
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            outputs = model(batch_X)
            optimizer.zero_grad()
            loss = smape_loss(batch_Y, outputs)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], SMAPE: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), 'gru_model.pth')
else:
    # Load the model
    print('Loading saved model...')
    model.load_state_dict(torch.load('gru_model.pth'))
    model.eval()

# Testing the model
print('Evaluating model...')
model.eval()

with torch.no_grad():  # Disable gradient calculation for evaluation
    train_predict = model(X_train)
    test_predict = model(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict.cpu().numpy())
Y_train = scaler.inverse_transform(Y_train.cpu().numpy().reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.cpu().numpy())
Y_test = scaler.inverse_transform(Y_test.cpu().numpy().reshape(-1, 1))

# Debugging: Print lengths of arrays
print(f"Length of train_predict: {len(train_predict)}")
print(f"Length of test_predict: {len(test_predict)}")
print(f"Length of trend_list[0]: {len(trend_list[0])}")
print(f"Length of seasonal_list[0]: {len(seasonal_list[0])}")

# Repeat trend and seasonal components to match prediction lengths
def repeat_to_length(arr, length):
    return np.tile(arr, length // len(arr) + 1)[:length]

trend_train = repeat_to_length(trend_list[0], len(train_predict))
seasonal_train = repeat_to_length(seasonal_list[0], len(train_predict))
trend_test = repeat_to_length(trend_list[0], len(test_predict))
seasonal_test = repeat_to_length(seasonal_list[0], len(test_predict))

# Debugging: Print lengths after adjustment
print(f"Length of trend_train: {len(trend_train)}")
print(f"Length of seasonal_train: {len(seasonal_train)}")
print(f"Length of trend_test: {len(trend_test)}")
print(f"Length of seasonal_test: {len(seasonal_test)}")

# Ensure lengths match before addition
if len(train_predict) == len(trend_train) == len(seasonal_train):
    train_predict_final = train_predict + trend_train + seasonal_train
else:
    print("Length mismatch in train data")

if len(test_predict) == len(trend_test) == len(seasonal_test):
    test_predict_final = test_predict + trend_test + seasonal_test
else:
    print("Length mismatch in test data")

# Calculate SMAPE on test data
if 'test_predict_final' in locals():
    test_smape = smape_loss(torch.Tensor(Y_test).to(device), torch.Tensor(test_predict_final).to(device)).item()
    print(f'Test SMAPE: {test_smape:.4f}')

    # Plotting
    print('Plotting data...')
    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(scaled_data), label='Original Time Series')
    plt.plot(np.arange(look_back, len(train_predict_final) + look_back), train_predict_final, label='Training Predictions')
    plt.plot(np.arange(len(train_predict_final) + (look_back * 2) + 1, len(scaled_data) - 1), test_predict_final, label='Testing Predictions')
    plt.title('GRU Model Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Clear memory
del X_train, X_test, Y_train, Y_test, train_loader, train_dataset, model, train_predict, test_predict, train_predict_final, test_predict_final
torch.cuda.empty_cache()
