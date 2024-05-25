import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_preprocessing import prepare_data
from models.model_factory import ModelFactory
from util import *

# Configurations
TEST_SIZE = 18                     # Set to 18 datapoints to be predicted
LOOK_BACK = 12                     # Set to number of datapoints to be accounted for during prediciton
MODEL = 'GRU'                      # Set to model to be used
TRAIN_MODEL = False                 # Set to False to skip training and load the saved model
VERIFY_PREPROCESSING = False       # Set to False to skip preprocessing verification

# Load the data
print('Loading data...')
data_file = 'M3C_Monthly.csv'
data = pd.read_csv(data_file)

# Preprocess the data
print('Preprocessing data...')
window = 12
original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = prepare_data(data, window)

if VERIFY_PREPROCESSING:
    print('  Verifying preprocessing...')
    verify_preprocessing(original_series_list, trend_list, seasonal_list, residual_list)
    exit()

# Split the residuals into training and testing sets
print('  Creating data splits...')
test_size = TEST_SIZE

train_residuals_list = [res[:-test_size] for res in residual_list]
test_residuals_list = [res[-test_size:] for res in residual_list]

# Normalize each residual individually and combine them
print('  Normalizing data...')
scaled_train_residuals_list = []
scaled_test_residuals_list = []
scalers = []

for train_res, test_res in zip(train_residuals_list, test_residuals_list):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_res = scaler.fit_transform(train_res.values.reshape(-1, 1))
    scaled_test_res = scaler.transform(test_res.values.reshape(-1, 1))
    
    scaled_train_residuals_list.append(scaled_train_res)
    scaled_test_residuals_list.append(scaled_test_res)
    scalers.append(scaler)

# Combine all normalized residuals
scaled_train_data = np.concatenate(scaled_train_residuals_list)
scaled_test_data = np.concatenate(scaled_test_residuals_list)

look_back = LOOK_BACK
X_train, Y_train = create_dataset(scaled_train_data, look_back)
X_test, Y_test = create_dataset(scaled_test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

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

model = ModelFactory.create_model(MODEL, input_size, hidden_size, num_layers, output_size).to(device)

if TRAIN_MODEL:
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    print('Training model...')
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            outputs = model(batch_X)
            loss = smape_loss(batch_Y, outputs)
            optimizer.zero_grad()
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

# Testing the model
print('Evaluating model...')
model.eval()

# Assuming X_test contains the initial sequences for each series you want to forecast
predictions = []

# Iterate over each test sequence
for i in range(X_test.shape[0]):
    # Take the initial sequence for rolling prediction
    current_sequence = X_test[i:i+1, :, :]  # Ensure it maintains three dimensions
    
    # Store predictions for current series
    series_predictions = []
    
    for _ in range(test_size):  # Predict 18 steps into the future
        # Make prediction for the next step
        with torch.no_grad():
            next_point = model(current_sequence)
        
        # Append the predicted point to the series predictions
        series_predictions.append(next_point.item())
        
        # Update the current sequence to include the new point
        current_sequence = torch.cat((current_sequence[:, 1:, :], next_point.unsqueeze(0)), dim=1)
    
    # Collect all predictions from current series
    predictions.append(series_predictions)

reconstructed_new_data = reconstruct_series(trend_list, seasonal_list, predictions, test_size)
plot_actual_vs_predicted(original_series_list, reconstructed_new_data, test_size)
