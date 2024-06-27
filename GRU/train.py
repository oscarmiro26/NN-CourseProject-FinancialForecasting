import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
base_path = os.path.dirname(os.path.abspath(__file__))
# Define the path for saving and loading the model in the GRU directory
gru_dir = os.path.join(base_path, '..', 'GRU')

os.makedirs(gru_dir, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(gru_dir, 'gru_model.pth')

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from preprocessing.data_preprocessing import prepare_data
from models.model_factory import ModelFactory
from util.util import *

# Define the base path as the directory containing this configuration file
base_path = os.path.dirname(os.path.abspath(__file__))

# Define the path to the input_data directory relative to the base path
input_data_path = os.path.join(base_path, '..', 'input_data')

# Define the full path to the data file relative to the input_data directory
DATA_FILE = os.path.join(input_data_path, 'M3C_Monthly_FINANCE.csv')


# Configurations
TEST_SIZE = 18                     # Set to 18 datapoints to be predicted
LOOK_BACK = 12                     # Set to number of datapoints to be accounted for during prediction
MODEL = 'GRU'                      # Set to model to be used
TRAIN_MODEL = True                 # Set to False to skip training and load the saved model
VERIFY_PREPROCESSING = False       # Set to False to skip preprocessing verification

# Load the data
print('Loading data...')
data_file = 'M3C_Monthly_FINANCE.csv'
data = pd.read_csv(DATA_FILE)

# Preprocess the data
print('Preprocessing data...')
window = 20
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
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
else:
    # Load the model
    print('Loading saved model...')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Testing the model
print('Evaluating model...')
model.eval()

# Use the last LOOK_BACK points from the training set for each series for rolling prediction
initial_sequences = []

for series in scaled_train_residuals_list:
    initial_sequence = series[-look_back:]  # Last LOOK_BACK points
    initial_sequences.append(initial_sequence)

# Convert initial sequences to PyTorch tensors
initial_sequences = np.array(initial_sequences)
initial_sequences = np.reshape(initial_sequences, (initial_sequences.shape[0], initial_sequences.shape[1], 1))
initial_sequences = torch.Tensor(initial_sequences).to(device)

predictions = []

# Iterate over each initial sequence
for i in range(initial_sequences.shape[0]):
    # Take the initial sequence for rolling prediction
    current_sequence = initial_sequences[i:i+1, :, :]  # Shape: (1, LOOK_BACK, 1)
    
    # Store predictions for current series
    series_predictions = []
    
    for _ in range(test_size):  # Predict 18 steps into the future
        # Make prediction for the next step
        with torch.no_grad():
            next_point = model(current_sequence)
        
        # Append the predicted point to the series predictions
        series_predictions.append(next_point.item())
        
        # Update the current sequence to include the new point
        next_point = next_point.view(1, 1, 1)  # Ensure next_point has the shape (1, 1, 1)
        current_sequence = torch.cat((current_sequence[:, 1:, :], next_point), dim=1)
    
    # Collect all predictions from current series
    predictions.append(series_predictions)

# Denormalize the predictions
denormalized_predictions = []

for scaler, prediction in zip(scalers, predictions):
    denormalized_prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    denormalized_predictions.append(denormalized_prediction)

# Evaluate predictions
mse_list, mae_list, r2_list, smape_list = evaluate_predictions(test_residuals_list, denormalized_predictions)

# Reconstruct series and plot
reconstructed_new_data = reconstruct_series(trend_list, seasonal_list, denormalized_predictions, test_size)
print(calculate_median_smape(original_series_list, reconstructed_new_data, TEST_SIZE))

plot_actual_vs_predicted(original_series_list, reconstructed_new_data, test_size)
