import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing.data_preprocessing import prepare_data
from models.gru_model import get_gru_model
from util.util import (
    create_dataset, 
    denormalize_predictions,
    plot_predictions, 
    evaluate_predictions
)

def plot_two_series(first_series, second_series, title):
    print(first_series.shape, second_series.shape)
    print(first_series)
    print(second_series)
    # Generate time labels
    time_labels_first = range(len(first_series))
    time_labels_second = range(len(first_series), len(first_series) + len(second_series))

    # Plot the data
    plt.figure(figsize=(16, 4))
    plt.plot(time_labels_first, first_series, label='First Series', color='blue')
    plt.plot(time_labels_second, second_series, label='Second Series', color='orange')
    plt.legend()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

# Parameters
test_size = 18
look_back = 12

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
df = pd.read_csv("input_data/M3C_Monthly_FINANCE.csv")

# Prepare the data
original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = prepare_data(df)

# Create the datasets for each series
train_datasets = []
test_datasets = []
scalers = []

for residual in residual_list:
    scaler = MinMaxScaler()
    scaled_residual = scaler.fit_transform(residual.values.reshape(-1, 1))
    scalers.append(scaler)
    X_train, y_train = create_dataset(scaled_residual, look_back)
    train_datasets.append((X_train, y_train))
    X_test, y_test = create_dataset(scaled_residual[-(look_back + test_size):], look_back)
    test_datasets.append((X_test, y_test))

# Combine the datasets for training
X_train_combined = np.vstack([x for x, y in train_datasets])
y_train_combined = np.hstack([y for x, y in train_datasets])

# Convert to tensors
X_train_combined = torch.tensor(X_train_combined, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_combined = torch.tensor(y_train_combined, dtype=torch.float32).to(device)

# Initialize model
gru_model = get_gru_model(1, 64, 1, 1).to(device)

# Define loss function and optimizer
def smape_loss(pred, true):
    return torch.mean(2 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-10))

loss_function = smape_loss
optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    gru_model.train()
    optimizer.zero_grad()
    outputs = gru_model(X_train_combined)
    loss = loss_function(outputs, y_train_combined)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Predict the last 18 points in the testing set using the sliding window approach
gru_model.eval()
predicted_residuals = []
with torch.no_grad():
    for X_test, y_test in test_datasets:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
        predictions = []

        # Use the initial look_back window to start predictions
        input_seq = X_test_tensor[0].unsqueeze(0)

        for _ in range(test_size):
            # Predict the next value
            predicted_value = gru_model(input_seq)
            predictions.append(predicted_value.item())

            # Update the input sequence by removing the first value and adding the predicted value
            input_seq = torch.cat((input_seq[:, 1:, :], predicted_value.unsqueeze(0).unsqueeze(0)), dim=1)

        predicted_residuals.append(predictions)

# Undo the normalization
denormalized_predictions = denormalize_predictions(predicted_residuals, scalers)

# Plot the actual and predicted residuals for the test set
plot_predictions([residual[-test_size:] for residual in residual_list], denormalized_predictions, test_size)

# Evaluate the predictions using SMAPE
evaluate_predictions([residual[-test_size:] for residual in residual_list], denormalized_predictions)
