import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_preprocessing import prepare_data
from models.model_factory import ModelFactory
from util import create_dataset, reconstruct_series, plot_actual_vs_predicted, verify_preprocessing, plot_prediction_errors

# Configurations and Hyperparameters
DATA_FILE = 'M3C_Monthly.csv'  # Path to the CSV file that contains the data

VALIDATION_SPLIT = 0.1          # Percentage of data to be used for validation, common practice is to allocate around 70-80% of the data for training, 10-15% for validation, and 10-15% for testing.
TEST_SPLIT = 0.1                # Percentage of data to be used for testing
PREDICTION_SIZE = 18            # Number of future datapoints to predict

LOOK_BACK = 12                  # Number of past datapoints to consider for prediction. For monthly stock price data, a look-back period of 12 months (1 year) to 24 months (2 years) is often reasonable.

MODEL = 'MLP'                   # Model we use (e.g., 'GRU', 'LSTM', 'MLP')

TRAIN_MODEL = True              # Set to False to skip training and load the saved model
VERIFY_PREPROCESSING = False    # Set to True to verify preprocessing steps

BATCH_SIZE = 64                 # Larger batch sizes can make training faster and more stable but may require more memory. Batch sizes between 32 and 128 are commonly used.
NUM_EPOCHS = 50                 # Number of epochs for training. The number of epochs should be large enough to allow the model to learn the data well but not so large that it overfits. We are using early stopping based on validation loss, so it can be bigger I guess.
HIDDEN_SIZE = 20                # Number of hidden units. More hidden units can capture more complex patterns but also increase the risk of overfitting and computational cost. For an MLP, values between 20 and 100 are common starting points.
LEARNING_RATE = 0.001           # Learning rate for the optimizer. The learning rate controls the step size of each update. A too high learning rate can cause the model to converge too quickly to a suboptimal solution, while a too low learning rate can make training very slow. Typical values range from 0.001 to 0.01.
PATIENCE = 6                    # Number of epochs with no improvement before stopping. Patience is used in early stopping to avoid premature termination of training. A typical range is between 5 to 10 epochs.

# Define loss function as a parameter
LOSS_FUNCTION = nn.MSELoss()

def preprocess_data(data, window):
    print('Preprocessing data...')
    result = prepare_data(data, window)
    original_series_list, trend_list, seasonal_list, residual_list, *_ = result
    
    if VERIFY_PREPROCESSING:
        print('  Verifying preprocessing...')
        verify_preprocessing(original_series_list, trend_list, seasonal_list, residual_list)
        exit()
    
    return original_series_list, trend_list, seasonal_list, residual_list

def split_data(residual_list, val_split, test_split):
    print('  Creating data splits...')
    val_size = int(len(residual_list[0]) * val_split)
    test_size = int(len(residual_list[0]) * test_split)
    train_residuals_list = [res[:-test_size-val_size] for res in residual_list]
    val_residuals_list = [res[-test_size-val_size:-test_size] for res in residual_list]
    test_residuals_list = [res[-test_size:] for res in residual_list]
    
    return train_residuals_list, val_residuals_list, test_residuals_list

def normalize_data(train_residuals_list, val_residuals_list, test_residuals_list):
    """
    Normalize the training, validation, and testing residuals using MinMaxScaler.

    Args:
        train_residuals_list (list): List of training residuals for each time series.
        val_residuals_list (list): List of validation residuals for each time series.
        test_residuals_list (list): List of testing residuals for each time series.

    Returns:
        Tuple: Normalized training, validation, and testing data, and the list of scalers used for normalization.
    """
    
    print('  Normalizing data...')  # Inform that normalization process has started.
    
    # Initialize empty lists to store scaled residuals and scalers
    scaled_train_residuals_list = []
    scaled_val_residuals_list = []
    scaled_test_residuals_list = []
    scalers = []

    # Iterate over the residuals of each time series for training, validation, and testing
    for train_res, val_res, test_res in zip(train_residuals_list, val_residuals_list, test_residuals_list):
        # Initialize the MinMaxScaler to scale data to the range [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit the scaler on the training residuals and transform them
        scaled_train_res = scaler.fit_transform(train_res.values.reshape(-1, 1))
        # Transform the validation and test residuals using the fitted scaler
        scaled_val_res = scaler.transform(val_res.values.reshape(-1, 1))
        scaled_test_res = scaler.transform(test_res.values.reshape(-1, 1))

        # Append the scaled residuals to their respective lists
        scaled_train_residuals_list.append(scaled_train_res)
        scaled_val_residuals_list.append(scaled_val_res)
        scaled_test_residuals_list.append(scaled_test_res)
        # Append the scaler to the list of scalers
        scalers.append(scaler)

    # Concatenate all scaled residuals to create a single array for each dataset
    scaled_train_data = np.concatenate(scaled_train_residuals_list)
    scaled_val_data = np.concatenate(scaled_val_residuals_list)
    scaled_test_data = np.concatenate(scaled_test_residuals_list)
    
    # Return the concatenated arrays and the list of scalers
    return scaled_train_data, scaled_val_data, scaled_test_data, scalers

"""
Explanation of Why This is Useful:

1. **Consistent Scaling**: Each time series is normalized independently, ensuring that all data is scaled to the same range [0, 1]. This is important because different stocks can have vastly different price ranges, and normalization ensures that each series contributes equally to the model training.

2. **Improved Model Performance**: Normalization helps in faster convergence during training and can improve model performance. Neural networks often perform better when input data is scaled.

3. **Handling Multiple Time Series**: By normalizing and then concatenating the data, we effectively create a larger dataset that the model can learn from, which is particularly useful when individual time series are short.

4. **Preservation of Data Characteristics**: Since each series is normalized separately and then concatenated, the unique characteristics of each series are preserved while allowing the model to learn from a combined dataset.

5. **Ease of Data Handling**: Concatenating the data into single arrays for training, validation, and testing makes it easier to handle and feed into the model during training and evaluation phases.
"""


def create_datasets(data, look_back, val_split, test_split):
    original_series_list, trend_list, seasonal_list, residual_list = preprocess_data(data, window=12)
    train_residuals_list, val_residuals_list, test_residuals_list = split_data(residual_list, val_split, test_split)
    scaled_train_data, scaled_val_data, scaled_test_data, scalers = normalize_data(train_residuals_list, val_residuals_list, test_residuals_list)

    X_train, Y_train = create_dataset(scaled_train_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    X_val, Y_val = create_dataset(scaled_val_data, look_back)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1]))

    X_test, Y_test = create_dataset(scaled_test_data, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    X_test = torch.Tensor(X_test).to(device)
    Y_train = torch.Tensor(Y_train).to(device)
    Y_val = torch.Tensor(Y_val).to(device)
    Y_test = torch.Tensor(Y_test).to(device)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, original_series_list, trend_list, seasonal_list, residual_list

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, loss_function, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    total_iterations = len(train_loader) * num_epochs
    pbar = tqdm(total=total_iterations, desc="Training Progress", leave=True)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            outputs = model(batch_X).view(-1)  # Ensure output tensor is flattened
            batch_Y = batch_Y.view(-1)  # Ensure target tensor is flattened
            loss = loss_function(outputs, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_Y in val_loader:
                val_outputs = model(val_X).view(-1)  # Ensure output tensor is flattened
                val_Y = val_Y.view(-1)  # Ensure target tensor is flattened
                val_loss += loss_function(val_outputs, val_Y).item()
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training MSE: {loss.item():.4f}, Validation MSE: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f'{MODEL}_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    pbar.close()

def evaluate_model(model, X_test, prediction_size):
    model.eval()
    predictions = []

    for i in range(X_test.shape[0]):
        current_sequence = X_test[i:i+1, :]
        series_predictions = []

        for _ in range(prediction_size):
            with torch.no_grad():
                next_point = model(current_sequence)
            
            series_predictions.append(next_point.item())
            next_point = next_point.view(1, 1)
            current_sequence = torch.cat((current_sequence[:, 1:], next_point), dim=1)
        
        predictions.append(series_predictions)

    return predictions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading data...')
    data = pd.read_csv(DATA_FILE)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, original_series_list, trend_list, seasonal_list, residual_list = create_datasets(data, LOOK_BACK, VALIDATION_SPLIT, TEST_SPLIT)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Creating model...')
    input_size = LOOK_BACK
    output_size = 1
    model = ModelFactory.create_model(MODEL, input_size, HIDDEN_SIZE, output_size).to(device)

    if TRAIN_MODEL:
        print('Training model...')
        train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, LOSS_FUNCTION, PATIENCE)
    else:
        print('Loading saved model...')
        model.load_state_dict(torch.load(f'{MODEL}_model.pth'))

    print('Evaluating model...')
    predictions = evaluate_model(model, X_test, PREDICTION_SIZE)

    reconstructed_new_data = reconstruct_series(trend_list, seasonal_list, predictions, PREDICTION_SIZE)
    plot_prediction_errors(original_series_list, reconstructed_new_data, PREDICTION_SIZE)
    #plot_actual_vs_predicted(original_series_list, reconstructed_new_data, PREDICTION_SIZE)
