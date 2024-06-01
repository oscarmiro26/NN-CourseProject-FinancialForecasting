import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_preprocessing import prepare_data
from models.model_factory import ModelFactory
from util import create_dataset, smape_loss, reconstruct_series, plot_actual_vs_predicted, verify_preprocessing


# train MLP created because MLPs expect input data to be in a flat 2D format WHILE
# GRUs require input data to be structured as a 3D tensor representing sequences of observations, so it was easier to split the training scripts for me, even though there are shared parts

# Configurations and Hyperparameters
DATA_FILE = 'M3C_Monthly.csv'  # Path to the CSV file that contains the data
TEST_SIZE = 18                 # Number of datapoints that we want to predict
LOOK_BACK = 12                 # Number of the past datapoints we should consider for prediction
MODEL = 'MLP'                  #  Model we use (e.g., 'GRU', 'LSTM', 'MLP')
TRAIN_MODEL = True             # Set to False to skip training and load the saved model
VERIFY_PREPROCESSING = False   # Set to True to verify preprocessing steps
BATCH_SIZE = 64                # Batch size for training- Larger batch sizes can make training faster and more stable, but may require more memory and might not always converge to the best solution.
NUM_EPOCHS = 20                # Number of epochs for training Early Stopping: Use early stopping to determine the optimal number of epochs.
                               #Learning Curves: Plot learning curves to visualize how the model performance changes with epochs and decide if more epochs are needed.
HIDDEN_SIZE = 10               # Number of hidden units - best to perform a search to optimize these
LEARNING_RATE = 0.0001          # Learning rate for the optimizer. Useful to start with a higher learning rate and gradually decrease it.


def preprocess_data(data, window):
    """
    Preprocess the data to extract residuals, trends, and seasonality.

    Args:
        data (pd.DataFrame): The input data.
        window (int): The window size for detrending.

    Returns:
        Tuple: Original series, trends, seasonal components, and residuals.
    """
    print('Preprocessing data...')
    original_series_list, trend_list, detrended_series_list, seasonal_list, residual_list = prepare_data(data, window)
    
    if VERIFY_PREPROCESSING:
        print('  Verifying preprocessing...')
        verify_preprocessing(original_series_list, trend_list, seasonal_list, residual_list)
        exit()
    
    return original_series_list, trend_list, seasonal_list, residual_list

def split_data(residual_list, test_size):
    """
    Split residuals into training and testing sets.

    Args:
        residual_list (list): List of residuals for each series.
        test_size (int): The number of test datapoints.

    Returns:
        Tuple: Training and testing residuals.
    """
    print('  Creating data splits...')
    train_residuals_list = [res[:-test_size] for res in residual_list]
    test_residuals_list = [res[-test_size:] for res in residual_list]
    
    return train_residuals_list, test_residuals_list

def normalize_data(train_residuals_list, test_residuals_list):
    """
    Normalize the training and testing residuals.

    Args:
        train_residuals_list (list): List of training residuals.
        test_residuals_list (list): List of testing residuals.

    Returns:
        Tuple: Normalized training and testing data, and the scalers.
    """
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
    
    return scaled_train_data, scaled_test_data, scalers

def create_datasets(data, look_back, test_size):
    """
    Prepare training and testing datasets.

    Args:
        data (pd.DataFrame): The input data.
        look_back (int): The number of past datapoints to consider for prediction.
        test_size (int): The number of test datapoints.

    Returns:
        Tuple: Training and testing datasets, and additional preprocessing results.
    """
    original_series_list, trend_list, seasonal_list, residual_list = preprocess_data(data, window=12)
    train_residuals_list, test_residuals_list = split_data(residual_list, test_size)
    scaled_train_data, scaled_test_data, scalers = normalize_data(train_residuals_list, test_residuals_list)

    # Create training datasets
    X_train, Y_train = create_dataset(scaled_train_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    # Create evaluation datasets
    X_test, Y_test = create_dataset(scaled_test_data, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.Tensor(X_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    Y_train = torch.Tensor(Y_train).to(device)
    Y_test = torch.Tensor(Y_test).to(device)

    return X_train, Y_train, X_test, Y_test, scalers, original_series_list, trend_list, seasonal_list, residual_list

def train_model(model, train_loader, num_epochs, learning_rate):
    """
    Train the model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        None
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_iterations = len(train_loader) * num_epochs
    pbar = tqdm(total=total_iterations, desc="Training Progress", leave=True)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            outputs = model(batch_X)
            loss = smape_loss(batch_Y, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], SMAPE: {loss.item():.4f}')
    
    pbar.close()
    torch.save(model.state_dict(), f'{MODEL}_model.pth')


def evaluate_model(model, X_test, test_size):
    """
    Evaluate the model and make predictions.

    Args:
        model (torch.nn.Module): The trained model.
        X_test (torch.Tensor): Test dataset.
        test_size (int): The number of test datapoints.

    Returns:
        list: Predictions for each series in the test dataset.
    """
    model.eval()
    predictions = []

    for i in range(X_test.shape[0]):
        current_sequence = X_test[i:i+1, :]  # Ensure it maintains two dimensions
        series_predictions = []

        for _ in range(test_size):
            with torch.no_grad():
                next_point = model(current_sequence)
            
            series_predictions.append(next_point.item())
            # Update the current sequence to include the new point
            next_point = next_point.view(1, 1)  # Ensure it has the same dimensions as the current sequence
            current_sequence = torch.cat((current_sequence[:, 1:], next_point), dim=1)
        
        predictions.append(series_predictions)

    return predictions



# Main script
if __name__ == "__main__":
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    print('Loading data...')
    data = pd.read_csv(DATA_FILE)

    # Create datasets
    X_train, Y_train, X_test, Y_test, scalers, original_series_list, trend_list, seasonal_list, residual_list = create_datasets(data, LOOK_BACK, TEST_SIZE)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get the model
    print('Creating model...')
    input_size = LOOK_BACK
    output_size = 1
    model = ModelFactory.create_model(MODEL, input_size, HIDDEN_SIZE, output_size).to(device)

    if TRAIN_MODEL:
        # Train the model
        print('Training model...')
        train_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    else:
        # Load the model
        print('Loading saved model...')
        model.load_state_dict(torch.load(f'{MODEL}_model.pth'))

    # Evaluate the model
    print('Evaluating model...')
    predictions = evaluate_model(model, X_test, TEST_SIZE)

    # Reconstruct and plot series
    reconstructed_new_data = reconstruct_series(trend_list, seasonal_list, predictions, TEST_SIZE)
    plot_actual_vs_predicted(original_series_list, reconstructed_new_data, TEST_SIZE)
