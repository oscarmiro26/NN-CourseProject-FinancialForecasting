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
from util import create_dataset, verify_preprocessing

# Configurations and Hyperparameters
DATA_FILE = 'M3C_Monthly_FINANCE.csv'  # Path to the CSV file that contains the data

VALIDATION_SPLIT = 0.1          # Percentage of data to be used for validation
TEST_SPLIT = 0.1                # Percentage of data to be used for testing
EVAL_PREDICTION_SIZE = 18       # Number of future datapoints to predict and hold out for evaluation

LOOK_BACK = 10                  # Number of past datapoints to consider for prediction

MODEL = 'MLP'                   # Model we use

TRAIN_MODEL = True              # Set to False to skip training and load the saved model
VERIFY_PREPROCESSING = False    # Set to True to verify preprocessing steps

BATCH_SIZE = 64                 # Larger batch sizes can make training faster and more stable but may require more memory
NUM_EPOCHS = 50                 # Number of epochs for training
HIDDEN_SIZE = 100                # Number of hidden units
LEARNING_RATE = 0.0001           # Learning rate for the optimizer
PATIENCE = 8                    # Number of epochs with no improvement before stopping

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
    
    return trend_list, seasonal_list, residual_list


def split_data(residual_list, val_split, test_split, eval_size):
    
    print('  Creating data splits...')

    # Hold out the last `eval_size` values from each series in residual_list for final evaluation
    eval_residuals_list = [res[-eval_size:] for res in residual_list]

    # The remaining data after holding out the evaluation set (we remove the last 18)
    all_residuals_except_eval = [res[:-eval_size] for res in residual_list]

    # Calculate the size of the validation set based on the percentage provided (val_split)
    val_size = int(len(all_residuals_except_eval[0]) * val_split)

    # Calculate the size of the test set based on the percentage provided (test_split)
    test_size = int(len(all_residuals_except_eval[0]) * test_split)

    # Split the remaining data into training set
    train_residuals_list = [res[:-test_size-val_size] for res in all_residuals_except_eval]

    # Split the remaining data into validation set
    val_residuals_list = [res[-test_size-val_size:-test_size] for res in all_residuals_except_eval]

    # Split the remaining data into test set
    test_residuals_list = [res[-test_size:] for res in all_residuals_except_eval]
    
    return train_residuals_list, val_residuals_list, test_residuals_list, eval_residuals_list, all_residuals_except_eval


def normalize_data(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_eval):
    print('  Normalizing data...')
    scaled_train_residuals_list = []
    scaled_val_residuals_list = []
    scaled_test_residuals_list = []
    scaled_all_residuals_list = [] 
    scalers = []

    for train_res, val_res, test_res, all_res in zip(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_eval):
        train_res_np = train_res.values.reshape(-1, 1)
        val_res_np = val_res.values.reshape(-1, 1)
        test_res_np = test_res.values.reshape(-1, 1)
        all_res_np = all_res.values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaled_train_res = scaler.fit_transform(train_res_np)
        scaled_val_res = scaler.transform(val_res_np)
        scaled_test_res = scaler.transform(test_res_np)
        scaled_all_res = scaler.transform(all_res_np)

        scaled_train_residuals_list.append(scaled_train_res)
        scaled_val_residuals_list.append(scaled_val_res)
        scaled_test_residuals_list.append(scaled_test_res)
        scaled_all_residuals_list.append(scaled_all_res)

        scalers.append(scaler)

    scaled_train_data = np.concatenate(scaled_train_residuals_list)
    scaled_val_data = np.concatenate(scaled_val_residuals_list)
    scaled_test_data = np.concatenate(scaled_test_residuals_list)

    return scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_residuals_list, scalers



def create_datasets(data, look_back, val_split, test_split, eval_size):
    trend_list, seasonal_list, residual_list = preprocess_data(data, window=12)
    
    train_residuals_list, val_residuals_list, test_residuals_list, eval_residuals_list, all_residuals_except_eval = split_data(residual_list, val_split, test_split, eval_size)
    
    scaled_train_data, scaled_val_data, scaled_test_data, scaled_all_data, scalers = normalize_data(train_residuals_list, val_residuals_list, test_residuals_list, all_residuals_except_eval)

    X_train, Y_train = create_dataset(scaled_train_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    X_val, Y_val = create_dataset(scaled_val_data, look_back)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1]))

    X_test, Y_test = create_dataset(scaled_test_data, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    
    X_train = torch.Tensor(X_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    X_test = torch.Tensor(X_test).to(device)
    Y_train = torch.Tensor(Y_train).to(device)
    Y_val = torch.Tensor(Y_val).to(device)
    Y_test = torch.Tensor(Y_test).to(device)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, trend_list, seasonal_list, eval_residuals_list, scaled_all_data

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


def generate_predictions(model, scaled_all_data, look_back, prediction_size):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []  # Initialize an empty list to store predictions for all sequences

    for data in scaled_all_data:
        if len(data) < look_back:
            raise ValueError(f"Not enough data points in sequence. Required: {look_back}, Available: {len(data)}")
        
        # Use the last 'look_back' points from the data to start the prediction
        current_sequence = data[-look_back:].reshape(1, look_back, 1)  # Reshape to match the model's expected input shape
        
        series_predictions = []  # Initialize a list to store predictions for the current sequence

        for _ in range(prediction_size):  # Loop to generate 'prediction_size' future points
            with torch.no_grad():  # Disable gradient calculation for inference
                next_point = model(torch.Tensor(current_sequence))  # Predict the next point in the sequence
                next_point = next_point.cpu().numpy()  # Move prediction back to CPU and numpy

            series_predictions.append(next_point.item())  # Append the predicted value to the list
            next_point = next_point.reshape(1, 1)  # Reshape the prediction to match the input shape

            # Update the current sequence by removing the oldest point and adding the new prediction
            current_sequence = np.append(current_sequence[:, 1:, :], next_point.reshape(1, 1, 1), axis=1)

        all_predictions.append(series_predictions)  # Append predictions for the current sequence to the main list

    return all_predictions  # Return the list of all predictions for all sequences



def denormalize_predictions(predictions, scalers):
    denormalized_predictions = []
    for pred, scaler in zip(predictions, scalers):
        pred_array = np.array(pred).reshape(-1, 1)
        denormalized_pred = scaler.inverse_transform(pred_array)
        denormalized_predictions.append(denormalized_pred.flatten().tolist())
    return denormalized_predictions

def plot_predictions(actual_list, predicted_list, num_points):
    for i, (actual, predicted) in enumerate(zip(actual_list, predicted_list)):
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_points), actual[-num_points:], label='Actual')
        plt.plot(range(num_points), predicted, label='Predicted')
        plt.legend()
        plt.title(f'Actual vs Predicted for Series {i+1} for last {num_points} points')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading data...')
    data = pd.read_csv(DATA_FILE)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, trend_list, seasonal_list, eval_residuals_list, scaled_all_data = create_datasets(data, LOOK_BACK, VALIDATION_SPLIT, TEST_SPLIT, EVAL_PREDICTION_SIZE)

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

    print('Generating predictions...')

    predictions = generate_predictions(model, scaled_all_data, LOOK_BACK, EVAL_PREDICTION_SIZE)
    
    print('Denormalizing predictions...')
    denormalized_predictions = denormalize_predictions(predictions, scalers)

    print('Plotting predictions...')
    plot_predictions(eval_residuals_list, denormalized_predictions, EVAL_PREDICTION_SIZE)
