import sys
import os

from GRU.config_GRU import INPUT_SIZE, NUM_LAYERS, OUTPUT_SIZE

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocessing.data_preprocessing import *
from models.model_factory import ModelFactory
from util.util import plot_predictions, denormalize_predictions, evaluate_predictions, plot_predictions_naive, reconstruct_series, plot_actual_vs_predicted, calculate_median_smape
from GRU.config_GRU import *


def train_model(model_str, train_loader, val_loader, num_epochs, loss_function, model_save_path, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the hyperparameter grid
    hidden_sizes = [32, 64, 128]
    num_layers = [1, 2, 3]
    learning_rates = [0.001, 0.005, 0.01]
    dropout_rates = [0.0, 0.2, 0.5]
    
    best_val_loss = float('inf')
    best_hyperparameters = None
    log = []

    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            for lr in learning_rates:
                for dropout in dropout_rates:
                    # Initialize the model with the current set of hyperparameters
                    model = ModelFactory.create_model(model_str, INPUT_SIZE, hidden_size, num_layer, OUTPUT_SIZE, dropout)
                    model.to(device)
                    
                    # Define the optimizer with the current learning rate
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    # Training loop
                    best_epoch_loss = float('inf')
                    epochs_no_improve = 0
                    
                    for epoch in range(num_epochs):
                        model.train()
                        epoch_loss = 0.0
                        
                        for X_batch, y_batch in train_loader:
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            
                            optimizer.zero_grad()
                            output = model(X_batch)
                            loss = loss_function(output, y_batch)
                            loss.backward()
                            optimizer.step()
                            
                            epoch_loss += loss.item() * X_batch.size(0)
                        
                        epoch_loss /= len(train_loader.dataset)
                        
                        # Validation loop
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for X_batch, y_batch in val_loader:
                                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                output = model(X_batch)
                                loss = loss_function(output, y_batch)
                                val_loss += loss.item() * X_batch.size(0)
                        
                        val_loss /= len(val_loader.dataset)
                        
                        # Early stopping
                        if val_loss < best_epoch_loss:
                            best_epoch_loss = val_loss
                            epochs_no_improve = 0
                            torch.save(model.state_dict(), model_save_path)
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                break

                        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
                    
                    # Log the results
                    log_entry = {
                        'hidden_size': hidden_size,
                        'num_layers': num_layer,
                        'learning_rate': lr,
                        'dropout': dropout,
                        'val_loss': best_epoch_loss
                    }
                    log.append(log_entry)
                    
                    # Update the best hyperparameters if the current validation loss is the best
                    if best_epoch_loss < best_val_loss:
                        best_val_loss = best_epoch_loss
                        best_hyperparameters = (hidden_size, num_layer, lr, dropout)
    
    # Print and save the best hyperparameters
    print("Best Hyperparameters:")
    print(f"Hidden Size: {best_hyperparameters[0]}")
    print(f"Number of Layers: {best_hyperparameters[1]}")
    print(f"Learning Rate: {best_hyperparameters[2]}")
    print(f"Dropout: {best_hyperparameters[3]}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Save the log to a file
    with open('hyperparameter_tuning_log.csv', 'w') as f:
        f.write("hidden_size,num_layers,learning_rate,dropout,val_loss\n")
        for entry in log:
            f.write(f"{entry['hidden_size']},{entry['num_layers']},{entry['learning_rate']},{entry['dropout']},{entry['val_loss']:.4f}\n")

    # Load the best model
    model = ModelFactory.create_model(model_str, INPUT_SIZE, hidden_size, num_layer, OUTPUT_SIZE, dropout)
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    
    return model


def generate_predictions(model, scaled_all_data, look_back, prediction_size, device):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []  # Initialize an empty list to store predictions for all sequences

    for data in scaled_all_data:
        if len(data) < look_back:
            raise ValueError(f"Not enough data points in sequence. Required: {look_back}, Available: {len(data)}")

        # Use the last 'look_back' points from the data to start the prediction
        current_sequence = data[-look_back:].reshape(1, look_back, 1)  # Reshape to match the model's expected input shape

        # Convert current_sequence to a torch tensor and move to the correct device
        current_sequence = torch.Tensor(current_sequence).to(device)

        series_predictions = []  # Initialize a list to store predictions for the current sequence

        for _ in range(prediction_size):  # Loop to generate 'prediction_size' future points
            with torch.no_grad():  # Disable gradient calculation for inference
                next_point = model(current_sequence)  # Predict the next point in the sequence
                next_point = next_point.cpu().numpy()  # Move prediction back to CPU and numpy

            series_predictions.append(next_point.item())  # Append the predicted value to the list
            next_point = next_point.reshape(1, 1, 1)  # Reshape the prediction to match the input shape

            # Update the current sequence by removing the oldest point and adding the new prediction
            current_sequence = torch.cat((current_sequence[:, 1:, :], torch.Tensor(next_point).to(device)), dim=1)

        all_predictions.append(series_predictions)  # Append predictions for the current sequence to the main list

    return all_predictions  # Return the list of all predictions for all sequences


def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    # Create data loaders for training, validation, and test sets
    # DataLoader helps manage batches of data efficiently during training and evaluation
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def start_to_train_model(model_str, train_loader, val_loader):
    # Train the model if the TRAIN_MODEL flag is set to True
    if TRAIN_MODEL:
        print('Training model...')
        model = train_model(model_str, train_loader, val_loader, NUM_EPOCHS, LOSS_FUNCTION, MODEL_SAVE_PATH, PATIENCE)
    else:
        # Load the pre-trained model weights
        print('Loading saved model...')
        model = ModelFactory.create_model(model_str)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    return model


def naive_predictor(eval_residuals_list, prediction_size):
    naive_predictions = []
    for residuals in eval_residuals_list:
        last_value = residuals[-1]
        naive_predictions.append([last_value] * prediction_size)
    return naive_predictions


def main():
    print('Lookback:', LOOK_BACK)
    # Determine if CUDA (GPU) is available; if not, use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess the data and create training, validation, and test datasets
    print('Creating datasets...')
    X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, trend_list, seasonal_list, eval_residuals_list, scaled_all_data, original_series = create_datasets(LOOK_BACK)
    train_loader, val_loader, test_loader = create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test)
   
    # Initialize the model
    # The ModelFactory class abstracts the creation of different models
    # input_size and output_size are set based on the problem requirements
    print('Creating model...')
    model = start_to_train_model(MODEL, train_loader, val_loader)

    # Generate predictions for future data points using the trained model
    print('Generating predictions...')
    predictions = generate_predictions(model, scaled_all_data, LOOK_BACK, PREDICTION_SIZE, device)

    # Denormalize the predictions to convert them back to the original scale
    print('Denormalizing predictions...')
    denormalized_predictions = denormalize_predictions(predictions, scalers)

    # Evaluate the model's predictions against the actual data of the last 18 points
    print('Evaluating predicted vs actual residual predictions...')
    mse_list, mae_list, r2_list, smape_list = evaluate_predictions(eval_residuals_list, denormalized_predictions)

    # Generate naive predictions
    print('Generating naive predictions...')
    naive_predictions = naive_predictor(eval_residuals_list, PREDICTION_SIZE)

    # Plot the predictions against the actual data to visualize performance
    print('Plotting residual predictions vs actual...')
    for true, pred, naive in zip(eval_residuals_list, denormalized_predictions, naive_predictions):
        #plot_predictions_naive(true, pred, naive, 'Residuals: True vs Predicted vs Naive')
        pass

    # Reconstructing the series
    reconstructed_series = reconstruct_series(trend_list, seasonal_list, denormalized_predictions, PREDICTION_SIZE)

    # Uncomment the following line if you want to plot the actual vs predicted series
    # plot_actual_vs_predicted(original_series, reconstructed_series, PREDICTION_SIZE)
    
    print("Final SMAPE score:")
    print(calculate_median_smape(original_series, reconstructed_series, PREDICTION_SIZE))
    
    # Calculate and print the median SMAPE of the naive residuals versus the true residuals
    naive_smape_list = [calculate_median_smape([true], [naive], PREDICTION_SIZE) for true, naive in zip(eval_residuals_list, naive_predictions)]

    median_naive_smape = np.median(naive_smape_list)
    median_pred_smape = np.median(smape_list)

    print("Median SMAPE for naive predictions:")
    print(median_naive_smape)

    print("Median SMAPE for model predictions:")
    print(median_pred_smape)


if __name__ == "__main__":
    main()