import sys
import os
import itertools
import json

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
from util.util import plot_predictions, denormalize_predictions, evaluate_predictions, reconstruct_series, plot_actual_vs_predicted, calculate_median_smape, naive_predictor, plot_prediction_errors, calculate_mean_smape, plot_combined_predictions
from MLP.config_mlp import *

# Define the training function
def train_model(model, train_loader, val_loader, num_epochs, weight_decay, loss_function, model_save_path, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    total_iterations = len(train_loader) * num_epochs  # Total number of iterations

    pbar = tqdm(total=total_iterations, desc="Training Progress", leave=True)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            outputs = model(batch_X).view(-1)  # Ensure output tensor is flattened
            batch_Y = batch_Y.view(-1)  # Ensure target tensor is flattened
            loss = loss_function(outputs, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'Training Loss': train_loss / (pbar.n + 1)})
            pbar.update(1)
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_Y in val_loader:
                val_outputs = model(val_X).view(-1)  # Ensure output tensor is flattened
                val_Y = val_Y.view(-1)  # Ensure target tensor is flattened
                val_loss += loss_function(val_outputs, val_Y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update the progress bar description to include the current epoch's training and validation losses
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Training: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save the model if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
                break

    pbar.close()
    print()
    print(f"BEST Validation Loss: {best_val_loss:.4f}")  # Print the best validation loss after training stops
    print()
    return train_losses, val_losses

# Function to perform grid search
def grid_search(lookback_values, weight_decays, num_layers_list, num_nodes_list, num_epochs, loss_function, model_save_path, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_params = None
    best_val_loss = float('inf')
    results = []

    for lookback in lookback_values:
        print(f"Creating datasets for lookback={lookback}...")
        datasets = create_datasets(lookback)
        X_train, Y_train, X_val, Y_val, trend_list, seasonal_list, test_residuals_list, original_series_list, residual_list, train_scalers, val_scalers, test_scalers, scaled_all_residuals_list = datasets

        for weight_decay, num_layers, num_nodes in itertools.product(weight_decays, num_layers_list, num_nodes_list):
            print(f"Training with lookback={lookback}, lr={lr}, num_layers={num_layers}, num_nodes={num_nodes}")

            # Create the model
            model = ModelFactory.create_model(MODEL, lookback, num_nodes, PREDICTED_DATA_POINTS, num_layers).to(device)

            # Create data loaders
            train_loader, val_loader = create_dataloaders(X_train, Y_train, X_val, Y_val)

            train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, weight_decay, loss_function, model_save_path, patience)

            results.append({
                'lookback': lookback,
                'weight_decay': weight_decay,
                'num_layers': num_layers,
                'num_nodes': num_nodes,
                'train_losses': train_losses,
                'val_losses': val_losses
            })

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_params = (lookback, weight_decay, num_layers, num_nodes)
                best_model_state = model.state_dict()
                best_dataset = datasets

    print(f"Best params: {best_params} with validation loss: {best_val_loss:.4f}")
    return results, best_params, best_model_state, best_dataset

# Function to save results to a log file
def save_results_to_log(results, log_file_path):
    with open(log_file_path, 'w') as log_file:
        json.dump(results, log_file)


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





def start_to_train_model(model, train_loader, val_loader):
    # Train the model if the TRAIN_MODEL flag is set to True
    if TRAIN_MODEL:
        print('Training model...')
        train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, LOSS_FUNCTION, MODEL_SAVE_PATH, PATIENCE)
    else:
        # Load the pre-trained model weights
        print('Loading saved model...')
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))



def create_dataloaders(X_train, Y_train, X_val, Y_val):
     # Create data loaders for training and validation sets
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameter grid
    lookback_values = [4, 8, 12, 16]
    weight_decay_list = [1, 0.1, 0.001]
    num_layers_list = [1, 2]
    num_nodes_list = [2, 4, 8, 16, 32, 64]

    # Perform grid search
    print('Starting grid search...')
    results, best_params, best_model_state, best_dataset = grid_search(lookback_values, weight_decay_list, num_layers_list, num_nodes_list, NUM_EPOCHS, LOSS_FUNCTION, MODEL_SAVE_PATH, PATIENCE)

    # Save results to log file
    log_file_path = 'grid_search_results.json'
    save_results_to_log(results, log_file_path)
    print(f'Results saved to {log_file_path}')

    # Evaluate best model
    best_lookback, best_lr, best_num_layers, best_num_nodes = best_params
    print('Creating model...')
    best_model = ModelFactory.create_model(MODEL, best_lookback, best_num_nodes, PREDICTED_DATA_POINTS, best_num_layers).to(device)
    best_model.load_state_dict(best_model_state)

    X_train, Y_train, X_val, Y_val, trend_list, seasonal_list, test_residuals_list, original_series_list, residual_list, train_scalers, val_scalers, test_scalers, scaled_all_residuals_list = best_dataset

    print('Generating predictions for comparison with test set...')
    predictions = generate_predictions(best_model, scaled_all_residuals_list, best_lookback, PREDICTION_SIZE, device)
    
    if not predictions:
        print("No predictions generated.")
        return

    denormalized_predictions = denormalize_predictions(predictions, test_scalers)
    naive_preds = naive_predictor(residual_list, PREDICTION_SIZE)
    eval_metrics = evaluate_predictions(test_residuals_list, denormalized_predictions, naive_preds)

    # Print evaluation metrics for GRU
    gru_mse_list, gru_mae_list, gru_r2_list, gru_smape_list = eval_metrics["model"]
    print(f"{MODEL} RESIDUAL MEAN SMAPE: {np.mean(gru_smape_list):.4f}")

    # Print evaluation metrics for Naive
    naive_mse_list, naive_mae_list, naive_r2_list, naive_smape_list = eval_metrics["naive"]
    print(f"NAIVE RESIDUAL MEAN SMAPE: {np.mean(naive_smape_list):.4f}")

    reconstructed_series = reconstruct_series(trend_list, seasonal_list, denormalized_predictions, PREDICTION_SIZE)
    print()
    print(f"{MODEL} Final reconstruction")
    print(f"{MODEL} Median SMAPE: {calculate_median_smape(original_series_list, reconstructed_series, PREDICTION_SIZE)}")
    print(f"{MODEL} Mean SMAPE: {calculate_mean_smape(original_series_list, reconstructed_series, PREDICTION_SIZE)}")

    print()
    print("NAIVE: Final reconstruction")
    reconstructed_naive_series = reconstruct_series(trend_list, seasonal_list, naive_preds, PREDICTION_SIZE)
    print(f"NAIVE Median SMAPE: {calculate_median_smape(original_series_list, reconstructed_naive_series, PREDICTION_SIZE)}")
    print(f"NAIVE Mean SMAPE: {calculate_mean_smape(original_series_list, reconstructed_naive_series, PREDICTION_SIZE)}")


if __name__ == "__main__":
    main()
