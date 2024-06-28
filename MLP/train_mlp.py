import sys
import os

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
from util.util import plot_predictions, denormalize_predictions, evaluate_predictions, reconstruct_series, plot_actual_vs_predicted, calculate_median_smape, naive_predictor, plot_prediction_errors, calculate_mean_smape
from config_mlp import *


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, loss_function, model_save_path, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    total_iterations = len(train_loader) * num_epochs  # Total number of iterations

    pbar = tqdm(total=total_iterations, desc="Training Progress", leave=True)
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_Y in val_loader:
                val_outputs = model(val_X).view(-1)  # Ensure output tensor is flattened
                val_Y = val_Y.view(-1)  # Ensure target tensor is flattened
                val_loss += loss_function(val_outputs, val_Y).item()
        val_loss /= len(val_loader)

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

    # Preprocess the data and create training, validation, and test datasets
    print('Creating datasets...')
    datasets = create_datasets(LOOK_BACK)
    X_train, Y_train, X_val, Y_val, X_test, Y_test, trend_list, seasonal_list, test_residuals_list, original_series_list, residual_list, train_scalers, val_scalers, test_scalers, scaled_all_residuals_list = datasets
    
    train_loader, val_loader = create_dataloaders(X_train, Y_train, X_val, Y_val)
   
    print('Creating model...')
    model = ModelFactory.create_model('MLP', LOOK_BACK, HIDDEN_SIZE, PREDICTED_DATA_POINTS).to(device)

    start_to_train_model(model, train_loader, val_loader)


    print('Generating predictions...')
    predictions = generate_predictions(model, scaled_all_residuals_list, LOOK_BACK, PREDICTION_SIZE, device)
    
    if not predictions:
        print("No predictions generated.")
        return

    
    print('Denormalizing predictions...')
    denormalized_predictions = denormalize_predictions(predictions, test_scalers)
    
    print('Generating naive predictions...')
    naive_preds = naive_predictor(residual_list, PREDICTION_SIZE)

    print('Evaluating predicted vs actual residual predictions...')
    # Accessing the first series from all lists
    first_test_residuals = test_residuals_list[0]
    first_denormalized_predictions = denormalized_predictions[0]
    first_naive_preds = naive_preds[0]

    # Printing the first series
    print("First series of test residuals:")
    print(first_test_residuals)

    print("\nFirst series of denormalized predictions:")
    print(first_denormalized_predictions)

    print("\nFirst series of naive predictions:")
    print(first_naive_preds)

    eval_metrics = evaluate_predictions(test_residuals_list, denormalized_predictions, naive_preds)

    # Print evaluation metrics for MLP
    mlp_mse_list, mlp_mae_list, mlp_r2_list, mlp_smape_list = eval_metrics["mlp"]
    print("MLP Predictions:")
    print(f"Mean MSE: {np.mean(mlp_mse_list):.4f}")
    print(f"Mean MAE: {np.mean(mlp_mae_list):.4f}")
    print(f"Mean R2: {np.mean(mlp_r2_list):.4f}")
    print(f"Mean SMAPE: {np.mean(mlp_smape_list):.4f}")

    # Print evaluation metrics for Naive
    naive_mse_list, naive_mae_list, naive_r2_list, naive_smape_list = eval_metrics["naive"]
    print("Naive Predictions:")
    print(f"Mean MSE: {np.mean(naive_mse_list):.4f}")
    print(f"Mean MAE: {np.mean(naive_mae_list):.4f}")
    print(f"Mean R2: {np.mean(naive_r2_list):.4f}")
    print(f"Mean SMAPE: {np.mean(naive_smape_list):.4f}")

    #print('Plotting residual predictions vs actual...')
    #plot_prediction_errors(residual_list, denormalized_predictions, PREDICTION_SIZE)
    """plot_predictions(
        actual_full_list=residual_list,  # The actual full residuals
        predicted_list=denormalized_predictions,  # The denormalized model predictions
        naive_predictions=naive_preds,  # The naive predictions
        num_points=PREDICTION_SIZE,  # The number of prediction points
        extra_context_points=30  # The number of extra context points (adjust as needed)
    ) """

    reconstructed_series = reconstruct_series(trend_list, seasonal_list, denormalized_predictions, PREDICTION_SIZE)
    #plot_prediction_errors(original_series_list, reconstructed_series, PREDICTION_SIZE)
    
    print("Final SMAPE score MLP:")
    print(calculate_median_smape(original_series_list, reconstructed_series, PREDICTION_SIZE))
    print(calculate_mean_smape(original_series_list, reconstructed_series, PREDICTION_SIZE))
    #plot_actual_vs_predicted(original_series_list, reconstructed_series, PREDICTION_SIZE)

    print("Final SMAPE score Naive:")
    reconstructed_naive_series = reconstruct_series(trend_list, seasonal_list, naive_preds, PREDICTION_SIZE)
    print(calculate_median_smape(original_series_list, reconstructed_naive_series, PREDICTION_SIZE))
    print(calculate_mean_smape(original_series_list, reconstructed_naive_series, PREDICTION_SIZE))
    #plot_actual_vs_predicted(original_series_list, reconstructed_naive_series, PREDICTION_SIZE)

if __name__ == "__main__":
    main()
