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
from util.util import plot_predictions, denormalize_predictions, evaluate_predictions, reconstruct_series, plot_actual_vs_predicted, calculate_median_smape
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




def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
     # Create data loaders for training, validation, and test sets
    # DataLoader helps manage batches of data efficiently during training and evaluation
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader




def start_to_train_model(model, train_loader, val_loader):
    # Train the model if the TRAIN_MODEL flag is set to True
    if TRAIN_MODEL:
        print('Training model...')
        train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, LOSS_FUNCTION, MODEL_SAVE_PATH, PATIENCE)
    else:
        # Load the pre-trained model weights
        print('Loading saved model...')
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))




def main():
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
   
    model = ModelFactory.create_model(MODEL, LOOK_BACK, HIDDEN_SIZE_1, HIDDEN_SIZE_2, PREDICTED_DATA_POINTS).to(device)

    # start the training process
    start_to_train_model(model, train_loader, val_loader)

    # Generate predictions for future data points using the trained model
    print('Generating predictions...')
    predictions = generate_predictions(model, scaled_all_data, LOOK_BACK, EVAL_PREDICTION_SIZE, device)

    # Denormalize the predictions to convert them back to the original scale
    print('Denormalizing predictions...')
    denormalized_predictions = denormalize_predictions(predictions, scalers)

    # Evaluate the model's predictions against the actual data of the last 18 points
    print('Evaluating predicted vs actual residual predictions...')
    mse_list, mae_list, r2_list, smape_list = evaluate_predictions(eval_residuals_list, denormalized_predictions)
    
    # Plot the predictions against the actual data to visualize performance
    print('Plotting residual predictions vs actual...')
    #plot_predictions(eval_residuals_list, denormalized_predictions, EVAL_PREDICTION_SIZE)

    # reconstructing the series
    reconstructed_series = reconstruct_series(trend_list, seasonal_list, denormalized_predictions, EVAL_PREDICTION_SIZE)

    #plot_actual_vs_predicted(original_series, reconstructed_series, EVAL_PREDICTION_SIZE)
    
    print("Final SMAPE score:")
    print(calculate_median_smape(original_series, reconstructed_series, EVAL_PREDICTION_SIZE))


if __name__ == "__main__":
    main()
