import torch
import numpy as np
from util import *


def evaluate_single(model, data, labels):
    # Assuming X_test contains the initial sequences for each series you want to forecast
    predictions = []

    # Iterate over each test sequence
    for i in range(data.shape[0]):
        # Take the initial sequence for rolling prediction
        current_sequence = data[i:i+1, :, :]  # Ensure it maintains three dimensions

        
        # Make prediction for the next step
        with torch.no_grad():
            next_point = model(current_sequence)
            
        
        # Collect all predictions from current series
        predictions.append(next_point)


    print(len(predictions), len(labels.tolist()))
    sml = smape_loss(torch.tensor(predictions), labels)
    print(sml.item())


def evaluate_series(model, full_data, scalers, device, test_size=18, look_back=12):
    predictions = []
    actuals = []
    smape_scores = []

    for idx, series in enumerate(full_data):
        # Ensure there is enough data for the initial sequence plus the test size
        if len(series) < look_back + test_size:
            print(f"Skipping series {idx+1} due to insufficient data.")
            continue

        # Prepare the initial sequence from the last known points before the test segment
        current_sequence = torch.tensor(series[-(look_back + test_size):-test_size].reshape(1, look_back, 1), dtype=torch.float32).to(device)

        series_predictions = []
        for _ in range(test_size):
            with torch.no_grad():
                next_point = model(current_sequence)

            # Update the sequence with the newly predicted point
            next_point_value = next_point.cpu().numpy().flatten()[0]
            new_point_np = np.array([[next_point_value]])
            new_sequence = np.append(current_sequence.cpu().numpy().flatten()[1:], new_point_np).reshape(1, look_back, 1)
            current_sequence = torch.tensor(new_sequence, dtype=torch.float32).to(device)

            series_predictions.append(next_point_value)

        # Denormalize predictions
        denormalized_preds = scalers[idx].inverse_transform(np.array(series_predictions).reshape(-1, 1)).flatten()
        predictions.append(denormalized_preds)

        # Retrieve actual values and denormalize them
        actual_values = series[-test_size:]  # Last test_size elements of the series
        denormalized_actuals = scalers[idx].inverse_transform(actual_values.reshape(-1, 1)).flatten()
        actuals.append(denormalized_actuals)

        # Calculate SMAPE score
        smape_value = smape_loss_np(denormalized_actuals, denormalized_preds)
        smape_scores.append(smape_value)
        print(f"Time Series {idx+1} - SMAPE: {smape_value:.2f}%")

    return predictions, actuals, smape_scores
