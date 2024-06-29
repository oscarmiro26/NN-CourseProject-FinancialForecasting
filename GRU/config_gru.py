
import os
import torch
import torch.nn as nn

base_path = os.path.dirname(os.path.abspath(__file__))

LOOK_BACK = 27                  # Number of past datapoints to consider for training at once
PREDICTED_DATA_POINTS = 1

MODEL = 'GRU'                   # Model we use


class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, y_pred, y_true):
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        diff = torch.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0  # Avoid division by zero
        return torch.mean(diff) * 100
    

# Define loss function as a parameter
LOSS_FUNCTION = SMAPE()

# Hyperparameters
BATCH_SIZE = 100                 # Larger batch sizes can make training faster and more stable but may require more memory
NUM_EPOCHS = 200                 # Number of epochs for training
INPUT_SIZE = 1
NUM_LAYERS = 1
HIDDEN_SIZE_1 = 50               # Number of hidden units
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001          # Learning rate for the optimizer
PATIENCE = 50                    # Number of epochs with no improvement before stopping

# Grid search parameters
GRID_SEARCH_PARAMS = {
    'batch_size': [32, 50],
    'num_epochs': [50, 100],
    'hidden_size': [10, 32, 50],
}

# Define the path for saving and loading the model in the MLP directory
mlp_dir = os.path.join(base_path, '..', 'GRU')

os.makedirs(mlp_dir, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(mlp_dir, 'gru_model.pth')

TRAIN_MODEL = True
VERIFY_PREPROCESSING = False