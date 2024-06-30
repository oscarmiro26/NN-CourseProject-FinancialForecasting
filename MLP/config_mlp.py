
import os
import torch.nn as nn

from util.util import smape_loss
base_path = os.path.dirname(os.path.abspath(__file__))

LOOK_BACK = 8                 # Number of past datapoints to consider for training at once
PREDICTED_DATA_POINTS = 1

MODEL = 'MLP'                   # Model we use

# Define loss function as a parameter
LOSS_FUNCTION = smape_loss

# Hyperparameters
BATCH_SIZE = 64                 # Larger batch sizes can make training faster and more stable but may require more memory
NUM_EPOCHS = 100                # Number of epochs for training
HIDDEN_SIZE = 2                # Number of hidden units (only one hidden layer now)
LEARNING_RATE = 0.001           # Learning rate for the optimizer
PATIENCE = 50                   # Number of epochs with no improvement before stopping

# Grid search parameters
GRID_SEARCH_PARAMS = {
    'batch_size': [32, 50],
    'num_epochs': [50, 100],
    'hidden_size': [10, 32, 50],
}

# Define the path for saving and loading the model in the MLP directory
import os
base_path = os.path.dirname(os.path.abspath(__file__))
mlp_dir = os.path.join(base_path, '..', 'MLP')
os.makedirs(mlp_dir, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(mlp_dir, 'mlp_model_tuned.pth')

TRAIN_MODEL = True
VERIFY_PREPROCESSING = False