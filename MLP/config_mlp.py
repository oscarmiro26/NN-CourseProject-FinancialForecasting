
import os
import torch.nn as nn

base_path = os.path.dirname(os.path.abspath(__file__))

LOOK_BACK = 5                  # Number of past datapoints to consider for training at once
PREDICTED_DATA_POINTS = 1

MODEL = 'MLP'                   # Model we use

# Define loss function as a parameter
LOSS_FUNCTION = nn.MSELoss()


# Hyperparameters
BATCH_SIZE = 100                 # Larger batch sizes can make training faster and more stable but may require more memory
NUM_EPOCHS = 200                 # Number of epochs for training
HIDDEN_SIZE_1 = 4               # Number of hidden units
HIDDEN_SIZE_2 = 2 
LEARNING_RATE = 0.001          # Learning rate for the optimizer
PATIENCE = 50                    # Number of epochs with no improvement before stopping

# Grid search parameters
GRID_SEARCH_PARAMS = {
    'batch_size': [32, 50],
    'num_epochs': [50, 100],
    'hidden_size': [10, 32, 50],
}

# Define the path for saving and loading the model in the MLP directory
mlp_dir = os.path.join(base_path, '..', 'MLP')

os.makedirs(mlp_dir, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(mlp_dir, 'mlp_model.pth')

TRAIN_MODEL = True
VERIFY_PREPROCESSING = False