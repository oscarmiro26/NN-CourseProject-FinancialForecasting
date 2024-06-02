import os
import torch.nn as nn

# Define the base path as the directory containing this configuration file
base_path = os.path.dirname(os.path.abspath(__file__))

# Define the path to the input_data directory relative to the base path
input_data_path = os.path.join(base_path, '..', 'input_data')

# Define the full path to the data file relative to the input_data directory
DATA_FILE = os.path.join(input_data_path, 'M3C_Monthly_FINANCE.csv')

# Configuration parameters
VALIDATION_SPLIT = 0.1          # Percentage of data to be used for validation
TEST_SPLIT = 0.1                # Percentage of data to be used for testing
EVAL_PREDICTION_SIZE = 18       # Number of future datapoints to predict and hold out for evaluation

TREND_CALCULATION_WINDOW = 12   # The rolling mean used for trend calculation
LOOK_BACK = 25                  # Number of past datapoints to consider for prediction

MODEL = 'MLP'                   # Model we use

VERIFY_PREPROCESSING = False    # Set to True to verify preprocessing steps

# Define loss function as a parameter
LOSS_FUNCTION = nn.MSELoss()

# Hyperparameters
BATCH_SIZE = 64                 # Larger batch sizes can make training faster and more stable but may require more memory
NUM_EPOCHS = 50                 # Number of epochs for training
HIDDEN_SIZE = 20               # Number of hidden units
LEARNING_RATE = 0.001          # Learning rate for the optimizer
PATIENCE = 8                    # Number of epochs with no improvement before stopping

# Grid search parameters
GRID_SEARCH_PARAMS = {
    'batch_size': [32, 50, 64],
    'num_epochs': [50, 100, 150],
    'hidden_size': [16, 32, 50],
    'learning_rate': [0.001, 0.0001],
    'patience': [5, 10]
}

# Define the path for saving and loading the model in the MLP directory
mlp_dir = os.path.join(base_path, '..', 'MLP')

os.makedirs(mlp_dir, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(mlp_dir, 'mlp_model.pth')

TRAIN_MODEL = True
VERIFY_PREPROCESSING = False
