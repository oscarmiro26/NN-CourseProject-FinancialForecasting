# Configuration for data preprocessing
import os
# Define the base path as the directory containing this configuration file
base_path = os.path.dirname(os.path.abspath(__file__))

# Define the path to the input_data directory relative to the base path
input_data_path = os.path.join(base_path, '..', 'input_data')

# Define the full path to the data file relative to the input_data directory
DATA_FILE = os.path.join(input_data_path, 'M3C_Monthly_FINANCE.csv')




# Window size for moving average
TREND_CALCULATION_WINDOW = 12  

# Verification flag
VERIFY_PREPROCESSING = False

# Configuration parameters
VALIDATION_SPLIT = 0.1          # Percentage of data to be used for validation
PREDICTION_SIZE = 18       # Number of future datapoints to predict and hold out for evaluation
