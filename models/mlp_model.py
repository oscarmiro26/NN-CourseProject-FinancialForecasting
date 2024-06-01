# flake8: noqa
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.hidden_size = hidden_size

        # Define the MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Flatten the input tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Function to create and return the model
def get_mlp_model(input_size, hidden_size, output_size):
    return MLPModel(input_size, hidden_size, output_size)
