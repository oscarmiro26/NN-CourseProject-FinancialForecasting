import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1):
        super(MLPModel, self).__init__()
        self.hidden_size = hidden_size

        # Define the input layer
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        # Define the hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Define the output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out

# Function to create and return the model
def get_mlp_model(input_size, hidden_size, output_size, num_hidden_layers=1):
    return MLPModel(input_size, hidden_size, output_size, num_hidden_layers)
