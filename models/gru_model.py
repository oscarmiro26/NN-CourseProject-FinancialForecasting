# flake8: noqa
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1,
                 output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device)

        # Forward propagate the GRU
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Function to create and return the model
def get_gru_model(input_size=1, hidden_size=50, num_layers=1, output_size=1):
    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    return model
