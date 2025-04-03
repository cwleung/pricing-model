import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetworkMean(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            self.hidden_layers.append(nn.ReLU())
            # self.hidden_layers.append(nn.Dropout(0.1))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x