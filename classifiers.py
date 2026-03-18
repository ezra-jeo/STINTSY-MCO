import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list, output_size: int, dropout=0.3):
        super().__init__()

        layers = []

        sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(sizes) - 2):
            layers.extend([
                nn.Linear(sizes[i], sizes[i+1]),
                nn.ReLU(), # relu used by default
                nn.Dropout(dropout)
            ])

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        # final acitvation (softmax/sigmoid) is handled by the loss function 

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)