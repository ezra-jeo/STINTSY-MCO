import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list, output_size: int): 
        super().__init__()

        layer = []

        layer.append(nn.Flatten())

        sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(sizes) - 2):
            layers = [
                nn.Linear(sizes[i], sizes[i+1]),
                nn.ReLU()
            ]
            layer.extend(layers)

        layer.append(nn.Linear(sizes[-2], sizes[-1]))

        self.model = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.model(x)