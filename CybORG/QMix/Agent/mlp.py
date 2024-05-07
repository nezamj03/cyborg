import torch.nn as nn
import torch.nn.functional as F

class MLPAgent(nn.Module):

    def __init__(self, input_shape, n_actions, args):
        super(MLPAgent, self).__init__()
        self.args = args
        layer_sizes = [input_shape] + self.args.hidden_layers + [n_actions]
        self.layers = nn.ModuleList()
        if self.args.activation == 'relu':
            self.activation = F.relu
        else:
            raise KeyError(f'support for {self.args.activation} not yet implemented')

        # Create layers based on the configuration
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i < len(self.layers) - 1:  # Apply activation function to all but the output layer
                inputs = self.activation(inputs)
        return inputs
