import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from collections import OrderedDict

torch.manual_seed(1234)

class DenseNN(nn.Module):
    def __init__(self, layers = [2, 40, 40, 40, 40, 1], init_func=nn.init.normal_):
        super(DenseNN, self).__init__()

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


        self.depth = len(layers) - 1
        self.activation = nn.LeakyReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth-1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict).to(self.device)

        l = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                l += 1
                in_dim = layers[l-1]
                out_dim = layers[l]
                std = np.sqrt(2./(in_dim + out_dim))
                init_func(layer.weight, mean=0, std=std)
                torch.nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        x = x.to(self.device)
        out = self.layers(x)
        return out
    
    def predict(self, X):
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            predictions = self.forward(X).cpu().numpy()
        return predictions

    def predict(self, x, t):
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            t_tensor = torch.from_numpy(t).float().to(self.device)
            inputs = torch.cat((x_tensor, t_tensor), dim=1).to(self.device)
            predictions = self.forward(inputs).cpu().numpy()
        return predictions
    
    def get_weights_and_biases(self):
        weights = []
        biases = []

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                weights.append(layer.state_dict()['weight'])
                biases.append(layer.state_dict()['bias'])

        return weights, biases
    
    def print_weights(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                print(layer.state_dict()['weight'])

    def print_biases(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                print(layer.state_dict()['bias'])

    def print_weights_and_biases(self):
        self.print_weights()
        self.print_biases()

    def set_weights_and_biases(self, weights, biases):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = weights.pop(0)
                layer.bias.data = biases.pop(0)