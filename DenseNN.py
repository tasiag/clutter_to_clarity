import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict

torch.manual_seed(1234)

class DenseNN(nn.Module):
    def __init__(self, layers = [2, 40, 40, 40, 40, 1]):
        super(DenseNN, self).__init__()

        self.depth = len(layers) - 1
        self.activation = nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth-1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        out = self.layers(x)
        return out

    def predict(self, x, t):
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float()
            t_tensor = torch.from_numpy(t).float()
            inputs = torch.cat((x_tensor, t_tensor), dim=1)
            predictions = self.forward(inputs).numpy()
        return predictions
    
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