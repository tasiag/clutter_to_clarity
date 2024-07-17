import torch
import torch.nn as nn
import numpy as np

from DenseNN import DenseNN

class DeepONet():
    def __init__(self, torch_data_type, trunk_layers, branch_layers):
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.data_type = torch_data_type

        self.trunk_dnn = DenseNN(trunk_layers)
        self.branch_dnn = DenseNN(branch_layers)

        self.params = []
        self.params.extend(self.trunk_dnn.parameters())
        self.params.extend(self.branch_dnn.parameters())


    def fnn_T(self, X, Xmin, Xmax):

        X = X.to(self.device)
        Xmin = Xmin.to(self.device)
        Xmax = Xmax.to(self.device)

        A = 2.0*(X - Xmin)/(Xmax - Xmin) - 1.0 
        Y = self.trunk_dnn(A)
        return Y
    
    def fnn_B(self, X):
        X = X.to(self.device)
        return self.branch_dnn(X)
    
    def __call__(self, X, F, F_norm, Xmin, Xmax):
        X = X.to(self.device)
        F = F.to(self.device)
        Xmin = Xmin.to(self.device)
        Xmax = Xmax.to(self.device)
        
        u_B = self.fnn_B(F)
        u_T = self.fnn_T(X, Xmin, Xmax)
        u_nn = torch.einsum('bij,nj->bnj', u_B, u_T)
        u_nn = torch.sum(u_nn, axis=-1, keepdims=True)

        return u_nn*F_norm

    
    
    
