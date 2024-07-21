"""
Utilities for Oscillator dataset. 
Includes shuffling and reorganizing data matrix.
"""

import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from scipy.integrate import ode
from scipy import interpolate

import matthews as mint

torch.set_default_dtype(torch.float32)


# this shuffles a 3d matrix
def shufflematrix(matrix):
    # Create a copy of the tensor to avoid modifying the original
    shuffled_tensor = np.copy(matrix)

    indices_0 = np.arange(matrix.shape[0])
    np.random.shuffle(indices_0)
    shuffled_tensor = shuffled_tensor[indices_0, :, :]

    indices_1 = np.arange(matrix.shape[1])
    np.random.shuffle(indices_1)
    shuffled_tensor = shuffled_tensor[:, indices_1, :]

    indices_2 = np.arange(matrix.shape[2])
    np.random.shuffle(indices_2)
    shuffled_tensor = shuffled_tensor[:, :, indices_2]

    return indices_0, indices_1, indices_2, shuffled_tensor

def reorganizematrix(matrix, indices_0, indices_1, indices_2):
    # Create a copy of the tensor to avoid modifying the original
    deshuffled_tensor = np.copy(matrix)

    deshuffled_tensor = deshuffled_tensor[np.argsort(indices_0), :, :]
    deshuffled_tensor = deshuffled_tensor[:, np.argsort(indices_1), :]
    deshuffled_tensor = deshuffled_tensor[:, :, np.argsort(indices_2)]

    return deshuffled_tensor