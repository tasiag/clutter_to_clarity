'''

Utilities for 
(a) simulating the advection-diffusion equation
and 
(b) shuffling and reorganizing the resulting datasets.

'''

import numpy as np

from scipy import special

# constants
TIME = 2 # minutes
LENGTH = 1 # meters

# dimensions
NT = 100
NL = 80
NC = 8 # will have NC^2 experiments

def shufflematrix2D(matrix):
	'''
	Shuffles a 2D matrix. Returns the proper ordering 
	for each dimension, as well as the shuffled matrix.
	'''
	
	# Avoid modifying the original
	shuffled_tensor = np.copy(matrix)

	indices_0 = np.arange(matrix.shape[0])
	np.random.shuffle(indices_0)
	shuffled_tensor = shuffled_tensor[indices_0, :]

	indices_1 = np.arange(matrix.shape[1])
	np.random.shuffle(indices_1)
	shuffled_tensor = shuffled_tensor[:, indices_1]

	return indices_0, indices_1, shuffled_tensor

def shufflematrix(matrix):
	'''
	Shuffles a 3D matrix. Returns the proper ordering 
	for each dimension, as well as the shuffled matrix.
	'''
	
	# Avoid modifying the original
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
	'''
	Reorganizes a scrambled 3D matrix given proper labels for 
	each axis. Returns reorganized matrix.
	'''
	deshuffled_tensor = np.copy(matrix)
	deshuffled_tensor = deshuffled_tensor[np.argsort(indices_0), :, :]
	deshuffled_tensor = deshuffled_tensor[:, np.argsort(indices_1), :]
	deshuffled_tensor = deshuffled_tensor[:, :, np.argsort(indices_2)]
	return deshuffled_tensor


def c(z, t, U, D):
	'''
	Solves for tracer concentration at a given z, t
	'''
	Pe = U*LENGTH/D
	x = (z-U*t)/np.sqrt(4*D*t)
	y = (z/LENGTH - t/TIME)/np.sqrt(4*t/TIME/Pe)
	if not np.isnan(np.max(x-y)):
		assert np.max(x-y) < 1E-12, f"oops {np.max(x-y)}"

	return 0.5*(1-special.erf(y))


def simulate(P):
	'''
	Solves for tracer concentration at all z, t given 
	a parameter P=[diffusion coefficient, velocity]
	'''
	data = []
	time = np.linspace(0,TIME,NT)
	for z in np.linspace(0.001,LENGTH,NL):
		c_zt = c(z, time, P[0], P[1])
		data.append(c_zt)
	return np.array(data)

def plot(axs, data, P, label_peclet = True):
	axs.imshow(data.T, origin="lower", extent = [0, LENGTH, 0, TIME])
	axs.set_ylabel("time (m)")
	axs.set_xlabel("z (m)")
	if label_peclet: axs.set_title(f"Pe:{round(P[0]*LENGTH/P[1],2)}")
	return axs