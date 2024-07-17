'''
Generates a single experiment for the advection-diffusion dataset: 
a 1D pipe.

1 experiments (at Pe = 2)
80 sensors along the pipe 
100 time steps

If SAVE = True, saves to specified folder. 
Note: data is already supplied in respective folders; 
	  rerunning this code will generate a different 
	  (but similar) dataset.
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import special
from scipy import io
from scipy.stats import qmc

from copy import copy, deepcopy
from utils import shufflematrix, reorganizematrix

from utils import shufflematrix2D, reorganizematrix, plot, simulate, NC, NL, NT, LENGTH

FOLDER = "./1_80_100/"
SAVE = False

# initial parameters
# dimensionless variable: Pe = U*L/D
U = .2 # m/min
D = .1 # m2/min

if __name__ == "__main__":

	condition = [U, D]
	data = simulate(condition)
	print(data.shape)

	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,3))
	plot(axes, data, condition)
	plt.show()

	peclet = (condition[0]*LENGTH/condition[1])
	original_dataset = np.copy(data)

	z_label, t_label, shuffled_dataset = shufflematrix2D(data)
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,3))
	plot(axes, shuffled_dataset, condition)
	plt.show()

	if SAVE:
		io.savemat(f"{FOLDER}labels.mat", {'z':z_label, 't': t_label})
		io.savemat(f'{FOLDER}2dpipe_Peclet2.mat', {'Data': shuffled_dataset})
		io.savemat(f'{FOLDER}2dpipe_Peclet2_original.mat', {'Data': original_dataset})

