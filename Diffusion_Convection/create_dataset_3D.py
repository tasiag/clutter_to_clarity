'''
Creates the advection-diffusion dataset: 
a 1D pipe.

64 experiments (varying D & U)
80 sensors along the pipe 
100 time steps

If SAVE = True, saves to specified folder. 
Note: data is already supplied in respective folders; 
	  rerunning this code will generate a different 
	  (but similar) dataset.
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import io
from scipy.stats import qmc

from copy import copy, deepcopy
from utils import shufflematrix, reorganizematrix, plot, simulate, NC, NL, NT, LENGTH


FOLDER = "./64_80_100/"
SAVE = False

if __name__ == "__main__":

	dataset = np.empty((NC**2, NL, NT))
	print("Original shape: ", dataset.shape)

	sampler = qmc.LatinHypercube(d=2)
	sample = sampler.random(n=NC**2)

	l_bounds = [0.1, 0.005] # bounds of diffusion coefficient
	u_bounds = [3.0, 1] # bounds of velocity 
	experiment_set = qmc.scale(sample, l_bounds, u_bounds)
	experiment_set += np.random.normal(0, 0.01, size=(NC**2,2))

	peclet = [None] * len(experiment_set)

	for i, condition in enumerate(experiment_set):
		zandt = simulate(condition)
		dataset[i, :, :] = zandt
		peclet[i] = (condition[0]*LENGTH/condition[1])

	# obtain peclet numbers for each dataset and sort by peclet
	peclet_order = np.argsort(peclet)
	peclet = np.array(peclet)
	peclet_sorted = peclet[peclet_order]

	# obtain original, sorted dataset
	original_dataset = np.copy(dataset[peclet_order,:,:])

	# shuffled dataset
	ex_label, z_label, t_label, shuffled_dataset = shufflematrix(dataset)


	# visualize original data
	fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(12,4))
	for i, experiment_no in enumerate((0, 4, 12, 23, 34, 45, 51, 59, 60)):
		plot(axes[i], dataset[experiment_no, :, :].squeeze(), experiment_set[experiment_no])
	plt.suptitle("Original Data")
	plt.tight_layout()
	plt.savefig("OriginalData.png")
	plt.show()

	# visualize some shuffled experiments
	fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(12,4))
	for i, experiment_no in enumerate((0, 4, 12, 23, 34, 45, 51, 59, 60)):
		plot(axes[i], shuffled_dataset[experiment_no, :, :].squeeze(), experiment_set[experiment_no], label_peclet=False)
	plt.suptitle("Shuffled data")
	plt.tight_layout()
	plt.savefig("ExperimentsShuffled.png")
	plt.show()

	if SAVE:

		io.savemat(f"{FOLDER}labels.mat", {'z':z_label, 't': t_label, 'ex': ex_label})
		io.savemat(f"{FOLDER}peclet_ordered.mat", {'peclet': peclet_sorted})
		io.savemat(f'{FOLDER}3dpipe.mat', {'Data': shuffled_dataset})
		io.savemat(f'{FOLDER}3dpipe_original.mat', {'Data': original_dataset})
		io.savemat(f'{FOLDER}2dpipe_1experiment.mat', {'matrix': shuffled_dataset[0:1,:,:].squeeze()})



