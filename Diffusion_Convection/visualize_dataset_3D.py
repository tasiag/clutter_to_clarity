'''
Visualizes the advection-diffusion dataset (a 1D pipe).

64 experiments (varying D & U)
80 sensors along the pipe 
100 time steps

Uses data created by create_dataset_3D.py
and in Diffusion_Convection/64_80_100 folder
as well as embedding outputs from questionnaires
in Diffusion_Convection/64_80_100/Output folder.

Saves all figures to specified folder:
Diffusion_Convection/64_80_100/Figures

If SAVE = True, saves out embeddings (from questionnaires)
in format fit for neural networks.
'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial.distance import pdist

from copy import copy, deepcopy

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', family='serif')

SAVE = False

FOLDER = "./Diffusion_Convection/64_80_100/"

TIME = 2 # minutes
LENGTH = 1 # meters

NT = 100
NL = 80

## ex x z x t

## ORIGINAL DATA (preshuffled)
original = scipy.io.loadmat(f"{FOLDER}3dpipe_original.mat")
peclet = scipy.io.loadmat(f"{FOLDER}peclet_ordered.mat")["peclet"][0]
data_original = original["Data"]

## SHUFFLED DATA

# "truth" behind shuffled data (our "truth" that is hidden)
labels = scipy.io.loadmat(f"{FOLDER}labels.mat") # the correct ordering
ex = labels["ex"][0] # this is the index order of which the peclet numbers have been sorted. To get which peclets refer to which, use peclet[ex]
z = labels["z"][0] # index order of space
t = labels["t"][0] # index order of time

## questionnaire labels (diffusion coordinates))
shuffled = scipy.io.loadmat(f"{FOLDER}3dpipe.mat")["Data"]
embedding = scipy.io.loadmat(f"{FOLDER}Output/embedding.mat") 
ex_embedding = embedding["embedding"][0, 0]
z_embedding = embedding["embedding"][1, 0]
t_embedding = embedding["embedding"][2, 0]


def plot(axs, data, P, label_peclet = True, title=""):
	axs.imshow(data.T, origin="lower")
	plt.suptitle(title)
	if label_peclet: axs.set_title(f"{round(P,3)}")
	return axs

def conv_diff_plot(original, shuffled, pe):
	fig = plt.figure()
	gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

	axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, i]) for i in range(3)]
	for i in range(3):
		im = axes[i].imshow(original[i, :, :].squeeze().T, origin="lower", extent=[0, LENGTH, 0, TIME], aspect='auto')
		axes[i].set_title(f'Pe: {round(pe[i], 2)}')
		if i == 0: 
			axes[i].set_yticks([0, 2])
		else: 
			axes[i].set_yticks([])
		axes[i].set_xticks([0, 1])
		axes[i].set_xlabel(r'$\hat{z}$', labelpad=-5)

	for i in range(3):
		axes[i + 3].imshow(shuffled[i, :, :].squeeze().T, origin="lower", aspect='auto')
		if i == 0:
			axes[i + 3].set_yticks([0, 100])
		else: 
			axes[i + 3].set_yticks([])
		axes[i + 3].set_xticks([0, 80])
		axes[i + 3].set_xlabel("z index", labelpad=-5)

    # Adding common y-axis labels
	fig.text(0.1, 0.75, r'$\hat{t}$', ha='center', va='center', rotation='vertical')
	fig.text(0.1, 0.3, 't index', ha='center', va='center', rotation='vertical')

    # Add the color bar
	cax = fig.add_axes([0.86, 0.15, 0.02, 0.75])
	cbar = plt.colorbar(im, cax=cax)
	cbar.ax.set_ylabel(r'$\hat{C}$', labelpad=-10)  # Set the label for the color bar
	cbar.set_ticks([0, 0.99])

	plt.subplots_adjust(hspace=0.3, wspace=0.32, top=0.9, bottom=0.15, left=0.15, right=0.82)
	plt.show()

	fig.savefig(f"{FOLDER}/Figures/conv_diff_data.pdf", dpi=300, format='pdf')

def embedding_plot_truth(ex_embedding, z_embedding, t_embedding, ex, z, t):
	# diffusion map embeddings
	fig, axs = plt.subplots(nrows=1, ncols=3)
	axs[0].scatter(ex_embedding[:, 0], ex, c=ex)
	axs[0].set_xlabel(r'$\hat{\pi}$')
	axs[0].set_ylabel(r'Péclet')

	axs[1].scatter(z_embedding[:, 0], z, c=z)
	axs[1].set_xlabel(r'$\hat{\zeta}$')
	axs[1].set_ylabel(r'$\hat{z}$')

	axs[2].scatter(t_embedding[:, 0], t, c=t)
	axs[2].set_xlabel(r'$\hat{\tau}$')
	axs[2].set_ylabel(r'$\hat{t}$')
	plt.subplots_adjust(hspace=0.6, wspace=0.5, top=0.5, bottom=0.14, left=0.1, right=0.99)

	plt.show()
	fig.savefig(f"{FOLDER}/Figures/conv_diff_emb_truth.pdf", dpi=300, format='pdf')


def original_vs_embedding_plot(original, ex_embedding, z_embedding, t_embedding, partially_shuffled, or_x, or_t):

	print(original.shape)
	print(partially_shuffled.shape)

	x = np.zeros(len(z_embedding)*len(t_embedding))
	t = np.zeros(len(z_embedding)*len(t_embedding))
	U = np.zeros(len(z_embedding)*len(t_embedding))

	count = 0
	for i, x_ev in enumerate(z_embedding):
		for j, t_ev in enumerate(t_embedding):
			U[count] = partially_shuffled[i, j]
			x[count] = x_ev[0]
			t[count] = t_ev[0]
			count +=1

	X = np.array(np.vstack((x, t)).T)

	# plot to make sure it's the same
	fig, axs = plt.subplots(nrows=1, ncols=2)
	axs[1].scatter(x, t, c=U, s=1)
	axs[1].set_xlabel(r"$\hat{\zeta}$")
	axs[1].set_ylabel(r"$\hat{\tau}$")
	axs[1].set_aspect('equal')#, 'box')


	count = 0
	U = np.zeros(len(or_x)*len(or_t))
	for i, x_ev in enumerate(or_x):
		for j, t_ev in enumerate(or_t):
			U[count] = original[i, j]
			x[count] = x_ev
			t[count] = t_ev
			count +=1

	X = np.array(np.vstack((x, t)).T)

	axs[0].scatter(x, t, c=U, s=1)
	axs[0].set_xlabel(r"$\hat{z}$")
	axs[0].set_ylabel(r"$\hat{t}$")
	axs[0].set_aspect('equal')#, 'box')


	plt.subplots_adjust(hspace=0.6, wspace=-0.5, top=0.5, bottom=0.14, left=0.1, right=0.99)

	plt.show()
	fig.savefig(f"{FOLDER}/Figures/original_vs_embedd.pdf", dpi=300, format='pdf')



	plt.show()

'''
##################################################
		 
		PLOTS FOR PUBLICATION

##################################################
'''

indices_to_keep = [1, 23, 55]

partially_shuffled = np.copy(shuffled[np.argsort(ex),:,:])
conv_diff_plot(data_original[indices_to_keep, :, :], partially_shuffled[indices_to_keep, :, :], peclet[indices_to_keep])

time = np.linspace(0,TIME,NT)
space = np.linspace(0.001,LENGTH,NL)

embedding_plot_truth(ex_embedding, z_embedding, t_embedding, peclet[ex], space[z], time[t])

keep = 43
original_vs_embedding_plot(data_original[keep, :, :], ex_embedding, z_embedding, t_embedding, partially_shuffled[keep, :, :], space, time)

# U: shuffled 80 x 100
# f: ex_embedding[:,0] (first eigenvector)
# x: z_embedding[:,0] (first eigenvector)
# t: t_embedding[:,0] (first eigenvector)
print(shuffled.shape)

if SAVE:
	np.save(f'{FOLDER}ForNN/U.npy', shuffled)
	np.save(f'{FOLDER}ForNN/x_evec.npy', z_embedding[:,0])
	np.save(f'{FOLDER}ForNN/t_evec.npy', t_embedding[:,0])
	np.save(f'{FOLDER}ForNN/ex_evec.npy', ex_embedding[:,0])
