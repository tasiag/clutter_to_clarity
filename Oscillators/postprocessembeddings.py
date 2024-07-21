"""
Script to post-processes questionnaires embedding in order 
to take 2D time and space embeddings to single dimension
by running diffusion maps again.

To save post-processed embeddings, make SAVE = True

Run from Oscillators folder (or change PATH).
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import math

import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from scipy.spatial.distance import pdist

PATH = "./dat_0to35_1.08to1.8"

SAVE = True

if __name__ == "__main__":

    data = np.load(f"{PATH}/3D_data.npy")
    shuffled_data = np.load(f"{PATH}/3D_data_shuffled.npy")
    shuffled_data = shuffled_data.reshape(data.shape[0], -1)

    data = np.load(f"{PATH}/3D_data.npy")
    shuffled_data = np.load(f"{PATH}/3D_data_shuffled.npy")

    # load true labels
    K = np.load(f"{PATH}/True_K.npy")
    K_labels = io.loadmat(f"{PATH}/labels.mat")["K"][0]
    space_labels = io.loadmat(f"{PATH}/labels.mat")["x"][0]
    time_labels = io.loadmat(f"{PATH}/labels.mat")["t"][0]

    # load embeddings
    K_emb = io.loadmat(f"{PATH}/Output/embedding.mat")["embedding"][0,0]
    X_emb = io.loadmat(f"{PATH}/Output/embedding.mat")["embedding"][1,0][:,0:2]
    T_emb = io.loadmat(f"{PATH}/Output/embedding.mat")["embedding"][2,0][:,0:3]

    # plot space #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].set_title("Space")
    scatter0 = axes[0].scatter(X_emb[:, 0], X_emb[:, 1], c=space_labels)
    fig.colorbar(scatter0, ax=axes[0])
    axes[0].set_xlabel(r"$\phi_1$")
    axes[0].set_ylabel(r"$\phi_2$")

    axes[1].set_title("Space")
    scatter1 = axes[1].scatter(X_emb[:, 0], space_labels, c=space_labels)
    fig.colorbar(scatter1, ax=axes[1])
    axes[1].set_xlabel(r"$\phi_1$")
    axes[1].set_ylabel(r"space")

    axes[2].set_title("Space")
    scatter2 = axes[2].scatter(X_emb[:, 1], space_labels, c=space_labels)
    fig.colorbar(scatter2, ax=axes[2])
    axes[2].set_xlabel(r"$\phi_2$")
    axes[2].set_ylabel(r"space")

    plt.tight_layout()
    plt.show()

    # diffusion maps to extract arclength #

    X_pcm = pfold.PCManifold(X_emb)
    print("Space epsilon: ", X_pcm.optimize_parameters()[0])
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=X_pcm.optimize_parameters()[0], distance=dict(cut_off=1)),
        n_eigenpairs=2,
    )
    dmap = dmap.fit(X_pcm)
    x_evecs, x_evals = dmap.eigenvectors_, dmap.eigenvalues_

    plt.title("Space")
    plt.scatter(x_evecs[:,1], space_labels, c=space_labels)
    plt.colorbar()
    plt.xlabel(r"$\phi_{\phi_1}$ with datafold")
    plt.ylabel(r"x_label")
    plt.show()

    # plot time #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].set_title("Time")
    scatter0 = axes[0].scatter(T_emb[:, 0], T_emb[:, 1], c=time_labels)
    fig.colorbar(scatter0, ax=axes[0])
    axes[0].set_xlabel(r"$\psi_1$")
    axes[0].set_ylabel(r"$\psi_2$")

    axes[1].set_title("Time")
    scatter1 = axes[1].scatter(T_emb[:, 0], time_labels, c=time_labels)
    fig.colorbar(scatter1, ax=axes[1])
    axes[1].set_xlabel(r"$\psi_1$")
    axes[1].set_ylabel(r"time")

    axes[2].set_title("Time")
    scatter2 = axes[2].scatter(T_emb[:, 1], time_labels, c=time_labels)
    fig.colorbar(scatter2, ax=axes[2])
    axes[2].set_xlabel(r"$\psi_2$")
    axes[2].set_ylabel(r"time")

    plt.tight_layout()
    plt.show()

    # diffusion maps to extract arclength #
    T_pcm = pfold.PCManifold(T_emb)
    print("Time epsilon: ", T_pcm.optimize_parameters()[0])
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=T_pcm.optimize_parameters()[0]/100, distance=dict(cut_off=.1)),
        n_eigenpairs=2,
    )
    dmap = dmap.fit(T_pcm)
    t_evecs, t_evals = dmap.eigenvectors_, dmap.eigenvalues_

    plt.title("Time")
    plt.scatter(t_evecs[:,1], time_labels, c=time_labels)
    plt.colorbar()
    plt.xlabel(r"$\psi_{\psi_1}$ with datafold")
    plt.ylabel(r"t_label")
    plt.show()


    if SAVE:
        np.save(f'{PATH}/ForNN/U.npy', shuffled_data)
        np.save(f'{PATH}/ForNN/x_evec.npy', x_evecs[:,1])
        np.save(f'{PATH}/ForNN/t_evec.npy', t_evecs[:,1])
        np.save(f'{PATH}/ForNN/K_evec.npy', K_emb[:,0:2])

