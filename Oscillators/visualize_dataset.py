"""
Create figures for Oscillators. 
Run from oscillators folder.

Includes:
- data 
- embedding results

Does not include:
- DeepONet results

"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

import time
import pickle
import scipy.spatial as sp
import scipy.sparse.linalg as spsp
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize

from utils import reorganizematrix, shufflematrix

plt.rc('text', usetex=False); 
plt.rc('xtick', labelsize=14); 
plt.rc('ytick', labelsize=14);
plt.rc('axes', labelsize=16, titlesize=16); 
plt.rc('figure', titlesize=20); 
plt.rc('font', family='serif');
plt.rc('font', family='serif');

PATH = "./dat_0to35_1.08to1.8" 

SAVE = False

def calculate_angle(x, y):
    angles = np.arctan2(y, x)
    angles = np.mod(angles, 2 * np.pi)
    return angles

def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates to polar coordinates."""
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    """Convert polar coordinates to Cartesian coordinates."""
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

def transform_coordinates(x, y, scale=4):
    """Transform coordinates by converting to polar, scaling the radius, and converting back to Cartesian."""
    # Convert to polar coordinates
    r, theta = cartesian_to_polar(x, y)
    
    # Scale the radius
    r *= scale
    
    # Convert back to Cartesian coordinates
    new_x, new_y = polar_to_cartesian(r, theta)
    
    return new_x, new_y

def plot_x_embedding(X_emb, X2_emb, space_labels):

    fig, axes = plt.subplots(1, 2)
    plt.rc('axes', labelsize=18, titlesize=18); 

    scatter0 = axes[0].scatter(X_emb[:, 0], X_emb[:, 1], c=space_labels)
    axes[0].set_xlabel(r"$\zeta_1$")
    axes[0].set_ylabel(r"$\zeta_2$")

    scatter1 = axes[1].scatter(X2_emb, space_labels, c=space_labels)
    axes[1].set_xlabel(r"$\zeta$")
    axes[1].set_ylabel(f"emergent space")
    #axes[1].set_ticks([0, 250, 500])
    
    plt.subplots_adjust(hspace=0.7, wspace=0.45, top=0.87, bottom=0.07, left=0.135, right=0.89)

    cbar = plt.colorbar(scatter1, ax=axes[:], location="bottom", pad=0.2)
    cbar.ax.set_xlabel(f'emergent space', labelpad=-10)  # Set the label for the color bar
    cbar.set_ticks([0, 512])

    plt.show()
    fig.savefig(f"{PATH}/Figures/space_embedding2d.pdf", dpi=300, format='pdf')

def plot_t_embedding_rawtime(T_emb, T2_emb, time_labels):

    fig, axes = plt.subplots(1, 2)
    plt.rc('axes', labelsize=18, titlesize=18); 

    scatter0 = axes[0].scatter(T_emb[:, 0], T_emb[:, 1], c=time_labels)
    axes[0].set_xlabel(r"$\tau_1$")
    axes[0].set_ylabel(r"$\tau_2$")

    scatter1 = axes[1].scatter(T2_emb, time_labels, c=time_labels)
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(f"time")
    
    plt.subplots_adjust(hspace=0.6, wspace=0.45, top=0.87, bottom=0.07, left=0.135, right=0.89)


    cbar = plt.colorbar(scatter1, ax=axes[:], location="bottom", pad=0.2)
    cbar.ax.set_xlabel(f'time', labelpad=-10)  # Set the label for the color bar
    cbar.set_ticks([0, 35])

    plt.show()
    fig.savefig(f"{PATH}/Figures/time_embedding2d.pdf", dpi=300, format='pdf')

def plot_combined_embedding(K_emb, k_labels1, k_labels2):

    # Create a figure
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    plt.rc('axes', labelsize=18, titlesize=18); 

    # Create 3D plot in the first column
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    scatter3d = ax3d.scatter(K_emb[:, 0], K_emb[:, 1], K_emb[:, 2], c=k_labels1, cmap='viridis')
    ax3d.set_xlabel(r"$\kappa_1$")
    ax3d.set_ylabel(r"$\kappa_2$")
    ax3d.set_zlabel(r"$\kappa_3$")
    ax3d.grid(False)
    ax3d.view_init(elev=20., azim=30)

    cbar3d = plt.colorbar(scatter3d, ax=ax3d, pad=0.1, location="bottom")
    cbar3d.ax.set_xlabel('K')
    cbar3d.set_ticks([round(np.min(k_labels1), 2), round(np.max(k_labels1), 2)])

    # Create 2D plots in the second and third columns
    ax2d_1 = fig.add_subplot(gs[0, 1])
    scatter2d_1 = ax2d_1.scatter(K_emb[:, 0], K_emb[:, 1], c=k_labels1)
    ax2d_1.set_xlabel(r"$\kappa_1$")
    ax2d_1.set_ylabel(r"$\kappa_2$")

    cbar2d_1 = plt.colorbar(scatter2d_1, ax=ax2d_1, location="bottom", pad=0.15)
    cbar2d_1.ax.set_xlabel('K')
    cbar2d_1.set_ticks([round(np.min(k_labels1), 2), round(np.max(k_labels1), 2)])

    ax2d_2 = fig.add_subplot(gs[0, 2])
    scatter2d_2 = ax2d_2.scatter(K_emb[:, 0], K_emb[:, 1], c=k_labels2)
    ax2d_2.set_xlabel(r"$\kappa_1$")
    ax2d_2.set_ylabel(r"$\kappa_2$")

    cbar2d_2 = plt.colorbar(scatter2d_2, ax=ax2d_2, location="bottom", pad=0.15)
    cbar2d_2.ax.set_xlabel('phase shift')
    cbar2d_2.set_ticks([round(np.min(k_labels2), 1), round(np.max(k_labels2), 1)])
    cbar2d_2.set_ticklabels(['0', r'$2\pi$'])

    plt.subplots_adjust(hspace=0.4, wspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)

    # Show and save the plot
    plt.show()
    fig.savefig(f"{PATH}/Figures/K_embedding_combined.pdf", dpi=300, format='pdf')


def oscillator_plot(original, shuffled, pe, vmin=-0.4, vmax=0.4, cmap="viridis"): #"PRGn", "viridis"

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Plotting the images in the top row with aspect ratio adjustment
    for i in range(3):
        im = axes[i].imshow(original[i, :, :].real.squeeze().T, origin="lower", extent=[0, 512, 0, 35], aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
        axes[i].set_title(f'K: {round(pe[i], 2)}')
        axes[i].set_yticks([0, 35])
        axes[i].set_xticks([0, 512])

    # Plotting the images in the bottom row with aspect ratio adjustment
    for i in range(3):
        axes[i + 3].imshow(shuffled[i, :, :].real.squeeze().T, origin="lower", aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
        axes[i + 3].set_yticks([0, 500])
        axes[i + 3].set_xticks([0, 512])

    # Adding common x-axis labels
    fig.text(0.5, 0.52, r'$\zeta$ order', ha='center', va='center')
    fig.text(0.5, 0.08, 'space index', ha='center', va='center')

    # Adding common y-axis labels
    fig.text(0.09, 0.75, r'$t$', ha='center', va='center', rotation='vertical')
    fig.text(0.09, 0.3, 't index', ha='center', va='center', rotation='vertical')

    # Add the color bar
    cax = fig.add_axes([0.86, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(r'Re$W$')  # Set the label for the color bar
    cbar.set_ticks([vmin, 0, vmax])

    plt.subplots_adjust(hspace=0.39, wspace=0.38, top=0.9, bottom=0.15, left=0.15, right=0.82)
    plt.show()

    fig.savefig(f"{PATH}/Figures/oscillator_data.pdf", dpi=300, format='pdf')

def plot_one_trajectory_3D(data_dict, elev=17, azim=-51):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(data_dict["data"].shape[1]):
        re_data = data_dict["data"][:-1, i].real
        im_data = data_dict["data"][:-1, i].imag

        ax.plot(re_data[0:200], im_data[0:200], data_dict["tt"][0:200], 
                label=f'Trajectory {i + 1}' if i < 10 else None)

    ax.set_xlabel(r'$Re W$')
    ax.set_ylabel(r'$Im W$')
    ax.set_zlabel('Time')

    # Adjust rotation
    ax.view_init(elev=elev, azim=azim)

    # Set major ticks to 3 in x, y, and z directions
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

    #plt.legend()
    plt.show()
    fig.savefig(f"{PATH}/Figures/single_traj.pdf", dpi=300, format='pdf')

def plot_initial_conditions2(data_dict_list, ax):
    init_x = data_dict_list[0]["init"].real
    init_y = data_dict_list[0]["init"].imag

    scatter = ax.scatter(init_x, init_y, c=init_x)
    ax.set_xlabel(r"Re$W$", labelpad=10)
    ax.set_ylabel(r"Im$W$", labelpad=-5)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_aspect("equal", adjustable="box")

def plot_one_trajectory_3D2(data_dict, ax, elev=17, azim=-51):
    for i in range(data_dict["data"].shape[1]):
        re_data = data_dict["data"][:-1, i].real
        im_data = data_dict["data"][:-1, i].imag

        ax.plot(re_data[0:200], im_data[0:200], data_dict["tt"][0:200], 
                label=f'Trajectory {i + 1}' if i < 10 else None)

    ax.set_xlabel(r'Re$ W$', labelpad=10)
    ax.set_ylabel(r'Im$ W$', labelpad=10)
    ax.set_zlabel('Time', labelpad=5)
    ax.zaxis.label.set_rotation(90)

    # Adjust rotation
    ax.view_init(elev=elev, azim=azim)

    # Set major ticks to 3 in x, y, and z directions
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

def plot_single_image_3d2(data, index_3d, scrambed_space_label, time, fig, gs):
    if time is None:
        time = np.arange(data.shape[1])

    # Create the first 3D plot
    ax1 = fig.add_subplot(gs[1, 0], projection='3d')
    X, Y = np.meshgrid(time, np.arange(data.shape[1]))
    Z1 = data[index_3d, scrambed_space_label, :]
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis')
    ax1.set_ylabel(r'Im$W_0$ order', labelpad=10)
    ax1.set_xlabel('Time', labelpad = 10)
    ax1.set_zlabel(r'Re$W$', labelpad=10)

    # Create the second 3D plot
    ax2 = fig.add_subplot(gs[1, 1], projection='3d')
    Z2 = data[index_3d, :, :]
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis')
    ax2.set_ylabel(r'${\zeta}$ order', labelpad=10)
    ax2.set_xlabel('Time', labelpad = 10)
    ax2.set_zlabel(r'Re$W$', labelpad=10)

    # Adjust the layout to add a colorbar spanning both subplots
    cbar_ax = fig.add_axes([0.08, 0.15, 0.6, 0.02])  # Adjust these values to position the colorbar correctly
    cbar = fig.colorbar(surf2, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xlabel(r'Re$W$')

def main(data_dict_list, data_dict, data, index_3d, scrambed_space_label, time):
    fig = plt.figure(figsize=(10, 10))
    
    # Adjusting GridSpec layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # Adjusting padding for the top left plot
    ax1 = fig.add_subplot(gs[0, 0])
    pos1 = ax1.get_position()  # Get the original position
    pos1 = ax1.get_position()  # Get the original position
    pos2 = [pos1.x0 + 0.5, pos1.y0 + 0.5, pos1.width * 0.2, pos1.height * 0.2]  
    plot_initial_conditions2(data_dict_list, ax1)

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_one_trajectory_3D2(data_dict, ax2)

    plot_single_image_3d2(data, index_3d, scrambed_space_label, time, fig, gs)

    plt.tight_layout(rect=[0, 0.2, 0.7, 0.92])  # Adjust rect to make space for the colorbar
    plt.subplots_adjust(wspace=0.3, hspace=0.01)  # Adjust spacing

    plt.savefig(f"{PATH}/Figures/oscillator_explore.pdf", dpi=300, format='pdf')
    plt.show()


if __name__ == "__main__":

    ## Load Data. ## 
    data = np.load(f"{PATH}/3D_data.npy")
    shuffled_data = np.load(f"{PATH}/3D_data_shuffled.npy")

    K = np.load(f"{PATH}/True_K.npy")
    K_labels = io.loadmat(f"{PATH}/labels.mat")["K"][0]
    space_labels = io.loadmat(f"{PATH}/labels.mat")["x"][0]
    time_labels = io.loadmat(f"{PATH}/labels.mat")["t"][0]

    K_emb = io.loadmat(f"{PATH}/Output/embedding.mat")["embedding"][0,0]
    phi1 = K_emb[:,0]
    phi2 = K_emb[:,1]
    X_emb = io.loadmat(f"{PATH}/Output/embedding.mat")["embedding"][1,0][:,0:2]
    T_emb = io.loadmat(f"{PATH}/Output/embedding.mat")["embedding"][2,0][:,0:3]

    x_evec = np.load(f'{PATH}/ForNN/x_evec.npy')
    t_evec = np.load(f'{PATH}/ForNN/t_evec.npy')
    K_evecs = np.load(f'{PATH}/ForNN/K_evec.npy')

    with open(f"{PATH}/data_dict_list.pkl", 'rb') as f:
        data_dict_list = pickle.load(f)

    time = np.concatenate((np.array([0]), data_dict_list[0]["tt"]))
    angles = calculate_angle(phi1, phi2)

    INDEX = 400
    initial_imag = [value.imag for value in data_dict_list[INDEX]["data"][0]]
    initial_imag_labels = np.argsort(initial_imag)

    indices_to_keep = [10, 150, 490]
    oscillator_plot(data[indices_to_keep, :, :], shuffled_data[indices_to_keep, :, :], K[indices_to_keep])
    main(data_dict_list, data_dict_list[INDEX], data, INDEX, initial_imag_labels, time)
    plot_x_embedding(X_emb, x_evec, space_labels)
    plot_t_embedding_rawtime(T_emb, t_evec, time[time_labels])
    plot_combined_embedding(K_emb, K[K_labels], angles)