"""
Creates oscillator dataset using Matthews.py
and the specified config file (config.cfg)

Must run from inside Oscillators folder.

Adapted from https://github.com/fkemeth/lpde
"""
import os
import pickle
import shutil
import configparser
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch

from scipy import io
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

import utils as utils

import matthews as mint

torch.set_default_dtype(torch.float32)

def make_plots_for3D(config, path, verbose=False):
    '''
    Visualize N**2 instances of the dataset.
    '''

    N = 2

    with open(f'{path}/data_dict_list.pkl', 'rb') as input_file:
        data_dict_list = pickle.load(input_file)

    time = data_dict_list[0]["tt"]
    K_list = np.array([dictt["K"] for dictt in data_dict_list])

    data = np.load(f"{path}/3D_data.npy")
    fig, axes = plt.subplots(N, N, figsize=(9, 9))

    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            img_index = i * N + j
            cax = ax.imshow(data[img_index, :, :].real, aspect='auto', extent=[time.min(), time.max(), 0, data.shape[1]], origin='lower')
            
            # Add color bar for each subplot
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label(r'$Re W$')

            ax.set_xlabel('Time')
            ax.set_ylabel('Index')
            ax.set_title(f'K= {round(K_list[img_index],3)}')

    plt.tight_layout()
    plt.savefig(config["GENERAL"]["save_dir"]+'/Figures/'+"OriginalData.png")
    plt.show()

    data_shuffled = np.load(f"{path}/3D_data_shuffled.npy")

    fig, axes = plt.subplots(N, N, figsize=(9, 9))
    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            img_index = i * N + j
            cax = ax.imshow(data_shuffled[img_index, :, :].real, aspect='auto', origin='lower')
            
            # Add color bar for each subplot
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label(r'$Re W$')

            ax.set_xlabel('Shuffled Time')
            ax.set_ylabel('Shuffled Space')
            ax.set_title(f'Shuffled K')

    plt.tight_layout()
    plt.savefig(config["GENERAL"]["save_dir"]+'/Figures/'+'ShuffledData.png')
    plt.show()

def integrate_system(config, path, verbose=False):
    """Integrate Matthews system."""
    pars = {}
    pars["gamma"] = float(config["gamma"])
    pars["omega"] = np.linspace(-pars["gamma"], pars["gamma"], int(config["N_int"])) + \
        float(config["gamma_off"])
    pars["K"] = float(config["K"])
    data_dict = mint.integrate(pars=pars,
                               dt=float(config["dt"]), N=int(config["N_int"]), T=int(config["T"]),
                               tmin=float(config["tmin"]), tmax=float(config["tmax"]),
                               gamma_off=float(config["gamma_off"]),
                               append_init=True, 
                               ic='grid')#'synchronized')

    output = open('{path}data_dict.pkl', 'wb')
    pickle.dump(data_dict, output)

def integrate_system_K(config, path, verbose=False, no=500):
    """Integrate Matthews system with varying K."""

    K_list = np.linspace(float(config["K_start"]), float(config["K_end"]), no)
    data_dict_list = [None] * len(K_list)
    data_list = [None] * len(K_list)


    for i, K in enumerate(tqdm(K_list)):
        if i%10 == 0: print("K: "+ str(K))
        pars = {}
        pars["K"] = float(K)
        pars["gamma"] = float(config["gamma"])
        pars["omega"] = np.linspace(-pars["gamma"], pars["gamma"], int(config["N_int"])) + \
                                    float(config["gamma_off"])
        data_dict = mint.integrate(pars=pars,
                                   dt=float(config["dt"]), N=int(config["N_int"]), T=int(config["T"]),
                                   tmin=float(config["tmin"]), tmax=float(config["tmax"]),
                                   gamma_off=float(config["gamma_off"]),
                                   append_init=True,
                                   ic='grid',)
        data_dict_list[i] = data_dict

        time = data_dict["tt"]
        data = data_dict["data"]  # This contains both real and imaginary

        data_list[i] = data.T

        # Plotting the 2D image (every 100 K values)
        if verbose and i % 100 == 0:
            fig, ax = plt.subplots()
            
            # Use imshow to plot the data
            cax = ax.imshow(data.real.T, aspect='auto', extent=[time.min(), time.max(), 0, data.shape[1]], origin='lower')

            # Adding color bar
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label(r'$Re W$')

            ax.set_xlabel('Time')
            ax.set_ylabel('Index')
            plt.title('Trajectories')

            plt.show()

    output = open(f'{path}/data_dict_list.pkl', 'wb')
    pickle.dump(data_dict_list, output)

    data_array = np.array(data_list)
    print("DATA ARRAY SHAPE: ", data_array.shape)

    param_idx, space_idx, time_idx, shuffled_dataset = utils.shufflematrix(data_array)
    np.save(f"{path}/3D_data.npy", data_array)
    np.save(f"{path}/3D_data_shuffled.npy", shuffled_dataset)
    np.save(f"{path}/True_K.npy", np.array(K_list))

    ### save out as matlab files
    io.savemat(f"{path}/labels.mat", {'x':space_idx, 't': time_idx, 'K': param_idx})
    io.savemat(f"{path}/True_K.mat", {'K': K_list})
    io.savemat(f'{path}/3D_data_shuffled.mat', {'Data': shuffled_dataset})
    io.savemat(f'{path}/3D_data_original.mat', {'Data': data_array})

def reshuffle_datadiclist(path):
    '''
    Takes path to data_dict_list.pkl, loads in data, 
    and reshuffles the matrix and saves out necessary 
    components for questionnaires.
    '''
    with open(f'{path}/data_dict_list.pkl', 'rb') as input_file:
        data_dict_list = pickle.load(input_file)

    data_list = [None] * len(data_dict_list)
    K_list = [None] * len(data_dict_list)

    for i, data_dict in enumerate(data_dict_list):
        time = data_dict["tt"]
        data = data_dict["data"]
        data_list[i] = data.T
        K_list[i] = data_dict["K"]
    
    data_array = np.array(data_list)
    print("DATA ARRAY SHAPE: ", data_array.shape)

    param_idx, space_idx, time_idx, shuffled_dataset = utils.shufflematrix(data_array)
    np.save(f"{path}/3D_data.npy", data_array)
    np.save(f"{path}/3D_data_shuffled.npy", shuffled_dataset)
    np.save(f"{path}/True_K.npy", np.array(K_list))

    ### save out as matlab files
    io.savemat(f"{path}/labels.mat", {'x':space_idx, 't': time_idx, 'K': param_idx})
    io.savemat(f"{path}/True_K.mat", {'K': K_list})
    io.savemat(f'{path}/3D_data_shuffled.mat', {'Data': shuffled_dataset})
    io.savemat(f'{path}/3D_data_original.mat', {'Data': data_array})


def turn3D_to_2D_singletime(path, index=10, verbose=False):

    data_array = np.load("{path}3D_data.npy")
    print(data_array.shape)

    if verbose:
        for i in [0, 100, 200, 300, 400, 500]:

            time_snapshot = data_array[:,:,i]
            print(time_snapshot.shape)

            fig, ax = plt.subplots()
            cax = ax.imshow(time_snapshot, aspect='auto', origin='lower')

            # Adding color bar
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label(r'$Re W$')

            ax.set_xlabel('Space Index')
            ax.set_ylabel('K')
            plt.title(f'Time Index = {i}')

            plt.show()

    time_snapshot = data_array[:,:,index:index+1]
    param_idx, space_idx, time_idx, shuffled_dataset = utils.shufflematrix(time_snapshot)
    np.save(f"{path}/2D_data_{index}.npy", time_snapshot.squeeze())
    np.save(f"{path}/2D_data_{index}_shuffled.npy", shuffled_dataset.squeeze())

    ### save out as matlab files
    io.savemat(f"{path}/labels_2D_{index}.mat", {'x':space_idx, 'K': param_idx})
    io.savemat(f'{path}/2D_data_{index}_shuffled.mat', {'matrix': shuffled_dataset.squeeze()})
    io.savemat(f'{path}/2D_data_{index}_original.mat', {'matrix': time_snapshot.squeeze()})


def main(config):
    """Integrate system and train model."""

    verbose = config["GENERAL"].getboolean("verbose")

    # Create data folders
    if not os.path.exists(config["GENERAL"]["save_dir"]):
        os.makedirs(config["GENERAL"]["save_dir"])

    # Create dataset
    if config["MODEL"].getboolean("use_param"):

        if config["MODEL"]["vary"] == "K":
            integrate_system_K(config["SYSTEM"], config["GENERAL"]["save_dir"], verbose=verbose)

    else:
        integrate_system(config["SYSTEM"],
                         config["GENERAL"]["save_dir"],
                         verbose=verbose)

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.cfg')

    # Create dataset.
    main(config)

    # Create figures.
    if not os.path.exists(config["GENERAL"]["save_dir"]+'/Figures'):
        os.makedirs(config["GENERAL"]["save_dir"]+'/Figures')
    
    # make_plots_for3D(config,
    #                  config["GENERAL"]["save_dir"],
    #                  verbose=config["GENERAL"].getboolean("verbose"))
