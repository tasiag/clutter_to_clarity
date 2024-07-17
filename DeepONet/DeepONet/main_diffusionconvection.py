'''
Builds a DeepONet model on the advection-diffusion dataset.
'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import shutil
import argparse

from model_run import Runner
from dataset_evecs_diffusionconvection import DataSet

np.random.seed(1234)
torch_data_type = torch.float

def main(save_index):
    # Create directories
    current_directory = os.getcwd()    
    case = "Case_"
    name = "DiffusionConvection/"
    folder_index = str(save_index)
    results_dir = "/" + "DeepONet/" + name + case + folder_index +"/Results"
    variable_dir = "/" + "DeepONet/" + name + case + folder_index +"/Variables"
    save_results_to = current_directory + results_dir
    save_variables_to = current_directory + variable_dir

    # Remove existing results
    if os.path.exists(save_results_to):
        shutil.rmtree(save_results_to)
        shutil.rmtree(save_variables_to)

    os.makedirs(save_results_to) 
    os.makedirs(save_variables_to) 
    print(f"making directory: {save_results_to}")
    
    p = 100
    h = 12 
    num = 1
    
    hyperparameters = {"B_net": [num, 40, 40, 40, 40, p], "T_net": [2, 40, 40, 40, 40, p], 
                       "bs": 12, "tsbs": 12, "epochs": 500000, "num":h,
                       "device": "mps"}
    
    io.savemat(save_variables_to+'/hyperparameters.mat', mdict=hyperparameters)
    
    param = DataSet(num, hyperparameters["bs"], save_results_to)
    network = Runner(torch_data_type, param)
    network.run(hyperparameters, save_results_to, save_variables_to)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--Case_', default=1, type=int, help='prior_sigma')
    args = parser.parse_args()
    Case = args.Case_
    main(Case)