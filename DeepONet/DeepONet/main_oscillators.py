
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import shutil
import argparse

from model_run import Runner
from dataset_evecs_oscillators import DataSet

# np.random.seed(1234)
torch_data_type = torch.float

def main(save_index):
    # Create directories
    current_directory = os.getcwd()   
    datapath = './Oscillators/dat_0to35_1.08to1.8_real/' 
    case = "Case_"
    name = "Oscillators/"
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
    
    p = 500
    h = 12 
    num = 2 # number of sensors 
    
    hyperparameters = {"B_net": [num, 40, 40, 40, p], 
                       "T_net": [2, 100, 100, 100, 100, 100, 100, p], 
                       "bs": 20, "tsbs": 20, "epochs": 120000, "num":h} #120000
    
    io.savemat(save_variables_to+'/hyperparameters.mat', mdict=hyperparameters)
    
    param = DataSet(num, hyperparameters["bs"], save_results_to, path=datapath)
    network = Runner(torch_data_type, param)
    network.run(hyperparameters, save_results_to, save_variables_to)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--Case_', default=3, type=int, help='prior_sigma')
    args = parser.parse_args()
    Case = args.Case_
    main(Case)