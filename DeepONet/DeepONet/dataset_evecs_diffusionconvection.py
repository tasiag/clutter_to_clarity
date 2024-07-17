'''
Data loader for advection-diffusion dataset.

Change FILE_PATH (uses absolute file paths). 
'''

import torch
import numpy as np
import scipy.io as io
import sys
from model_predict import plot_solution

import matplotlib.pyplot as plt
np.random.seed(1234)

FILE_PATH = '/Users/anastasia/Documents/Education/PhD/EN.560.617 Deep Learning/Project/DeepLearningProject/Diffusion_Convection/64_80_100/ForNN'
SAVE_PATH = '/Users/anastasia/Documents/Education/PhD/EN.560.617 Deep Learning/Project/DeepLearningProject/Test'

class DataSet:
    def __init__(self, num, bs, save_results_to):
        self.save_results_to = save_results_to

        self.num = num
        self.bs = bs
        self.F_train, self.U_train, self.F_test, self.U_test, \
            self.X, self.u_mean, self.u_std = self.load_data()
        
        self.Xmin = np.min(self.X, axis=0) 
        self.Xmax = np.max(self.X, axis=0)

        print(f"xmin {self.Xmin}")
        print(f"xmax {self.Xmax}")

        print(f"fmin {self.Fmin}")
        print(f"fmax {self.Fmax}")
        
        print(f"loaded into self shape of ftrain: {self.F_train.shape}")
        print(f"loaded into self shape of ftest: {self.F_test.shape}")


    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x
    
    def load_data(self, plot=False):

        U_or = np.load(f'{FILE_PATH}/U.npy')
        ex_dc = np.load(f'{FILE_PATH}/ex_evec.npy')
        x_dc = np.load(f'{FILE_PATH}/x_evec.npy')
        t_dc = np.load(f'{FILE_PATH}/t_evec.npy')
        
        if plot: 
            plt.imshow(U_or[0,:,:].squeeze())
            plt.show()

        self.Fmin = np.min(ex_dc)
        self.Fmax = np.max(ex_dc)
            
        nex, nx, nt = len(ex_dc), len(x_dc), len(t_dc)

        self.nx = nx
        self.nt = nt

        F = np.expand_dims(ex_dc, axis = (1,2))
        print(f"F.shape: {F.shape}")

        U = U_or.reshape(nex, -1)
        U = np.expand_dims(U, axis = 2)
        print(f"U.shape: {U.shape}")

        # reorganize matrix
        x = np.zeros(nx*nt)
        t = np.zeros(nx*nt)
        count = 0
        for i, x_ev in enumerate(x_dc):
            for j, t_ev in enumerate(t_dc):
                    x[count] = x_ev
                    t[count] = t_ev
                    count +=1

        X = np.stack((x, t), axis=-1)
        print(f"X.shape: {X.shape}")


        print("plotting truth")
        if plot: plot_solution(X, F, U, title="Raw data", save_results_to=self.save_results_to)

        sorted_indices = np.argsort(F, axis=0)[::-1]
        if plot: plot_solution(X, F[sorted_indices, ...], U[sorted_indices, ...], title="Raw Data, Sorted", save_results_to=self.save_results_to)

        TEST_PERCENT = 0.1875

        num_train = int(nex*(1-TEST_PERCENT))
        num_test = nex-num_train

        
        f_train = F[:num_train, :, :]
        u_train = U[:num_train, :, :]

        f_test = F[num_train:num_train+num_test, :, :]
        u_test = U[num_train:num_train+num_test, :, :] 

        # compute mean values
        f_train_mean = np.mean(f_train, 0)
        f_train_std = np.std(f_train, 0)
        u_train_mean = np.mean(u_train, 0)
        u_train_std = np.std(u_train, 0)
        
        ns = 1 # number of sensors
        num_res = nt*nx # total output dimension
        
        # Reshape
        f_train_mean = np.reshape(f_train_mean, (-1, 1, ns))
        f_train_std = np.reshape(f_train_std, (-1, 1, ns))
        u_train_mean = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std = np.reshape(u_train_std, (-1, num_res, 1))

        F_train = np.reshape(f_train, (-1, 1, 1))
        U_train = np.reshape(u_train, (-1, num_res, 1))
        F_test = np.reshape(f_test, (-1, 1, 1))
        U_test = np.reshape(u_test, (-1, num_res, 1))

        print(F_train.shape) # 51 x 1 x 1
        print(U_train.shape) # 51 x 8000 x 1
        print(X.shape)       # 8000 x 2

        print(F_test.shape) # 13 x 1 x 1
        print(U_test.shape) # 13 x 8000 x 1

        return F_train, U_train, F_test, U_test, \
            X, u_train_mean, u_train_std

    
    def minibatch_minigrid(self, gridsize=10000):
        return self.minibatch()

        
    def minibatch(self):

        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)

        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]

        x_train = self.X

        Xmin = np.array([ min(x_train[:,0]), min(x_train[:,1])]).reshape((-1, 2))
        Xmax = np.array([ max(x_train[:,0]), max(x_train[:,1])]).reshape((-1, 2)) 

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):
        
        batch_id = np.arange(num_test)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X
        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
    
    def printbatch(self, bs=None, plot=False):
        if bs is None:
            bs = self.bs
        batch_id = np.random.choice(self.F_test.shape[0], bs, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))
        if plot: plot_solution(x_test, f_test, u_test, title="a random test batch", save_results_to=self.save_results_to) 

        return x_test, f_test, u_test, batch_id
    
    def all_X(self):

        N = 100

        phi_wholespace = np.linspace(self.Xmin[0], self.Xmax[0], N)
        psi_wholespace = np.linspace(self.Xmin[1], self.Xmax[1], N)
        phi_W, psi_W = np.meshgrid(phi_wholespace, psi_wholespace)
        allpoints = np.hstack((phi_W.flatten()[:,None], psi_W.flatten()[:,None]))   
        return allpoints
    
    def get_random_F(self):
        return np.array([[[ np.random.random()*(self.Fmax-self.Fmin)+self.Fmin]]])
    
    def format_F(self, f):
        return np.array([[[f]]])
    
    

if __name__ == "__main__":


    #batch_size
    bs = 12 # goes evenly into both test & train sets

    #size of input for Trunk net
    nx = 80
    nt = 100
    x_num = nt*nx

    data = DataSet(nx, bs, save_results_to=SAVE_PATH)
    data.printbatch()
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch()    
    print(x_train.shape, f_train.shape, u_train.shape, Xmin, Xmax)

    print(data.get_random_F())
   
    
    
    
    
