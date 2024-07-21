import torch
import numpy as np
import scipy.io as io
import sys
from model_predict import plot_solution
import matplotlib.pyplot as plt
np.random.seed(1234)

class DataSet:
    def __init__(self, num, bs, save_results_to, path='./Oscillators/dat_0to35_1.08to1.8_real/'):
        self.save_results_to = save_results_to
        self.path = path

        self.num = num
        self.bs = bs
        self.F_train, self.U_train, self.F_test, self.U_test, \
            self.X, self.u_mean, self.u_std = self.load_data()
        
        self.Xmin = np.min(self.X, axis=0) # of all data
        self.Xmax = np.max(self.X, axis=0) # of all data

    def decoder(self, x):
        
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        
        return x
    
    def load_data(self, plot=False):

        print("loading data")
        PATH = self.path

        U_or = np.load(f'{PATH}/3D_data_shuffled.npy') 
        ex_dc = np.load(f'{PATH}/ForNN/K_evec.npy')
        x_dc = np.load(f'{PATH}/ForNN/x_evec.npy')
        t_dc = np.load(f'{PATH}/ForNN/t_evec.npy')
        print("data loaded.")
        
        self.Fmin = np.min(ex_dc, axis=0)
        self.Fmax = np.max(ex_dc, axis=0)
            
        nex, nx, nt = len(ex_dc), len(x_dc), len(t_dc)

        F = np.expand_dims(ex_dc, axis = (1))
        U = U_or.reshape(nex, -1)
        U = np.expand_dims(U, axis = 2)
        U = U.real ### use only real part

        x = np.zeros(nx*nt)
        t = np.zeros(nx*nt)
        count = 0
        for i, x_ev in enumerate(x_dc):
            for j, t_ev in enumerate(t_dc):
                    x[count] = x_ev
                    t[count] = t_ev
                    count +=1

        X = np.stack((x, t), axis=-1)

        plt.scatter(X[:,0], X[:,1], c=U[400,:].squeeze(), s=1)
        plt.show()

        if plot: plot_solution(X, F, U, title="Raw data", save_results_to=self.save_results_to)
    
        F1 = F[:,0,0]
        sorted_indices = np.argsort(F1)
        if plot: plot_solution(X, F[sorted_indices, ...], U[sorted_indices, ...], title="Raw Data, Sorted on F_1", save_results_to=self.save_results_to)

        TEST_PERCENT = 0.2

        num_train = int(nex*(1-TEST_PERCENT))
        num_test = nex-num_train
        
        f_train = F[:num_train, :, :]
        u_train = U[:num_train, :, :]

        if plot: plot_solution(X, f_train, u_train, title="Training data, before mean normalization", save_results_to=self.save_results_to)

        f_test = F[num_train:num_train+num_test, :, :]
        u_test = U[num_train:num_train+num_test, :, :]

        # compute mean values
        f_train_mean = np.mean(f_train, axis=0)
        f_train_std = np.std(f_train, axis=0)
        print(f"f_train_mean.shape: {f_train_mean.shape}")
        u_train_mean = np.mean(u_train, axis=0)
        u_train_std = np.std(u_train, axis=0)
        
        ns = F.shape[-1] # number of sensors
        print(ns)
        num_res = nt*nx # total output dimension
        
        # Reshape
        f_train_mean = np.reshape(f_train_mean, (-1, 1, ns))
        f_train_std = np.reshape(f_train_std, (-1, 1, ns))
        u_train_mean = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std = np.reshape(u_train_std, (-1, num_res, 1))

        F_train = np.reshape(f_train, (-1, 1, ns))
        U_train = np.reshape(u_train, (-1, num_res, 1))

        if plot: plot_solution(X, F_train, U_train,
                               title="Training data, NOT Mean Normalized",
                               save_results_to=self.save_results_to)

        F_test = np.reshape(f_test, (-1, 1, ns))
        U_test = np.reshape(u_test, (-1, num_res, 1))

        print(F_train.shape) # 400 x 1 x 2
        print(U_train.shape) # 400 x 256512 x 1
        print(X.shape)       # 256512 x 2

        print(F_test.shape) # 100 x 1 x 2
        print(U_test.shape) # 100 x 256512 x 1

        return F_train, U_train, F_test, U_test, \
            X, u_train_mean, u_train_std

        
    def minibatch(self):

        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)

        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]

        x_train = self.X

        Xmin = np.array([ min(x_train[:,0]), min(x_train[:,1])]).reshape((-1, 2))
        Xmax = np.array([ max(x_train[:,0]), max(x_train[:,1])]).reshape((-1, 2))  # I didn't rescale to be between 0 to 1

        return x_train, f_train, u_train, Xmin, Xmax
    

    def minibatch_minigrid(self, gridsize=10000):

        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        grid_id = np.random.choice(self.X.shape[0], gridsize, replace=False)

        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id] 
        u_train = u_train[:, grid_id, :]

        x_train = self.X[grid_id]

        Xmin = np.array([ min(x_train[:,0]), min(x_train[:,1])]).reshape((-1, 2))
        Xmax = np.array([ max(x_train[:,0]), max(x_train[:,1])]).reshape((-1, 2))  # I didn't rescale to be between 0 to 1

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):
        
        batch_id = np.arange(num_test)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))
        return batch_id, x_test, f_test, u_test
    
    def printbatch(self, bs = None, batch_id = None):
        if bs is None: bs = self.bs
        if batch_id is None:
            batch_id = np.random.choice(self.F_test.shape[0], bs, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))
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
    bs = 20 # goes evenly into both test & train sets

    #size of input for Trunk net
    nx = 512
    nt = 501
    x_num = nt*nx

    data = DataSet(nx, bs, 
                   save_results_to='/Users/anastasia/Documents/Education/PhD/EN.560.617 Deep Learning/Project/clutter_to_clarity/DeepONet/Oscillators/Test',
                  )

    x_train, f_train, u_train, Xmin, Xmax = data.minibatch_minigrid(gridsize=10000) #data.minibatch() 
