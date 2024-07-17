import torch
import numpy as np
import time
import scipy.io as io

from DeepONet import DeepONet
from model_train import Train_Adam
from model_error import Error_Test
from model_plot import Plot
from model_predict import Predict, plot_solution

class Runner:
    def __init__(self, torch_data_type, param):
        self.data_type = torch_data_type
        self.param = param
    
    def run(self, hyperparameters, save_results_to, save_variables_to):
        B_net = hyperparameters['B_net'] # layers
        T_net = hyperparameters['T_net'] # layers
        bs = hyperparameters['bs']
        num = hyperparameters['num']
        tsbs = hyperparameters['tsbs']
        epochs = hyperparameters['epochs']

        # this will initialize DeepONet, plus the trunk and branch
        model = DeepONet(self.data_type, T_net, B_net) 

        Train_Model_Adam = Train_Adam(model, self.data_type)
        Test_error = Error_Test(model, self.data_type, save_results_to)

        optimiser = torch.optim.Adam(model.params)
    
        n = 0
        start_time = time.perf_counter()
        time_step_0 = time.perf_counter()

        train_loss = np.zeros((epochs+1, 1))
        test_loss = np.zeros((epochs+1, 1))   
        while n <= epochs:
            
            # these are all numpy arrays, you need to make them tensors
            x_train, f_train, u_train, Xmin, Xmax = self.param.minibatch_minigrid()
            x_train = torch.from_numpy(x_train).float()
            f_train = torch.from_numpy(f_train).float()
            u_train = torch.from_numpy(u_train).float()
            Xmin = torch.from_numpy(Xmin).float()
            Xmax = torch.from_numpy(Xmax).float()
            
            lr = 0.0001
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr
            train_dict = Train_Model_Adam.nn_train(optimiser, x_train, f_train, u_train, 1, bs, Xmin, Xmax)
            loss = train_dict['loss']

            if n%1000 == 0:
                batch_id, x_test, f_test, u_test = self.param.testbatch(bs)

                x_test = torch.from_numpy(x_test).float()
                f_test = torch.from_numpy(f_test).float()
                u_test_pyt = torch.from_numpy(u_test).float()
                u_pred = Train_Model_Adam.call(x_test, f_test, 1, Xmin, Xmax)

                temp = (u_test - u_pred.detach().cpu().numpy()) ** 2 / (u_test ** 2 + 1e-4)
                err = np.mean(temp)
                err = np.reshape(err, (-1, 1))
                time_step_1000 = time.perf_counter()
                T = time_step_1000 - time_step_0
                print('Step: %d, Loss: %.4e, Test L2 error: %.4f, Time (secs): %.4f'%(n, loss, err, T))
                time_step_0 = time.perf_counter()     
            
            train_loss[n,0] = loss
            test_loss[n,0] = err
            n += 1

        x_print, f_print, u_print, batch_id = self.param.printbatch()

        x_print = torch.from_numpy(x_print).float()
        f_print = torch.from_numpy(f_print).float()
        u_print = torch.from_numpy(u_print).float()
        batch_id, f_print, u_print, u_pred = Test_error.nn_error_test(x_print, f_print, u_print, 1, tsbs, Xmin, Xmax, batch_id, self.param)
        err_before_mean = (u_print.detach().cpu().numpy() - u_pred.detach().cpu().numpy()) ** 2 / (u_print.detach().cpu().numpy() ** 2 + 1e-4)
        print(f"shape of u: {u_pred.shape}")
        print(f"shape of error before mean {err_before_mean.shape}")
        err = np.mean(err_before_mean)
        err = np.reshape(err, (-1, 1))
        np.savetxt(save_results_to+'/err', err, fmt='%e')       
        io.savemat(save_results_to+'/Darcy.mat', 
                    mdict={'test_id': batch_id,
                            'x_test': f_print.detach().cpu().numpy(),
                            'y_test': u_print.detach().cpu().numpy(), 
                            'y_pred': u_pred.detach().cpu().numpy()})
        stop_time = time.perf_counter()
        print('Elapsed time (secs): %.3f'%(stop_time - start_time))

        # Save variables (weights + biases)
        w_T, b_T = model.trunk_dnn.get_weights_and_biases()
        w_B, b_B = model.branch_dnn.get_weights_and_biases()
    
        W_b_dict_save = {'W_br_fnn': w_B, 'b_br_fnn': b_B,
                        'W_tr': w_T, 'b_tr': b_T}
        torch.save(W_b_dict_save, save_variables_to+'/Weight_bias.pt')

        print('Complete storing')
        
        print('Save')

        plot = Plot()
        plot.Plotting(train_loss, test_loss, save_results_to)

        print('Running some test predictions')
        predict_test = Predict(model, self.data_type, save_results_to)

        X_predict = torch.from_numpy(self.param.all_X()).float()
        F_predict = torch.from_numpy(predict_test.get_random_F(self.param.Fmin, self.param.Fmax, num=26)).float()
        U_predict = predict_test.predict_field(X_predict, F_predict, 1, Xmin, Xmax)
        print(U_predict.shape)

        plot_solution(X_predict, F_predict, U_predict, title="Prediction", save_results_to=save_results_to)


        