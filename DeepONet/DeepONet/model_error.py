import torch

class Error_Test:
    def __init__(self, model, torch_data_type, saveresultsto):
        self.model = model
        self.data_type = torch_data_type
        self.save_results_to = saveresultsto
    
    def nn_error_test(self, x_test, f_test, u_test, f_norm_test, bs, Xmin, Xmax, batch_id, param):
        
        # Branch FNN
        u_B = self.model.fnn_B(f_test) # is this what's supposed to go here?

        # Trunk FNN
        u_T = self.model.fnn_T(x_test, Xmin, Xmax)
        
        u_nn = torch.einsum('ijk,lk->il', u_B, u_T)
        u_nn = torch.unsqueeze(u_nn, dim=-1)
        u_pred = u_nn*f_norm_test
        
        return batch_id, f_test, u_test, u_pred
        

