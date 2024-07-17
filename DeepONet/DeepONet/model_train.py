import torch
    
class Train_Adam:
    def __init__(self, model, torch_data_type):
        self.model = model
        self.data_type = torch_data_type
        torch_type = str(self.data_type)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def call(self, X, F, F_norm, Xmin, Xmax):

        X = X.to(self.device)
        F = F.to(self.device)
        Xmin = Xmin.to(self.device)
        Xmax = Xmax.to(self.device)
              
        # Branch FNN
        u_B = self.model.fnn_B(F)

        # Trunk FNN
        u_T = self.model.fnn_T(X, Xmin, Xmax)
        
        u_nn = torch.einsum('bij,nj->bnj', u_B, u_T)
        u_nn = torch.sum(u_nn, axis=-1, keepdims=True)

        u_pred = u_nn*F_norm
        
        return u_pred
    
    def nn_train(self, optimizer, X, F, U, F_norm, bs, Xmin, Xmax):

        X = X.to(self.device)
        F = F.to(self.device)
        U = U.to(self.device)
        Xmin = Xmin.to(self.device)
        Xmax = Xmax.to(self.device)

        optimizer.zero_grad()

        u_pred  = self.call(X, F, F_norm, Xmin, Xmax).to((self.device))
        loss = torch.mean(torch.square(U-u_pred)/(torch.square(U)+ 1e-4))
        loss.backward()
        optimizer.step()

        loss_dict = {'loss': loss, 'U_pred': u_pred}
        return loss_dict
                                                  
