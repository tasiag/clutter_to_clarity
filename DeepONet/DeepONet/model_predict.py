import torch
import numpy as np
import matplotlib.pyplot as plt

class Predict:
    def __init__(self, model, torch_data_type, saveresultsto):
        self.model = model
        self.data_type = torch_data_type
        self.save_results_to = saveresultsto

    def predict_field(self, X, F, F_norm, Xmin, Xmax):
        return self.model(X, F, F_norm, Xmin, Xmax)
        

    def get_random_F(self, F_min, F_max, num=1):
        if isinstance(F_min, (float, np.float64)):
            ns = 1
        else:
            ns = F_min.shape[-1]
        return (np.random.random(num*ns).reshape(num,ns)*(F_max-F_min)+F_min).reshape(num, 1, ns)
    

def plot_solution(X, F, U, title=None, save_results_to=None):

    if not isinstance(X, np.ndarray):
        X = X.cpu().detach().numpy()

    if not isinstance(U, np.ndarray):
        U = U.cpu().detach().numpy()
    U = U.reshape(F.shape[0], -1)

    if not isinstance(F, np.ndarray):
        F = F.cpu().detach().numpy()
    print(f" f shape {F.shape}")

    n = int(np.sqrt(F.shape[0]))
    if n > 2: n = 2
    print(f"n: {n}")
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(8,8), constrained_layout=True)
    axes = axes.ravel()
    for i in range(len(axes)):
        axes[i].scatter(X[:,0], X[:,1], c=U[i,:].squeeze(), s=1)
        axes[i].set_ylim(max(X[:,1]), min(X[:,1]))
        # f = round(F[i],2)
        if F.shape[-1] > 1:
            mini_title = [round(num, 2) for num in F[i,...].squeeze()]
        else: 
            mini_title = round(F.squeeze().reshape(-1)[i],2)
        axes[i].set_title(mini_title)
    fig.supxlabel("z embedding")
    fig.supylabel("t embedding")
    if title is not None:
        fig.suptitle(title)

    if save_results_to is None:
        save_results_to = '/Users/anastasia/Documents/Education/PhD/EN.560.617 Deep Learning/Project/DeepLearningProject/DeepONet/Case_1/Results'
    fig.savefig(f'{save_results_to}/{title}.pdf', dpi=300, format='pdf')
    plt.show()

    
        

