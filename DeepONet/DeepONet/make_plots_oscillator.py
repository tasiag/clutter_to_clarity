import matplotlib.pyplot as plt  
import numpy as np  
import scipy.io as io
import torch

from model_predict import Predict, plot_solution
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator

from DeepONet import DeepONet
from dataset_evecs_oscillators import DataSet

PATH = "./DeepONet/Oscillators/Case_3"

def plot_three_old(X, F, U, U_P, U_E, error_min=0.2, title=None, save_results_to=None):
    if not isinstance(X, np.ndarray):
        X = X.cpu().detach().numpy()

    if not isinstance(U, np.ndarray):
        U = U.cpu().detach().numpy()
    U = U.reshape(F.shape[0], -1)

    if not isinstance(U_P, np.ndarray):
        U_P = U_P.cpu().detach().numpy()
    U_P = U_P.reshape(F.shape[0], -1)

    if not isinstance(U_E, np.ndarray):
        U_E = U_E.cpu().detach().numpy()
    U_E = U_E.reshape(F.shape[0], -1)

    if not isinstance(F, np.ndarray):
        F = F.cpu().detach().numpy()
    print(f" f shape {F.shape}")

    # Create custom colormap with white at 0
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    nodes = [0.0, 0.5, 1.0]  # Define the positions of the colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    # Normalize colors to ensure 0 is white
    norm = Normalize(vmin=-error_min, vmax=error_min)  # Adjust vmin and vmax as needed

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))#, constrained_layout=True)
    axes = axes.ravel()

    for i in range(3):
        sc = axes[i].scatter(X[:, 0], X[:, 1], c=U[i, :].squeeze(), s=0.2)
        if i == 0:
            axes[i].set_xticks([])
            axes[i].yaxis.set_major_locator(MaxNLocator(2))
            axes[i].set_ylabel(r"$\tau$", labelpad=0)
        else: 
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        if F.shape[-1] > 1:
            mini_title = [round(num, 2) for num in F[i, ...].squeeze()]
        else:
            mini_title = round(F.squeeze().reshape(-1)[i], 2)
        axes[i].set_title(r"$\kappa$: " + str(mini_title))

    for i, v in enumerate(range(3, 6)):
        sc = axes[v].scatter(X[:, 0], X[:, 1], c=U_P[i, :].squeeze(), s=0.2)
        if i == 0:
            axes[v].set_xticks([])
            axes[v].yaxis.set_major_locator(MaxNLocator(2))
            axes[v].set_ylabel(r"$\tau$", labelpad=0)
        else: 
            axes[v].set_xticks([])
            axes[v].set_yticks([])
        if F.shape[-1] > 1:
            mini_title = [round(num, 2) for num in F[i, ...].squeeze()]
        else:
            mini_title = round(F.squeeze().reshape(-1)[i], 2)

    for i, v in enumerate(range(6, 9)):
        sc = axes[v].scatter(X[:, 0], X[:, 1], c=U_E[i, :].squeeze(), s=0.2, cmap=custom_cmap, norm=norm)
        if i == 0:
            axes[v].xaxis.set_major_locator(MaxNLocator(2))
            axes[v].yaxis.set_major_locator(MaxNLocator(2))
            axes[v].set_ylabel(r"$\tau$", labelpad=0)
        else: 
            axes[v].xaxis.set_major_locator(MaxNLocator(2))
            axes[v].set_yticks([])
        axes[v].set_xlabel(r"$\zeta$", labelpad=1)
        if F.shape[-1] > 1:
            mini_title = [round(num, 2) for num in F[i, ...].squeeze()]
        else:
            mini_title = round(F.squeeze().reshape(-1)[i], 2)

    # Common colorbar at the bottom for the error values
    cbar_ax_bottom = fig.add_axes([0.1, 0.1, 0.6, 0.02])  
    cbar_bottom = fig.colorbar(sc, cax=cbar_ax_bottom, orientation='horizontal')
    cbar_bottom.set_label('Error', labelpad=-10)
    cbar_bottom.set_ticks([-error_min, error_min])

    # Common colorbar at the top for the first two rows
    adjust = 0.4
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-adjust, vmax=adjust))
    sm._A = []  # Dummy array for the ScalarMappable
    cbar_ax_top = fig.add_axes([0.1, 0.9, 0.6, 0.02])  
    cbar_top = fig.colorbar(sm, cax=cbar_ax_top, orientation='horizontal')
    cbar_top.set_label(r'Re$W$', labelpad=-26)
    cbar_top.ax.xaxis.set_label_position('top')
    cbar_top.set_ticks([-adjust, adjust])

    # Add vertical labels
    fig.text(0.72, 0.72, 'True', va='center', rotation='vertical', fontsize=12)
    fig.text(0.72, 0.51, 'Predicted', va='center', rotation='vertical', fontsize=12)
    fig.text(0.72, 0.29, 'Error', va='center', rotation='vertical', fontsize=12)

    # Adjust padding
    plt.subplots_adjust(hspace=0.3, wspace=0.21, top=0.82, bottom=0.2, left=0.1, right=0.706)

    if save_results_to is None:
        save_results_to = '/Users/anastasia/Documents/Education/PhD/EN.560.617 Deep Learning/Project/clutter_to_clarity/DeepONet/Case_2/Results'
    fig.savefig(f'{save_results_to}/{title}.pdf', dpi=300, format='pdf')
    plt.show()


plt.rc('text', usetex=False); 
plt.rc('font', family='serif');
plt.rc('font', family='serif');

### TRAINING AND LOSS PLOTS 
train_loss = np.loadtxt(f"{PATH}/Results/loss_train")
test_loss = np.loadtxt(f"{PATH}/Results/loss_test")
epochs = np.loadtxt(f"{PATH}/Results/epochs")

fig, axs = plt.subplots()

axs.semilogy(epochs[::100], train_loss[::100], label='Training Loss')
axs.semilogy(epochs[::100], test_loss[::100], label='Testing Loss')
axs.set_xlabel('Epochs', fontsize=20)
axs.set_ylabel('Loss', fontsize=20)

axs.set_xticks([0, 60000, 120000])
axs.set_xticklabels(['0', '60k', '120k'], fontsize=16)
axs.set_yticks([0.6, 0.4, 0.3 ,0.2])
axs.set_yticklabels(['0.6', '0.4', '0.3', '0.2'], fontsize=16)
axs.tick_params(axis='y', labelsize=20)
axs.tick_params(axis='x', labelsize=20)

plt.legend(fontsize=16)
plt.tight_layout()
plt.grid(True, which='both', axis='both')
fig.savefig(f'{PATH}/Results/LossPlot.pdf', dpi=300, format='pdf')
plt.show()


####### Representative DeepONet Plots  ########

hyperparameters = io.loadmat(f'{PATH}/Variables/hyperparameters.mat')
hyperparameters_dict = {
    "B_net": hyperparameters["B_net"].tolist()[0],
    "T_net": hyperparameters["T_net"].tolist()[0],
    "bs": int(hyperparameters["bs"]),
    "tsbs": int(hyperparameters["tsbs"]),
    "epochs": int(hyperparameters["epochs"]),
    "num": int(hyperparameters["num"])
}

param = DataSet(hyperparameters_dict["num"], hyperparameters_dict["bs"], f"{PATH}/Results")
x_train, f_train, u_train, Xmin, Xmax = param.minibatch()

weights_and_biases = torch.load(f'{PATH}/Variables/Weight_bias.pt')

model = DeepONet(torch.float, hyperparameters_dict["T_net"], hyperparameters_dict["B_net"]) 
model.trunk_dnn.set_weights_and_biases(weights_and_biases['W_tr'], weights_and_biases['b_tr'])
model.branch_dnn.set_weights_and_biases(weights_and_biases['W_br_fnn'], weights_and_biases['b_br_fnn'])

print('Running some test predictions')
predict_test = Predict(model, torch.float, f"{PATH}/Results")

X_predict = torch.from_numpy(param.all_X()).float()
F_predict = torch.from_numpy(predict_test.get_random_F(param.Fmin, param.Fmax, num=4)).float()
U_predict = predict_test.predict_field(X_predict, F_predict, 1, torch.from_numpy(Xmin).float(), torch.from_numpy(Xmax).float())

x_test, f_test, u_test, batch_id = param.printbatch(bs=4, batch_id=[5,89, 21])
u_predict = predict_test.predict_field(torch.from_numpy(x_test).float(), torch.from_numpy(f_test).float(), 1, torch.from_numpy(Xmin).float(), torch.from_numpy(Xmax).float())

plot_three_old(x_test, f_test, u_test, u_predict, u_test - u_predict.cpu().detach().numpy(), title="Comparison", save_results_to=f"{PATH}/Results")


