'''
Script to generate plots for the advection-diffusion example.

Must point to correct folder path -- uses absolute paths.
'''

import matplotlib.pyplot as plt  
import numpy as np  
import scipy.io as io
import torch

from scipy.interpolate import griddata

from model_predict import Predict, plot_solution

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator

from matplotlib import gridspec
from dataset_evecs_diffusionconvection import DataSet

from DeepONet import DeepONet

FOLDER_PATH = "/Users/anastasia/Documents/Education/PhD/EN.560.617 Deep Learning/Project/DeepLearningProject"
RESULTS_PATH = f"{FOLDER_PATH}/DeepONet/DiffusionConvection/Case_0/"
DHPM_PATH = f"{FOLDER_PATH}/HiddenPhysics/Data"

def plot_three_imshow(X, F, U, U_P, U_E, error_min=0.05, title=None, save_results_to=None):
    if not isinstance(X, np.ndarray):
        X = X.cpu().detach().numpy()

    if not isinstance(U, np.ndarray):
        U = U.cpu().detach().numpy()

    if not isinstance(U_P, np.ndarray):
        U_P = U_P.cpu().detach().numpy()

    if not isinstance(U_E, np.ndarray):
        U_E = U_E.cpu().detach().numpy()

    if not isinstance(F, np.ndarray):
        F = F.cpu().detach().numpy()
    print(f" f shape {F.shape}")

    # Create custom colormap with white at 0
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    nodes = [0.0, 0.5, 1.0]  
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    # Normalize colors to ensure 0 is white
    norm = Normalize(vmin=-error_min, vmax=error_min) 

    # Create grid data for interpolation
    xi = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yi = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    fig, axes = plt.subplots(nrows=3, ncols=3)
    figsize = fig.get_size_inches()
    print("Figure size: width = {}, height = {}".format(figsize[0], figsize[1]))

    axes = axes.ravel()

    for i in range(3):
        zi = griddata(X, U[i, :], (xi, yi), method='cubic')
        im = axes[i].imshow(zi, aspect='auto', extent=(X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()), origin='upper')
        if i == 0:
            axes[i].set_xticks([]) 
            axes[i].yaxis.set_major_locator(MaxNLocator(2))
            axes[i].set_ylabel(r"$\hat{\tau}$", labelpad=1)
        else: 
            axes[i].set_xticks([]) 
            axes[i].set_yticks([]) 
        if F.shape[-1] > 1:
            mini_title = [round(num, 2) for num in F[i, ...].squeeze()]
        else:
            mini_title = round(F.squeeze().reshape(-1)[i], 2)
        axes[i].set_title(r"$\hat{\pi}$: " + str(mini_title))

    for i, v in enumerate(range(3, 6)):
        zi = griddata(X, U_P[i, :], (xi, yi), method='cubic')
        im = axes[v].imshow(zi, aspect='auto', extent=(X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()), origin='upper')
        if i == 0:
            axes[v].set_xticks([])
            axes[v].yaxis.set_major_locator(MaxNLocator(2))
            axes[v].set_ylabel(r"$\hat{\tau}$", labelpad=1)
        else: 
            axes[v].set_xticks([]) 
            axes[v].set_yticks([]) 
        if F.shape[-1] > 1:
            mini_title = [round(num, 2) for num in F[i, ...].squeeze()]
        else:
            mini_title = round(F.squeeze().reshape(-1)[i], 2)

    for i, v in enumerate(range(6, 9)):
        zi = griddata(X, U_E[i, :], (xi, yi), method='cubic')
        im = axes[v].imshow(zi, aspect='auto', extent=(X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()), origin='upper', cmap=custom_cmap, norm=norm)
        if i == 0:
            axes[v].xaxis.set_major_locator(MaxNLocator(2))
            axes[v].yaxis.set_major_locator(MaxNLocator(2))
            axes[v].set_ylabel(r"$\hat{\tau}$", labelpad=1)
        else: 
            axes[v].xaxis.set_major_locator(MaxNLocator(2))
            axes[v].set_yticks([]) 
        axes[v].set_xlabel(r"$\hat{\zeta}$", labelpad=1)
        if F.shape[-1] > 1:
            mini_title = [round(num, 2) for num in F[i, ...].squeeze()]
        else:
            mini_title = round(F.squeeze().reshape(-1)[i], 2)

    # Add a common colorbar at the bottom for the error values
    cbar_ax_bottom = fig.add_axes([0.1, 0.06, 0.6, 0.02])  
    cbar_bottom = fig.colorbar(im, cax=cbar_ax_bottom, orientation='horizontal')
    cbar_bottom.set_label('Error', labelpad=-10)
    cbar_bottom.set_ticks([-error_min, error_min])

    # Add a common colorbar at the top for the first two rows
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=U.min(), vmax=U.max()))
    sm._A = []  
    cbar_ax_top = fig.add_axes([0.1, 0.94, 0.6, 0.02])  
    cbar_top = fig.colorbar(sm, cax=cbar_ax_top, orientation='horizontal')
    cbar_top.set_label(r'$\hat{C}$', labelpad=-25)
    cbar_top.ax.xaxis.set_label_position('top')
    cbar_top.set_ticks([0, 0.94])

    # Add vertical labels
    fig.text(0.72, 0.74, 'Emergent\n Truth', va='center', rotation='vertical')
    fig.text(0.72, 0.51, 'Predicted', va='center', rotation='vertical')
    fig.text(0.72, 0.29, 'Error', va='center', rotation='vertical')
    
    plt.tight_layout()

    # Adjust padding
    plt.subplots_adjust(hspace=0.3, wspace=0.21, top=0.82, bottom=0.2, left=0.1, right=0.706)

    if save_results_to is None:
        save_results_to = '/Users/anastasia/Documents/Education/PhD/EN.560.617 Deep Learning/Project/DeepLearningProject/DeepONet/Case_1/Results'
    fig.savefig(f'{save_results_to}/{title}_imshow.pdf', dpi=300, format='pdf')
    plt.show()

def plot_dhpm(t_c, t, u_pred, U, x_c, x, save_results_to):
    # Determine common color limits
    vmin = min(U.min(), u_pred[:, 0].min())
    vmax = max(U.max(), u_pred[:, 0].max())

    # plt.rc('xtick', labelsize=16)
    # plt.rc('ytick', labelsize=16)
    # plt.rc('axes', labelsize=20, titlesize=20)
    # plt.rc('figure', titlesize=20)

    # Plot settings
    fig, axs = plt.subplots(3, 1, figsize=(6.4/3, 5.2-0.5))

    # True Solution
    X = np.concatenate((x.reshape(-1,1), t.reshape(-1,1)), axis=1)

    # Create grid data for interpolation
    xi = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yi = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(X, U, (xi, yi), method='cubic')
    sc1 = axs[0].imshow(zi, aspect='auto', extent=(X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()), origin='lower')
    
    axs[0].set_ylabel(r"$\hat{\tau}$", labelpad=-5)
    axs[0].set_xticks([])

    # Colocation Points
    sc11 = axs[0].scatter(x_c, t_c, color="lightgray", marker="+", zorder=100, s=2)

    # Set fewer x and y ticks
    axs[0].yaxis.set_major_locator(MaxNLocator(2))  # Set max 5 y-ticks

    # Predicted Solution
    zi = griddata(X, u_pred, (xi, yi), method='cubic')
    sc2 = axs[1].imshow(zi, aspect='auto', extent=(X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()), origin='lower')

    axs[1].set_ylabel(r"$\hat{\tau}$", labelpad=-5)
    axs[1].set_xticks([])
    axs[1].yaxis.set_major_locator(MaxNLocator(2)) 

    # Error Plot
    from matplotlib.colors import LinearSegmentedColormap

    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    nodes = [0.0, 0.5, 1.0]  # Define the positions of the colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    error = U - u_pred[:, 0]
    zi = griddata(X, error, (xi, yi), method='cubic')
    sc3 = axs[2].imshow(zi, aspect='auto', extent=(X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()), origin='lower',
                        cmap="coolwarm", vmin=-0.6, vmax=0.6)
    axs[2].set_xlabel(r"$\hat{\zeta}$", labelpad=1)
    axs[2].set_ylabel(r"$\hat{\tau}$", labelpad=-5)

    # Set fewer x and y ticks
    axs[2].xaxis.set_major_locator(MaxNLocator(2))  # Set max 5 x-ticks
    axs[2].yaxis.set_major_locator(MaxNLocator(2))  # Set max 5 y-ticks

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create colorbar for the first plot at the top
    cbar_ax_top = fig.add_axes([0.22, 0.94, 0.6, 0.02])  
    cbar_top = fig.colorbar(sc1, cax=cbar_ax_top, orientation='horizontal')
    cbar_top.ax.xaxis.set_label_position('top')
    cbar_top.set_label(r"$\hat{C}$", labelpad=-25)
    cbar_top.set_ticks([0, 0.84])

    # Create colorbar for the third plot spanning across the bottom
    cbar_ax_bottom = fig.add_axes([0.22, 0.05, 0.6, 0.02])  
    cbar_bottom = fig.colorbar(sc3, cax=cbar_ax_bottom, orientation='horizontal')
    cbar_bottom.set_label('Error', labelpad=-10)
    cbar_bottom.set_ticks([-0.6, 0.6])

    # Add vertical labels for each subplot
    fig.text(0.85, 0.73, ' ', va='center', rotation='vertical')
    fig.text(0.85, 0.51, ' ', va='center', rotation='vertical')
    fig.text(0.85, 0.28, ' ', va='center', rotation='vertical')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.283, wspace=0.21, top=0.82, bottom=0.19, right=0.8, left=0.252)


    plt.show()
    fig.savefig(f"{save_results_to}/hp_pipe_error_vertical.pdf", dpi=300, format='pdf')

def plot_loss(train_loss, test_loss, epochs):

    fig = plt.figure()

    # plt.rc('xtick', labelsize=16); 
    # plt.rc('ytick', labelsize=16);
    # plt.rc('axes', labelsize=20, titlesize=20); 
    # plt.rc('figure', titlesize=20); 

    fig, ax = plt.subplots()
    ax.semilogy(epochs[::100], train_loss[::100], label='Training Loss')
    ax.semilogy(epochs[::100], test_loss[::100], label='Testing Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=16)
    ax.grid(True, which='both', axis='both')

    # Setting specific x-axis ticks
    ax.set_xticks([0, 250000, 500000])
    ax.set_xticklabels(['0', '250k', '500k'])

    plt.tight_layout()
    fig.savefig(f'{RESULTS_PATH}/Figures/LossPlot.pdf', dpi=300, format='pdf')
    plt.show()

if __name__ == "__main__":

    plt.rc('text', usetex=False); 
    plt.rc('font', family='serif');
    plt.rc('font', family='serif');

    ###### TRAINING AND LOSS PLOTS  #######
    train_loss = np.loadtxt(f"{RESULTS_PATH}/Results/loss_train")
    test_loss = np.loadtxt(f"{RESULTS_PATH}/Results/loss_test")
    epochs = np.loadtxt(f"{RESULTS_PATH}/Results/epochs")
    plot_loss(train_loss, test_loss, epochs)
    ##########


    ###### PREDICTION PLOTS  #######

    hyperparameters = io.loadmat(f'{RESULTS_PATH}/Variables/hyperparameters.mat')
    hyperparameters_dict = {
        "B_net": hyperparameters["B_net"].tolist()[0],
        "T_net": hyperparameters["T_net"].tolist()[0],
        "bs": int(hyperparameters["bs"]),
        "tsbs": int(hyperparameters["tsbs"]),
        "epochs": int(hyperparameters["epochs"]),
        "num": int(hyperparameters["num"])
    }

    param = DataSet(hyperparameters_dict["num"], hyperparameters_dict["bs"], f"{RESULTS_PATH}/Results")
    x_train, f_train, u_train, Xmin, Xmax = param.minibatch()

    weights_and_biases = torch.load(f'{RESULTS_PATH}/Variables/Weight_bias.pt')

    model = DeepONet(torch.float, hyperparameters_dict["T_net"], hyperparameters_dict["B_net"]) 
    model.trunk_dnn.set_weights_and_biases(weights_and_biases['W_tr'], weights_and_biases['b_tr'])
    model.branch_dnn.set_weights_and_biases(weights_and_biases['W_br_fnn'], weights_and_biases['b_br_fnn'])

    print('Running some test predictions')
    predict_test = Predict(model, torch.float, f"{RESULTS_PATH}/Results")

    X_predict = torch.from_numpy(param.all_X()).float()
    F_predict = torch.from_numpy(predict_test.get_random_F(param.Fmin, param.Fmax, num=26)).float()

    U_predict = predict_test.predict_field(X_predict, F_predict, 1, torch.from_numpy(Xmin).float(), torch.from_numpy(Xmax).float())
    x_test, f_test, u_test, batch_id = param.printbatch(bs=4)
    u_predict = predict_test.predict_field(torch.from_numpy(x_test).float(), torch.from_numpy(f_test).float(), 1, torch.from_numpy(Xmin).float(), torch.from_numpy(Xmax).float())

    plot_three_imshow(x_test, f_test, u_test, u_predict, u_test - u_predict.cpu().detach().numpy(), title="Comparison", save_results_to=f"{RESULTS_PATH}/Figures")

    dhpm_t_c = np.load(f"{DHPM_PATH}/t_c.npy")
    dhpm_t = np.load(f"{DHPM_PATH}/t.npy")
    dhpm_u_pred = np.load(f"{DHPM_PATH}/u_pred.npy")
    dhpm_U = np.load(f"{DHPM_PATH}/U.npy")
    dhpm_x_c = np.load(f"{DHPM_PATH}/x_c.npy")
    dhpm_x = np.load(f"{DHPM_PATH}/x.npy")

    plot_dhpm(dhpm_t_c, dhpm_t, dhpm_u_pred, dhpm_U, dhpm_x_c, dhpm_x, save_results_to=f"{RESULTS_PATH}/Figures")

    ##########