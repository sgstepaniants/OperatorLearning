import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import h5py
import random

import os.path
import sys
sys.path.append("../..")
from losses import relative_error
from rkhs_functions import RKHSFunction
from helper import trapezoid_rule, standard_deviation

from training_functions import train_rkhs_pde

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


###############################################################################
#   Generate Poisson/Helmholtz PDE Forcings and Solutions
###############################################################################

kernel_width_train = 0.01
sigma = 0.0

L = 1
left_dbc = -0.1
right_dbc = 0.1
w = 0
#w = 20

ind_divisor = 0

# whether bias term exists or not
bias = True

# true Green's kernel
mx_true = 500
x_true = torch.linspace(0, L, mx_true).type(tensor)
X, Y = torch.meshgrid(x_true, x_true, indexing="ij")
G_true = torch.zeros((mx_true, mx_true)).type(tensor)
if np.abs(w) < 1e-15:
    G_true = (X + Y - torch.abs(Y - X)) / 2 - X * Y
else:
    modes = 100
    for k in range(1, modes):
        p = math.pi*k/L
        G_true += 2/L * torch.sin(p*X) * torch.sin(p*Y) / (p**2 - w**2)

beta_true = None
if bias:
    # true bias term
    beta_true = (right_dbc - left_dbc) * x_true + left_dbc
    if np.abs(w) >= 1e-15:
        A = (right_dbc - left_dbc * math.cos(w)) / math.sin(w)
        B = left_dbc
        beta_true = A*torch.sin(w*x_true) + B*torch.cos(w*x_true)



###############################################################################
#   Sweep over Kernel Hyperparameters (kernel, spacing, width, regularization)
###############################################################################
num_train = 500
epochs = 1000
batch_size = 100

greens_out_meshes_true = (x_true, x_true)
bias_out_meshes_true = (x_true,)

kernels = ["Exponential", "SquaredExponential", "Matern3/2", "Matern5/2"]
mx_weights = np.arange(10, 101, 10)
rkhs_kernel_widths = np.hstack([np.arange(1, 10) * 1e-3, np.arange(1, 10) * 1e-2, np.arange(1, 10) * 1e-1, 1])
lambdas = [1e-6] #[0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

sweeps_saved = None
sweeps_file = "poisson_sweep.npz"
#sweeps_file = "helmholtz_sweep.npz"
if os.path.exists(sweeps_file):
    sweeps_saved = np.load(sweeps_file)

sweep_num = int(sys.argv[1])
total_sweeps = int(sys.argv[2])

combinations = []
for i in range(len(kernels)):
    for j in range(len(mx_weights)):
        for k in range(len(rkhs_kernel_widths)):
            for l in range(len(lambdas)):
                if sweeps_saved is None or np.isnan(sweeps_saved["G_res"][i, j, k, l]):
                    combinations.append((i, j, k, l))
random.Random(4).shuffle(combinations)

total_combinations = len(combinations)
sweep_size = math.ceil(total_combinations / total_sweeps)

sweep_combinations = combinations[(sweep_num-1)*sweep_size:sweep_num*sweep_size]
print(len(sweep_combinations), flush=True)


mx_interps = [100, 250, 500, 750, 1000]

for combination in sweep_combinations:
    i, j, k, l = combination
    kernel = kernels[i]
    mx_weight = mx_weights[j]
    rkhs_kernel_width = rkhs_kernel_widths[k]
    lmbda = lambdas[l]
    
    print(f"Kernel: {kernel}", flush=True)
    print(f"Weight Mesh: {mx_weight}", flush=True)
    print(f"RKHS Kernel Width: {rkhs_kernel_width}", flush=True)
    print(f"Lambda: {lmbda}", flush=True)
    
    mx_train = mx_weight
    data = h5py.File("../../generate_data/poisson_gaussianKLE1D_dirbc_small.hdf5", "r")
    #data = h5py.File("../../generate_data/helmholtz_gaussianKLE1D_dirbc_small.hdf5", "r")
    x_train = data[f"mesh{mx_train}"].attrs["mesh"][0]
    
    fs_train = data[f"mesh{mx_train}"][f"kernelwidth{kernel_width_train}"]["forcings"][:num_train, :]
    us_train_clean = data[f"mesh{mx_train}"][f"kernelwidth{kernel_width_train}"]["solutions"][:num_train, :]
    
    x_train = torch.from_numpy(x_train).type(tensor)
    fs_train = torch.from_numpy(fs_train).type(tensor)
    us_train_clean = torch.from_numpy(us_train_clean).type(tensor)
    
    # add noise to solutions at a given SNR
    us_train = us_train_clean + sigma * standard_deviation(us_train_clean) * torch.randn(num_train, mx_train).type(tensor)
    
    greens_out_meshes = (x_train, x_train)
    bias_out_meshes = (x_train,)
    
    x_weight = x_train
    greens_weight_meshes = (x_weight, x_weight)
    bias_weight_meshes = (x_weight,)
    greens_function, bias_term, _, _, res, G_res, beta_res, G_norms, beta_norms = \
                train_rkhs_pde(fs_train, us_train, ind_divisor, kernel, rkhs_kernel_width,
                               greens_out_meshes, greens_weight_meshes, greens_out_meshes_true=greens_out_meshes_true,
                               bias_out_meshes=bias_out_meshes, bias_weight_meshes=bias_weight_meshes, bias_out_meshes_true=bias_out_meshes_true,
                               G_true=G_true, beta_true=beta_true,
                               greens_lmbda=lmbda, bias_lmbda=lmbda,
                               batch_size=batch_size, epochs=epochs,
                               greens_learning_rate=None, bias_learning_rate=None,
                               init_re_thresh=1,
                               verbal=True)
    
    best_ind = np.argmin(res)
    print(f"RE ({i}, {j}, {k}, {l}): {res[best_ind]}", flush=True)
    print(f"Green's Function RE ({i}, {j}, {k}, {l}): {G_res[best_ind]}", flush=True)
    print(f"Bias Term RE ({i}, {j}, {k}, {l}): {beta_res[best_ind]}", flush=True)
    for m in range(len(mx_interps)):
        mx_interp = mx_interps[m]
        x_interp = torch.linspace(x_train.min(), x_train.max(), mx_interp).type(tensor)
        
        greens_out_meshes_interp = (x_interp, x_interp)
        f_quadrature_interp = trapezoid_rule(greens_out_meshes_interp[:ind_divisor+1]).flatten()
        f_size_interp = math.prod([len(x) for x in greens_out_meshes_interp[:ind_divisor+1]])
        u_quadrature_interp = trapezoid_rule(greens_out_meshes_interp[ind_divisor+1:]).flatten()
        u_size_interp = math.prod([len(x) for x in greens_out_meshes_interp[ind_divisor+1:]])
        
        fs_train_interp = np.zeros((num_train, mx_interp))
        us_train_interp = np.zeros((num_train, mx_interp))
        for it in range(num_train):
            fs_train_interp[it, :] = np.interp(x_interp.detach().cpu().numpy(), x_train.detach().cpu().numpy(), fs_train[it, :].detach().cpu().numpy())
            us_train_interp[it, :] = np.interp(x_interp.detach().cpu().numpy(), x_train.detach().cpu().numpy(), us_train[it, :].detach().cpu().numpy())
        fs_train_interp = torch.from_numpy(fs_train_interp).type(tensor)
        us_train_interp = torch.from_numpy(us_train_interp).type(tensor)
        
        greens_function.update_mesh((greens_out_meshes_interp))
        G = greens_function()
        G = torch.reshape(G, (f_size_interp, u_size_interp))
        
        us_hat = (fs_train_interp * f_quadrature_interp) @ G
        if bias_term is not None:
            bias_out_meshes_interp = (x_interp,)
            bias_term.update_mesh(bias_out_meshes_interp)
            us_hat += bias_term()
        
        interp_re = relative_error(us_hat.cpu(), us_train_interp.cpu(), agg="mean").item()
        print(f"Combination ({i}, {j}, {k}, {l}, {m}): {interp_re}", flush=True)
    print("", flush=True)
