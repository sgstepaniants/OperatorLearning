import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from scipy.special import gamma, kv

from losses import squared_error, relative_error
from rkhs_functions import RKHSFunction
from helper import trapezoid_rule


def decrement_learning_rate(lr, vals = np.array([1, 5])):
    log_lr = math.log10(lr)
    all_lrs = vals * 10**math.floor(log_lr)
    less_inds = np.where(all_lrs < lr)[0]
    if len(less_inds) == 0:
        return vals[-1] * 10**(math.floor(log_lr)-1)
    return vals[less_inds[-1]] * 10**math.floor(log_lr)


def train_rkhs_pde(fs_train, us_train, ind_divisor, kernel, sigma,
                   greens_out_meshes, greens_weight_meshes, greens_out_meshes_true=None, G_true=None,
                   bias_out_meshes=None, bias_weight_meshes=None, bias_out_meshes_true=None, beta_true=None,
                   greens_weight_quadrature=None, bias_weight_quadrature=None, f_quadrature=None, u_quadrature=None,
                   greens_lmbda=0, bias_lmbda=0,
                   batch_size=None, epochs=1000,
                   greens_learning_rate=None, bias_learning_rate=None, init_re_thresh=1, init_runs_max=20,
                   greens_weight_parametrizations=None, bias_weight_parametrizations=None,
                   greens_transform_output=None, bias_transform_output=None,
                   plotting_function=None, plotting_freq=math.inf, verbal=True):
    num_train = fs_train.shape[0]
    dtype = fs_train.type()
    assert num_train == us_train.shape[0]
    
    # whether or not to include a bias term in the fit
    bias = True
    if bias_out_meshes is None or bias_weight_meshes is None:
        bias = False
    
    #f_size = math.prod([len(x) for x in greens_out_meshes[:ind_divisor+1]])
    #u_size = math.prod([len(x) for x in greens_out_meshes[ind_divisor+1:]])
    f_size = fs_train.shape[1]
    u_size = us_train.shape[1]
    
    if f_quadrature is None:
        f_quadrature = trapezoid_rule(greens_out_meshes[:ind_divisor+1]).flatten()
    if u_quadrature is None:
        u_quadrature = trapezoid_rule(greens_out_meshes[ind_divisor+1:]).flatten()
    if greens_weight_quadrature is None:
        greens_weight_quadrature = trapezoid_rule(greens_weight_meshes).flatten()
    
    greens_function = RKHSFunction(greens_out_meshes, greens_weight_meshes,
                                   kernel=kernel, sigma=sigma, dtype=dtype,
                                   weight_parametrizations=greens_weight_parametrizations,
                                   transform_output=greens_transform_output)
    
    num = math.prod([len(x) for x in greens_weight_meshes])
    ones = torch.ones(num).type(dtype)
    greens_K_frob = torch.sqrt(torch.sum((greens_function.K_norm**2) @ (greens_weight_quadrature**2))).item()
    greens_prefactor = math.sqrt(num) / greens_K_frob
    
    greens_learning_rate_sweep = False
    bias_learning_rate_sweep = False
    if greens_learning_rate is None:
        greens_learning_rate_sweep = True
        greens_learning_rate = 100
    
    # add Green's function weight to optimization parameters
    parameters = [{'params': list(greens_function.parameters())[0], 'lr': greens_prefactor * greens_learning_rate}]
    
    bias_term = None
    if bias:
        if bias_weight_quadrature is None:
            bias_weight_quadrature = trapezoid_rule(bias_weight_meshes).flatten()
        bias_term = RKHSFunction(bias_out_meshes, bias_weight_meshes,
                                 kernel=kernel, sigma=sigma, dtype=dtype,
                                 weight_parametrizations=bias_weight_parametrizations,
                                 transform_output=bias_transform_output)
        
        num = math.prod([len(x) for x in bias_weight_meshes])
        ones = torch.ones(num).type(dtype)
        bias_K_frob = torch.sqrt(torch.sum((bias_term.K_norm**2) @ (bias_weight_quadrature**2))).item()
        bias_prefactor = math.sqrt(num) / bias_K_frob
        
        if bias_learning_rate is None:
            bias_learning_rate_sweep = True
            bias_learning_rate = 100
        
        # add bias term weight to optimization parameters
        parameters += [{'params': list(bias_term.parameters())[0], 'lr': bias_prefactor * bias_learning_rate}]
    
    # print learning rates
    if verbal:
        print(f'Green\'s Function Learning Rate: {greens_learning_rate}', flush=True)
        if bias:
            print(f'Bias Term Learning Rate: {bias_learning_rate}', flush=True)
    
    # initialize optimizer
    optimizer = torch.optim.Adam(parameters, amsgrad=True)
    
    # set batch size
    if batch_size is None:
        batch_size = num_train
    
    # save best Green's function and bias term
    best_re = math.inf
    best_greens_function = greens_function.deepcopy()
    best_bias_term = None
    if bias:
        best_bias_term = bias_term.deepcopy()
    
    # iterate to find the optimal network parameters
    freq = 10
    res = []
    G_res = []
    beta_res = []
    G_norms = []
    beta_norms = []
    init_runs = 1
    epoch = 1
    while epoch <= epochs:
        perm = torch.randperm(num_train)
        for i in range(0, num_train, batch_size):
            inds = perm[i:i+batch_size]
            fs_batch = fs_train[inds, :]
            us_batch = us_train[inds, :]
            
            G = greens_function()
            G = torch.reshape(G, (f_size, u_size))
            
            us_hat = (fs_batch * f_quadrature) @ G
            if bias:
                beta = bias_term()
                us_hat += beta

            loss = squared_error(us_hat, us_batch, quadrature=u_quadrature, agg="mean")
            loss += greens_lmbda * greens_function.square_norm()
            if bias:
                 loss += bias_lmbda * bias_term.square_norm()

            optimizer.zero_grad()
            # prevent gradient measurement to accumulate
            loss.backward()

            # calculate gradient in each iteration
            optimizer.step()

        # recompute Green's function and bias term after optimization
        G = greens_function()
        G = torch.reshape(G, (f_size, u_size))
        if bias:
            beta = bias_term()
        
        us_train_hat = (fs_train * f_quadrature) @ G
        if bias:
            us_train_hat += beta
        
        if epoch % freq == 0 and plotting_function is not None:
            plotting_function(us_train_hat)
        
        # compute train re over all data
        train_re = relative_error(us_train_hat.cpu(), us_train.cpu(), agg="mean").item()
        if verbal:
            print(f'Epoch {epoch} Relative Error {train_re}', flush=True)
        
        if (greens_learning_rate_sweep or bias_learning_rate_sweep) and \
            epoch == 1 and train_re > init_re_thresh and init_runs < init_runs_max:
            if greens_learning_rate_sweep:
                greens_learning_rate = decrement_learning_rate(greens_learning_rate)
                greens_function = RKHSFunction(greens_out_meshes, greens_weight_meshes,
                                   kernel=kernel, sigma=sigma, dtype=dtype,
                                   weight_parametrizations=greens_weight_parametrizations,
                                   transform_output=greens_transform_output)
                parameters = [{'params': list(greens_function.parameters())[0], 'lr': greens_prefactor * greens_learning_rate}]

            if bias and bias_learning_rate_sweep:
                bias_term = RKHSFunction(bias_out_meshes, bias_weight_meshes,
                                 kernel=kernel, sigma=sigma, dtype=dtype,
                                 weight_parametrizations=bias_weight_parametrizations,
                                 transform_output=bias_transform_output)
                bias_learning_rate = decrement_learning_rate(bias_learning_rate)
                parameters += [{'params': list(bias_term.parameters())[0], 'lr': bias_prefactor * bias_learning_rate}]
            
            # reinitialize optimizer with new learning rates
            optimizer = torch.optim.Adam(parameters,
                                         amsgrad=True)
            
            if verbal:
                print(f'First epoch had re = {train_re} > {init_re_thresh}', flush=True)
                print(f'Green\'s Function Learning Rate: {greens_learning_rate}', flush=True)
                if bias:
                    print(f'Bias Term Learning Rate: {bias_learning_rate}', flush=True)

            init_runs += 1
            continue
        
        # add train relative error to list of relative errors, update best Green's function and bias term
        res.append(train_re)
        if train_re < best_re:
            best_re = train_re
            best_greens_function = greens_function.deepcopy()
            if bias:
                best_bias_term = bias_term.deepcopy()
        
        # increment epoch counter
        epoch += 1
        
        # compute re between true and predicted Green's kernel and bias
        G_norm = greens_function.square_norm().cpu().detach().item()
        G_norms.append(G_norm)
        if greens_out_meshes_true is not None and G_true is not None:
            greens_function.update_mesh(greens_out_meshes_true)
            G = greens_function()
            G = torch.reshape(G, G_true.shape)
            G_re = relative_error(G, G_true, agg="none", dim=(0, 1)).item()
            G_res.append(G_re)
            greens_function.update_mesh(greens_out_meshes)
        
        if bias:
            beta_norm = bias_term.square_norm().cpu().detach().item()
            beta_norms.append(beta_norm)
            if bias_out_meshes_true is not None and beta_true is not None:
                bias_term.update_mesh(bias_out_meshes_true)
                beta = bias_term()
                beta_re = relative_error(beta, beta_true, agg="none", dim=0).item()
                beta_res.append(beta_re)
                bias_term.update_mesh(bias_out_meshes)
    
    return best_greens_function, best_bias_term, greens_function, bias_term, res, G_res, beta_res, G_norms, beta_norms
