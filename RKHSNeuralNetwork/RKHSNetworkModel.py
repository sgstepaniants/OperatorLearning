import math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import cvxopt
from fft_conv import fft_conv
from scipy.optimize import least_squares
from scipy.linalg import svd

from abc import abstractmethod
import copy
import matplotlib.pyplot as plt

from ProdRKHSLayer import *
from BasisRKHSLayer import *
from ConvRKHSLayer import *
from helper_functions import *


##############################################################################
#   Functional Network and RKHS Layer Parent Classes
##############################################################################

class FunctionalNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        self.layers = layers
        
        # scaling constants to multiply neural network inputs and outputs by
        self.in_scale = 1
        self.out_scale = 1
        
        # initialize weights so that each layer has constant L^2 norm
        self.initialize_weights()
    
    def initialize_weights(self):
        K1 = None
        scale = 1
        Sigma = None
        
        # 1^TWAx = 1 so 1^TWA = 1^T so A^Ts = 1 where s is the sum of the rows of W
        for layer in self.layers:
            if isinstance(layer, RKHSLayer) and isinstance(layer.weight, ProdRKHSWeight):
                print('initializing layer')
                lw = layer.weight
                K2 = lw.K2
                scale *= math.prod(lw.delta_y) * math.prod(lw.delta_eta)
                
                if True: #Sigma is None: # assuming all layers have the same reproducing kernel
                    if K1:
                        A = torch.matmul(K2.t(), K1).numpy()**2
                    else:
                        A = K2.t().numpy()**2
                    
                    s0 = np.zeros(math.prod(lw.eta_sizes))
                    f = lambda s : A.T.dot(s) - 1
                    jac = lambda s : A.T
                    res = least_squares(f, s0, jac, bounds=(math.prod(lw.eta_sizes)*[0], np.inf))
                    s = torch.from_numpy(res.x)
                    Sigma = torch.sqrt(s.repeat(math.prod(lw.xi_sizes), 1) / math.prod(lw.xi_sizes))
                
                W = torch.randn([math.prod(lw.xi_sizes), math.prod(lw.eta_sizes)], dtype=torch.float64)   
                W = W * Sigma
                W = W / scale
                W = W * math.sqrt(2) # for ReLU activations
                
                # since the weight matrix is symmetric, only keep its upper/lower triangles
                def triangle_hook(W):
                    if lw.symmetries:
                        W = torch.reshape(W, lw.xi_sizes + lw.eta_sizes)
                        W = triangular_mask(W, lw.symmetries)
                        if lw.causal_pair is not None:
                            W = triangular_mask(W, [lw.causal_pair])
                        W = torch.reshape(W, (math.prod(lw.xi_sizes), math.prod(lw.eta_sizes)))
                    return W
                lw.W = nn.parameter.Parameter(triangle_hook(W))
                #lw.W.register_hook(triangle_hook) # only update upper/lower triangles of gradient
                
                K1 = lw.K1
                scale = math.prod(lw.delta_xi)
            else:
                K1 = None
                scale = 1
    
    def forward(self, f):
        f = f.clone().double()
        f *= self.in_scale
        #print(torch.sqrt(torch.sum(f**2)*math.prod(self.layers[0].delta_y)))
        
        # run new data points through network
        for layer in self.layers:
            f = layer(f).clone()
        #print(torch.sqrt(torch.sum(f**2)*math.prod(self.layers[0].delta_x)))

        f *= self.out_scale
        #print(torch.sqrt(torch.sum(f**2)*math.prod(self.layers[0].delta_x)))
        return f
    
    def set_resolution(self, new_layer_meshes):
        l = 0
        # iterate over each RKHSLayer and update its input and output meshes
        for layer in self.layers:
            if isinstance(layer, RKHSLayer):
                layer.set_resolution(new_layer_meshes[l+1], new_layer_meshes[l])
                l += 1
    
    def set_basis_shape(self, new_in_basis_shapes, new_out_basis_shapes):
        l = 0
        # iterate over each BasisRKHSWeight and update its input and output meshes
        for layer in self.layers:
            if isinstance(layer.weight, BasisRKHSWeight):
                layer.weight.set_basis_shape(new_in_basis_shapes[l], new_out_basis_shapes[l])
                l += 1
    
    def compute_regularization(self, weight_lambdas=None, bias_lambdas=None):
        regularization = 0
        l = 0
        for layer in self.layers:
            if isinstance(layer, RKHSLayer):
                if weight_lambdas is not None:
                    regularization += weight_lambdas[l] * layer.weight.rkhs_norm()
                if bias_lambdas is not None and layer.bias is not None:
                    regularization += bias_lambdas[l] * layer.bias.rkhs_norm()
        return regularization
    
    def rescale(self, in_scale, out_scale):
        self.in_scale = in_scale
        self.out_scale = out_scale
    
    def to(self, device):
        super().to(device)
        self.layers = self.layers.to(device)
