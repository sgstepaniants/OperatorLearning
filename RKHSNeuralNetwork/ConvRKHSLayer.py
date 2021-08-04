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

from RKHSNetworkModel import *


##############################################################################
#   Convolutional RKHS Layers
##############################################################################

# conv layer has no eta mesh because we assume that input and output domains are the same
class ConvRKHSWeight(nn.Module):
    def __init__(self, x_mesh, xi_mesh, bounds, symmetries=[], causal_pair=None):
        #super().__init__()
        
        # create kernel for convolutional Green's function
        self.register_buffer('K', None)
        
        # set properties of meshes for third and fourth coordinates
        self.xi_mesh = xi_mesh
        self.xi_dim, self.xi_sizes, self.xi_mesh_stack, self.xi_mesh_flat, self.delta_xi = mesh_stats(self.xi_mesh)
        
        # symmetrize xi mesh
        self.xi_symm_mesh = symmetrize_mesh(self.xi_mesh)
        self.xi_symm_dim, self.xi_symm_sizes, self.xi_symm_mesh_stack, self.xi_symm_mesh_flat, self.delta_xi_symm = mesh_stats(self.xi_symm_mesh)
        
        # set properties of input/output mesh
        self.x_dim = None
        self.x_mesh = None
        self.x_sizes = None
        self.x_mesh_stack = None
        self.x_mesh_flat = None
        self.delta_x = None
        
        self.bounds = bounds
        self.filter_mesh = None
        self.filter_dim = None
        self.filter_sizes = None
        self.filter_mesh_stack = None
        self.filter_mesh_flat = None
        self.delta_filter = None
        self.set_resolution(x_mesh)
        
        # weights of neural network before convolved with kernels
        w = torch.randn(math.prod(self.xi_symm_sizes), dtype=torch.float64)
        self.w = nn.Parameter(w)
        
        # null space coefficients
        self.ds = None
        nullspan = self.nullspan()
        if nullspan is not None:
            ds = torch.zeros(nullspan.shape[-1], dtype=torch.float64)
            self.ds = nn.Parameter(ds)
    
    def forward(self, f):
        f = preprocess_input(f, self.x_sizes)
        
        #G = self.G()
        #output = torch.matmul(G, f) * math.prod(self.delta_x)
        
        n = f.shape[1]
        f = torch.reshape(f, self.x_sizes + (n,))
        f = f.permute([self.x_dim] + list(range(0, self.x_dim))).unsqueeze(1)
        
        filt = self.get_filter().unsqueeze(0).unsqueeze(1)
        padding = tuple(map(lambda i: i//2, self.filter_sizes))
        
        #output = nn.functional.conv1d(f, filt, padding=padding) * math.prod(self.delta_x)
        output = nn.functional.conv2d(f, filt, padding=padding) * math.prod(self.delta_x)
        
        output = output.squeeze(1).permute(list(range(1, self.x_dim+1)) + [0])
        output = torch.reshape(output, (math.prod(self.x_sizes), n))
        
        return output
    
    @abstractmethod
    def set_kernel(self):
        pass
    
    def set_resolution(self, x_mesh):
        self.x_mesh = x_mesh
        self.x_dim, self.x_sizes, self.x_mesh_stack, self.x_mesh_flat, self.delta_x = mesh_stats(x_mesh)
        
        if self.x_dim != self.xi_dim:
            ValueError('x_mesh and xi_mesh must be of the same dimension')
        
        # create filter mesh
        self.filter_mesh = symmetrize_mesh(self.x_mesh)
        self.filter_mesh = bound_mesh(self.filter_mesh, self.bounds)
        self.filter_dim, self.filter_sizes, self.filter_mesh_stack, self.filter_mesh_flat, self.delta_filter = mesh_stats(self.filter_mesh)
        
        # set the resolution of the kernel
        self.set_kernel()
    
    # think about whether symmetry for a convolutional kernel makes sense
    def make_symmetric(self):
        pass
    
    def get_filter(self):
        # add Green's function component in H1
        g = torch.matmul(self.K, self.w) * math.prod(self.delta_xi_symm)
        g = torch.reshape(g, self.filter_sizes)
        
        # add Green's function component in H0
        if self.ds is not None:
            g_null_terms = torch.sum(self.nullspan() * self.ds[None, None, :], dim=2)
            g += g_null_terms
        
        return g
    
    def to_matrix(self):
        G = torch.reshape(hankel(self.get_filter()), (math.prod(self.x_sizes), math.prod(self.x_sizes)))
        return G
    
    @abstractmethod
    def nullspan(self):
        pass
    
    # no way to regularize a convolutional Green's function
    def rkhs_norm(self):
        return 0


class GaussianConvRKHSWeight(ConvRKHSWeight):
    def __init__(self, Sigma, x_mesh, xi_mesh, bounds, symmetries=[], causal_pair=None):
        nn.Module.__init__(self)
        
        # create covariance matrix for Gaussian mixture convolutional kernel
        self.register_buffer('Sigma', Sigma.double())
        
        super().__init__(x_mesh, xi_mesh, bounds, symmetries, causal_pair)
    
    def set_kernel(self):
        self.K = gaussian_kernel(self.filter_mesh_flat, self.filter_sizes, self.xi_symm_mesh_flat, self.xi_symm_sizes, self.Sigma)
        
    def nullspan(self):
        return None


class GaussianConvRKHSLayer(RKHSLayer):
    def __init__(self, Sigma, x_mesh, xi_mesh, bounds, bias=False, symmetries=[], causal_pair=None):
        nn.Module.__init__(self)
        super().__init__()
        self.weight = GaussianConvRKHSWeight(Sigma, x_mesh, xi_mesh, bounds, symmetries, causal_pair)
        self.bias = None
        if bias:
            self.bias = GaussianSimpleRKHSBias(Sigma, x_mesh, xi_mesh, symmetries)
