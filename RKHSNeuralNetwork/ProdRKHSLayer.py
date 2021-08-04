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
from RKHSLayer import *


##############################################################################
#   Prod RKHS Layers
##############################################################################

class ProdRKHSWeight(RKHSWeight):
    def __init__(self, x_mesh, y_mesh, xi_mesh, eta_mesh, symmetries=[], causal_pair=None):
        # create two kernels for (x, xi) and (y, eta)
        self.register_buffer('K1', None)
        self.register_buffer('K2', None)
        
        # set properties of meshes for third and fourth coordinates
        self.xi_mesh = xi_mesh
        self.xi_dim, self.xi_sizes, self.xi_mesh_stack, self.xi_mesh_flat, self.delta_xi = mesh_stats(xi_mesh)
        self.eta_mesh = eta_mesh
        self.eta_dim, self.eta_sizes, self.eta_mesh_stack, self.eta_mesh_flat, self.delta_eta = mesh_stats(eta_mesh)
        
        # set properties of input and output meshes
        self.x_dim = None
        self.y_dim = None
        self.x_mesh = None
        self.y_mesh = None
        self.x_sizes = None
        self.y_sizes = None
        self.x_mesh_stack = None
        self.y_mesh_stack = None
        self.x_mesh_flat = None
        self.y_mesh_flat = None
        self.delta_x = None
        self.delta_y = None
        self.set_resolution(x_mesh, y_mesh)
        
        # weights of neural network before convolved with kernels
        W = torch.randn([math.prod(self.xi_sizes), math.prod(self.eta_sizes)], dtype=torch.float64)
        self.W = nn.Parameter(W)
        #nn.init.kaiming_normal_(self.W)
        
        # null space coefficients
        self.ds = None
        nullspan = self.nullspan()
        if nullspan is not None:
            ds = torch.zeros(nullspan.shape[-1], dtype=torch.float64)
            self.ds = nn.Parameter(ds)
        
        # constraints on the integral operators
        self.symmetries = symmetries
        self.symmx, self.symmy, self.symmxy = filter_symmetries(self.symmetries, self.x_dim, self.y_dim)
        self.causal_pair = causal_pair
        if self.symmetries:
            self.make_symmetric()
            #nn.utils.parametrize.register_parametrization(layer, "weight", Symmetric(self.xi_sizes, self.eta_sizes, self.symmetries))
    
    def forward(self, f):
        f = preprocess_input(f, self.y_sizes)
        
        # add Green's function
        G = self.to_matrix()
        output = torch.matmul(G, f) * math.prod(self.delta_y)
        
        return output
    
    @abstractmethod
    def set_kernel(self):
        pass
    
    def set_resolution(self, x_mesh, y_mesh):
        self.x_mesh = x_mesh
        self.x_dim, self.x_sizes, self.x_mesh_stack, self.x_mesh_flat, self.delta_x = mesh_stats(x_mesh)
        
        self.y_mesh = y_mesh
        self.y_dim, self.y_sizes, self.y_mesh_stack, self.y_mesh_flat, self.delta_y = mesh_stats(y_mesh)
        
        if self.x_dim != self.xi_dim:
            ValueError('x_mesh and xi_mesh must be of the same dimension')
        if self.y_dim != self.eta_dim:
            ValueError('y_mesh and eta_mesh must be of the same dimension')
        
        # set the resolution of the kernel
        self.set_kernel()
    
    def make_symmetric(self):
        # symmetrize kernel 1
        K1symm = torch.reshape(self.K1, self.x_sizes + self.xi_sizes)
        K1symm = symmetrize(K1symm, self.symmx)
        K1symm = torch.reshape(K1symm, (math.prod(self.x_sizes), math.prod(self.xi_sizes))).t()
        K1symm = torch.reshape(K1symm, self.xi_sizes + self.x_sizes)
        K1symm = symmetrize(K1symm, self.symmx)
        K1symm = torch.reshape(K1symm, (math.prod(self.xi_sizes), math.prod(self.x_sizes))).t()
        self.K1 = K1symm
        
        # symmetrize kernel 2
        K2symm = torch.reshape(self.K2, self.y_sizes + self.eta_sizes)
        K2symm = symmetrize(K2symm, self.symmy)
        K2symm = torch.reshape(K2symm, (math.prod(self.y_sizes), math.prod(self.eta_sizes))).t()
        K2symm = torch.reshape(K2symm, self.eta_sizes + self.y_sizes)
        K2symm = symmetrize(K2symm, self.symmy)
        K2symm = torch.reshape(K2symm, (math.prod(self.eta_sizes), math.prod(self.y_sizes))).t()
        self.K2 = K2symm
    
    def to_matrix(self):
        W = self.W
        # symmetrize W in the coordinates where it is symmetric
        if self.symmetries:
            W = torch.reshape(W, self.xi_sizes + self.eta_sizes)
            W = symmetrize(W, self.symmetries)
            W = torch.reshape(W, (math.prod(self.xi_sizes), math.prod(self.eta_sizes)))
        
        # add Green's function component in H1
        G = torch.matmul(torch.matmul(self.K1, W), self.K2.t()) * math.prod(self.delta_xi) * math.prod(self.delta_eta)
        
        # add Green's function component in H0
        if self.ds is not None:
            G_null_terms = torch.sum(self.nullspan() * self.ds[None, None, :], dim=2)
            G += G_null_terms
        
        # symmetrize G in the coordinates where it is symmetric
        if self.symmxy:
            G = torch.reshape(G, self.x_sizes + self.y_sizes)
            G = symmetrize(G, self.symmxy) # maybe should be self.symmetries?
            G = torch.reshape(G, (math.prod(self.x_sizes), math.prod(self.y_sizes)))
        if self.causal_pair is not None:
            G = torch.reshape(G, self.x_sizes + self.y_sizes)
            G = triangular_mask(G, [self.causal_pair])
            G = torch.reshape(G, (math.prod(self.x_sizes), math.prod(self.y_sizes)))
        
        return G
    
    @abstractmethod
    def nullspan(self):
        # add symmetry and causal constraints
        pass
    
    # assumes that (x, y) and (xi, eta) grids coincide
    def rkhs_norm(self):
        W = self.W
        # symmetrize W in the coordinates where it is symmetric
        if self.symmetries:
            W = torch.reshape(W, self.xi_sizes + self.eta_sizes)
            W = symmetrize(W, self.symmetries)
            W = torch.reshape(W, (math.prod(self.xi_sizes), math.prod(self.eta_sizes)))
        
        return torch.sum(self.to_matrix() * W) * math.prod(self.delta_x) * math.prod(self.delta_y)


class SimpleRKHSBias(RKHSBias):
    def __init__(self, x_mesh, xi_mesh, symmetries=[]):
        # create (x, xi) kernel for bias term
        self.register_buffer('K', None)
        
        # set properties of meshes for third and fourth coordinates
        self.xi_mesh = xi_mesh
        self.xi_dim, self.xi_sizes, self.xi_mesh_stack, self.xi_mesh_flat, self.delta_xi = mesh_stats(xi_mesh)
        
        # set properties of input and output meshes
        self.x_dim = None
        self.x_mesh = None
        self.x_sizes = None
        self.x_mesh_stack = None
        self.x_mesh_flat = None
        self.delta_x = None
        self.set_resolution(x_mesh)
        
        # bias term weights
        b = torch.zeros(math.prod(self.xi_sizes), dtype=torch.float64)
        self.b = nn.Parameter(b)
        
        # null space coefficients
        self.ds = None
        nullspan = self.nullspan()
        if nullspan is not None:
            ds = torch.zeros(nullspan.shape[-1], dtype=torch.float64)
            self.ds = nn.Parameter(ds)
        
        # symmetry constraints on the bias term
        self.symmetries = symmetries
        if self.symmetries:
            self.make_symmetric()
    
    @abstractmethod
    def set_kernel(self):
        pass
    
    def set_resolution(self, x_mesh):
        self.x_mesh = x_mesh
        self.x_dim, self.x_sizes, self.x_mesh_stack, self.x_mesh_flat, self.delta_x = mesh_stats(x_mesh)
        
        # set the resolution of the kernel
        self.set_kernel()
    
    def make_symmetric(self):
        # symmetrize bias kernel
        Ksymm = torch.reshape(self.K, self.x_sizes + self.xi_sizes)
        Ksymm = symmetrize(Ksymm, self.symmetries)
        Ksymm = torch.reshape(Ksymm, (math.prod(self.x_sizes), math.prod(self.xi_sizes))).t()
        Ksymm = torch.reshape(Ksymm, self.xi_sizes + self.x_sizes)
        Ksymm = symmetrize(Ksymm, self.symmetries)
        Ksymm = torch.reshape(Ksymm, (math.prod(self.xi_sizes), math.prod(self.x_sizes))).t()
        self.K = Ksymm
    
    def to_vector(self):
        # add symmetry constraints
        if self.b is None:
            return None
        
        # add bias component in B1
        nu = torch.mv(self.K, self.b) * math.prod(self.delta_xi)
        
        # add bias component in B0
        if self.ds is not None:
            nu_null_terms = torch.matmul(self.nullspan(), self.ds)
            nu += nu_null_terms
        
        return nu
    
    @abstractmethod
    def nullspan(self):
        # add symmetry constraints
        pass
    
    # assumes that x and xi grids coincide
    def rkhs_norm(self):
        return torch.dot(torch.mv(self.K, self.b), self.b) * math.prod(self.delta_x) * math.prod(self.delta_xi)


class GaussianProdRKHSWeight(ProdRKHSWeight):
    def __init__(self, Sigma1, Sigma2, x_mesh, y_mesh, xi_mesh, eta_mesh, symmetries=[], causal_pair=None):
        nn.Module.__init__(self)
        
        # create covariance matrices for Gaussian kernels
        self.register_buffer('Sigma1', Sigma1.double())
        self.register_buffer('Sigma2', Sigma2.double())
        
        super().__init__(x_mesh, y_mesh, xi_mesh, eta_mesh, symmetries, causal_pair)
    
    def set_kernel(self):
        self.K1 = gaussian_kernel(self.x_mesh_flat, self.x_sizes, self.xi_mesh_flat, self.xi_sizes, self.Sigma1)
        self.K2 = gaussian_kernel(self.y_mesh_flat, self.y_sizes, self.eta_mesh_flat, self.eta_sizes, self.Sigma2)
    
    def nullspan(self):
        return None


class GaussianSimpleRKHSBias(SimpleRKHSBias):
    def __init__(self, Sigma, x_mesh, xi_mesh, symmetries=[]):
        nn.Module.__init__(self)
        
        # create covariance matrices for Gaussian kernels
        self.register_buffer('Sigma', Sigma.double())
        
        super().__init__(x_mesh, xi_mesh, symmetries)
    
    def set_kernel(self):
        self.K = gaussian_kernel(self.x_mesh_flat, self.x_sizes, self.xi_mesh_flat, self.xi_sizes, self.Sigma)
    
    def nullspan(self):
        return None


class GaussianProdRKHSLayer(RKHSLayer):
    def __init__(self, Sigma1, Sigma2, x_mesh, y_mesh, xi_mesh, eta_mesh, bias=False, symmetries=[], causal_pair=None):
        nn.Module.__init__(self)
        super().__init__()
        self.weight = GaussianProdRKHSWeight(Sigma1, Sigma2, x_mesh, y_mesh, xi_mesh, eta_mesh, symmetries, causal_pair)
        self.bias = None
        if bias:
            self.bias = GaussianSimpleRKHSBias(Sigma1, x_mesh, xi_mesh, symmetries)
