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
#   Basis RKHS Layers
##############################################################################

class BasisRKHSWeight(RKHSWeight):
    def __init__(self, x_mesh, y_mesh, func_to_basis, basis_to_func,
                 in_basis_shape=None, out_basis_shape=None,
                 W_scale_fn=None,
                 symmetries=[], causal_pair=None):
        
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
        
        self.func_to_basis, self.basis_to_func = func_to_basis, basis_to_func
        if in_basis_shape is None:
            self.in_basis_shape = y_mesh[0].shape
        else:
            self.in_basis_shape = in_basis_shape
        if out_basis_shape is None:
            self.out_basis_shape = x_mesh[0].shape
        else:
            self.out_basis_shape = out_basis_shape
        
        self.in_basis_size = math.prod(self.in_basis_shape)
        self.out_basis_size = math.prod(self.out_basis_shape)
        
        # set the weighting function for the Greens function coefficients
        self.W_scale_fn = W_scale_fn
        self.W_scale = torch.ones(self.out_basis_size, self.in_basis_size, dtype=torch.float64)
        if self.W_scale_fn is not None:
            self.W_scale = self.W_scale_fn(self.out_basis_shape, self.in_basis_shape)
        
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
        
        # initialize network weights
        W = torch.zeros((self.out_basis_size, self.in_basis_size), dtype=torch.float64)
        self.W = nn.Parameter(W)
        nn.init.xavier_normal_(self.W)
        self.W = nn.Parameter(self.W_scale * W)
        #self.make_symmetric()
    
    def forward(self, f):
        f = preprocess_input(f, self.y_sizes)
        
        in_basis_coeffs = torch.flatten(self.func_to_basis(f, self.in_basis_shape, self.y_sizes), end_dim=-2)
        
        W = self.W
        # symmetrize W in the coordinates where it is symmetric
        if self.symmetries:
            W = torch.reshape(W, self.out_basis_shape + self.in_basis_shape)
            W = symmetrize(W, self.symmetries)
            W = torch.reshape(W, (self.out_basis_size, self.in_basis_size))
        
        # add Green's function component in H1
        out_basis_coeffs = torch.matmul(W, in_basis_coeffs)
        #print(out_basis_coeffs)
        
        output = self.basis_to_func(out_basis_coeffs, self.out_basis_shape, self.x_sizes)
        
        # sanity check!
        #S = torch.sin(math.pi*torch.outer(torch.arange(self.in_basis_shape[0])+1, self.y_mesh[0]))
        #w = torch.matmul(S, f) * math.prod(self.delta_y)
        #w_prime = torch.matmul(W_scale * self.W, w)
        #S_prime = torch.sin(math.pi*torch.outer(self.x_mesh[0], torch.arange(self.out_basis_shape[0])+1))
        #f_prime = torch.matmul(S_prime, w_prime)
        
        # add Green's function component in H0
        if self.ds is not None:
            G_null_terms = torch.sum(self.nullspan() * self.ds[None, None, :], dim=2)
            output += torch.matmul(G_null_terms, f) * math.prod(self.delta_y)
        
        #G = self.to_matrix()
        #output = torch.matmul(G, f) * math.prod(self.delta_y)
        
        return output
    
    def set_resolution(self, x_mesh, y_mesh):
        self.x_mesh = x_mesh
        self.x_dim, self.x_sizes, self.x_mesh_stack, self.x_mesh_flat, self.delta_x = mesh_stats(x_mesh)
        
        self.y_mesh = y_mesh
        self.y_dim, self.y_sizes, self.y_mesh_stack, self.y_mesh_flat, self.delta_y = mesh_stats(y_mesh)
    
    def set_basis_shape(self, in_basis_shape, out_basis_shape):
        self.in_basis_shape = in_basis_shape
        self.out_basis_shape = out_basis_shape
        self.in_basis_size = math.prod(in_basis_shape)
        self.out_basis_size = math.prod(out_basis_shape)
        
        self.W = nn.parameter.Parameter(crop_pad(self.W, (self.out_basis_size, self.in_basis_size)))
    
    def make_symmetric(self):
        # symmetrize weight matrix along specified axes
        W = self.W
        W_scale = self.W_scale
        if self.symmetries:
            W = torch.reshape(W, self.out_basis_shape + self.in_basis_shape)
            W = triangular_mask(W, self.symmetries)
            W = torch.reshape(W, (self.out_basis_size, self.in_basis_size))
            W_scale = torch.reshape(W_scale, self.out_basis_shape + self.in_basis_shape)
            W_scale = symmetrize(W_scale, self.symmetries)
            W_scale = torch.reshape(W_scale, (self.out_basis_size, self.in_basis_size))
        self.W = nn.Parameter(W / W_scale)
        self.W_scale = W_scale
        
        # since the weight matrix is symmetric, only keep its upper/lower triangles
        def triangle_hook(Wgrad):
            Wgrad = torch.reshape(Wgrad, self.out_basis_shape + self.in_basis_shape)
            Wgrad = triangular_mask(Wgrad, self.symmetries)
            Wgrad = torch.reshape(Wgrad, (self.out_basis_size, self.in_basis_size))
            return Wgrad
        #self.W.register_hook(triangle_hook)
    
    def to_matrix(self):
        W = self.W
        if self.symmetries:
            W = torch.reshape(W, self.out_basis_shape + self.in_basis_shape)
            W = symmetrize(W, self.symmetries)
            W = torch.reshape(W, (self.out_basis_size, self.in_basis_size))
        
        # add Green's function component in H1
        I = torch.eye(math.prod(self.y_sizes)) / math.prod(self.delta_y)
        in_basis_coeffs = torch.flatten(self.func_to_basis(I, self.in_basis_shape, self.y_sizes), end_dim=-2)
        out_basis_coeffs = torch.matmul(self.W_scale * W, in_basis_coeffs)
        G = self.basis_to_func(out_basis_coeffs, self.out_basis_shape, self.x_sizes)
        
        # add Green's function component in H0
        if self.ds is not None:
            G_null_terms = torch.sum(self.nullspan() * self.ds[None, None, :], dim=2)
            G += G_null_terms
        
        # satisfy causality constraints (symmetries naturally satisfied through W)
        if self.causal_pair is not None:
            G = torch.reshape(G, self.x_sizes + self.y_sizes)
            G = triangular_mask(G, [self.causal_pair])
            G = torch.reshape(G, (math.prod(self.x_sizes), math.prod(self.y_sizes)))
        
        return G
    
    @abstractmethod
    def nullspan(self):
        pass
    
    def rkhs_norm(self):
        W = self.W
        if self.symmetries:
            W = torch.reshape(W, self.out_basis_shape + self.in_basis_shape)
            W = symmetrize(W, self.symmetries)
            if self.causal_pair is not None:
                W = symmetrize(W, [self.causal_pair])
            W = torch.reshape(W, (self.out_basis_size, self.in_basis_size))
        
        # add symmetry and causal constraints
        return torch.sum(W**2 / self.W_scale)


# RKHS of Sobolev Green's functions with Dirichlet (zero) boundary conditions
class SobolevRKHSWeight_Dir(BasisRKHSWeight):
    def __init__(self, x_mesh, y_mesh,
                 in_basis_shape=None, out_basis_shape=None,
                 symmetries=[], causal_pair=None):
        nn.Module.__init__(self)
        
        def scale_fn(sizes):
            mesh = torch.meshgrid([torch.arange(1, k+1)**2 for k in sizes])
            scale = 4 / (math.pi**2 * torch.sum(torch.stack(mesh, dim=-1), dim=-1))
            return scale
        
        def W_scale_fn(out_shape, in_shape):
            return scale_fn(out_shape + in_shape).view(math.prod(out_shape), math.prod(in_shape))
        
        dst_type = 1
        def func_to_basis(f, in_basis_shape, y_mesh_shape):
            return dst(f, in_basis_shape, y_mesh_shape, dst_type)
        
        def basis_to_func(w, out_basis_shape, x_mesh_shape):
            return idst(w, out_basis_shape, x_mesh_shape, dst_type)
        
        super().__init__(x_mesh, y_mesh, func_to_basis, basis_to_func,
                         in_basis_shape, out_basis_shape,
                         None, symmetries, causal_pair)
    
    def nullspan(self):
        return None


# RKHS of Sobolev bias functions of one real variable
class SobolevRKHSBias_1D(SimpleRKHSBias):
    def __init__(self, x_mesh, xi_mesh, symmetries=[]):
        nn.Module.__init__(self)
        
        super().__init__(x_mesh, xi_mesh, symmetries)
    
    def set_kernel(self):
        # # compute min(x, xi)
        # x_inds = torch.arange(self.x_mesh_flat.shape[0])
        # xi_inds = torch.arange(self.xi_mesh_flat.shape[0])
        # prodinds = torch.cartesian_prod(x_inds, xi_inds)
        
        # x_min = torch.min(self.x_mesh_flat)
        # xi_min = torch.min(self.xi_mesh_flat)
        # x_xi_prod = torch.cat((self.x_mesh_flat[prodinds[:, 0], :] - x_min, self.xi_mesh_flat[prodinds[:, 1], :] - xi_min), 1)
        # x_xi_min = torch.min(x_xi_prod, dim=-1)[0]
        
        # self.Kb = torch.reshape(x_xi_min, (math.prod(self.x_sizes), math.prod(self.xi_sizes)))
        
        x_xi_diff = torch.flatten(self.x_mesh_flat.unsqueeze(1) - self.xi_mesh_flat, end_dim=-2).double()
        
        self.K = torch.reshape(torch.abs(x_xi_diff), (math.prod(self.x_sizes), math.prod(self.xi_sizes)))
        self.K = 1/2 * (self.K**2 - self.K + 1/6)
        self.K += torch.outer(self.x_mesh[0] - 1/2, self.xi_mesh[0] - 1/2)
    
    def nullspan(self):
        nu_const = torch.ones([math.prod(self.x_sizes), 1], dtype=torch.float64)
        return nu_const


# RKHS layer of Sobolev Green's function and bias term mapping functions of one
# real variable to functions of one real variable
class SobolevRKHSLayer_1D(RKHSLayer):
    def __init__(self, x_mesh, y_mesh, xi_mesh,
                 in_basis_shape=None, out_basis_shape=None,
                 bias=False, 
                 symmetries=[], causal_pair=None):
        nn.Module.__init__(self)
        super().__init__()
        self.weight = SobolevRKHSWeight_Dir(x_mesh, y_mesh, in_basis_shape, out_basis_shape, symmetries, causal_pair)
        self.bias = None
        if bias:
            self.bias = SobolevRKHSBias_1D(x_mesh, xi_mesh, symmetries)
