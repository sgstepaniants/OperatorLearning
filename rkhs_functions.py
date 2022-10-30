import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from pykeops.numpy import LazyTensor as NumpyLazyTensor
from pykeops.torch import LazyTensor as TorchLazyTensor
from scipy.special import gamma, kv
from helper import trapezoid_rule

def make_kernel(out_meshes, weight_meshes, kernel="Exponential", sigma=0.1, dtype=torch.float64, lazytensor="torch"):
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.type(dtype)
    
    if lazytensor == "torch":
        out_mesh_grid = torch.meshgrid(out_meshes)
        out_pts = torch.vstack([mesh.ravel() for mesh in out_mesh_grid]).T.contiguous().type(dtype) / sigma
        out_pts_lazy = TorchLazyTensor(out_pts.unsqueeze(-2))
    else:
        out_mesh_grid = np.meshgrid(*out_meshes)
        out_pts = np.ascontiguousarray(np.vstack([mesh.ravel() for mesh in out_mesh_grid]).T, dtype=dtype) / sigma
        out_pts_lazy = NumpyLazyTensor(np.expand_dims(out_pts, axis=-2))
    
    if lazytensor == "torch":
        weight_mesh_grid = torch.meshgrid(weight_meshes)
        weight_pts = torch.vstack([mesh.ravel() for mesh in weight_mesh_grid]).T.contiguous().type(dtype) / sigma
        weight_pts_lazy = TorchLazyTensor(weight_pts.unsqueeze(-3))
    else:
        weight_mesh_grid = np.meshgrid(*weight_meshes)
        weight_pts = np.ascontiguousarray(np.vstack([mesh.ravel() for mesh in weight_mesh_grid]).T, dtype=dtype) / sigma
        weight_pts_lazy = NumpyLazyTensor(np.expand_dims(weight_pts, axis=-3))
    
    D2 = ((out_pts_lazy - weight_pts_lazy)**2).sum(-1) 
    D = D2.sqrt()

    if kernel == "Linear":
        K = (1 - D)
        K = (K + K.abs()) / 2
    elif kernel == "Exponential":
        K = (-D).exp()
    elif kernel == "SquaredExponential":
        K = (-D2/2).exp()
    elif kernel == "Matern3/2":
        K = (1 + math.sqrt(3)*D) * (-math.sqrt(3)*D).exp()
    elif kernel == "Matern5/2":
        K = (1 + math.sqrt(5)*D + 5*D2/3) * (-math.sqrt(5)*D).exp()
    else:
        raise ValueError("Reproducing kernel must be Exponential, SquaredExponential, Matern3/2, or Matern5/2")

    return K

class RKHSFunction(nn.Module):
    def __init__(self, out_meshes, weight_meshes, weight_quadrature=None, kernel="Exponential", sigma=0.1, dtype=torch.float64, weight_parametrizations=None, transform_output=None):
        super().__init__()
        
        self.out_meshes = out_meshes
        self.weight_meshes = weight_meshes
        self.kernel = kernel
        self.sigma = sigma
        self.dtype = dtype
        
        self.K = make_kernel(out_meshes, weight_meshes, kernel=kernel, sigma=sigma, dtype=dtype)
        self.K_norm = make_kernel(weight_meshes, weight_meshes, kernel=kernel, sigma=sigma, dtype=dtype)
        
        self.weight_quadrature = weight_quadrature
        if weight_quadrature is None:
            self.weight_quadrature = trapezoid_rule(weight_meshes).flatten()
        self.weight = torch.nn.Parameter(torch.zeros(math.prod([len(x) for x in weight_meshes])).type(dtype))
        if weight_parametrizations is not None:
            for parameterization in weight_parametrizations:
                parametrize.register_parametrization(self, "weight", parameterization)
        
        self.transform_output = None
        if transform_output is not None:
            self.transform_output = transform_output
    
    def forward(self):
        f = self.K @ (self.weight * self.weight_quadrature)
        if self.transform_output is not None:
            f = self.transform_output(f, self.out_meshes)
        return f
    
    def square_norm(self):
        prod = self.weight * self.weight_quadrature
        norm = torch.dot(prod, self.K_norm @ prod)
        return norm
    
    def update_mesh(self, out_meshes):
        self.out_meshes = out_meshes
        self.K = make_kernel(out_meshes, self.weight_meshes, kernel=self.kernel, sigma=self.sigma, dtype=self.dtype)
        self.K_norm = make_kernel(self.weight_meshes, self.weight_meshes, kernel=self.kernel, sigma=self.sigma, dtype=self.dtype)
   
    def deepcopy(self):
        parametrizations = None
        if hasattr(self, "parametrizations"):
            parametrizations = self.parametrizations.weight
        
        self_copy = RKHSFunction(copy.deepcopy(self.out_meshes), copy.deepcopy(self.weight_meshes), copy.deepcopy(self.weight_quadrature), copy.deepcopy(self.kernel), copy.deepcopy(self.sigma), copy.deepcopy(self.dtype), copy.deepcopy(parametrizations), copy.deepcopy(self.transform_output))
        
        if hasattr(self, "parametrizations"):
            self_copy.parametrizations.weight.original = self.parametrizations.weight.original
        else:
            self_copy.weight = self.weight
        return self_copy
