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
#   Vanilla Fully Connected Neural Network
##############################################################################

class VanillaNetwork(nn.Module):
    def __init__(self, x_mesh, y_mesh, bias=False):
        nn.Module.__init__(self)
        
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
        
        # scaling constants to multiply neural network inputs and outputs by
        self.in_scale = 1
        self.out_scale = 1
        
        W = torch.zeros(math.prod(self.x_sizes), math.prod(self.y_sizes), dtype=torch.float64)
        self.W = nn.Parameter(W)
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
        # bias terms
        self.b = None
        if bias:
            b = torch.zeros(math.prod(self.x_sizes), dtype=torch.float64)
            self.b = nn.Parameter(b)
    
    def forward(self, f):
        f = f.clone().double()
        # reshape input f to be of the right size
        if f.shape[0] == math.prod(self.y_sizes):
            if len(f.shape) == 1:
                f = f.unsqueeze(1)
            elif len(f.shape) > 2:
                raise Exception('more dimensions in input data than in mesh')
        elif f.shape == self.y_sizes:
            f = torch.flatten(f).unsqueeze(-1)
        elif f.shape[:-1] == self.y_sizes:
            if len(f.shape) > len(self.y_sizes) + 1:
                raise Exception('more dimensions in input data than in mesh')
            f = torch.flatten(f, end_dim=-2)
        
        f *= self.in_scale
        
        # feed input function to discrete linear layer
        output = torch.matmul(self.W, f) * math.prod(self.delta_y)
        
        # add bias term
        if self.b is not None:
            output = output + self.b[:, None]
        
        output *= self.out_scale
        return output
    
    def set_resolution(self, x_mesh, y_mesh):
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.x_dim = len(self.x_mesh)
        self.y_dim = len(self.y_mesh)
        
        self.x_sizes = tuple(self.x_mesh[0].shape)
        self.y_sizes = tuple(self.y_mesh[0].shape)
        
        self.x_mesh_stack = torch.stack(self.x_mesh, dim=-1)
        self.y_mesh_stack = torch.stack(self.y_mesh, dim=-1)
        
        self.x_mesh_flat = torch.flatten(self.x_mesh_stack, end_dim=-2)
        self.y_mesh_flat = torch.flatten(self.y_mesh_stack, end_dim=-2)
        
        self.delta_x = torch.tensor([1]) if self.x_mesh[0].numel() == 1 else self.x_mesh_stack[(1,)*self.x_dim + (slice(None),)] - self.x_mesh_stack[(0,)*self.x_dim + (slice(None),)]
        self.delta_y = torch.tensor([1]) if self.y_mesh[0].numel() == 1 else self.y_mesh_stack[(1,)*self.y_dim + (slice(None),)] - self.y_mesh_stack[(0,)*self.y_dim + (slice(None),)]
    
    def rescale(self, in_scale, out_scale):
        self.in_scale = in_scale
        self.out_scale = out_scale