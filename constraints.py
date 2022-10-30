import math
import torch
import torch.nn as nn
import numpy as np
from helper import partial_index
import matplotlib.pyplot as plt

class Symmetric(nn.Module):
    def __init__(self, dims, i, j, tri="both"):
        if i == j:
            raise ValueError("Indices i, j to symmetrize cannot be equal")
        
        n = len(dims)
        if i > n or j > n:
            raise ValueError("Indices i, j cannot exceed the total number of dimensions")
        
        if dims[i] != dims[j]:
            raise ValueError("Dimensions at indices i, j are not the same")
        
        super().__init__()
        self.i = min(i, j)
        self.j = max(i, j)
        self.dims = torch.tensor(dims)
        self.n = n
        self.tri = tri
    
    def forward(self, w, dims=None, tri=None):
        if dims is None:
            dims = self.dims
        else:
            dims = torch.tensor(dims)
        if tri is None:
            tri = self.tri
        
        # expand weight matrix into a tensor
        W = w.reshape(tuple(dims.tolist()))
        
        # symmetrize dimensions i and j
        S = torch.zeros_like(W)
        S[:] = W
        if tri == "upper":
            triu_inds1, triu_inds2 = torch.triu_indices(S.shape[0], S.shape[1], offset=1)
            S[partial_index([self.i, self.j], [triu_inds2, triu_inds1], self.n)] = S[partial_index([self.i, self.j], [triu_inds1, triu_inds2], self.n)]
        elif tri == "lower":
            tril_inds1, tril_inds2 = torch.tril_indices(S.shape[0], S.shape[1], offset=-1)
            S[partial_index([self.i, self.j], [tril_inds2, tril_inds1], self.n)] = S[partial_index([self.i, self.j], [tril_inds1, tril_inds2], self.n)]
        else:
            S = (S + S.transpose(self.i, self.j)) / 2
        
        # reshape weight tensor back into a matrix
        w = S.reshape(w.shape)
        return w

class FlipSymmetric(nn.Module):
    def __init__(self, dims, inds, direction="both"):
        n = len(dims)
        
        super().__init__()
        self.inds = torch.sort(torch.unique(torch.tensor(inds))).values
        if self.inds[-1] >= n:
            raise ValueError("Indices cannot exceed the total number of dimensions")
        
        self.dims = torch.tensor(dims)
        self.n = n
        self.direction = direction
    
    def forward(self, w, dims=None, direction=None):
        if dims is None:
            dims = self.dims
        else:
            dims = torch.tensor(dims)
        if direction is None:
            direction = self.direction
        
        # expand weight matrix into a tensor
        W = w.reshape(tuple(dims.tolist()))
        
        # symmetrize dimensions i and j
        S = torch.zeros_like(W)
        S[:] = W
        S = (S + S.flip(tuple(self.inds.tolist()))) / 2
        '''
        if direction == "horizontal":
            S[partial_index([self.i], [range(torch.div(dims[self.i]+1, 2, rounding_mode='floor'), dims[self.i])], self.n)] = S[partial_index([self.i], [range(torch.div(dims[self.i], 2, rounding_mode='floor'))], self.n)].flip(self.i, self.j)
        elif direction == "vertical":
            S[partial_index([self.j], [range(torch.div(dims[self.j]+1, 2, rounding_mode='floor'), dims[self.j])], self.n)] = S[partial_index([self.j], [range(torch.div(dims[self.j], 2, rounding_mode='floor'))], self.n)].flip(self.i, self.j)
        else:
            S = (S + S.flip(self.i, self.j)) / 2
        '''
        
        # reshape weight tensor back into a matrix
        w = S.reshape(w.shape)
        return w

class Causal(nn.Module):
    def __init__(self, dims, i, j, anticausal=False):
        if i == j:
            raise ValueError("Causal indices i -> j cannot be equal")
        
        n = len(dims)
        if i > n or j > n:
            raise ValueError("Indices i, j cannot exceed the total number of dimensions")
        
        if dims[i] != dims[j]:
            raise ValueError("Dimensions at indices i, j are not the same")
        
        super().__init__()
        self.i = min(i, j)
        self.j = max(i, j)
        self.dims = torch.tensor(dims)
        self.n = n
        self.anticausal = anticausal
    
    def forward(self, w, dims=None):
        if dims is None:
            dims = self.dims
        else:
            dims = torch.tensor(dims)
        
        # expand weight matrix into a tensor
        W = w.reshape(tuple(dims.tolist()))
        
        # set causal or anticausal constraints on i and j
        C = torch.zeros_like(W)
        C[:] = W
        triu_inds1, triu_inds2 = torch.triu_indices(C.shape[0], C.shape[1], offset=1)
        if self.anticausal:
            C[partial_index([self.i, self.j], [triu_inds1, triu_inds2], self.n)] = 0
        else:
            C[partial_index([self.i, self.j], [triu_inds2, triu_inds1], self.n)] = 0
        
        # reshape weight tensor back into a matrix
        w = C.reshape(w.shape)
        return w

class DomainMask(nn.Module):
    def __init__(self, dims, mask):
        super().__init__()
        self.dims = torch.tensor(dims)
        self.n = len(dims)
        
        if type(mask) is tuple:
            if len(torch.unique(torch.tensor([len(x) for x in torch.where(S > T)]))) != 1:
                raise ValueError("Mask passed as a list of indices contains different number of indices in each dimension")
            self.mask = mask
        elif type(mask) is torch.Tensor:
            if torch.equal(torch.tensor(mask.size()), self.dims):
                self.mask = torch.where(mask)
            else:
                raise ValueError("Mask passed as a list of indices contains different number of indices in each dimension")
        else:
            raise ValueError("Mask must be passed as a list of indices or a binary tensor")
    
    def forward(self, w, dims=None, mask=None):
        if dims is None:
            dims = self.dims
        else:
            dims = torch.tensor(dims)
        if type(mask) is torch.Tensor:
            self.mask = torch.where(mask)
        
        # expand weight matrix into a tensor
        W = w.reshape(tuple(dims.tolist()))
        
        # set causal or anticausal constraints on i and j
        D = torch.zeros_like(W)
        D[mask] = W[mask]
        
        # reshape weight tensor back into a matrix
        w = D.reshape(w.shape)
        return w
