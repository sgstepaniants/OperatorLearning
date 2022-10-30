import math
import numpy as np
import torch

def partial_index(dims, dim_inds, n):
    dims = np.array(dims)
    if max(dims) >= n:
        raise ValueError("Largest index must be less than n")
    dims = np.sort(np.unique(dims))
    pindex = np.repeat((slice(None),), n)
    pindex[dims] = dim_inds
    return tuple(pindex)

def hexagonal_potential(x, y, L=1, mag=1):
    a = L/4
    xc = L/2
    yc = L/2
    idx = np.abs(x-xc) < (a*math.sqrt(3)/2)
    idx *= math.tan(math.pi/6) * (x-xc) + a > (y-yc)
    idx *= math.tan(math.pi/6) * (x-xc) - a < (y-yc)
    idx *= -math.tan(math.pi/6) * (x-xc) - a < (y-yc)
    idx *= -math.tan(math.pi/6) * (x-xc) + a > (y-yc)
    pot = mag * idx
    return pot

def trapezoid_rule(meshes):
    tensor = meshes[0].type()
    quadrature = torch.tensor([1]).type(tensor)
    for i in range(len(meshes)):
        x = meshes[i]
        dx = torch.diff(x)
        x_quadrature = torch.zeros(len(x)).type(tensor)
        x_quadrature[:-1] += dx
        x_quadrature[1:] += dx
        x_quadrature /= 2

        if i > 0:
            quadrature = torch.unsqueeze(quadrature, dim=-1)
        quadrature = torch.tensordot(quadrature, x_quadrature[None, :], dims=([-1], [0]))
    return quadrature

def standard_deviation(fs):
    return torch.sqrt(torch.nanmean((fs - torch.nanmean(fs, 0, True))**2))