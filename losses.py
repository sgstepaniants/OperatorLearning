import torch

# output and target must be of size m x n (grid points by samples)
def relative_error(output, target, quadrature=None, agg=None, dim=None):
    loss = torch.sqrt(relative_squared_error(output, target, quadrature, agg, dim))
    return loss

def relative_squared_error(output, target, quadrature=None, agg=None, dim=None):
    if dim is None:
        dim = tuple(range(1, output.ndim))
    
    if quadrature is None:
        loss = torch.nansum((output - target)**2, dim=dim) / torch.nansum(target**2, dim=dim)
    else:
        loss = torch.nansum((output - target)**2 * quadrature, dim=dim) / torch.nansum(target**2 * quadrature, dim=dim)
    
    if agg == "mean":
        loss = torch.mean(loss)
    elif agg == "sum":
        loss = torch.sum(loss)
    return loss

def squared_error(output, target, quadrature=None, agg=None, dim=None):
    if dim is None:
        dim = tuple(range(1, output.ndim))
    
    if quadrature is None:
        loss = torch.nansum((output - target)**2, dim=dim)
    else:
        loss = torch.nansum((output - target)**2 * quadrature, dim=dim)
    
    if agg == "mean":
        loss = torch.mean(loss)
    elif agg == "sum":
        loss = torch.sum(loss)
    return loss