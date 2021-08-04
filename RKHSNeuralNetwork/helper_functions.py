import math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import cvxopt
from fft_conv import fft_conv
from scipy.optimize import least_squares
from scipy.linalg import svd
from scipy import sparse
from sparseqr import qr

from abc import abstractmethod
import copy
import matplotlib.pyplot as plt


##############################################################################
#   Helper Functions
##############################################################################

def preprocess_input(f, sizes):
    f = f.double()
    # reshape input f to be of the right size
    if f.shape[0] == math.prod(sizes):
        if len(f.shape) == 1:
            f = f.unsqueeze(1)
        elif len(f.shape) > 2:
            raise Exception('more dimensions in input data than in mesh')
    elif f.shape == sizes:
        f = torch.flatten(f).unsqueeze(-1)
    elif f.shape[:-1] == sizes:
        if len(f.shape) > len(sizes) + 1:
            raise Exception('more dimensions in input data than in mesh')
        f = torch.flatten(f, end_dim=-2)
    
    return f

def mesh_stats(mesh):
    dim = len(mesh)
    sizes = tuple(mesh[0].shape)
    mesh_stack = torch.stack(mesh, dim=-1)
    mesh_flat = torch.flatten(mesh_stack, end_dim=-2)
    
    delta = None
    if mesh[0].numel() == 1:
        delta = torch.tensor([1]) 
    else:
        delta = mesh_stack[(1,)*dim + (slice(None),)] - mesh_stack[(0,)*dim + (slice(None),)]
    
    return dim, sizes, mesh_stack, mesh_flat, delta

def gaussian_kernel(x_mesh_flat, x_sizes, y_mesh_flat, y_sizes, Sigma):
    dim = Sigma.shape[0]
    x_dim = x_mesh_flat.shape[1]
    y_dim = y_mesh_flat.shape[1]
    
    if x_dim != dim or y_dim != dim:
        raise Exception('dimensions of meshes and size of covariance matrix are not equal')
    
    # compute the inner products (x-y)'*Sigma^{-1}*(x-y)
    x_y_diff = torch.flatten(x_mesh_flat.unsqueeze(1) - y_mesh_flat, end_dim=-2).double()
    
    if dim == 1:
        x_y_squared = torch.squeeze(x_y_diff**2 / Sigma)
    else:
        x_y_squared = torch.sum(x_y_diff * torch.transpose(torch.solve(torch.transpose(x_y_diff, 0, 1), Sigma)[0], 0, 1), dim=1)
    
    # create (x, y) Gaussian kernel
    det = Sigma if dim == 1 else torch.det(Sigma)
    K = torch.reshape(torch.exp(-1/2*x_y_squared) / math.sqrt((2*math.pi)**dim * det), (math.prod(x_sizes), math.prod(y_sizes)))
    return K

def removedup(s):
    counter = Counter(s)
    return ''.join(ch for ch, count in counter.most_common() if count == 1)

def symmetrize_mesh(mesh):
    dim = len(mesh)
    mesh_symm = []
    for i in range(dim):
        X = mesh[i].double()
        X_symm = X
        for j in range(dim):
            shape = list(X_symm.shape)
            j_size = shape[j]
            if j_size > 1:
                shape[j] = 2*j_size - 1
                X_symm_new = torch.zeros(shape, dtype=torch.float64)
                if j == i:
                    X_symm_new[j*(slice(None),) + (range(j_size),) + (dim-j-1)*(slice(None),)] = -X_symm.flip(j)
                else:
                    X_symm_new[j*(slice(None),) + (range(j_size),) + (dim-j-1)*(slice(None),)] = X_symm
                X_symm_new[j*(slice(None),) + (range(j_size-1, 2*j_size-1),) + (dim-j-1)*(slice(None),)] = X_symm
                X_symm = X_symm_new
        mesh_symm.append(X_symm)
    return tuple(mesh_symm)

def torch_pinv(A, rcond=1e-15):
    try:
        Apinv = torch.pinverse(A, rcond)
    except:
        print(A)
        Apinv = torch.from_numpy(gepinv(A.numpy(), rcond))
    return Apinv

def gepinv(A, rcond=1e-15):
    U, s, Vh = svd(A, lapack_driver='gesvd')
    s[np.abs(s) < rcond] = 0
    s[np.abs(s) > 0] = 1 / s[np.abs(s) > 0]
    Apinv = (np.matrix.getH(Vh) * s).dot(np.matrix.getH(U))
    return Apinv

def reginv(B, A, lmbda):
    Gamma = torch.matmul(A.t(), A)
    Gamma = Gamma + lmbda * torch.eye(Gamma.shape[0])
    X = torch.solve(torch.matmul(A.t(), B), Gamma)[0]
    return X

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetries
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def quadratic_minimizer(B, A, lmbda):
    _, n = B.shape
    X = np.zeros((A.shape[1], n))
    for i in range(n):
        P = A.T.dot(A) + lmbda * np.eye(A.shape[1])
        #print(np.linalg.cond(P))
        #print(np.linalg.eigh(P))
        q = -A.T.dot(B[:, i])
        X[:, i] = cvxopt_solve_qp(P, q)
    return X

def crop_pad(X, new_shape):
    shape = X.shape
    X_new = torch.zeros(new_shape, dtype=X.dtype)
    inds = tuple()
    for i in range(len(new_shape)):
        inds += (slice(min(new_shape[i], shape[i])),)
    X_new[inds] = X[inds]
    return X_new

def bound_mesh(mesh, bounds):
    d = len(mesh)
    bounding_box = tuple()
    for i in range(d):
        x_i = mesh[i][i*(0,) + (slice(None),) + (d-i-1)*(0,)]
        b_i = bounds[i]
        
        min_ind = 0
        less_inds = torch.where(x_i < -b_i)[0]
        if len(less_inds) > 0:
            min_ind = less_inds[-1] + 1
        
        max_ind = len(x_i) - 1
        greater_inds = torch.where(x_i > b_i)[0]
        if len(greater_inds) > 0:
            max_ind = greater_inds[0] - 1
        
        bounding_box += (slice(min_ind, max_ind+1),)
    
    mesh_bound = tuple()
    for i in range(d):
        mesh_bound += (mesh[i][bounding_box],)
    
    return mesh_bound

def rfft(f, in_basis_shape, y_mesh):
    if f.ndim == 1:
        f = f.unsqueeze(1)
    n = f.shape[1]
    f = torch.reshape(f, y_mesh[0].shape+(n,))
    w = torch.fft.rfftn(f, dim=tuple(range(len(f.shape)-1)))
    
    # crop or zero-pad the fft coefficients of y_mesh to match in_basis_shape
    w = crop_pad(w, in_basis_shape+(n,))
    #print(w)
    return torch.flatten(w, end_dim=-2)

def irfft(w, out_basis_shape, x_mesh):
    if w.ndim == 1:
        w = w.unsqueeze(1)
    n = w.shape[1]
    w = torch.reshape(w, out_basis_shape+(n,))
    #print(w)
    
    # crop or zero-pad the fft coefficients w to match out_basis_shape//2+1
    coeff_shape = x_mesh[0].shape
    coeff_shape = coeff_shape[:-1] + (coeff_shape[-1]//2+1,)
    w = crop_pad(w, coeff_shape+(n,))
    
    f = torch.fft.irfftn(w, dim=tuple(range(len(w.shape)-1)))
    return torch.flatten(f, end_dim=-2)

def dst(f, in_basis_shape, y_mesh_shape, dst_type=1):
    f = f.double()
    if f.ndim == 1:
        f = f.unsqueeze(1)
    n = f.shape[1]
    #f = torch.reshape(f, y_mesh[0].shape+(n,))
    
    Nshift = 0
    nshift = 0
    kshift = 0
    if dst_type == 1:
        Nshift = 1
        nshift = 1
        kshift = 1
    elif dst_type == 2:
        Nshift = 0
        nshift = 1/2
        kshift = 1
    elif dst_type == 3:
        Nshift = 0
        nshift = 1
        kshift = 1/2
    elif dst_type == 4:
        Nshift = 0
        nshift = 1/2
        kshift = 1/2
    
    w_shape = ()
    for i in range(len(y_mesh_shape)):
        N = y_mesh_shape[i]
        K = min(N, in_basis_shape[i])
        w_shape += (K,)
        Si = torch.sin(math.pi/(N+Nshift) * torch.outer(torch.arange(K)+kshift, torch.arange(N)+nshift))
        if dst_type == 3:
            Si[:, N-1] = Si[:, N-1]/2
        Si *= 2 / (N + Nshift)
        
        if i == 0:
            S = Si
        else:
            S = torch.kron(S, Si)
    w = torch.matmul(S.double(), f)
    w = torch.reshape(w, w_shape+(n,))
    
    # crop or zero-pad the dst coefficients of y_mesh to match in_basis_shape
    w = crop_pad(w, in_basis_shape+(n,))
    #print(w)
    return torch.flatten(w, end_dim=-2)

def idst(w, out_basis_shape, x_mesh_shape, dst_type=1):
    w = w.double()
    if w.ndim == 1:
        w = w.unsqueeze(1)
    #w = torch.real(w).double()
    
    Nshift = 0
    nshift = 0
    kshift = 0
    if dst_type == 1:
        Nshift = 1
        nshift = 1
        kshift = 1
    elif dst_type == 2:
        Nshift = 0
        nshift = 1/2
        kshift = 1
    elif dst_type == 3:
        Nshift = 0
        nshift = 1
        kshift = 1/2
    elif dst_type == 4:
        Nshift = 0
        nshift = 1/2
        kshift = 1/2
    
    for i in range(len(out_basis_shape)):
        K = out_basis_shape[i]
        N = x_mesh_shape[i]
        Si = torch.sin(math.pi/(N+Nshift) * torch.outer(torch.arange(N)+nshift, torch.arange(K)+kshift))
        if dst_type == 2:
            Si[N-1, :] = Si[N-1, :]/2
        
        if i == 0:
            S = Si
        else:
            S = torch.kron(S, Si)
    f = torch.matmul(S.double(), w)
    
    return f

# f = torch.sin(torch.linspace(0, 10*math.pi, 100))
# y_mesh_shape = (100,)
# in_basis_shape = (100,)
# dst_type = 4
# w = dst(f, in_basis_shape, y_mesh_shape, dst_type)

# out_basis_shape = in_basis_shape
# x_mesh_shape = y_mesh_shape
# f_prime = idst(w, out_basis_shape, x_mesh_shape, dst_type)
# print(torch.norm(f.unsqueeze(1) - f_prime))


def filter_symmetries(symmetries, xdim, ydim):
    symmx = []
    symmy = []
    symmxy = []
    for pair in symmetries:
        ind1 = pair[0]
        ind2 = pair[1]
        if ind1 >= xdim + ydim or ind2 >= xdim + ydim:
            raise Exception('symmetric coordinate index exceeds number of coordinates in function')
        
        if ind1 < xdim and ind2 < xdim:
            symmx.append((ind1, ind2))
        elif ind1 >= xdim and ind2 >= xdim:
            symmy.append((ind1 - xdim, ind2 - xdim))
        else:
            symmxy.append((ind1, ind2))
    
    return symmx, symmy, symmxy

def symmetrize(X, symmetries):
    if not symmetries:
        return X
    
    pair = symmetries[0]
    ind1 = pair[0]
    ind2 = pair[1]
    X = (X + torch.transpose(X, ind1, ind2)) / 2
    return symmetrize(X, symmetries[1:])

# if pairs contains tuple (i, j), set all entries of X to zero at indices
# where the ith index is less than the jth index
def triangular_mask(X, pairs):
    if not pairs:
        return X
    
    pair = pairs[0]
    if pair[0] == pair[1]:
        raise Exception('pairs cannot contain duplicate coordinates')
    
    x = 1
    if pair[0] < pair[1]:
        x = 0
    
    i = min(pair)
    j = max(pair)
    d = len(X.shape)
    n = X.shape[i]
    m = X.shape[j]
    idxs = torch.triu_indices(n, m, offset=1)
    
    X[i*(slice(None),) + (idxs[x, :],) + (j-i-1)*(slice(None),) + (idxs[1-x, :],) + (d-j-1)*(slice(None),)] = 0
    return triangular_mask(X, pairs[1:])

def hankel(fsymm):
    sizes = tuple((s+1)//2 for s in fsymm.shape)
    n = len(sizes)
    fhankel = torch.zeros(sizes + sizes, dtype=torch.float64)
    for ind1 in np.ndindex(sizes):
        indblock = tuple(slice(x[0], x[0] + x[1]) for x in zip(ind1, sizes))
        fhankel[ind1 + n*(slice(None),)] = fsymm[indblock]
    return fhankel

def kronIdentity(A, sizes, ind):
    n = math.prod(sizes[:ind])
    s = sizes[ind]
    m = math.prod(sizes[ind+1:])
    
    if any(s <= 0 for s in sizes):
        raise Exception('sizes must be positive')
    if sizes[ind] != A.shape[0] or sizes[ind] != A.shape[1]:
        raise Exception('specified size at ind does not equal dimension of square matrix A')
    
    # simulate I_n kron A
    kronIA = torch.zeros((n, s, n, s), dtype=torch.float64)
    rn = torch.arange(n)
    kronIA[rn, :, rn, :] = A
    kronIA = torch.reshape(kronIA, (n*s, n*s))
    
    # simluate I_n kron A kron I_m
    kronIAI = torch.zeros((n*s, m, n*s, m), dtype=torch.float64)
    rm = torch.arange(m)
    kronIAI[:, rm, :, rm] = kronIA
    kronIAI = torch.reshape(kronIAI, (n*s*m, n*s*m))
    
    return kronIAI

def nullspace(A, rcond=None):
    m, n = A.shape
    spA = sparse.csr_matrix(A.detach().numpy())
    # k = 500
    # k = min(k, min(spA.shape)-1)
    # sL = sparse.linalg.svds(spA, k=1, which='LM', return_singular_vectors=False, solver='lobpcg')[0]
    # _, s, Vh = sparse.linalg.svds(spA, k=k, which='SM', return_singular_vectors='vh', solver='lobpcg')
    # print('bye')
    # s = torch.from_numpy(s)
    # Vh = torch.from_numpy(Vh)
    # if rcond is None:
    #     rcond = torch.finfo(s.dtype).eps * max(m, n)
    # tol = sL * rcond
    # num = torch.sum(s > tol, dtype=int)
    # nullspace = Vh[:num, :].t().cpu().conj()
    
    Q, _, _,r = qr(spA.transpose())
    N = torch.from_numpy(Q.todense()[:, r:])
    return N

# for every point in mesh_flat with d columns, compute a
# monomial of the form x_1^a_1 * ... * x_d^a_d where a_i <= order_i.
def monomial_basis(mesh_flat, order):
    monomials = []
    for ks in np.ndindex(order):
        power = torch.tensor(ks, dtype=torch.float64)
        monomials.append(torch.prod(torch.pow(mesh_flat, power), 1, keepdim=True))
    basis = torch.cat(monomials, dim=1)
    return basis

class Symmetrize(nn.Module):
    def __init__(self, xi_sizes, eta_sizes, symmetries):
        self.xi_sizes = xi_sizes
        self.eta_sizes = eta_sizes
        self.symmetries = symmetries
    
    def forward(self, W):
        # symmetrize W in the coordinates where it is symmetric
        W = torch.reshape(W, self.xi_sizes + self.eta_sizes)
        W = symmetrize(W, self.symmetries)
        W = torch.reshape(W, (math.prod(self.xi_sizes), math.prod(self.eta_sizes)))
