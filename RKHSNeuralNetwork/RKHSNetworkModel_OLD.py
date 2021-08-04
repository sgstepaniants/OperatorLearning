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
import matplotlib.pyplot as plt


##############################################################################
#   Helper Functions
##############################################################################

def removedup(s):
    counter = Counter(s)
    return ''.join(ch for ch, count in counter.most_common() if count == 1)

def symmetrize_mesh(mesh):
    dim = len(mesh)
    mesh_symm = []
    for i in range(dim):
        X = mesh[i]
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
    P = .5 * (P + P.T)  # make sure P is symmetric
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

def rfft(f, in_basis_shape, y_mesh):
    n = f.shape[1]
    f = torch.reshape(f, y_mesh[0].shape+(n,))
    w = torch.fft.rfftn(f, dim=tuple(range(len(f.shape)-1)))
    
    # crop or zero-pad the fft coefficients of y_mesh to match in_basis_shape
    w = crop_pad(w, in_basis_shape+(n,))
    #print(w)
    return torch.flatten(w, end_dim=-2)

def irfft(w, out_basis_shape, x_mesh):
    n = w.shape[1]
    w = torch.reshape(w, out_basis_shape+(n,))
    #print(w)
    
    # crop or zero-pad the fft coefficients w to match out_basis_shape//2+1
    coeff_shape = x_mesh[0].shape
    coeff_shape = coeff_shape[:-1] + (coeff_shape[-1]//2+1,)
    w = crop_pad(w, coeff_shape+(n,))
    
    f = torch.fft.irfftn(w, dim=tuple(range(len(w.shape)-1)))
    return torch.flatten(f, end_dim=-2)

##############################################################################
#   Functional Network and RKHS Layer Parent Classes
##############################################################################

class FunctionalNetwork(nn.Module):
    def __init__(self, layers, fs_train):
        super().__init__()
        
        self.layers = layers
        self.register_buffer('fs_train', torch.flatten(fs_train, end_dim=-2).double())

        self.n_train = self.fs_train.shape[1]
        
        self.continuous_mode = False
        
        #self.intermediate_train_outputs()
        self.initialize_weights()
    
    def intermediate_train_outputs(self):
        fs = self.fs_train
        for layer in self.layers:
            if isinstance(layer, RKHSLayer):
                layer.prev_fs = fs
            fs = layer(fs)
    
    # def initialize_weights(self):
    #     fs = self.fs_train
    #     prev_rkhs_fs = self.fs_train
    #     for layer in self.layers:
    #         if isinstance(layer, RKHSLayer):
    #             layer.prev_fs = fs
    #             prev_norm = torch.mean(torch.norm(prev_rkhs_fs, dim=0))
    #             curr_fs = layer(fs)
    #             norm = torch.mean(torch.norm(curr_fs, dim=0))
    #             #print(prev_norm / norm)
    #             scale = (prev_norm/norm).detach()
    #             layer.cs = nn.parameter.Parameter(scale * layer.cs)
    #             #layer.W = nn.parameter.Parameter(scale * layer.W)
    #             fs = layer(fs)
    #             #plt.plot(fs[:, 0].detach().numpy())
    #             #plt.show()
    #             #print(torch.mean(torch.norm(fs, dim=0)))
    #             prev_rkhs_fs = fs
    #         else:
    #             fs = layer(fs)
    
    def initialize_weights(self):
        K1 = None
        scale = 1
        Sigma = None
        
        fs = self.fs_train
        
        for layer in self.layers:
            print('initializing layer')
            if isinstance(layer, RKHSLayer):
                layer.prev_fs = fs
                K2 = layer.K2
                scale *= math.prod(layer.delta_y) * math.prod(layer.delta_eta)
                
                if layer.continuous_mode:
                    f_bar = layer.prev_fs.detach()
                    if K1:
                        A = torch.matmul(f_bar.t(), torch.matmul(K2.t(), K1)).numpy()**2
                    else:
                        A = torch.matmul(f_bar.t(), K2.t()).numpy()**2
                    
                    s0 = np.zeros(layer.n_train)
                    f = lambda s : A.T.dot(s) - 1
                    jac = lambda s : A.T
                    res = least_squares(f, s0, jac, bounds=(layer.n_train*[0], np.inf))
                    s = torch.from_numpy(res.x)
                    Sigma = torch.sqrt(s.repeat(math.prod(layer.xi_sizes), 1) / math.prod(layer.xi_sizes))
                    
                    cs = torch.randn([math.prod(layer.xi_sizes), layer.n_train], dtype=torch.float64)
                    cs = cs * Sigma
                    cs = cs / scale
                    cs = cs * math.sqrt(2) # for ReLU activations
                    
                    layer.cs = nn.parameter.Parameter(cs)
                else:
                    if Sigma is None: # assuming all layers have the same reproducing kernel
                        if K1:
                            A = torch.matmul(K2.t(), K1).numpy()**2
                        else:
                            A = K2.t().numpy()**2
                        
                        s0 = np.zeros(math.prod(layer.eta_sizes))
                        f = lambda s : A.T.dot(s) - 1
                        jac = lambda s : A.T
                        res = least_squares(f, s0, jac, bounds=(math.prod(layer.eta_sizes)*[0], np.inf))
                        s = torch.from_numpy(res.x)
                        Sigma = torch.sqrt(s.repeat(math.prod(layer.xi_sizes), 1) / math.prod(layer.xi_sizes))
                    
                    W = torch.randn([math.prod(layer.xi_sizes), math.prod(layer.eta_sizes)], dtype=torch.float64)   
                    W = W * Sigma
                    W = W / scale
                    W = W * math.sqrt(2) # for ReLU activations
                    
                    layer.W = nn.parameter.Parameter(W)
                
                K1 = layer.K1
                scale = math.prod(layer.delta_xi)
            else:
                K1 = None
                scale = 1
            
            if self.continuous_mode:
                fs = layer(fs)
    
    def make_continuous(self):
        self.continuous_mode = True
        for layer in self.layers:
            if isinstance(layer, RKHSLayer):
                layer.make_continuous()
    
    def revert_to_discrete(self):
        self.make_discrete()
        self.intermediate_train_outputs()
    
    def make_discrete(self):
        self.continuous_mode = False
        for layer in self.layers:
            if isinstance(layer, RKHSLayer):
                layer.make_discrete()
    
    def add_refs(self, fs_ref):
        fs_ref = fs_ref.double()
        m, n = fs_ref.shape
        self.fs_train = torch.cat((self.fs_train, fs_ref), dim=1)
        new_prev_fs = fs_ref
        for layer in self.layers:
            if isinstance(layer, RKHSLayer):
                new_cs = torch.randn(m, n, dtype=torch.float64)
                layer.cs = nn.parameter.Parameter(torch.cat((layer.cs, new_cs), dim=1))
                layer.prev_fs = torch.cat((layer.prev_fs, new_prev_fs), dim=1)
                print(layer.prev_fs.shape)
                layer.n_train += n
            new_prev_fs = layer(new_prev_fs)
    
    def forward(self, f):
        # run training data through network to obtain intermediate values
        if self.training:
            self.intermediate_train_outputs()
        
        # run new data points through network
        for layer in self.layers:
            f = layer(f)
        
        return f
    
    def set_resolution(self, new_layer_meshes):
        l = 0
        # iterate over each RKHSLayer and update its input and output meshes
        for layer in self.layers:
            if isinstance(layer, RKHSLayer) or isinstance(layer, BasisLayer):
                layer.set_resolution(new_layer_meshes[l+1], new_layer_meshes[l])
                l += 1
    
    def set_basis_shape(self, new_in_basis_shapes, new_out_basis_shapes):
        l = 0
        # iterate over each BasisLayer and update its input and output meshes
        for layer in self.layers:
            if isinstance(layer, BasisLayer):
                layer.set_basis_shape(new_in_basis_shapes[l], new_out_basis_shapes[l])
                l += 1
    
    def compute_regularization(self, lambdas):
        regularization = 0
        l = 0
        for layer in self.layers:
            if isinstance(layer, RKHSLayer):
                regularization += lambdas[l] * layer.RKHSnorm()
        return regularization

    def to(self, device):
        super().to(device)
        self.layers = self.layers.to(device)

class BasisLayer(nn.Module):
    def __init__(self, x_mesh, y_mesh, func_to_basis, basis_to_func, in_basis_shape=None, out_basis_shape=None, bias=False):
        super().__init__()
        
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
        BasisLayer.set_resolution(self, x_mesh, y_mesh)
        
        self.func_to_basis, self.basis_to_func = func_to_basis, basis_to_func
        if in_basis_shape is None:
            self.in_basis_shape = y_mesh[0].shape
        else:
            self.in_basis_shape = in_basis_shape
        if out_basis_shape is None:
            self.out_basis_shape = x_mesh[0].shape
        else:
            self.out_basis_shape = out_basis_shape
        
        self.in_basis_size = math.prod(in_basis_shape)
        self.out_basis_size = math.prod(out_basis_shape)
        
        # initialize network basis coefficent parameters
        W = torch.zeros((self.out_basis_size, self.in_basis_size), dtype=torch.complex128)
        self.W = nn.Parameter(W)
        nn.init.xavier_normal_(self.W)
        
        self.b = None
        if bias:
            b = torch.zeros(self.out_basis_size, dtype=torch.complex128)
            self.b = nn.Parameter(b)
    
    def forward(self, f):
        if len(f.shape) == 1:
            f = f.unsqueeze(1)
        in_basis_coeffs = torch.flatten(self.func_to_basis(f, self.in_basis_shape, self.y_mesh), end_dim=-2)
        #print(in_basis_coeffs)
        #print(self.W)
        out_basis_coeffs = torch.matmul(self.W, in_basis_coeffs)
        #print(out_basis_coeffs)
        if self.b is not None:
            out_basis_coeffs = out_basis_coeffs + self.b[:, None]
        return self.basis_to_func(out_basis_coeffs, self.out_basis_shape, self.x_mesh)
    
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
    
    def set_basis_shape(self, in_basis_shape, out_basis_shape):
        self.in_basis_shape = in_basis_shape
        self.out_basis_shape = out_basis_shape
        self.in_basis_size = math.prod(in_basis_shape)
        self.out_basis_size = math.prod(out_basis_shape)
        
        self.W = nn.parameter.Parameter(crop_pad(self.W, (self.out_basis_size, self.in_basis_size)))
        self.b = nn.parameter.Parameter(crop_pad(self.b, (self.out_basis_size,)))

class RKHSLayer(nn.Module):
    def __init__(self, x_mesh, y_mesh, xi_mesh, eta_mesh, n_train, bias=False):
        #super().__init__()
        
        self.n_train = n_train
        
        # set properties of meshes for third and fourth coordinates
        self.xi_mesh = xi_mesh
        self.eta_mesh = eta_mesh
        self.xi_dim = len(self.xi_mesh)
        self.eta_dim = len(self.eta_mesh)
        self.xi_sizes = tuple(self.xi_mesh[0].shape)
        self.eta_sizes = tuple(self.eta_mesh[0].shape)
        self.xi_mesh_stack = torch.stack(self.xi_mesh, dim=-1)
        self.eta_mesh_stack = torch.stack(self.eta_mesh, dim=-1)
        self.xi_mesh_flat = torch.flatten(self.xi_mesh_stack, end_dim=-2)
        self.eta_mesh_flat = torch.flatten(self.eta_mesh_stack, end_dim=-2)
        self.delta_xi = torch.tensor([1]) if self.xi_mesh[0].numel() == 1 else self.xi_mesh_stack[(1,)*self.xi_dim + (slice(None),)] - self.xi_mesh_stack[(0,)*self.xi_dim + (slice(None),)]
        self.delta_eta = torch.tensor([1]) if self.eta_mesh[0].numel() == 1 else self.eta_mesh_stack[(1,)*self.eta_dim + (slice(None),)] - self.eta_mesh_stack[(0,)*self.eta_dim + (slice(None),)]
        
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
        RKHSLayer.set_resolution(self, x_mesh, y_mesh)
        
        # training functions to integrate against
        self.register_buffer('prev_fs', None)
        
        # initialize layer weights given by representer theorem
        cs = torch.randn([math.prod(self.xi_sizes), self.n_train], dtype=torch.float64)
        self.cs = nn.Parameter(cs)
        self.ds = None
        
        self.b = None
        if bias:
            b = torch.zeros(math.prod(self.xi_sizes), dtype=torch.float64)
            self.b = nn.Parameter(b)
        
        # set to True if layer is using continuous weights
        self.continuous_mode = False
        
        # weights of discrete neural network used to initialize cs
        W = torch.randn([math.prod(self.xi_sizes), math.prod(self.eta_sizes)], dtype=torch.float64)
        self.W = nn.Parameter(W)
        #nn.init.kaiming_normal_(self.W)
    
    def forward(self, f):
        f = f.double()
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
        
        if self.continuous_mode:
            A = torch.matmul(self.K1, self.cs) * math.prod(self.delta_xi)
            B = torch.matmul(self.K2, self.prev_fs) * math.prod(self.delta_eta)
            #B = B.detach()
            G = torch.matmul(A, B.t())
            output = torch.matmul(G, f) * math.prod(self.delta_y) #self.kernel(None, f, self.cs, self.prev_fs, 'ijkk')
        else:
            #f_bar = self.prev_fs.detach()
            #f_bar_pinv = torch_pinv(f_bar)
            #P = torch.matmul(f_bar, f_bar_pinv)
            #WP = torch.matmul(self.W, P)
            G = torch.matmul(torch.matmul(self.K1, self.W), self.K2.t()) * math.prod(self.delta_xi) * math.prod(self.delta_eta)
            output = torch.matmul(G, f) * math.prod(self.delta_y)
            
            # tol = 1e-4
            # U, S, _ = torch.svd(f_bar)
            # num = torch.sum(S >= tol)
            # WU = torch.matmul(self.W[:, :num], U[:, :num].t())
            # output = torch.matmul(torch.matmul(torch.matmul(self.K1, WU), self.K2.t()), f)
            # output = output * math.prod(self.delta_xi) * math.prod(self.delta_eta)
        
        if self.b is not None:
            output = output + torch.mv(self.K1, self.b)[:, None] * math.prod(self.delta_xi)
            
        if self.ds is not None:
            null_terms = torch.matmul(self.nullspan(None, f), self.ds)
            output = output + null_terms
        
        return output
    
    @abstractmethod
    def nullspan(self, fx, fy):
        pass
    
    @abstractmethod
    def kernel(self, fx, fy, gx, gy, ein='ijkl'):
        pass
    
    @abstractmethod
    def set_kernel(self):
        pass
    
    def set_resolution(self, x_mesh, y_mesh):
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.x_dim = len(self.x_mesh)
        self.y_dim = len(self.y_mesh)
        
        if self.x_dim != self.xi_dim:
            ValueError('x_mesh and xi_mesh must be of the same dimension')
        if self.y_dim != self.eta_dim:
            ValueError('y_mesh and eta_mesh must be of the same dimension')
        
        self.x_sizes = tuple(self.x_mesh[0].shape)
        self.y_sizes = tuple(self.y_mesh[0].shape)
        
        self.x_mesh_stack = torch.stack(self.x_mesh, dim=-1)
        self.y_mesh_stack = torch.stack(self.y_mesh, dim=-1)
        
        self.x_mesh_flat = torch.flatten(self.x_mesh_stack, end_dim=-2)
        self.y_mesh_flat = torch.flatten(self.y_mesh_stack, end_dim=-2)
        
        self.delta_x = torch.tensor([1]) if self.x_mesh[0].numel() == 1 else self.x_mesh_stack[(1,)*self.x_dim + (slice(None),)] - self.x_mesh_stack[(0,)*self.x_dim + (slice(None),)]
        self.delta_y = torch.tensor([1]) if self.y_mesh[0].numel() == 1 else self.y_mesh_stack[(1,)*self.y_dim + (slice(None),)] - self.y_mesh_stack[(0,)*self.y_dim + (slice(None),)]
        
        # set the resolution of the kernel
        self.set_kernel()
    
    def make_continuous(self):
        self.continuous_mode = True
        self.cs = nn.parameter.Parameter(self.cs_from_W(self.W))
    
    def make_discrete(self):
        self.continuous_mode = False
    
    @abstractmethod
    def cs_from_W(self, W):
        pass
    
    def RKHSnorm(self):
        return self.kernel(self.cs, self.prev_fs, self.cs, self.prev_fs, 'iijj')

    #def to(self, device):
    #    super().to(device)
        


##############################################################################
#   Product RKHS Layers
##############################################################################

class GaussianRKHSLayer(RKHSLayer):
    def __init__(self, Sigma1, Sigma2, x_mesh, y_mesh, xi_mesh, eta_mesh, n_train, bias=False):
        nn.Module.__init__(self)
        
        # create covariance matrices for Gaussian kernels
        self.register_buffer('Sigma1', Sigma1.double())
        self.register_buffer('Sigma2', Sigma2.double())
        
        # create two Gaussian kernels for (x, xi) and (y, eta)
        self.register_buffer('K1', None)
        self.register_buffer('K2', None)
        
        super().__init__(x_mesh, y_mesh, xi_mesh, eta_mesh, n_train, bias)
    
    def kernel(self, fx, fy, fxi, feta, ein='ijkl'):
        # compute product of first (x, xi) Gaussian kernel
        prod1 = self.K1
        if fx is not None:
            fx_flat = torch.flatten(fx, end_dim=-2) if len(fx.shape) > 1 else fx.unsqueeze(-1)
            prod1 = torch.matmul(torch.transpose(fx_flat, 0, 1).double(), prod1) * torch.prod(self.delta_x)
        if fxi is not None:
            fxi_flat = torch.flatten(fxi, end_dim=-2) if len(fxi.shape) > 1 else fxi.unsqueeze(-1)
            prod1 = torch.matmul(prod1, fxi_flat.double()) * torch.prod(self.delta_xi)
        
        # compute product of second (y, eta) Gaussian kernel
        prod2 = self.K2
        if fy is not None:
            fy_flat = torch.flatten(fy, end_dim=-2) if len(fy.shape) > 1 else fy.unsqueeze(-1)
            prod2 = torch.matmul(torch.transpose(fy_flat, 0, 1).double(), prod2) * torch.prod(self.delta_y)
        if feta is not None:
            feta_flat = torch.flatten(feta, end_dim=-2) if len(feta.shape) > 1 else feta.unsqueeze(-1)
            prod2 = torch.matmul(prod2, feta_flat.double()) * torch.prod(self.delta_eta)
        
        # multiply both kernel products together
        ein_in = ein[0] + ein[2] + ',' + ein[1] + ein[3]
        ein_out = removedup(ein)
        eq = ein_in + '->' + ein_out
        prod = torch.einsum(eq, prod1, prod2)
        
        return prod
    
    def set_kernel(self):
        # compute the inner products (x-xi)'*Sigma1^{-1}*(x-xi) and (y-eta)'*Sigma2^{-1}*(y-eta)
        x_xi_diff = torch.flatten(self.x_mesh_flat.unsqueeze(1) - self.xi_mesh_flat, end_dim=-2).double()
        y_eta_diff = torch.flatten(self.y_mesh_flat.unsqueeze(1) - self.eta_mesh_flat, end_dim=-2).double()
        
        # move covariance matrices onto the same device as the meshes
        #self.Sigma1 = self.Sigma1.to(x_xi_diff.device)
        #self.Sigma2 = self.Sigma2.to(y_eta_diff.device)

        if self.x_dim == 1:
            x_xi_squared = torch.squeeze(x_xi_diff**2 / self.Sigma1)
        else:
            x_xi_squared = torch.sum(x_xi_diff * torch.transpose(torch.solve(torch.transpose(x_xi_diff, 0, 1), self.Sigma1)[0], 0, 1), dim=1)
        if self.y_dim == 1:
            y_eta_squared = torch.squeeze(y_eta_diff**2 / self.Sigma2)
        else:
            y_eta_squared = torch.sum(y_eta_diff * torch.transpose(torch.solve(torch.transpose(y_eta_diff, 0, 1), self.Sigma2)[0], 0, 1), dim=1)
        
        # create first (x, xi) Gaussian kernel
        dim1 = self.Sigma1.shape[0]
        det1 = self.Sigma1 if dim1 == 1 else torch.det(self.Sigma1)
        #self.K1 = torch.reshape(torch.exp(-1/2*x_xi_squared) / math.sqrt((2*math.pi)**dim1 * det1), (self.x_sizes+self.xi_sizes))
        self.K1 = torch.reshape(torch.exp(-1/2*x_xi_squared) / math.sqrt((2*math.pi)**dim1 * det1), (math.prod(self.x_sizes), math.prod(self.xi_sizes)))
        
        # create second (y, eta) Gaussian kernel
        dim2 = self.Sigma2.shape[0]
        det2 = self.Sigma2 if dim2 == 1 else torch.det(self.Sigma2)
        #self.K2 = torch.reshape(torch.exp(-1/2*y_eta_squared) / math.sqrt((2*math.pi)**dim2 * det2), (self.y_sizes+self.eta_sizes))
        self.K2 = torch.reshape(torch.exp(-1/2*y_eta_squared) / math.sqrt((2*math.pi)**dim2 * det2), (math.prod(self.y_sizes), math.prod(self.eta_sizes)))
    
    def cs_from_W(self, W):
        A = self.K1 * math.prod(self.delta_xi)
        B = self.K2 * math.prod(self.delta_eta)
        f_bar = self.prev_fs.detach()
        print(torch.linalg.cond(f_bar))
        
        lmbda = 1e-5 * math.prod(self.delta_xi).item()
        
        #AB = torch.kron(A, B)
        #cs = torch.reshape(reginv(torch.flatten(W)[:, None], AB, lmbda), (math.prod(self.xi_sizes), self.n_train))
        
        P = torch.matmul(f_bar, torch.pinverse(f_bar))
        WP = torch.matmul(W, P)
        K1WPK2 = torch.matmul(torch.matmul(A, WP), B.t())
        #cs = torch.matmul(torch.pinverse(f_bar, rcond=1e-15), WP.t()).t()
        #cs = reginv(WP.t(), f_bar, lmbda).t()
        cs = torch.tensor(quadratic_minimizer(WP.t().detach().numpy(), f_bar.numpy(), lmbda)).t().double()
        #print(torch.mean(torch.norm(cs, dim=0)))
        #print(torch.norm(WP - torch.matmul(cs, f_bar.t())) / torch.norm(WP))
        #print(torch.norm(K1WPK2 - torch.matmul(A, torch.matmul(torch.matmul(cs, f_bar.t()), B.t()))) / torch.norm(K1WPK2))
        cs = cs / math.prod(self.delta_y)
        return cs

##############################################################################
#   Convolutional RKHS Layers
##############################################################################

class ConvRKHSLayer(RKHSLayer):
    def __init__(self, x_mesh, y_mesh, xi_mesh, eta_mesh, n_train, bias=False): # must have x and y, xi and eta meshes equal
        nn.Module.__init__(self)
        
        # convolution kernel
        self.k1 = None
        self.k2 = None
        
        self.K1 = None
        self.K2 = None
        
        self.K = None
        
        super().__init__(x_mesh, y_mesh, xi_mesh, eta_mesh, n_train, bias)
    
    def kernel(self, fx, fy, fxi, feta, ein='ijkl'):
        # compute product of first (x, y) Gaussian kernel
        prod1 = self.K1
        if fx is not None:
            fx_flat = torch.flatten(fx, end_dim=-2) if len(fx.shape) > 1 else fx.unsqueeze(-1)
            prod1 = torch.matmul(torch.transpose(fx_flat, 0, 1), prod1) * torch.prod(self.delta_x)
        if fy is not None:
            fy_flat = torch.flatten(fy, end_dim=-2) if len(fy.shape) > 1 else fy.unsqueeze(-1)
            prod1 = torch.matmul(prod1, fy_flat) * torch.prod(self.delta_y)
        
        # compute product of second (xi, eta) Gaussian kernel
        prod2 = self.K2
        if fxi is not None:
            fxi_flat = torch.flatten(fxi, end_dim=-2) if len(fxi.shape) > 1 else fxi.unsqueeze(-1)
            prod2 = torch.matmul(torch.transpose(fxi_flat, 0, 1), prod2) * torch.prod(self.delta_xi)
        if feta is not None:
            feta_flat = torch.flatten(feta, end_dim=-2) if len(feta.shape) > 1 else feta.unsqueeze(-1)
            prod2 = torch.matmul(prod2, feta_flat) * torch.prod(self.delta_eta)
        
        # multiply both kernel products together
        ein_in = ein[0] + ein[1] + ',' + ein[2] + ein[3]
        ein_out = removedup(ein)
        eq = ein_in + '->' + ein_out
        prod = torch.einsum(eq, prod1, prod2)
        
        return prod
    
    def kernel2(self, fx, fy, fxi, feta, ein='ijkl'):
        # compute product of second (x, y) Gaussian kernel
        if fx is not None:
            fx_flat = torch.transpose(torch.flatten(fx, end_dim=-2), 0, 1).unsqueeze(1) if len(fx.shape) > 1 else fx[None, None, :]
            prod1 = fft_conv(fx_flat, self.k1[None, None, :], padding=len(self.k1)//2).squeeze(1).double() * torch.prod(self.delta_x)
            if fy is not None:
                fy_flat = torch.flatten(fy, end_dim=-2) if len(fy.shape) > 1 else fy.unsqueeze(-1)
                prod1 = torch.matmul(prod1, fy_flat) * torch.prod(self.delta_y)
        elif fy is not None:
            fy_flat = torch.transpose(torch.flatten(fy, end_dim=-2), 0, 1).unsqueeze(1) if len(fy.shape) > 1 else fy[None, None, :]
            prod1 = fft_conv(fy_flat, self.k1.flip(0)[None, None, :], padding=len(self.k1)//2) * torch.prod(self.delta_y)
            prod1 = torch.transpose(prod1.squeeze(1), 0, 1).double()
        
        # compute product of second (xi, eta) Gaussian kernel
        if fxi is not None:
            fxi_flat = torch.transpose(torch.flatten(fxi, end_dim=-2), 0, 1).unsqueeze(1) if len(fxi.shape) > 1 else fxi[None, None, :]
            prod2 = fft_conv(fxi_flat, self.k2[None, None, :], padding=len(self.k2)//2).squeeze(1).double() * torch.prod(self.delta_xi)
            if feta is not None:
                feta_flat = torch.flatten(feta, end_dim=-2) if len(feta.shape) > 1 else feta.unsqueeze(-1)
                prod2 = torch.matmul(prod2, feta_flat) * torch.prod(self.delta_xi)
        elif feta is not None:
            feta_flat = torch.transpose(torch.flatten(feta, end_dim=-2), 0, 1).unsqueeze(1) if len(feta.shape) > 1 else feta[None, None, :]
            prod2 = fft_conv(feta_flat, self.k2.flip(0)[None, None, :], padding=len(self.k2)//2) * torch.prod(self.delta_eta)
            prod2 = torch.transpose(prod2.squeeze(1), 0, 1).double()
        
        # multiply both kernel products together
        ein_in = ein[0] + ein[1] + ',' + ein[2] + ein[3]
        ein_out = removedup(ein)
        eq = ein_in + '->' + ein_out
        prod = torch.einsum(eq, prod1, prod2)
        
        return prod
    
    def kernel3(self, fx, fy, fxi, feta, ein='ijkl'):
        if (fy is not None) and (feta is not None):
            fy_flat = torch.flatten(fy, end_dim=-2) if len(fy.shape) > 1 else fy.unsqueeze(-1)
            feta_flat = torch.flatten(feta, end_dim=-2) if len(feta.shape) > 1 else feta.unsqueeze(-1)
            
            imgs = torch.einsum('aj,bl->jlab', fy_flat, feta_flat).float()
            kernel_filter = self.k[None, None, :, :].flip(2).flip(3).float()
            padH = (kernel_filter.shape[2]-1)//2
            padW = (kernel_filter.shape[3]-1)//2
            
            convs = nn.functional.conv2d(torch.flatten(imgs, end_dim=1)[:, None, :, :], kernel_filter, padding=(padH, padW)).squeeze(1).double()
            # convs = torch.zeros(imgs.shape, dtype=torch.float64)
            # print(kernel_filter.shape)
            # for i in range(imgs.shape[0]):
            #     print(i)
            #     #print('i: ' + str(i))
            #     #for j in range(imgs.shape[1]):
            #         #print('j: ' + str(j))
            #         #convs[i, j, :, :] = fft_conv(imgs[i, j, None, None, :, :], kernel_filter, padding=(padH, padW)).squeeze(1)
            #     convs[i, :, :, :] = nn.functional.conv2d(imgs[i, :, None, :, :], kernel_filter, padding=(padH, padW)).squeeze(1)
            convs *= torch.prod(self.delta_x) * torch.prod(self.delta_xi)
            
            k_fy_feta = torch.reshape(convs, (fy_flat.shape[1], feta_flat.shape[1], imgs.shape[2], imgs.shape[3]))
            
            mats = [k_fy_feta]
            ein_in = ein[1] + ein[3] + 'ab'
            ein_out = ''
            scale_factor = 1
            if fx is not None:
                fx_flat = torch.flatten(fx, end_dim=-2) if len(fx.shape) > 1 else fx.unsqueeze(-1)
                mats.append(fx_flat)
                ein_in += ',a' + ein[0]
                ein_out += ein[0]
                scale_factor *= torch.prod(self.delta_x)
            else:
                ein_out += 'a'
            ein_out += ein[1]
            if fxi is not None:
                fxi_flat = torch.flatten(fxi, end_dim=-2) if len(fxi.shape) > 1 else fxi.unsqueeze(-1)
                mats.append(fxi_flat)
                ein_in += ',b' + ein[2]
                ein_out += ein[2]
                scale_factor *= torch.prod(self.delta_xi)
            else:
                ein_out += 'b'
            ein_out += ein[3]
            ein_out = removedup(ein_out)
            
            eq = ein_in + '->' + ein_out
            prod = scale_factor * torch.einsum(eq, mats)
        
        # mats = [self.k]
        # ein_in = 'abcd'
        # ein_out = ''
        # scale_factor = 1
        # if fx is not None:
        #     fx_flat = torch.flatten(fx, end_dim=-2) if len(fx.shape) > 1 else fx.unsqueeze(-1)
        #     mats += [fx_flat]
        #     ein_in += ',a' + ein[0]
        #     ein_out += ein[0]
        #     scale_factor *= torch.prod(self.delta_x)
        # else:
        #     ein_out += 'a'
        # if fy is not None:
        #     fy_flat = torch.flatten(fy, end_dim=-2) if len(fy.shape) > 1 else fy.unsqueeze(-1)
        #     mats += [fy_flat]
        #     ein_in += ',b' + ein[1]
        #     ein_out += ein[1]
        #     scale_factor *= torch.prod(self.delta_y)
        # else:
        #     ein_out += 'b'
        # if fxi is not None:
        #     fxi_flat = torch.flatten(fxi, end_dim=-2) if len(fxi.shape) > 1 else fxi.unsqueeze(-1)
        #     mats += [fxi_flat]
        #     ein_in += ',c' + ein[2]
        #     ein_out += ein[2]
        #     scale_factor *= torch.prod(self.delta_xi)
        # else:
        #     ein_out += 'c'
        # if feta is not None:
        #     feta_flat = torch.flatten(feta, end_dim=-2) if len(feta.shape) > 1 else feta.unsqueeze(-1)
        #     mats += [feta_flat]
        #     ein_in += ',d' + ein[3]
        #     ein_out += ein[3]
        #     scale_factor *= torch.prod(self.delta_eta)
        # else:
        #     ein_out += 'd'
        # ein_out = removedup(ein_out)
        
        # eq = ein_in + '->' + ein_out
        # prod = scale_factor * torch.einsum(eq, mats)
        
        return prod

class ConvGaussianRKHSLayer(ConvRKHSLayer):
    def __init__(self, Sigma1, Sigma2, x_mesh, y_mesh, xi_mesh, eta_mesh, n_train, bias=False):
        nn.Module.__init__(self)
        
        # covariance of Gaussian kernels
        self.Sigma1 = Sigma1.double()
        self.Sigma2 = Sigma2.double()
        
        super().__init__(x_mesh, y_mesh, xi_mesh, eta_mesh, n_train, bias)
    
    def set_kernel(self):
        dim1 = self.Sigma1.shape[0]
        det1 = self.Sigma1 if dim1 == 1 else torch.det(self.Sigma1)
        
        dim2 = self.Sigma2.shape[0]
        det2 = self.Sigma2 if dim2 == 1 else torch.det(self.Sigma2)
        
        # symmetrize the x and xi meshes
        x_mesh_symm = symmetrize_mesh(self.x_mesh)
        xi_mesh_symm = symmetrize_mesh(self.xi_mesh)
        
        x_symm_sizes = tuple(x_mesh_symm[0].shape)
        xi_symm_sizes = tuple(xi_mesh_symm[0].shape)
        
        x_mesh_symm_stack = torch.stack(x_mesh_symm, dim=-1)
        xi_mesh_symm_stack = torch.stack(xi_mesh_symm, dim=-1)
        
        x_mesh_symm_flat = torch.flatten(x_mesh_symm_stack, end_dim=-2)
        xi_mesh_symm_flat = torch.flatten(xi_mesh_symm_stack, end_dim=-2)
        
        # define the convolutional reproducing kernel over these symmetrized meshes
        if dim1 == 1:
            x_squared = torch.squeeze(x_mesh_symm_flat**2 / self.Sigma1)
        else:
            x_squared = torch.sum(x_mesh_symm_flat * torch.transpose(torch.solve(torch.transpose(x_mesh_symm_flat, 0, 1), self.Sigma1)[0], 0, 1), dim=0)
        if dim2 == 1:
            xi_squared = torch.squeeze(xi_mesh_symm_flat**2 / self.Sigma2)
        else:
            xi_squared = torch.sum(xi_mesh_symm_flat * torch.transpose(torch.solve(torch.transpose(xi_mesh_symm_flat, 0, 1), self.Sigma2)[0], 0, 1), dim=0)
        
        self.k1 = torch.reshape(torch.exp(-1/2*x_squared) / math.sqrt((2*math.pi)**dim1 * det1), (math.prod(x_symm_sizes),))
        self.k2 = torch.reshape(torch.exp(-1/2*xi_squared) / math.sqrt((2*math.pi)**dim2 * det2), (math.prod(xi_symm_sizes),))
        
        
        
        
        x_y_diff = torch.flatten(self.x_mesh_flat.unsqueeze(1) - self.y_mesh_flat, end_dim=-2)
        xi_eta_diff = torch.flatten(self.xi_mesh_flat.unsqueeze(1) - self.eta_mesh_flat, end_dim=-2)
        
        if self.x_dim == 1:
            x_y_squared = torch.squeeze(x_y_diff**2 / self.Sigma1)
        else:
            x_y_squared = torch.sum(x_y_diff * torch.transpose(torch.solve(torch.transpose(x_y_diff, 0, 1), self.Sigma1)[0], 0, 1), dim=0)
        if self.y_dim == 1:
            xi_eta_squared = torch.squeeze(xi_eta_diff**2 / self.Sigma2)
        else:
            xi_eta_squared = torch.sum(xi_eta_diff * torch.transpose(torch.solve(torch.transpose(xi_eta_diff, 0, 1), self.Sigma2)[0], 0, 1), dim=0)
        
        # create first (x, xi) Gaussian kernel
        dim1 = self.Sigma1.shape[0]
        det1 = self.Sigma1 if dim1 == 1 else torch.det(self.Sigma1)
        self.K1 = torch.reshape(torch.exp(-1/2*x_y_squared) / math.sqrt((2*math.pi)**dim1 * det1), (math.prod(self.x_sizes), math.prod(self.y_sizes)))
        
        # create second (y, eta) Gaussian kernel
        dim2 = self.Sigma2.shape[0]
        det2 = self.Sigma2 if dim2 == 1 else torch.det(self.Sigma2)
        self.K2 = torch.reshape(torch.exp(-1/2*xi_eta_squared) / math.sqrt((2*math.pi)**dim2 * det2), (math.prod(self.xi_sizes), math.prod(self.eta_sizes)))
        
        
        x_y_xi_eta_diff = torch.flatten(x_mesh_symm_flat.unsqueeze(1) - xi_mesh_symm_flat, end_dim=-2)
        
        if dim1 == 1:
            x_y_xi_eta_squared = torch.squeeze(x_y_xi_eta_diff**2 / self.Sigma1)
        else:
            x_y_xi_eta_squared = torch.sum(x_y_xi_eta_diff * torch.transpose(torch.solve(torch.transpose(x_y_xi_eta_diff, 0, 1), self.Sigma1)[0]), dim=0)
        
        self.k = torch.reshape(torch.exp(-1/2*x_y_xi_eta_squared) / math.sqrt((2*math.pi)**dim1 * det1), (math.prod(x_symm_sizes), math.prod(xi_symm_sizes)))
        center_x = math.floor(self.k.shape[0]//2)
        center_xi = math.floor(self.k.shape[1]//2)
        
        kernel_size = (10, 10)
        self.k = self.k[center_x-kernel_size[0]:center_x+kernel_size[0]+1, center_xi-kernel_size[1]:center_xi+kernel_size[1]+1]
        #self.k = torch.outer(self.k1, self.k2)


##############################################################################
#   Network Examples
##############################################################################

class FullyConnectedBasisNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, fs_train):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        in_basis_shape = (10,) #(layer_meshes[0][0].shape[0] // 2 + 1,)
        out_basis_shape = (10,) #(layer_meshes[0][0].shape[0] // 2 + 1,)
        
        # create FFT basis layers
        self.fftlayer1 = BasisLayer(layer_meshes[1], layer_meshes[0],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        self.fftlayer2 = BasisLayer(layer_meshes[2], layer_meshes[1],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        self.fftlayer3 = BasisLayer(layer_meshes[3], layer_meshes[2],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        self.fftlayer4 = BasisLayer(layer_meshes[4], layer_meshes[3],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        self.fftlayer5 = BasisLayer(layer_meshes[5], layer_meshes[4],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        # list all of the layers in this network
        self.layers = nn.ModuleList([
            self.fftlayer1,
            self.relu1,
            self.fftlayer2,
            #self.relu2,
            #self.fftlayer3,
            #self.relu3,
            #self.fftlayer4,
            #self.relu4,
            #self.fftlayer5
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(self.layers, fs_train)

class FullyConnectedGaussianNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes, fs_train):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        n_train = fs_train.shape[1]
        
        sigma = 1e-4
        
        # create Gaussian RKHS layers
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        self.gaussianRKHS1 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               n_train, bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        self.gaussianRKHS2 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[2], layer_meshes[1],
                                               weight_meshes[1], train_meshes[1],
                                               n_train, bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        self.gaussianRKHS3 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[3], layer_meshes[2],
                                               weight_meshes[2], train_meshes[2],
                                               n_train, bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        self.gaussianRKHS4 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[4], layer_meshes[3],
                                               weight_meshes[3], train_meshes[3],
                                               n_train, bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        self.gaussianRKHS5 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[5], layer_meshes[4],
                                               weight_meshes[4], train_meshes[4],
                                               n_train, bias=True)
        
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        # self.relu1 = nn.LeakyReLU()
        # self.relu2 = nn.LeakyReLU()
        # self.relu3 = nn.LeakyReLU()
        # self.relu4 = nn.LeakyReLU()
        
        # list all of the layers in this network
        self.layers = nn.ModuleList([
            self.gaussianRKHS1,
            self.relu1,
            self.gaussianRKHS2,
            self.relu2,
            self.gaussianRKHS3,
            self.relu3,
            self.gaussianRKHS4,
            self.relu4,
            self.gaussianRKHS5
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(self.layers, fs_train)

class FullyConnectedGaussianNetwork2D(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes, fs_train):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        n_train = fs_train.shape[1]
        
        sigma = 1e-3
        
        # create Gaussian RKHS layers
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        self.gaussianRKHS1 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               n_train, bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        self.gaussianRKHS2 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[2], layer_meshes[1],
                                               weight_meshes[1], train_meshes[1],
                                               n_train, bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        self.gaussianRKHS3 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[3], layer_meshes[2],
                                               weight_meshes[2], train_meshes[2],
                                               n_train, bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        self.gaussianRKHS4 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[4], layer_meshes[3],
                                               weight_meshes[3], train_meshes[3],
                                               n_train, bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        self.gaussianRKHS5 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[5], layer_meshes[4],
                                               weight_meshes[4], train_meshes[4],
                                               n_train, bias=True)
        
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        # list all of the layers in this network
        self.layers = nn.ModuleList([
            self.gaussianRKHS1,
            self.relu1,
            self.gaussianRKHS2,
            self.relu2,
            self.gaussianRKHS3,
            self.relu3,
            self.gaussianRKHS4,
            self.relu4,
            self.gaussianRKHS5
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(self.layers, fs_train)


class ConvolutionalGaussianNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes, fs_train):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        n_train = fs_train.shape[1]
        
        # create Gaussian RKHS layers
        Sigma1 = torch.tensor([5e-3])
        Sigma2 = torch.tensor([5e-3])
        self.gaussianRKHS1 = ConvGaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               n_train)
        Sigma1 = torch.tensor([5e-3])
        Sigma2 = torch.tensor([5e-3])
        self.gaussianRKHS2 = ConvGaussianRKHSLayer(Sigma1, Sigma2,
                                              layer_meshes[2], layer_meshes[1],
                                              weight_meshes[1], train_meshes[1],
                                              n_train)
        Sigma1 = torch.tensor([5e-3])
        Sigma2 = torch.tensor([5e-3])
        self.gaussianRKHS3 = ConvGaussianRKHSLayer(Sigma1, Sigma2,
                                              layer_meshes[3], layer_meshes[2],
                                              weight_meshes[2], train_meshes[2],
                                              n_train)
        
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        # list all of the layers in this network
        self.layers = nn.ModuleList([
            self.gaussianRKHS1,
            self.relu1,
            self.gaussianRKHS2,
            self.relu2,
            self.gaussianRKHS3
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(self.layers, fs_train)

class DiscriminatorNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes, fs_train):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        n_train = fs_train.shape[1]
        
        # create Gaussian RKHS layers
        Sigma1 = 1e-3 * torch.eye(2)
        Sigma2 = 1e-3 * torch.eye(2)
        self.gaussianRKHS1 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               n_train, bias=True)
        Sigma1 = 1e-3 * torch.eye(2)
        Sigma2 = 1e-3 * torch.eye(2)
        self.gaussianRKHS2 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[2], layer_meshes[1],
                                               weight_meshes[1], train_meshes[1],
                                               n_train, bias=True)
        Sigma1 = torch.tensor([1])
        Sigma2 = 1e-3 * torch.eye(2)
        self.gaussianRKHS3 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[3], layer_meshes[2],
                                               weight_meshes[2], train_meshes[2],
                                               n_train, bias=True)
        
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU()
        # self.relu2 = nn.LeakyReLU()
        
        # list all of the layers in this network
        self.layers = nn.ModuleList([
            self.gaussianRKHS1,
            self.relu1,
            self.gaussianRKHS2,
            self.relu2,
            self.gaussianRKHS3,
            self.sigmoid
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(self.layers, fs_train)

class GeneratorNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes, fs_train):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        n_train = fs_train.shape[1]
        
        # create Gaussian RKHS layers
        Sigma1 = 1e-3 * torch.eye(2)
        Sigma2 = torch.tensor([1])
        self.gaussianRKHS1 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               n_train, bias=True)
        Sigma1 = 1e-3 * torch.eye(2)
        Sigma2 = 1e-3 * torch.eye(2)
        self.gaussianRKHS2 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[2], layer_meshes[1],
                                               weight_meshes[1], train_meshes[1],
                                               n_train, bias=True)
        Sigma1 = 1e-3 * torch.eye(2)
        Sigma2 = 1e-3 * torch.eye(2)
        self.gaussianRKHS3 = GaussianRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[3], layer_meshes[2],
                                               weight_meshes[2], train_meshes[2],
                                               n_train, bias=True)
        
        # activation functions
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # list all of the layers in this network
        self.layers = nn.ModuleList([
            self.gaussianRKHS1,
            self.relu1,
            self.gaussianRKHS2,
            self.relu2,
            self.gaussianRKHS3,
            self.tanh
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(self.layers, fs_train)
