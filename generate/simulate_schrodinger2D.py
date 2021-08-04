import numpy as np
from scipy import sparse

def simulate_schrodinger2D(L, N, pot, bcs):
    # create grid
    h = L / (N-1)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x[1:N-1], y[1:N-1])
    
    # get potential
    V = pot(X, Y)
    
    # get forcing
    f = np.zeros([N-2, N-2])
    
    # build Hamiltonian
    diag = np.ones([N-2])
    diags = np.array([diag, -2*diag, diag])
    D = sparse.spdiags(diags, np.array([-1,0,1]), N-2, N-2) / h**2
    T = 1/2 * sparse.kronsum(D, D)
    C = sparse.diags(V.reshape((N-2)**2), (0))
    H = T - C
    
    # enforce the four boundary/gauge conditions
    m = bcs.shape[2]
    fs = np.repeat(f[:, :, np.newaxis], m, axis=2)
    fs[0, :, :] = -bcs[1:N-1, 0, :] / (2*h**2)
    fs[:, -1, :] = -bcs[1:N-1, 1, :] / (2*h**2)
    fs[-1, :, :] = -bcs[1:N-1, 2, :] / (2*h**2)
    fs[:, 0, :] = -bcs[1:N-1, 3, :] / (2*h**2)
    fs = np.reshape(fs, [(N-2)**2, m])
    
    # solve for the solution
    sols = sparse.linalg.spsolve(H, fs)
    sols = np.reshape(sols, [N-2, N-2, m])
    us = np.zeros([N, N, m])
    us[1:N-1, 1:N-1, :] = sols
    us[0, :, :] = bcs[:, 0, :]
    us[:, -1, :] = bcs[:, 1, :]
    us[-1, :, :] = bcs[:, 2, :]
    us[:, 0, :] = bcs[:, 3, :]
    
    return us
