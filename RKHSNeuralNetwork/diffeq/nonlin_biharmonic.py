import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import griddata

def nonlinBiharmonic(f, x_mesh, bv=[[0, 0], [0, 0]], eps=0.4):
    force = lambda x : griddata(x_mesh, f, x)
    p = lambda x : 0.5 * np.sin(x) - 3
    dp = lambda x : 0.5 * np.cos(x)
    d2p = lambda x : -0.5 * np.sin(x)
    q = lambda x : 0.6 * np.sin(x) - 2
    
    fun = lambda x, y : np.vstack((y[1], y[2], y[3], (-2*dp(x)*y[2]-d2p(x)*y[1]+q(x)*(y[0]+eps*y[0]**3)-force(x))/p(x)))
    bc = lambda ya, yb : np.concatenate((ya[:2] - bv[:, 0], yb[:2] - bv[:, 1]))
    
    y0 = np.zeros((4, x_mesh.size))
    res = solve_bvp(fun, bc, x_mesh, y0)
    sol = res.sol(x_mesh)[0]
    return sol
