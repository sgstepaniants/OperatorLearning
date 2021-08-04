import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import griddata

def nonlinHelmholtz(f, x_mesh, bv=[0, 0], alpha=-1, eps=-0.3):
    force = lambda x : griddata(x_mesh, f, x)
    fun = lambda x, y : np.vstack((y[1], -alpha*y[0]-eps*y[0]**3+force(x)))
    bc = lambda ya, yb : np.array([ya[0] - bv[0], yb[0] - bv[1]])
    
    y0 = np.zeros((2, x_mesh.size))
    res = solve_bvp(fun, bc, x_mesh, y0)
    sol = res.sol(x_mesh)[0]
    return sol
