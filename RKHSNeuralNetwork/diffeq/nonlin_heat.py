from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
from fipy.viewers.matplotlibViewer.matplotlib1DViewer import Matplotlib1DViewer as Viewer
from builtins import range
import numpy as np

## This script was derived from
## 'examples/diffusion/mesh1D.py'
def nonlinHeatEq(phi_init, grid, T, bc=[0, 0], diffusivity=1, alpha=0):
    valueLeft = float(bc[0])
    valueRight = float(bc[1])
    
    dt = 1e-3
    steps = int(T // dt)
    
    mesh = Grid1D(nx = grid.size-2, dx = (grid[1]-grid[0])) + (grid[1]-grid[0])/2
    
    # initial condition
    phi = CellVariable(name="solution variable",
                        mesh=mesh,
                        hasOld=1)
    
    phi.setValue(phi_init)
    phi.constrain(valueLeft, where=mesh.facesLeft)
    phi.constrain(valueRight, where=mesh.facesRight)
    
    eq = TransientTerm() == DiffusionTerm(coeff=diffusivity*np.exp(alpha*phi))
    
    for step in range(steps):
        #print('Step: ' + str(step))
        # only move forward in time once per time step
        phi.updateOld()
        
        res = 1e+10
        while res > 1e-6:
            res = eq.sweep(var=phi, dt=dt)
        #print(res)
    
    #viewer = Viewer(vars=phi)
    #viewer.plot()
    
    return phi.value
