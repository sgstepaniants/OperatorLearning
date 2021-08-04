from fipy import CellVariable, Grid2D, GaussianNoiseVariable, DiffusionTerm, TransientTerm, ImplicitSourceTerm, Viewerfrom fipy.tools import numeriximport numpy as npimport matplotlib.pyplot as pltnx = 50ny = 50mesh = Grid2D(nx=nx, ny=ny, dx=0.25, dy=0.25)phi = CellVariable(name=r"$\phi$", mesh=mesh)psi = CellVariable(name=r"$\psi$", mesh=mesh)noise = GaussianNoiseVariable(mesh=mesh,                              mean=0.5,                              variance=0.01).valuephi[:] = noiseD = a = epsilon = 1.dfdphi = a**2 * phi * (1 - phi) * (1 - 2 * phi)dfdphi_ = a**2 * (1 - phi) * (1 - 2 * phi)d2fdphi2 = a**2 * (1 - 6 * phi * (1 - phi))eq1 = (TransientTerm(var=phi) == DiffusionTerm(coeff=D, var=psi))eq2 = (ImplicitSourceTerm(coeff=1., var=psi)       == ImplicitSourceTerm(coeff=d2fdphi2, var=phi) - d2fdphi2 * phi + dfdphi       - DiffusionTerm(coeff=epsilon**2, var=phi))eq3 = (ImplicitSourceTerm(coeff=1., var=psi)       == ImplicitSourceTerm(coeff=dfdphi_, var=phi)       - DiffusionTerm(coeff=epsilon**2, var=phi))eq = eq1 & eq2dexp = -5elapsed = 0.duration = 1.sols = []while elapsed < duration:    print(elapsed)    dt = min(100, numerix.exp(dexp))    elapsed += dt    dexp += 0.01    eq.solve(dt=dt)    sols.append(np.copy(phi.value).reshape(nx, ny))