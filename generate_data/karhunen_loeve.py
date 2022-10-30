import math
import numpy as np

# Constructs n samples of a KL expansion for a 1D Brownian bridge on [xmin, xmax].
# The mesh to save the Brownian bridge is given by xs and is expected to lie in [xmin, xmax].
# modes determines the number of terms used in the KL expansion.
def brownian_bridge1D(xs, n, modes=100, xmin=0, xmax=1):
    if np.min(xs) < xmin or np.max(xs) > xmax:
        raise ValueError('mesh x lies outside interval [xmin, xmax]')
    L = xmax - xmin
    freqs = np.arange(1, modes+1)
    samples = math.sqrt(2*L) * (np.sin(math.pi*np.outer(xs, freqs)/L) / (math.pi*freqs)) @ np.random.randn(modes, n)
    return samples

def general_radial_kle(mesh, n, modes=100, f=lambda x: np.exp(-x**2)):
    modes = min(modes, np.prod(mesh[0].shape))
    d = len(mesh)
    mesh_stack = np.stack(mesh, axis=-1)
    mesh_flat = mesh_stack.reshape(-1, d)
    diffs = mesh_flat[:, None, :] - mesh_flat
    dists = np.linalg.norm(diffs, axis=2)
    K = f(dists)
    lmbda, phi = np.linalg.eigh(K)
    lmbda = (lmbda+np.abs(lmbda))/2
    lmbda = lmbda[-modes:]
    phi = np.real(phi[:, -modes:])

    xi = np.random.normal(size=(modes, n))
    samples = np.dot(phi*np.sqrt(lmbda)[None, :], xi)
    samples = np.real(samples)
    samples = samples.reshape(list(mesh[0].shape) + [n])
    return samples

#import matplotlib.pyplot as plt
#mx = 40
#mt = 40
#mesh = (np.linspace(0, 1, mx),)
#mesh = np.meshgrid(np.linspace(0, 1, mx), np.linspace(0, 1, mt))
#n = 1
#l = 1e-1
#modes = 1000
#samples = general_radial_kle(mesh, n, modes, f=lambda x: np.exp(-x/l))

#plt.plot(mesh[0], samples[:, 0])
#plt.show()

#plt.pcolormesh(mesh[0], mesh[1], samples[:, :, 0])
#plt.colorbar()
#plt.show()
