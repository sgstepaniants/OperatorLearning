import numpy as np
import matplotlib.pyplot as plt
import h5py

import sys
sys.path.append('../UQpy/StochasticProcess/KLE')
from UQpy.StochasticProcess import KLE

def generateKLE(n, mesh, mu, l, numeigs=None):
    d = len(mesh)
    mesh_stack = np.stack(mesh, axis=-1)
    mesh_flat = mesh_stack.reshape(-1, d)
    diffs = mesh_flat[:, None, :] - mesh_flat
    dists = np.linalg.norm(diffs, axis=2)
    K = np.exp(-dists / l)
    kle = KLE(n, K, None, numeigs)
    grfs = mu + np.transpose(np.squeeze(kle.samples, 1))
    return grfs

# # number of samples
# n = 100

# # mesh size
# mx = 40
# my = 40
# x = np.linspace(0, 1, mx)
# y = np.linspace(0, 1, my)
# mesh = np.meshgrid(x, y, indexing='ij')

# # generate Gaussian random fields
# mu = 0
# l = 0.1
# numeigs = 100
# grfs = generateKLE(n, mesh, mu, l, numeigs)
# grfs = np.exp(grfs) > 3

# # hf = h5py.File('./grfs/KLE{}_n={}_m={}x{}.h5'.format(numeigs, n, mx, my), 'w')
# # hf.create_dataset('grfs', data=grfs)
# # hf.create_dataset('x', data=x)
# # hf.create_dataset('y', data=y)
# # hf.close()

# plt.pcolormesh(mesh[0], mesh[1], np.reshape(grfs[:, 0], (mx, my)), shading='auto', cmap='gray')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()
