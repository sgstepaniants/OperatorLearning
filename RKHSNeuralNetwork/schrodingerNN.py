import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
from scipy import sparse

import sys
sys.path.append('../')
from generate.simulate_schrodinger2D import simulate_schrodinger2D
from NetworkExamples import BoundaryIntegralNetwork

params = {'font.family': 'Times New Roman',
         'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

###############################################################################
#   Function Declarations
###############################################################################

def relative_squared_error(output, target, mean=True, dim=None):
    if dim is None:
        dim = tuple(range(output.ndim-1))
    
    loss = torch.sum((output - target)**2, dim) / torch.sum(target**2, dim)
    if mean:
        loss = torch.mean(loss)
    return loss

def mean_squared_error(output, target):
    loss = torch.mean(torch.sum((output - target)**2, dim=tuple(range(output.ndim-1))))
    return loss


###############################################################################
#   Load Nonlinear Heat PDE Forcings and Solutions
###############################################################################

num_train = 1000
N = 50 # number of points on sidelength of 2D square domain
L = 1

# train mesh for boundary domain
mb_train = 4*N - 4
deltab_train = 4*L / (mb_train-1)
b_train = np.linspace(0, 4*L, mb_train)

# train mesh for interior domain
mx_train = N
deltax_train = L / (mx_train-1)
x_train = np.linspace(0, L, mx_train)

X, Y = np.meshgrid(x_train, x_train)

def potential(x, y):
    a = L/4
    xc = L/2
    yc = L/2
    idx = np.abs(x-xc) < (a*math.sqrt(3)/2)
    idx *= math.tan(math.pi/6) * (x-xc) + a > (y-yc)
    idx *= math.tan(math.pi/6) * (x-xc) - a < (y-yc)
    idx *= -math.tan(math.pi/6) * (x-xc) - a < (y-yc)
    idx *= -math.tan(math.pi/6) * (x-xc) + a > (y-yc)
    pot = 1e4 * idx
    
    #pot = -np.abs(x-xc)**2 - np.abs(y-yc)**2 + 0.1
    #pot[pot < 0] = 0
    return pot

# true potential function
V_true = potential(X, Y)

# generate train input boundary conditions as Brownian bridges
mag = 1
kl_modes = 20
freqs = np.arange(1, kl_modes+1) / L
fs_train = mag * math.sqrt(2) * (np.sin(math.pi*np.outer(b_train, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_train))

bcs_train = np.zeros([N, 4, num_train ])
bcs_train[:, 0, :] = fs_train[0:N, :]
bcs_train[:, 1, :] = fs_train[N-1:2*N-1, :]
bcs_train[:, 2, :] = np.flip(fs_train[2*N-2:3*N-2, :], axis=0)
bcs_train[:N-1, 3, :] = np.flip(fs_train[3*N-3:4*N-3, :], axis=0)
bcs_train[N-1, 3, :] = fs_train[0, :]
us_train = simulate_schrodinger2D(L, N, potential, bcs_train)

example_ind = 0

plt.figure(1)
#plt.plot(x_train, bcs_train[:, 0, example_ind])
#plt.plot(x_train, bcs_train[:, 1, example_ind])
#plt.plot(x_train, bcs_train[:, 2, example_ind])
plt.plot(x_train, bcs_train[:, 3, example_ind])
plt.show()

plt.figure(2)
#plt.plot(x_train, us_train[0, :, example_ind])
#plt.plot(x_train, us_train[:, -1, example_ind])
#plt.plot(x_train, us_train[-1, :, example_ind])
plt.plot(x_train, us_train[:, 0, example_ind])
plt.show()

plt.figure(3)
plt.pcolormesh(X, Y, us_train[:, :, example_ind], shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()


fs_train = torch.from_numpy(fs_train)
us_train = torch.reshape(torch.from_numpy(us_train), (N, N, num_train))
x_train = torch.from_numpy(x_train)
b_train = torch.from_numpy(b_train)
V_true = torch.from_numpy(V_true)

plt.figure(5)
plt.title('Potential Barrier', fontweight='bold')
plt.pcolormesh(x_train.detach(),
               x_train.detach(),
               V_true.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('x$_1$')
plt.ylabel('x$_2$')
plt.tight_layout()
plt.savefig('potential', dpi=300)
plt.show()

train_mesh = torch.meshgrid(x_train, x_train)
delta_train = (x_train[1] - x_train[0]) * (x_train[1] - x_train[0])

###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# normalize the data
f_norm = torch.mean(torch.sqrt(torch.sum(fs_train**2, dim=0)*delta_train))
u_norm = torch.mean(torch.sqrt(torch.sum(us_train**2, dim=0)*delta_train))

layer_meshes = [(b_train,), train_mesh]
weight_meshes = [train_mesh]
train_meshes = [(b_train,)]

# create functional neural network
model = BoundaryIntegralNetwork(layer_meshes, weight_meshes, train_meshes)
model.rescale(1/f_norm, u_norm)

# optimize using gradient descent
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=10, amsgrad=True)

# regularization weights
lambdas = 0 * torch.ones(1)

example_ind = 0

# iterate to find the optimal network parameters
epochs = 1000
batch_size = 100
freq = 10
rses = []
for epoch in range(epochs):
    perm = torch.randperm(num_train)
    batch_loss = 0
    batch_rse = 0
    for i in range(0, num_train, batch_size):
        inds = perm[i:i+batch_size]
        fs_batch = fs_train[:, inds]
        us_batch = us_train[:, :, inds]
        us_hat = torch.reshape(model(fs_batch), (mx_train, mx_train, batch_size))
        
        loss = mean_squared_error(us_hat, us_batch)
        loss += model.compute_regularization(lambdas)
        
        optimizer.zero_grad()
        # prevent gradient measurement to accumulate
        loss.backward()
        
        # calculate gradient in each iteration
        optimizer.step()
    
    if epoch % freq == 0:
        torch.save(model.state_dict(), 'model_state.pth')
        torch.save(model, 'model.pth')
        
        plt.figure(1)
        u = us_train[:, :, example_ind]
        u_hat = torch.reshape(model(fs_train[:, example_ind]), (mx_train, mx_train))
        diff = torch.abs(u - u_hat)
        diff = torch.reshape(diff, (mx_train, mx_train))
        plt.pcolormesh(train_mesh[0].detach().numpy(),
                       train_mesh[1].detach().numpy(),
                       diff.detach().numpy(),
                       shading='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.colorbar()
        plt.show()
    
    # average losses for this epoch
    us_hat = torch.reshape(model(fs_train), (mx_train, mx_train, num_train))
    rse = relative_squared_error(us_hat, us_train).item()
    print('Epoch {} Relative Squared Error {}'.format(epoch, rse))
    rses.append(rse)

# plot loss over iterations
plt.figure(2)
plt.title('Train Relative Squared Errors', fontweight='bold')
plt.plot(range(len(rses)), rses, label='Train RSE')
plt.xlabel('Epochs')
plt.ylabel('Relative Train Error')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('train_errors', dpi=300)
plt.show()

# compute the relative train error on the solutions
us_train_hat = torch.reshape(model(fs_train), (mx_train, mx_train, num_train))
train_rse = relative_squared_error(us_train_hat, us_train).item()
print('Train Relative Squared Error: ' + str(train_rse))

# plot best and worst case train predictions
train_rses = relative_squared_error(us_train_hat, us_train, mean=False)
best_rse, best_ind = torch.min(train_rses, 0)
worst_rse, worst_ind = torch.max(train_rses, 0)

best_u = us_train[:, :, best_ind]
best_u_hat = torch.reshape(model(fs_train[:, best_ind]), (mx_train, mx_train))
best_diff = torch.abs(best_u_hat - best_u)

plt.figure(5)
plt.title('Best Train Prediction (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               best_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('best_train_pred', dpi=300)
plt.show()

plt.figure(6)
plt.title('Best Train True Solution (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               best_u.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('best_train_true', dpi=300)
plt.show()

plt.figure(7)
plt.title('Best Train Difference (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               best_diff.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('best_train_diff', dpi=300)
plt.show()


worst_u = us_train[:, :, worst_ind]
worst_u_hat = torch.reshape(model(fs_train[:, worst_ind]), (mx_train, mx_train))
worst_diff = torch.abs(worst_u_hat - worst_u)

plt.figure(8)
plt.title('Worst Train Prediction (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               worst_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('worst_train_pred', dpi=300)
plt.show()

plt.figure(9)
plt.title('Worst Train True Solution (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               worst_u.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('worst_train_true', dpi=300)
plt.show()

plt.figure(10)
plt.title('Worst Train Difference (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               worst_diff.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('worst_train_diff', dpi=300)
plt.show()

# plot the learned Green's function
G = model.layers[0].G()
plt.figure(11)
plt.pcolormesh(G.detach())
plt.colorbar()

G_full = torch.reshape(G, (mx_train, mx_train, mb_train))
plt.figure(12)
plt.pcolormesh(G_full[10, :, :].detach())
plt.colorbar()



# compute the Laplacian of the Green's function
D = torch.diag(torch.ones(mx_train-1), diagonal=1) - 2*torch.diag(torch.ones(mx_train), diagonal=0) + torch.diag(torch.ones(mx_train-1), diagonal=-1)
D /= deltax_train**2
D = D.double()
dx2 = torch.matmul(D, torch.reshape(G, (mx_train, -1)))
dx2 = torch.reshape(dx2, (mx_train, mx_train, mb_train))
dy2 = torch.matmul(torch.reshape(torch.reshape(G, (mx_train, mx_train, mb_train)).permute(0, 2, 1), (-1, mx_train)), D.t())
dy2 = torch.reshape(dy2, (mx_train, mb_train, mx_train)).permute(0, 2, 1)
LoG = dx2 + dy2

# reconstruct the potential of the Schrodinger equation from the learned
# boundary integral Green's function
delta_y = math.prod(model.layers[0].delta_y)
V = 1/2 * torch.sum(LoG / torch.reshape(G, (mx_train, mx_train, mb_train)), dim=2) * delta_y

plt.figure(13)
plt.pcolormesh(train_mesh[0].detach(),
                train_mesh[1].detach(),
                V.detach(),
                shading='auto')
plt.colorbar()

u = us_train[:, :, 0]
Lu = torch.matmul(D, u) + torch.matmul(u, D)
denom = u.clone()
denom[torch.abs(denom) < 1e-7] = math.nan
plt.pcolormesh(1/2*(Lu/denom)[2:-2, 2:-2].detach())
plt.colorbar()
plt.show()

plt.pcolormesh(Lu.detach())
plt.colorbar()
plt.show()

plt.pcolormesh((us_train[:, :, 0] - us_train_hat[:, :, 0]).detach())
plt.colorbar()
plt.show()


###############################################################################
#   Test Functional Neural Network on New Mesh Sizes
###############################################################################

# indicate that we have finished training our model
model.eval()

# number of test samples
num_test = 1000

mesh_sizes = [50, 100, 150, 200, 250, 300]
test_mesh_rses = []
for N in mesh_sizes:
    print(N)
    
    # test mesh for boundary domain
    mb_test = 4*N - 4
    deltab_test = 4*L / (mb_test-1)
    b_test = np.linspace(0, 4*L, mb_test)
    
    # test mesh for interior domain
    mx_test = N
    deltax_test = L / (mx_test-1)
    x_test = np.linspace(0, L, mx_test)
    
    # generate test input boundary conditions as Brownian bridges
    freqs = np.arange(1, kl_modes+1) / L
    fs_test = mag * math.sqrt(2) * (np.sin(math.pi*np.outer(b_test, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_test))
    
    bcs_test = np.zeros([N, 4, num_test ])
    bcs_test[:, 0, :] = fs_test[0:N, :]
    bcs_test[:, 1, :] = fs_test[N-1:2*N-1, :]
    bcs_test[:, 2, :] = np.flip(fs_test[2*N-2:3*N-2, :], axis=0)
    bcs_test[:N-1, 3, :] = np.flip(fs_test[3*N-3:4*N-3, :], axis=0)
    bcs_test[N-1, 3, :] = fs_test[0, :]
    us_test = simulate_schrodinger2D(L, N, potential, bcs_test)
    
    fs_test = torch.from_numpy(fs_test)
    us_test = torch.reshape(torch.from_numpy(us_test), (N, N, num_test))
    x_test = torch.from_numpy(x_test)
    b_test = torch.from_numpy(b_test)
    
    test_mesh = torch.meshgrid(x_test, x_test)
    delta_test = (x_test[1] - x_test[0]) * (x_test[1] - x_test[0])
    
    # create list for test meshes
    test_meshes = [(b_test,), test_mesh]
    
    # change the mesh resolution at each layer of the network to the test set mesh
    model.set_resolution(test_meshes)
    
    # run the test input functions through the tested network
    us_test_hat = torch.reshape(model(fs_test), (mx_test, mx_test, num_test))
    
    # compute the relative test error on the solutions
    test_rse = relative_squared_error(us_test_hat, us_test).item()
    test_mesh_rses.append(test_rse)

# plot test error of model on new test meshes
plt.figure(7)
plt.title('Adaptation to Test Meshes (Trained on mxm = {}x{})'.format(mx_train, mx_train), y=1.05, fontweight='bold')
plt.plot(mesh_sizes, test_mesh_rses)
plt.xlabel('Mesh Size (mxm)')
plt.ylabel('Relative Test Error')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('mesh_adapt', dpi=300)
plt.show()

# plot best and worst case test predictions
test_rses = relative_squared_error(us_test_hat, us_test, mean=False)
best_rse, best_ind = torch.min(test_rses, 0)
worst_rse, worst_ind = torch.max(test_rses, 0)

best_u = us_test[:, :, best_ind]
best_u_hat = torch.reshape(model(fs_test[:, best_ind]), (mx_test, mx_test))
best_diff = best_u_hat - best_u

plt.figure(13)
plt.title('Best Test Prediction (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               best_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('best_test_pred', dpi=300)
plt.show()

plt.figure(14)
plt.title('Best Test True Solution (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               best_u.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('best_test_true', dpi=300)
plt.show()

plt.figure(15)
plt.title('Best Test Difference (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               best_diff.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('best_test_diff', dpi=300)
plt.show()


worst_u = us_test[:, :, worst_ind]
worst_u_hat = torch.reshape(model(fs_test[:, worst_ind]), (mx_test, mx_test))
worst_diff = worst_u_hat - worst_u

plt.figure(16)
plt.title('Worst Test Prediction (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               worst_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('worst_test_pred', dpi=300)
plt.show()

plt.figure(17)
plt.title('Worst Test True Solution (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               worst_u.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('worst_test_true', dpi=300)
plt.show()

plt.figure(18)
plt.title('Worst Test Difference (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               worst_diff.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.tight_layout()
plt.savefig('worst_test_diff', dpi=300)
plt.show()
