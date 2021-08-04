import math
import numpy as np
import torch
from scipy import sparse
from scipy import interpolate
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from RKHSNetworkModel import *

###############################################################################
#   Function Declarations
###############################################################################

# output and target must be of size m x n (grid points by samples)
def relative_squared_error(output, target, mean=True, dim=None):
    if dim is None:
        dim = tuple(range(output.ndim-1))
    
    loss = torch.sum((output - target)**2, dim) / torch.sum(target**2, dim)
    if mean:
        loss = torch.mean(loss)
    return loss

def mean_squared_error(output, target):
    loss = torch.mean(torch.sum((output - target)**2, 0))
    return loss

###############################################################################
#   Generate Helmholtz PDE Forcings and Solutions
###############################################################################

# number of train samples
num_train = 1000

# size of interval domain [0, L]
L = 1

# train mesh
m_train = 100
delta_train = L / (m_train-1)
x_train = np.linspace(0, L, m_train)

# mesh for kernel weights
m_weight = 100
h_weight = L / (m_weight-1)
x_weight = np.linspace(0, L, m_weight)

# wave number
#w = 0
w = 0

# Dirichlet (left and right endpoint) boundary conditions
a = 1
b = -1

# standard deviation of white noise added to ouputs (relative to their norm)
sigma = 0.1

# generate train input functions as Brownian bridges
mag = 1e2
kl_modes = 100
freqs = np.arange(1, kl_modes+1) / L
fs_train = mag * math.sqrt(2) * (np.sin(math.pi*np.outer(x_train, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_train))

# compute corresponding train and test solutions to PDE
us_train = np.zeros([m_train, num_train])
K_train = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(m_train-2, m_train-2))
K_train = K_train / delta_train**2 - w**2 * np.eye(m_train-2)
for k in range(num_train):
    print(k)
    f_pert = np.copy(fs_train[1:m_train-1, k])
    f_pert[0] += a/delta_train**2
    f_pert[-1] += b/delta_train**2
    us_train[:, k] = np.concatenate(([a], np.linalg.solve(K_train, f_pert), [b]))

fs_train = torch.from_numpy(fs_train)
us_train = torch.from_numpy(us_train)

# convert data to tensors
x_train = torch.from_numpy(x_train)
x_weight = torch.from_numpy(x_weight)

# true Green's kernel
X, Y = torch.meshgrid(x_train, x_train)
G_true = torch.zeros(m_train, m_train)
if w == 0:
    G_true = (X + Y - torch.abs(Y - X)) / 2 - X * Y
else:
    modes = 100
    for k in range(1, modes):
        p = math.pi*k/L
        G_true += 2/L * torch.sin(p*X) * torch.sin(p*Y) / (p**2 - w**2)

# true bias term
nu_true = (b - a) * x_train + a
if w != 0:
    A = (b - a * math.cos(w)) / math.sin(w)
    B = a
    nu_true = A*torch.sin(w*x_train) + B*torch.cos(w*x_train)

# add noise to solutions
noise_train = sigma * torch.mean(torch.std(us_train, dim=0)) * torch.randn(m_train, num_train)
#noise_train = sigma * torch.mean(torch.std(us_train - nu_true[:, None], dim=0)) * torch.randn(m_train, num_train)
us_train += noise_train

# compute the signal to noise of the solutions on the train data
snr = torch.mean(torch.sum((us_train - nu_true[:, None])**2, dim=0)) / torch.mean(torch.sum(noise_train**2, dim=0))
snr = torch.sqrt(snr)
print('Signal to Noise Ratio: ' + str(snr))

# fs_mean = torch.mean(fs_train, dim=1)
# us_mean = torch.mean(us_train, dim=1)
# b = us_mean - fs_mean

# fs_centered = fs_train - fs_mean[:, None]
# us_centered = us_train - us_mean[:, None]

# lmbda = 0
# Gamma1 = torch.matmul(fs_centered.t(), fs_centered) + lmbda*torch.eye(num_train)
# G_lsqr1 = torch.matmul(us_centered, torch.solve(fs_centered.t(), Gamma1)[0])

# Gamma2 = torch.matmul(fs_centered, fs_centered.t()) + lmbda*torch.eye(m_train)
# G_lsqr2 = torch.matmul(us_centered, torch.solve(fs_centered, Gamma2)[0].t())

###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# normalize the data
f_norm = torch.mean(torch.sqrt(torch.sum(fs_train**2, dim=0)*delta_train))
u_norm = torch.mean(torch.sqrt(torch.sum(us_train**2, dim=0)*delta_train))

# create list for weight and train meshes
train_mesh = torch.meshgrid(x_train)

# create functional neural network
#model = GreensKernelNetwork(train_meshes, weight_meshes, train_meshes[:-1])
#model = SobolevGreensFunction(train_meshes)
#model = ConvKernelNetwork(train_meshes, weight_meshes)
#model = FullyConnectedGaussianNetwork(train_meshes, weight_meshes, train_meshes[:-1])
#model = FullyConnectedBasisNetwork(train_meshes)
model = VanillaNetwork(train_mesh, train_mesh, bias=True)
model.rescale(1/f_norm, u_norm)

# optimize using gradient descent
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1, amsgrad=True)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e1, amsgrad=True)

# regularization weights
lambdas = [0]
#lambdas = [5e-3]
#lambdas = [1e-5] # use 0 to see artifacts!

# training example to plot
example_ind = 0

# iterate to find the optimal network parameters
epochs = 1000
batch_size = 100
freq = 10
rses = []
G_rses = []
nu_rses = []
for epoch in range(epochs):
    perm = torch.randperm(num_train)
    for i in range(0, num_train, batch_size):
        inds = perm[i:i+batch_size]
        fs_batch = fs_train[:, inds]
        us_batch = us_train[:, inds]
        us_hat = model(fs_batch)
        
        loss = delta_train * mean_squared_error(us_hat, us_batch)
        #loss += model.compute_regularization(lambdas, lambdas)
        #rse = relative_squared_error(us_hat, us_batch)
        
        optimizer.zero_grad()
        # prevent gradient measurement to accumulate
        loss.backward()
        
        # calculate gradient in each iteration
        optimizer.step()
    
    if epoch % freq == 0:
        u = us_train[:, example_ind]
        u_hat = model(fs_train[:, example_ind])
        plt.figure(1)
        plt.title('Training Output Example', fontweight='bold')
        plt.plot(x_train, u, label='$u$')
        plt.plot(x_train, u_hat.detach(), '--', label='$\widehat{u}$')
        plt.xlabel('x')
        plt.legend(loc='upper right')
        plt.show()
    
    # compute train rse over all data
    us_hat = model(fs_train)
    rse = relative_squared_error(us_hat, us_train).item()
    print('Epoch {} Relative Squared Error {}'.format(epoch, rse))
    rses.append(rse)
    
    # compute rse between true and predicted Green's kernel and bias
    #G = model.layers[0].G() * u_norm / f_norm
    G = model.W * u_norm / f_norm
    G_rse = relative_squared_error(G, G_true, dim=(0, 1)).item()
    G_rses.append(G_rse)
    
    #nu = model.layers[0].nu() * u_norm
    nu = model.b * u_norm
    nu_rse = relative_squared_error(nu, nu_true, dim=0).item()
    nu_rses.append(nu_rse)

# plot relative squared errors over iterations
plt.figure(2)
plt.title('Train Relative Squared Errors', fontweight='bold')
plt.plot(range(len(rses)), rses, label='Train RSE', zorder=2)
plt.plot(range(len(G_rses)), G_rses, label='$\widehat{G}(x, y)$ RSE', zorder=1)
plt.plot(range(len(nu_rses)), nu_rses, label='$\widehat{β}(x)$ RSE', zorder=0)
plt.xlabel('Epochs')
plt.ylabel('Relative Train Error')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('train_errors', dpi=300)
plt.show()

# plot learned Green's function
plt.figure(3)
plt.title('Predicted Green\'s Kernel $\widehat{G}(x, y)$' + ' (RSE: {:.3E})'.format(G_rses[-1]), fontweight='bold')
plt.pcolormesh(X, Y, G.detach(), shading='auto')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('pred_green', dpi=300)
plt.show()

# plot learned bias term
plt.figure(4)
plt.title('Predicted Bias Term $\widehat{β}(x)$' + ' (RSE: {:.3E})'.format(nu_rses[-1]), fontweight='bold')
plt.plot(x_train, nu_true.detach(), label='True $β(x)$')
plt.plot(x_train, nu.detach(), '--', label='Predicted $\widehat{β}(x)$')
plt.xlabel('x')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('pred_nu', dpi=300)
plt.show()

# compute the relative train error on the solutions
us_train_hat = model(fs_train)
train_rse = relative_squared_error(us_train_hat, us_train).item()
print('Train Relative Squared Error: ' + str(train_rse))

# plot best and worst case train predictions
train_rses = relative_squared_error(us_train_hat, us_train, mean=False)
best_rse, best_ind = torch.min(train_rses, 0)
worst_rse, worst_ind = torch.max(train_rses, 0)

plt.figure(5)
plt.title('Best Train Prediction (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.plot(x_train, us_train[:, best_ind])
plt.plot(x_train, us_train_hat[:, best_ind].detach().numpy(), '--')
plt.xlabel('x')
plt.tight_layout()
plt.savefig('best_train', dpi=300)
plt.show()

plt.figure(6)
plt.title('Worst Train Prediction (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.plot(x_train, us_train[:, worst_ind])
plt.plot(x_train, us_train_hat[:, worst_ind].detach().numpy(), '--')
plt.xlabel('x')
plt.tight_layout()
plt.savefig('worst_train', dpi=300)
plt.show()


###############################################################################
#   Test Functional Neural Network on New Mesh Sizes
###############################################################################

# indicate that we have finished training our model
model.eval()

# number of test samples
num_test = 1000

mesh_sizes = np.linspace(100, 1000, 10).astype(int)

test_mesh_rses = []
for m in mesh_sizes:
    print(m)
    
    # test mesh
    m_test = m
    delta_test = L / (m_test-1)
    x_test = np.linspace(0, L, m_test)
    
    # generate test input functions as Brownian bridges
    fs_test = mag * math.sqrt(2) * (np.sin(math.pi*np.outer(x_test, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_test))
    
    # compute corresponding test solutions to PDE
    us_test = np.zeros([m_test, num_test])
    K_test = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(m_test-2, m_test-2))
    K_test = K_test / delta_test**2 - w**2 * np.eye(m_test-2)
    for k in range(num_test):
        #print(k)
        f_pert = np.copy(fs_test[1:m_test-1, k])
        f_pert[0] += a/delta_test**2
        f_pert[-1] += b/delta_test**2
        us_test[:, k] = np.concatenate(([a], np.linalg.solve(K_test, f_pert), [b]))
    
    # convert data to tensors
    x_test = torch.from_numpy(x_test)
    fs_test = torch.from_numpy(fs_test)
    us_test = torch.from_numpy(us_test)
    
    # create list for test meshes
    test_mesh = torch.meshgrid(x_test)
    
    # run the test input functions through the trained network
    interp_W = interpolate.RectBivariateSpline(x_train, x_train, model.W.detach())
    W_test = torch.from_numpy(interp_W(x_test, x_test))
    interp_b = interpolate.interp1d(x_train, model.b.detach())
    b_test = torch.from_numpy(interp_b(x_test))
    us_test_hat = torch.matmul(W_test, fs_test/f_norm) * delta_test + b_test[:, None]
    us_test_hat *= u_norm
    
    # compute the relative test error on the solutions
    test_rse = relative_squared_error(us_test_hat, us_test).item()
    test_mesh_rses.append(test_rse)

# plot test error of model on new test meshes
plt.figure(7)
plt.title('Adaptation to Test Meshes (Trained on m = {})'.format(m_train), y=1.05, fontweight='bold')
plt.plot(mesh_sizes, test_mesh_rses)
plt.xlabel('Mesh Size (m)')
plt.ylabel('Relative Test Error')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('mesh_adapt', dpi=300)
plt.show()

# plot best and worst case test predictions
test_rses = relative_squared_error(us_test_hat, us_test, mean=False)
best_rse, best_ind = torch.min(test_rses, 0)
worst_rse, worst_ind = torch.max(test_rses, 0)

plt.figure(8)
plt.title('Best Test Prediction (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.plot(x_test, us_test[:, best_ind])
plt.plot(x_test, us_test_hat[:, best_ind].detach().numpy(), '--')
plt.xlabel('x')
plt.tight_layout()
plt.savefig('best_test', dpi=300)
plt.show()

plt.figure(9)
plt.title('Worst Test Prediction (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.plot(x_test, us_test[:, worst_ind])
plt.plot(x_test, us_test_hat[:, worst_ind].detach().numpy(), '--')
plt.xlabel('x')
plt.tight_layout()
plt.savefig('worst_test', dpi=300)
plt.show()
