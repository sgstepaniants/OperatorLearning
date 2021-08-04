import math
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from NetworkExamples import *
from VanillaNeuralNetwork import VanillaNetwork

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

# output and target must be of size m x n (grid points by samples)
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
w = 0 # for Poisson
#w = 20 # for Helmholtz

# Dirichlet (left and right endpoint) boundary conditions
a = 0
b = 0

# standard deviation of white noise added to ouputs (relative to their norm)
sigma = 0
#sigma = 0.3

# generate train input functions as Brownian bridges
mag = 1e1 # for Poisson
#mag = 1e4 # for Helmholtz
kl_modes = 100
freqs = np.arange(1, kl_modes+1) / L
fs_train = math.sqrt(2) * (np.sin(math.pi*np.outer(x_train, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_train))
fs_train += 1/2 * np.random.randn(num_train)[None, :]
fs_train *= mag

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


###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# normalize the data
f_norm = torch.mean(torch.sqrt(torch.sum(fs_train**2, dim=0)*delta_train))
u_norm = torch.mean(torch.sqrt(torch.sum(us_train**2, dim=0)*delta_train))

# create list for weight and train meshes
train_mesh = torch.meshgrid(x_train)
weight_mesh = torch.meshgrid(x_weight)

layer_meshes = 6*[train_mesh]
weight_meshes = 5*[weight_mesh]

# create functional neural network
#model = SobolevGreensFunction(layer_meshes, weight_meshes)
#model = VanillaNetwork(train_mesh, train_mesh, bias=True)
#model = ConvKernelNetwork(layer_meshes, weight_meshes)
#model = GreensKernelNetwork(layer_meshes, weight_meshes, weight_meshes)
model = FullyConnectedSobolevNetwork(layer_meshes, weight_meshes)
#model = FullyConnectedGaussianNetwork(layer_meshes, weight_meshes, layer_meshes[:-1])
model.rescale(1/f_norm, u_norm)

# optimize using gradient descent
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

# regularization weights
#lambdas = [1e-4]
#lambdas = [1e-7]
#lambdas = [0] # use 0 to see artifacts!

lambdas = 5*[0]

# training example to plot
example_ind = 0

# iterate to find the optimal network parameters
epochs = 1000
batch_size = 100
freq = 50
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
        loss += model.compute_regularization(lambdas, lambdas)
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
    G = model.layers[0].weight.to_matrix() * u_norm / f_norm
    G_rse = relative_squared_error(G, G_true, dim=(0, 1)).item()
    G_rses.append(G_rse)
    
    #nu = model.layers[0].bias.to_vector() * u_norm
    #nu_rse = relative_squared_error(nu, nu_true, dim=0).item()
    #nu_rses.append(nu_rse)

# plot relative squared errors over iterations
plt.figure(2)
#plt.title('Train Relative Squared Errors', fontweight='bold')
plt.plot(range(len(rses)), rses, label='Train RSE', zorder=2)
plt.plot(range(len(G_rses)), G_rses, label='$\widehat{G}$ RSE', zorder=1)
#plt.plot(range(len(nu_rses)), nu_rses, label='$\widehat{β}$ RSE', zorder=0)
plt.xlabel('Epochs')
plt.ylabel('Relative Train Error')
plt.yscale('log')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('train_errors', dpi=300)
plt.show()

# plot learned Green's function
plt.figure(3)
#plt.title('Predicted Green\'s Kernel $\widehat{G}(x, y)$' + ' (RSE: {:.3E})'.format(G_rses[-1]), fontweight='bold')
plt.pcolormesh(X, Y, G.detach().t(), shading='auto')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('pred_green', dpi=300)
plt.show()

# # plot learned bias term
# plt.figure(4)
# #plt.title('Predicted Bias Term $\widehat{β}(x)$' + ' (RSE: {:.3E})'.format(nu_rses[-1]), fontweight='bold')
# plt.plot(x_train, nu_true.detach(), label='True $β$')
# plt.plot(x_train, nu.detach(), '--', label='Predicted $\widehat{β}$')
# plt.xlabel('y')
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.savefig('pred_nu', dpi=300)
# plt.show()

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
    fs_test = math.sqrt(2) * (np.sin(math.pi*np.outer(x_test, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_test))
    fs_train += 1/2 * np.random.randn(num_train)[None, :]
    fs_test *= mag
    
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
    #noise_test = sigma * np.mean(np.std(us_test, axis=0)) * np.random.randn(m_test, num_test)
    #us_test += noise_test
    
    # convert data to tensors
    x_test = torch.from_numpy(x_test)
    fs_test = torch.from_numpy(fs_test)
    us_test = torch.from_numpy(us_test)
    
    # create list for test meshes
    test_mesh = torch.meshgrid(x_test)
    test_meshes = 2*[test_mesh]
    
    # change the mesh resolution at each layer of the network to the test set mesh
    model.set_resolution(test_meshes)
    
    # run the test input functions through the trained network
    us_test_hat = model(fs_test)
    
    # compute the relative test error on the solutions
    test_rse = relative_squared_error(us_test_hat, us_test).item()
    test_mesh_rses.append(test_rse)

# plot test error of model on new test meshes
plt.figure(7)
#plt.title('Adaptation to Test Meshes (Trained on m = {})'.format(m_train), y=1.05, fontweight='bold')
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
