import math
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from diffeq.nonlin_biharmonic import nonlinBiharmonic

from RKHSNetworkModel import FullyConnectedGaussianNetwork

###############################################################################
#   Function Declarations
###############################################################################

# output and target must be of size m x n (grid points by samples)
def relative_squared_error(output, target):
    loss = torch.mean(torch.sum((output - target)**2, 0) / torch.sum(target**2, 0))
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
h_train = L / (m_train-1)
x_train = np.linspace(0, L, m_train)

# mesh for kernel weights
m_weight = 100
h_weight = L / (m_weight-1)
x_weight = np.linspace(0, L, m_weight)

# nonlinearities
eps = 0.4

# Dirichlet (left and right endpoint) boundary conditions
bc = np.zeros((2, 2))

# standard deviation of white noise added to ouputs (relative to their norm)
sigma = 0

# generate train input functions as Brownian bridges
kl_modes = 100
freqs = np.arange(1, kl_modes+1) / L
fs_train = math.sqrt(2) * (np.sin(math.pi*np.outer(x_train, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_train))

# compute corresponding train and test solutions to PDE
us_train = np.zeros([m_train, num_train])
for k in range(num_train):
    print(k)
    us_train[:, k] = nonlinBiharmonic(fs_train[:, k], x_train, bc, eps)
noise_train = sigma * np.mean(np.std(us_train, axis=0)) * np.random.randn(m_train, num_train)
us_train += noise_train

# compute the signal to noise of the solutions on the train data
snr = 0

###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# convert data to tensors
x_train = torch.from_numpy(x_train)
x_weight = torch.from_numpy(x_weight)

fs_train = torch.from_numpy(fs_train)
us_train = torch.from_numpy(us_train)
f_norm = torch.mean(torch.sqrt(torch.sum(fs_train**2, dim=0)*h_train))
u_norm = torch.mean(torch.sqrt(torch.sum(us_train**2, dim=0)*h_train))
fs_train /= f_norm
us_train /= u_norm

# W = torch.randn(m_train, m_train, dtype=torch.float64)
# W = W * math.sqrt(1 / h_train)
# fs = fs_train
# us1 = torch.matmul(W, fs) * h_train
# print(torch.mean(torch.norm(fs, dim=0))*h_train)
# print(torch.mean(torch.norm(us1, dim=0))*h_train)

# relu = torch.nn.ReLU()
# us1 = relu(us1)

# W = torch.randn(m_train, m_train, dtype=torch.float64)
# W = W * math.sqrt(2 / h_train)
# us2 = torch.matmul(W, us1) * h_train
# print(torch.mean(torch.norm(us1, dim=0))*h_train)
# print(torch.mean(torch.norm(us2, dim=0))*h_train)

# create list for weight and train meshes
train_mesh = torch.meshgrid(x_train)
weight_mesh = torch.meshgrid(x_weight)

train_meshes = [train_mesh, train_mesh, train_mesh, train_mesh, train_mesh, train_mesh]
weight_meshes = [weight_mesh, weight_mesh, weight_mesh]

# create functional neural network
num_refs = 100
model = FullyConnectedGaussianNetwork(train_meshes, weight_meshes, train_meshes[:-1], fs_train[:, :num_refs])
#model = FullyConnectedBasisNetwork(train_meshes, fs_train[:, :num_refs])

# prev_fs = fs_train[:, :num_refs]
# K1 = model.layers[0].K1
# K2 = model.layers[0].K2
# A = torch.matmul(K2.t(), K1)**2
# #A = torch.matmul(prev_fs.t(), torch.matmul(K2.t(), K1))**2
# X = A.numpy()
# #s0 = np.zeros(m_train)
# s0 = np.zeros(num_refs)

# def f(s):
#     return X.T.dot(s) - 1

# def jac(s):
#     return X.T

# res = least_squares(f, s0, jac, bounds=(m_train*[0], np.inf))
# #res = least_squares(f, s0, jac, bounds=(num_refs*[0], np.inf))
# s = torch.from_numpy(res.x)
# Sigma = torch.sqrt(s.repeat(m_train, 1) / m_train)

# Y = torch.sum(torch.matmul(Sigma**2, A), 0)
# print(torch.norm(Y - 1))

# W = torch.randn(m_train, m_train, dtype=torch.float64)
# W = Sigma * W
# W = W / h_train**3

# cs = torch.randn(m_train, num_refs, dtype=torch.float64)
# cs = Sigma * cs
# cs = cs / h_train**3

# fs = torch.randn(m_train, 1000, dtype=torch.float64)
# us = torch.matmul(W, torch.matmul(K2.t(), torch.matmul(K1, fs))) * h_train**3
# #us = torch.matmul(cs, torch.matmul(prev_fs.t(), torch.matmul(K2.t(), torch.matmul(K1, fs)))) * h_train**3
# print(torch.mean(fs**2, 1))
# print(torch.mean(us**2, 1))

# optimize using gradient descent
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)

# regularization weights
#lambdas = 1e-6 * torch.ones(1)

example_ind = 0

plt.figure(1)
plt.plot(x_train, us_train[:, example_ind])
plt.plot(x_train, model(fs_train[:, example_ind]).detach().numpy())
plt.show()

# iterate to find the optimal network parameters
epochs = 10000
batch_size = 1000
losses = []
for epoch in range(epochs):
    perm = torch.randperm(num_train)
    batch_loss = 0
    batch_rse = 0
    for i in range(0, num_train, batch_size):
        inds = perm[i:i+batch_size]
        fs_train_batch = fs_train[:, inds]
        us_train_batch = us_train[:, inds]
        us_train_hat = model(fs_train_batch)
        
        loss = h_train * mean_squared_error(us_train_hat, us_train_batch)
        #loss += model.compute_regularization(lambdas)
        rse = relative_squared_error(us_train_hat, us_train_batch)
        
        optimizer.zero_grad()
        # prevent gradient measurement to accumulate
        loss.backward()
        
        # calculate gradient in each iteration
        optimizer.step()
        
        batch_loss += loss
        batch_rse += rse
    
    if epoch % 5 == 0:
        plt.figure(1)
        plt.plot(x_train, us_train[:, example_ind])
        plt.plot(x_train, model(fs_train[:, example_ind]).detach().numpy())
        plt.show()
    
    # average losses for this epoch
    denom = math.ceil(num_train / batch_size)
    batch_loss /= denom
    batch_rse /= denom
    losses.append(batch_loss.item())
    print('Epoch {} Relative Squared Error {}'.format(epoch, batch_rse))

# plot loss over iterations
plt.figure(1)
plt.plot(range(len(losses)),losses)

# compute the relative train error on the solutions
us_train_hat = model(fs_train)
train_rse = relative_squared_error(us_train_hat, us_train).item()
print('Train Relative Squared Error: ' + str(train_rse))

plt.figure(2)
ind = 0
plt.plot(x_train, us_train[:, ind])
plt.plot(x_train, us_train_hat[:, ind].detach().numpy())

#s = 9
#basis_shapes = [(s,), (s,), (s,), (s,), (s,)]
#model.set_basis_shape(basis_shapes, basis_shapes)


###############################################################################
#   Test Functional Neural Network on New Mesh Sizes
###############################################################################

# indicate that we have finished training our model
model.eval()

# number of test samples
num_test = 1000

mesh_sizes = np.linspace(100, 2000, 20).astype(int)

test_mesh_losses = []
for m in mesh_sizes:
    print(m)
    
    # test mesh
    m_test = m
    h_test = L / (m_test-1)
    x_test = np.linspace(0, L, m_test)
    
    # generate test input functions as Brownian bridges
    fs_test = math.sqrt(2) * (np.sin(math.pi*np.outer(x_test, freqs)) / (math.pi*freqs)).dot(np.random.randn(kl_modes, num_test))
    
    # compute corresponding test solutions to PDE
    us_test = np.zeros([m_test, num_test])
    for k in range(num_test):
        #print(k)
        us_test[:, k] = nonlinBiharmonic(fs_test[:, k], x_test, bc, eps)
    noise_test = sigma * np.mean(np.std(us_test, axis=0)) * np.random.randn(m_test, num_test)
    us_test += noise_test
    
    # convert data to tensors
    x_test = torch.from_numpy(x_test)
    fs_test = torch.from_numpy(fs_test)
    us_test = torch.from_numpy(us_test)
    
    # create list for test meshes
    test_mesh = torch.meshgrid(x_test)
    test_meshes = [test_mesh, test_mesh, test_mesh, test_mesh, test_mesh, test_mesh]
    
    # change the mesh resolution at each layer of the network to the test set mesh
    model.set_resolution(test_meshes)
    
    # run the test input functions through the trained network
    us_test_hat = model(fs_test / f_norm) * u_norm
    
    # compute the relative test error on the solutions
    test_rse = relative_squared_error(us_test_hat, us_test).item()
    test_mesh_losses.append(test_rse)

plt.figure(3)
ind = 10
plt.plot(x_test, us_test[:, ind])
plt.plot(x_test, us_test_hat[:, ind].detach().numpy())

plt.figure(3)
plt.title('Adaptation to Test Meshes (Trained on m = ' + str(m_train) + ')')
plt.plot(mesh_sizes, test_mesh_losses)
plt.xlabel('Mesh Size (m)')
plt.ylabel('Relative Test Error')
