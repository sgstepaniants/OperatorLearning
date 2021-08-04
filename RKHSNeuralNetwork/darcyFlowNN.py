import math
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import h5py

from RKHSNetworkModel import FullyConnectedGaussianNetwork2D

###############################################################################
#   Function Declarations
###############################################################################

# output and target must be of size m x n (grid points by samples)
def relative_squared_error(output, target):
    loss = torch.mean(torch.sum((output - target)**2, dim=tuple(range(output.ndim-1))) / torch.sum(target**2, dim=tuple(range(output.ndim-1))))
    return loss

def mean_squared_error(output, target):
    loss = torch.mean(torch.sum((output - target)**2, dim=tuple(range(output.ndim-1))))
    return loss

###############################################################################
# Permeability Input Fields and Pressure, Flux Output Fields from Darcy's Law
###############################################################################

field_num = 1

#train_data = h5py.File('datasets/32x32/kle100_lhs10000_train.hdf5', 'r')
#train_data = h5py.File('datasets/64x64/kle512_lhs10000_train.hdf5', 'r')
train_data = h5py.File('datasets/32x32/kle512_lhs1024_test.hdf5', 'r')
fs_train = torch.from_numpy(train_data['input'][:]).squeeze(1).permute(1, 2, 0)
us_train = torch.from_numpy(train_data['output'][:])[:, field_num, :, :].permute(1, 2, 0)

plt.figure(1)
plt.pcolormesh(fs_train[:, :, 0].detach().numpy())
plt.colorbar()

plt.figure(2)
plt.pcolormesh(us_train[:, :, 0].detach().numpy())
plt.colorbar()

mx_train = fs_train.shape[0]
my_train = fs_train.shape[1]

Lx = 1
Ly = 1
x_train = torch.linspace(0, Lx, mx_train)
y_train = torch.linspace(0, Ly, my_train)

train_mesh = torch.meshgrid(x_train, y_train)
delta_train = (x_train[1] - x_train[0]) * (y_train[1] - y_train[0])

num_train = 1000
fs_train = fs_train[:, :, :num_train]
us_train = us_train[:, :, :num_train]

sigma = 0
noise_train = sigma * torch.mean(torch.std(us_train, axis=(0, 1))) * torch.randn(mx_train, my_train, num_train)
us_train += noise_train

# compute the signal to noise of the solutions on the train data
snr = 0

###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# normalize the data
f_norms_train = torch.sqrt(torch.sum(fs_train**2, dim=(0, 1))*delta_train)
u_norms_train = torch.sqrt(torch.sum(us_train**2, dim=(0, 1))*delta_train)
f_norm_train = torch.mean(f_norms_train)
u_norm_train = torch.mean(u_norms_train)

fs_train /= f_norm_train
us_train /= u_norm_train

layer_meshes = 6*[train_mesh]
train_meshes = 5*[train_mesh]
weight_meshes = 5*[train_mesh]

# create functional neural network
model = FullyConnectedGaussianNetwork2D(layer_meshes, weight_meshes, train_meshes)

# optimize using gradient descent
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, amsgrad=True)

# regularization weights
#lambdas = 1e-6 * torch.ones(1)

example_ind = 0

# iterate to find the optimal network parameters
epochs = 10000
batch_size = 100
freq = 1
losses = []
for epoch in range(epochs):
    perm = torch.randperm(num_train)
    batch_loss = 0
    batch_rse = 0
    for i in range(0, num_train, batch_size):
        inds = perm[i:i+batch_size]
        fs_train_batch = fs_train[:, :, inds]
        us_train_batch = us_train[:, :, inds]
        us_train_hat = torch.reshape(model(fs_train_batch), (mx_train, my_train, batch_size))
        
        loss = mean_squared_error(us_train_hat, us_train_batch)
        #loss += model.compute_regularization(lambdas)
        rse = relative_squared_error(us_train_hat, us_train_batch)
        
        optimizer.zero_grad()
        # prevent gradient measurement to accumulate
        loss.backward()
        
        # calculate gradient in each iteration
        optimizer.step()
        
        batch_loss += loss
        batch_rse += rse
    
    if epoch % freq == 0:
        #torch.save(model.state_dict(), 'model_state.pth')
        #torch.save(model, 'model.pth')
        
        plt.figure(1)
        u = us_train[:, :, example_ind]
        u_hat = torch.reshape(model(fs_train[:, :, example_ind]), (mx_train, my_train))
        diff = u - u_hat
        diff = torch.reshape(diff, (mx_train, my_train))
        plt.pcolormesh(train_mesh[0], train_mesh[1], diff.detach().numpy(), shading='auto')
        plt.colorbar()
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
us_train_hat = torch.reshape(model(fs_train), (mx_train, my_train, num_train))
train_rse = relative_squared_error(us_train_hat, us_train).item()
print('Train Relative Squared Error: ' + str(train_rse))

ind = 200
u = us_train[:, :, ind]
u_hat = torch.reshape(model(fs_train[:, :, ind]), (mx_train, my_train))
diff = u_hat - u
rse = relative_squared_error(us_train_hat.unsqueeze(-1), us_train.unsqueeze(-1)).item()
print(rse)

plt.figure(2)
plt.title('Predicted Pressure Field')
plt.pcolormesh(train_mesh[0], train_mesh[1], u_hat.detach().numpy(), shading='auto')
plt.colorbar()
plt.show()

plt.figure(3)
plt.title('True Pressure Field')
plt.pcolormesh(train_mesh[0], train_mesh[1], u.detach().numpy(), shading='auto')
plt.colorbar()
plt.show()

plt.figure(4)
plt.title('Difference')
plt.pcolormesh(train_mesh[0], train_mesh[1], diff.detach().numpy(), shading='auto')
plt.colorbar()
plt.show()


###############################################################################
#   Test Functional Neural Network on New Mesh Sizes
###############################################################################

#model = FullyConnectedGaussianNetwork2D(layer_meshes, weight_meshes, train_meshes, fs_train[:, :num_refs])
#model.load_state_dict(torch.load('model_state.pth'), strict=False)
model = torch.load('model.pth', map_location=torch.device('cpu'))

# indicate that we have finished training our model
model.eval()

#test_data = h5py.File('datasets/32x32/kle100_lhs1000_test.hdf5', 'r')
test_data = h5py.File('datasets/64x64/kle512_lhs1000_test.hdf5', 'r')
#test_data = h5py.File('datasets/64x64/kle128_lhs1024_test.hdf5', 'r')
fs_test = torch.from_numpy(test_data['input'][:]).squeeze(1).permute(1, 2, 0)
us_test = torch.from_numpy(test_data['output'][:])[:, field_num, :, :].permute(1, 2, 0)

mx_test = fs_test.shape[0]
my_test = fs_test.shape[1]

x_test = torch.linspace(0, Lx, mx_test)
y_test = torch.linspace(0, Ly, my_test)

test_mesh = torch.meshgrid(x_test, y_test)
test_meshes = 6*[test_mesh]
delta_test = (x_test[1] - x_test[0]) * (y_test[1] - y_test[0])

model.set_resolution(test_meshes)

num_test = 1000
fs_test = fs_test[:, :, :num_test]
us_test = us_test[:, :, :num_test]

noise_test = sigma * torch.mean(torch.std(us_test, axis=(0, 1))) * torch.randn(mx_test, my_test, num_test)
us_test += noise_test

f_norms_test = torch.sqrt(torch.sum(fs_test**2, dim=(0, 1))*delta_test)
u_norms_test = torch.sqrt(torch.sum(us_test**2, dim=(0, 1))*delta_test)
f_norm_test = torch.mean(f_norms_test)
u_norm_test = torch.mean(u_norms_test)

# compute the relative train error on the solutions
us_test_hat = u_norm_test * torch.reshape(model(fs_test / f_norm_test), (mx_test, my_test, num_test))
test_rse = relative_squared_error(us_test_hat, us_test).item()
print('Test Relative Squared Error: ' + str(test_rse))

ind = 10
u = us_test[:, :, ind]
u_hat = u_norm_test * torch.reshape(model(fs_test[:, :, ind] / f_norm_test), (mx_test, my_test))
diff = u_hat - u
rse = relative_squared_error(u_hat.unsqueeze(-1), u.unsqueeze(-1)).item()
print(rse)

plt.figure(5)
plt.title('Predicted Field')
plt.pcolormesh(test_mesh[0], test_mesh[1], u_hat.detach().numpy(), shading='auto')
plt.colorbar()
plt.show()

plt.figure(6)
plt.title('True Field')
plt.pcolormesh(test_mesh[0], test_mesh[1], u.detach().numpy(), shading='auto')
plt.colorbar()
plt.show()

plt.figure(7)
plt.title('Difference')
plt.pcolormesh(test_mesh[0], test_mesh[1], diff.detach().numpy(), shading='auto')
plt.colorbar()
plt.show()
