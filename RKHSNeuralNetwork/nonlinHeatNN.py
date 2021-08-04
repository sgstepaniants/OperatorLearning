import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

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

numeigs = 100
n = 10000
mx_train = 20
mt_train = 20
train_data = h5py.File('../generate/heateq_KLE{}_n={}_m={}x{}.h5'.format(numeigs, n, mx_train, mt_train), 'r')
fs_train = torch.from_numpy(train_data['input'][:])
us_train = torch.from_numpy(train_data['output'][:])

x_train = torch.from_numpy(train_data['x'][:]).squeeze(1)
t_train = torch.from_numpy(train_data['t'][:]).squeeze(1)

L = torch.max(x_train) - torch.min(x_train)

fs_train = torch.reshape(fs_train, (mx_train, mt_train, n))
us_train = torch.reshape(us_train, (mx_train, mt_train, n))

train_mesh = torch.meshgrid(x_train, t_train)
delta_train = (x_train[1] - x_train[0]) * (t_train[1] - t_train[0])

plt.figure(1)
plt.pcolormesh(train_mesh[0].detach().numpy(),
               train_mesh[1].detach().numpy(),
               fs_train[:, :, 0].detach().numpy(),
               shading='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()

plt.figure(2)
plt.pcolormesh(train_mesh[0].detach().numpy(),
               train_mesh[1].detach().numpy(),
               us_train[:, :, 0].detach().numpy(),
               shading='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()

# true Green's kernel
X1, T1, X2, T2 = torch.meshgrid(x_train, t_train, x_train, t_train)
Tdiff = T1 - T2
G_true = torch.zeros(mx_train, mt_train, mx_train, mt_train)
modes = 20
D = 1e-2
for k in range(1, modes):
    p = math.pi*k/L
    G_true += 2/L * torch.sin(p*X1) * torch.sin(p*X2) * torch.exp(-D*p**2*Tdiff)
G_true[Tdiff < 0] = 0
G_true = G_true.view(mx_train*mt_train, mx_train*mt_train).double()

num_train = 1000
fs_train = fs_train[:, :, :num_train]
us_train = us_train[:, :, :num_train]

sigma = 0
noise_train = sigma * torch.mean(torch.std(us_train, axis=(0, 1))) * torch.randn(mx_train, mt_train, num_train)
us_train += noise_train

# compute the signal to noise of the solutions on the train data
snr = 0

###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# normalize the data
f_norm = torch.mean(torch.sqrt(torch.sum(fs_train**2, dim=0)*delta_train))
u_norm = torch.mean(torch.sqrt(torch.sum(us_train**2, dim=0)*delta_train))

layer_meshes = 2*[train_mesh]
train_meshes = [train_mesh]
weight_meshes = [train_mesh]

# create functional neural network
model = HeatKernelNetwork(layer_meshes, weight_meshes, train_meshes)
#model = VanillaNetwork(train_mesh, train_mesh)
model.rescale(1/f_norm, u_norm)

# optimize using gradient descent
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1, amsgrad=True)

# regularization weights
lambdas = 1e-2 * torch.ones(1)

example_ind = 0

# iterate to find the optimal network parameters
epochs = 1000
batch_size = 100
freq = 50
rses = []
for epoch in range(epochs):
    perm = torch.randperm(num_train)
    batch_loss = 0
    batch_rse = 0
    for i in range(0, num_train, batch_size):
        inds = perm[i:i+batch_size]
        fs_batch = fs_train[:, :, inds]
        us_batch = us_train[:, :, inds]
        us_hat = torch.reshape(model(fs_batch), (mx_train, mt_train, batch_size))
        
        loss = mean_squared_error(us_hat, us_batch)
        loss += model.compute_regularization(lambdas)
        
        optimizer.zero_grad()
        # prevent gradient measurement to accumulate
        loss.backward()
        
        # calculate gradient in each iteration
        optimizer.step()
    
    if epoch % freq == 0:
        #torch.save(model.state_dict(), 'model_state.pth')
        #torch.save(model, 'model.pth')
        
        plt.figure(1)
        u = us_train[:, :, example_ind]
        u_hat = torch.reshape(model(fs_train[:, :, example_ind]), (mx_train, mt_train))
        diff = torch.abs(u - u_hat)
        diff = torch.reshape(diff, (mx_train, mt_train))
        plt.pcolormesh(train_mesh[0].detach().numpy(),
                       train_mesh[1].detach().numpy(),
                       diff.detach().numpy(),
                       shading='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.colorbar()
        plt.show()
    
    # average losses for this epoch
    us_hat = torch.reshape(model(fs_train), (mx_train, mt_train, num_train))
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
us_train_hat = torch.reshape(model(fs_train), (mx_train, mt_train, num_train))
train_rse = relative_squared_error(us_train_hat, us_train).item()
print('Train Relative Squared Error: ' + str(train_rse))

# plot best and worst case train predictions
train_rses = relative_squared_error(us_train_hat, us_train, mean=False)
best_rse, best_ind = torch.min(train_rses, 0)
worst_rse, worst_ind = torch.max(train_rses, 0)

best_u = us_train[:, :, best_ind]
best_u_hat = torch.reshape(model(fs_train[:, :, best_ind]), (mx_train, mt_train))
best_diff = torch.abs(best_u_hat - best_u)

plt.figure(5)
plt.title('Best Train Prediction (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               best_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig('best_train_pred', dpi=300)
plt.show()

plt.figure(6)
plt.title('Best Train True Evolution (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               best_u.detach(),
               shading='auto')
plt.colorbar()
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
plt.tight_layout()
plt.savefig('best_train_diff', dpi=300)
plt.show()


worst_u = us_train[:, :, worst_ind]
worst_u_hat = torch.reshape(model(fs_train[:, :, worst_ind]), (mx_train, mt_train))
worst_diff = torch.abs(worst_u_hat - worst_u)

plt.figure(8)
plt.title('Worst Train Prediction (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               worst_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig('worst_train_pred', dpi=300)
plt.show()

plt.figure(9)
plt.title('Worst Train True Evolution (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               worst_u.detach(),
               shading='auto')
plt.colorbar()
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
plt.tight_layout()
plt.savefig('worst_train_diff', dpi=300)
plt.show()

# plot the learned Green's function
G = model.layers[0].G()
#G = model.W
plt.figure(11)
plt.pcolormesh(G.detach().numpy())
plt.colorbar()
plt.xlabel('(y, s)')
plt.ylabel('(x, t)')
plt.tight_layout()
plt.savefig('full_G', dpi=300)
plt.show()

G_full = torch.reshape(G, (mx_train, mt_train, mx_train, mt_train))
plt.figure(12)
plt.pcolormesh(G_full[10, :, 10, :].detach().numpy())
plt.colorbar()
plt.xlabel('s')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('sec_G', dpi=300)
plt.show()

# plot the true Green's function
plt.figure(13)
plt.pcolormesh(G_true.detach().numpy())
plt.colorbar()

G_true_full = torch.reshape(G_true, (mx_train, mt_train, mx_train, mt_train))
plt.figure(14)
plt.pcolormesh(G_true_full[10, :, 10, :].detach().numpy())
plt.colorbar()


###############################################################################
#   Test Functional Neural Network on New Mesh Sizes
###############################################################################

# indicate that we have finished training our model
model.eval()

# number of test samples
num_test = 9000

numeigs = 100
n = 10000

mesh_sizes = [20, 30, 40, 50]
test_mesh_rses = []
for m in mesh_sizes:
    print(m)
    
    mx_test = m
    mt_test = m
    test_data = h5py.File('../generate/heateq_KLE{}_n={}_m={}x{}.h5'.format(numeigs, n, mx_test, mt_test), 'r')
    fs_test = torch.from_numpy(test_data['input'][:])
    us_test = torch.from_numpy(test_data['output'][:])
    
    x_test = torch.from_numpy(test_data['x'][:]).squeeze(1)
    t_test = torch.from_numpy(test_data['t'][:]).squeeze(1)
    
    fs_test = torch.reshape(fs_test, (mx_test, mt_test, n))
    us_test = torch.reshape(us_test, (mx_test, mt_test, n))
    
    test_mesh = torch.meshgrid(x_test, t_test)
    delta_test = (x_test[1] - x_test[0]) * (t_test[1] - t_test[0])
    
    fs_test = fs_test[:, :, :num_test]
    us_test = us_test[:, :, :num_test]
    
    noise_test = sigma * torch.mean(torch.std(us_test, axis=(0, 1))) * torch.randn(mx_test, mt_test, num_test)
    us_test += noise_test
    
    # create list for test meshes
    test_meshes = 2*[test_mesh]
    
    # change the mesh resolution at each layer of the network to the test set mesh
    model.set_resolution(test_meshes)
    
    # run the test input functions through the tested network
    us_test_hat = torch.reshape(model(fs_test), (mx_test, mt_test, num_test))
    
    # compute the relative test error on the solutions
    test_rse = relative_squared_error(us_test_hat, us_test).item()
    test_mesh_rses.append(test_rse)

# plot test error of model on new test meshes
plt.figure(7)
plt.title('Adaptation to Test Meshes (Trained on mxm = {}x{})'.format(mx_train, mt_train), y=1.05, fontweight='bold')
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
best_u_hat = torch.reshape(model(fs_test[:, :, best_ind]), (mx_test, mt_test))
best_diff = best_u_hat - best_u

plt.figure(13)
plt.title('Best Test Prediction (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               best_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('best_test_pred', dpi=300)
plt.show()

plt.figure(14)
plt.title('Best Test True Evolution (RSE: {:.3E})'.format(best_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               best_u.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
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
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('best_test_diff', dpi=300)
plt.show()


worst_u = us_test[:, :, worst_ind]
worst_u_hat = torch.reshape(model(fs_test[:, :, worst_ind]), (mx_test, mt_test))
worst_diff = worst_u_hat - worst_u

plt.figure(16)
plt.title('Worst Test Prediction (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               worst_u_hat.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('worst_test_pred', dpi=300)
plt.show()

plt.figure(17)
plt.title('Worst Test True Evolution (RSE: {:.3E})'.format(worst_rse), fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               worst_u.detach(),
               shading='auto')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
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
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('worst_test_diff', dpi=300)
plt.show()
