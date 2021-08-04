import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import constants

import sys
sys.path.append('../')
from generate.gaussianKLE import generateKLE

sys.path.append('../fplanck')
from fplanck import fokker_planck, potential_from_data, boundary

from NetworkExamples import FundamentalSolutionNetwork

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

def mean_squared_error(output, target, mean=True, dim=None):
    if dim is None:
        dim = tuple(range(output.ndim-1))
    
    loss = torch.sum((output - target)**2, dim)
    if mean:
        loss = torch.mean(loss)
    return loss

def time_pred_error(model, time_series_batch, lossfn, mean=True):
    if time_series_batch.ndim == 2:
        time_series_batch = time_series_batch.unsqueeze(-1)
    mx = time_series_batch.shape[0]
    
    num_series = time_series_batch.shape[2]
    loss = torch.zeros(num_series)
    for i in range(num_series):
        time_series = time_series_batch[:, :, i]
        tlen = time_series.shape[1]
        for t in range(1):
            rho_true = time_series[:, t:]
            rho_pred = model(time_series[:, t]).view(mx, -1)[:, :tlen-t]
            loss[i] = lossfn(rho_pred, rho_true)
    
    if mean:
        loss = torch.mean(loss)
    return loss


###############################################################################
#   Load Nonlinear Heat PDE Forcings and Solutions
###############################################################################

L = 6
mx_train = 20
delta_x_train = L / (mx_train - 1)
x_train = torch.linspace(0, L, mx_train)
x_train -= torch.mean(x_train)

T = 1
mt_train = 20
delta_t_train = T / (mt_train - 1)
t_train = torch.linspace(0, T, mt_train)

delta_train = delta_x_train * delta_t_train

# generate intitial distributions as inputs
num_train = 1000
mu = 0
l = 1
numeigs = 10
x_mesh = (x_train.detach().numpy(),)
init_train = torch.from_numpy(generateKLE(num_train, x_mesh, mu, l, numeigs))
init_train = torch.exp(init_train)
init_train /= (delta_x_train * torch.sum(init_train, 0)[None, :])

D = 0.6
drag = 1 / D
temp = 1 / constants.k

U = lambda x: np.power(x, 4)/4 - 3*np.power(x, 2)/2

sim = fokker_planck(grid=x_mesh, temperature=temp, drag=drag,
                    boundary=boundary.reflecting, potential=U)

rhos_train = torch.zeros((mx_train, mt_train, num_train))
for i in range(num_train):
    print(i)
    pdf = potential_from_data(x_train.detach().numpy(), init_train[:, i].detach().numpy())
    rhos_train[:, :, i] = torch.from_numpy(sim.propagate_interval(pdf, T, Nsteps=mt_train, normalize=False)[1]).t()

train_mesh = torch.meshgrid(x_train, t_train)

plt.figure(1)
plt.plot(x_train.detach(), init_train[:, 0].detach())
plt.xlabel('x')
plt.show()

plt.figure(2)
plt.pcolormesh(train_mesh[0].detach().numpy(),
               train_mesh[1].detach().numpy(),
               rhos_train[:, :, 0].detach().numpy(),
               shading='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()

# Green's Function used in solver
G_sim = torch.zeros(mx_train, mt_train, mx_train)
for i in range(mt_train):
    G_sim[:, i, :] = torch.matrix_exp(t_train[i] * torch.from_numpy(sim.master_matrix.toarray()))
G_sim = G_sim.view(mx_train*mt_train, mx_train) / delta_x_train

# true Green's Function
X, Y, Z = torch.meshgrid(x_train, t_train, x_train)
G_true = torch.zeros(mx_train, mt_train, mx_train)
bc = 'reflecting'
# bc = 'absorbing'
modes = min(mx_train, mt_train)
xl = x_train.min()
xr = x_train.max()
if bc == 'absorbing':
    for k in range(0, modes):
        lmbda = math.pi*k/(xr - xl)
        G_true += 2/(xr - xl) * torch.sin(lmbda*(X-xl)) * torch.sin(lmbda*(Z-xl)) * torch.exp(-D*lmbda**2*Y)
elif bc == 'reflecting':
    G_true += 1/(xr - xl)
    for k in range(1, modes):
        lmbda = math.pi*k/(xr - xl)
        G_true += 2/(xr - xl) * torch.cos(lmbda*(X-xl)) * torch.cos(lmbda*(Z-xl)) * torch.exp(-D*lmbda**2*Y)
G_true = G_true.view(mx_train*mt_train, mx_train)

# plot true Green's function
plt.figure(3)
plt.title('True Flattened Green\'s Function', fontweight='bold')
plt.pcolormesh(G_true.detach().t(), shading='auto')
plt.colorbar()
plt.xlabel('(x, t)')
plt.ylabel('y')
plt.tight_layout()
#plt.savefig('true_green', dpi=300)
plt.show()

# plot simulated Green's function
plt.figure(4)
plt.title('True Flattened Green\'s Function', fontweight='bold')
plt.pcolormesh(G_sim.detach().t(), shading='auto')
plt.colorbar()
plt.xlabel('(x, t)')
plt.ylabel('y')
plt.tight_layout()
##plt.savefig('sim_green', dpi=300)
plt.show()

# true bias term
#nu_true = torch.zeros(mx_train, mt_train)

# plot true bias term
# plt.figure(3)
# plt.title('True Bias Term', fontweight='bold')
# plt.pcolormesh(X[:, :, 0], Y[:, :, 0], nu_true.detach(), shading='auto')
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('t')
# plt.tight_layout()
# plt.savefig('true_nu', dpi=300)
# plt.show()

# sigma = 0
# noise_train = sigma * torch.mean(torch.std(rhos_train, axis=(0, 1))) * torch.randn(mx_train, mt_train, num_train)
# rhos_train += noise_train

# # compute the signal to noise of the solutions on the train data
# snr = 0


###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# normalize the data
init_norm = torch.mean(torch.sqrt(torch.sum(rhos_train[:, 0, :]**2, dim=0)*delta_x_train))
rho_norm = torch.mean(torch.sqrt(torch.sum(rhos_train**2, dim=(0, 1))*delta_train))

layer_meshes = [(x_train,), train_mesh]
train_meshes = [(x_train,)]
weight_meshes = [train_mesh]

# create functional neural network
model = FundamentalSolutionNetwork(layer_meshes, weight_meshes, train_meshes)
model.rescale(1/init_norm, rho_norm)

# optimize using gradient descent
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, amsgrad=True)

# regularization weights
lambdas = [0]

example_ind = 0

# iterate to find the optimal network parameters
epochs = 1000
batch_size = 100
freq = 100
rses = []
G_rses = []
#nu_mses = []
for epoch in range(epochs):
    perm = torch.randperm(num_train)
    batch_loss = 0
    batch_rse = 0
    for i in range(0, num_train, batch_size):
        inds = perm[i:i+batch_size]
        rhos_batch = rhos_train[:, :, inds]
        
        lossfn = lambda output, target: mean_squared_error(output, target, dim=(0, 1))
        loss = time_pred_error(model, rhos_batch, lossfn)
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
        u = rhos_train[:, :, example_ind]
        u_hat = torch.reshape(model(u[:, 0]), (mx_train, mt_train))
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
    lossfn = lambda output, target: relative_squared_error(output, target, dim=(0, 1))
    rse = time_pred_error(model, rhos_train, lossfn).item()
    print('Epoch {} Relative Squared Error {}'.format(epoch, rse))
    rses.append(rse)
    
    # compute rse between true and predicted Green's Function and bias
    G = model.layers[0].G() * model.out_scale * model.in_scale
    G_rse = relative_squared_error(G, G_true, dim=(0, 1)).item()
    G_rses.append(G_rse)
    
    #nu = torch.reshape(model.layers[0].nu() * model.out_scale, (mx_train, mt_train))
    #nu_mse = mean_squared_error(nu, nu_true, dim=(0, 1)).item()
    #nu_mses.append(nu_mse)

# load existing model if necessary
#model = torch.load('model.pth', map_location=torch.device('cpu'))
#rses = np.load('rses.npy')
#G_true_rses = np.load('G_true_rses.npy')
#G_sim_rses = np.load('G_sim_rses.npy')
#G = model.layers[0].G() * model.out_scale * model.in_scale
#nu = torch.reshape(model.layers[0].nu() * model.out_scale, (mx_train, mt_train))

# plot relative squared errors over iterations
plt.figure(2)
plt.title('Train Errors', fontweight='bold')
plt.plot(range(len(rses)), rses, label='Train RSE', zorder=2)
#plt.plot(range(len(G_true_rses)), G_true_rses, label='$\widehat{G}(x, y)$ RSE', zorder=1)
#plt.plot(range(len(G_sim_rses)), G_sim_rses, label='$\widehat{G}(x, y)$ RSE', zorder=0)
#plt.plot(range(len(nu_mses)), nu_mses, label='$\widehat{β}(x)$ MSE', zorder=0)
plt.xlabel('Epochs')
plt.ylabel('Relative Train Error')
plt.yscale('log')
#plt.legend()
plt.tight_layout()
plt.savefig('train_errors', dpi=300)
plt.show()

im_min = min(G.min(), G.min())
im_max = min(G.max(), G.max())

# plot learned Green's function
plt.figure(3)
plt.title('Predicted Green\'s Function $\widehat{G}(x, y)$' + ' (RSE: {:.3E})'.format(relative_squared_error(G, G_sim, dim=(0, 1)).item()), fontweight='bold')
plt.pcolormesh(G.detach().t(), shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('(x, t)')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('pred_green', dpi=300)
plt.show()

# plot true Green's function
plt.figure(4)
plt.title('True Green\'s Function', fontweight='bold')
plt.pcolormesh(G_sim.detach().t(), shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('(x, t)')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('sim_green', dpi=300)
plt.show()

# # plot learned bias term
# plt.figure(4)
# plt.title('Predicted Bias Term $\widehat{β}(x)$' + ' (MSE: {:.3E})'.format(nu_mses[-1]), fontweight='bold')
# plt.pcolormesh(nu_true.detach(), shading='auto')
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('t')
# plt.tight_layout()
# #plt.savefig('pred_nu', dpi=300)
# plt.show()

# compute the relative train error on the solutions
lossfn = lambda output, target: relative_squared_error(output, target, dim=(0, 1))
train_rse = time_pred_error(model, rhos_train, lossfn).item()
print('Train Relative Squared Error: ' + str(train_rse))

# plot best and worst case train predictions
lossfn = lambda output, target: relative_squared_error(output, target, dim=(0, 1))
train_rses = time_pred_error(model, rhos_train, lossfn, mean=False)
best_rse, best_ind = torch.min(train_rses, 0)
worst_rse, worst_ind = torch.max(train_rses, 0)

best_rho = rhos_train[:, :, best_ind]
best_rho_hat = torch.reshape(model(best_rho[:, 0]), (mx_train, mt_train))
best_diff = torch.abs(best_rho_hat - best_rho)
im_min = min(best_rho.min().item(), best_rho_hat.min().item())
im_max = min(best_rho.max().item(), best_rho_hat.max().item())

plt.figure(5)
plt.title('Best Train Prediction', fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               best_rho_hat.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('best_train_pred', dpi=300)
plt.show()

plt.figure(6)
plt.title('Best Train True Evolution', fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               best_rho.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
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
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('best_train_diff', dpi=300)
plt.show()


worst_rho = rhos_train[:, :, worst_ind]
worst_rho_hat = torch.reshape(model(worst_rho[:, 0]), (mx_train, mt_train))
worst_diff = torch.abs(worst_rho_hat - worst_rho)
im_min = min(worst_rho.min().item(), worst_rho_hat.min().item())
im_max = min(worst_rho.max().item(), worst_rho_hat.max().item())

plt.figure(8)
plt.title('Worst Train Prediction', fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               worst_rho_hat.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('worst_train_pred', dpi=300)
plt.show()

plt.figure(9)
plt.title('Worst Train True Evolution', fontweight='bold')
plt.pcolormesh(train_mesh[0].detach(),
               train_mesh[1].detach(),
               worst_rho.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
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
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('worst_train_diff', dpi=300)
plt.show()

# plot the learned Green's function
G_full = torch.reshape(G, (mx_train, mt_train, mx_train))
t_ind = 0
plt.figure(11)
plt.pcolormesh(G_full[:, t_ind, :].detach(),
               shading='auto')
plt.colorbar()
plt.show()

G_true_full = torch.reshape(G_true, (mx_train, mt_train, mx_train))
plt.figure(12)
plt.pcolormesh(G_true_full[:, t_ind, :].detach(),
               shading='auto')
plt.colorbar()
plt.show()


plt.pcolormesh(model.layers[0].G().detach(),
               shading='auto')
plt.colorbar()
plt.show()

plt.pcolormesh(G_full[:, :, 10].detach())
plt.colorbar()
plt.show()

###############################################################################
#   Test Functional Neural Network on New Mesh Sizes
###############################################################################

# indicate that we have finished training our model
model.eval()

# number of test samples
num_test = 1000

mesh_sizes = [60, 70, 80, 90, 100]
test_mesh_rses = []
for m in mesh_sizes:
    print(m)
    
    mx_test = m
    mt_test = m
    delta_x_test = L / (mx_test - 1)
    x_test = torch.linspace(0, L, mx_test)
    x_test -= torch.mean(x_test)
    
    delta_t_test = T / (mt_test - 1)
    t_test = torch.linspace(0, T, mt_test)
    
    delta_test = delta_x_test * delta_t_test
    
    # generate intitial distributions as inputs
    x_mesh = (x_test.detach().numpy(),)
    init_test = torch.from_numpy(generateKLE(num_test, x_mesh, mu, l, m//2))
    init_test = torch.exp(init_test)
    init_test /= (delta_x_test * torch.sum(init_test, 0)[None, :])
    
    sim = fokker_planck(grid=x_mesh, temperature=temp, drag=drag,
                        boundary=boundary.reflecting, potential=U)
    
    rhos_test = torch.zeros((mx_test, mt_test, num_test))
    for i in range(num_test):
        pdf = potential_from_data(x_test.detach().numpy(), init_test[:, i].detach().numpy())
        rhos_test[:, :, i] = torch.from_numpy(sim.propagate_interval(pdf, T, Nsteps=mt_test, normalize=False)[1]).t()
    
    #init_norm = torch.mean(torch.sqrt(torch.sum(rhos_test[:, 0, :]**2, dim=0)*delta_x_test))
    #rho_norm = torch.mean(torch.sqrt(torch.sum(rhos_test**2, dim=(0, 1))*delta_test))
    
    test_mesh = torch.meshgrid(x_test, t_test)
    
    # create list for test meshes
    test_meshes = [(x_test,), test_mesh]
    
    # change the mesh resolution at each layer of the network to the test set mesh
    model.set_resolution(test_meshes)
    
    # compute the relative test error on the solutions
    lossfn = lambda output, target: relative_squared_error(output, target, dim=(0, 1))
    test_rse = time_pred_error(model, rhos_test, lossfn).item()
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
lossfn = lambda output, target: relative_squared_error(output, target, dim=(0, 1))
test_rses = time_pred_error(model, rhos_test, lossfn, mean=False)
best_rse, best_ind = torch.min(test_rses, 0)
worst_rse, worst_ind = torch.max(test_rses, 0)

best_rho = rhos_test[:, :, best_ind]
best_rho_hat = torch.reshape(model(best_rho[:, 0]), (mx_test, mt_test))
best_diff = torch.abs(best_rho_hat - best_rho)
im_min = min(best_rho.min().item(), best_rho_hat.min().item())
im_max = min(best_rho.max().item(), best_rho_hat.max().item())

plt.figure(5)
plt.title('Best Test Prediction', fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               best_rho_hat.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('best_test_pred', dpi=300)
plt.show()

plt.figure(6)
plt.title('Best Test True Evolution', fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               best_rho.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('best_test_true', dpi=300)
plt.show()

plt.figure(7)
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


worst_rho = rhos_test[:, :, worst_ind]
worst_rho_hat = torch.reshape(model(worst_rho[:, 0]), (mx_test, mt_test))
worst_diff = torch.abs(worst_rho_hat - worst_rho)
im_min = min(worst_rho.min().item(), worst_rho_hat.min().item())
im_max = min(worst_rho.max().item(), worst_rho_hat.max().item())

plt.figure(8)
plt.title('Worst Test Prediction', fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               worst_rho_hat.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('worst_test_pred', dpi=300)
plt.show()

plt.figure(9)
plt.title('Worst Test True Evolution', fontweight='bold')
plt.pcolormesh(test_mesh[0].detach(),
               test_mesh[1].detach(),
               worst_rho.detach(),
               shading='auto', vmin=im_min, vmax=im_max)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.tight_layout()
plt.savefig('worst_test_true', dpi=300)
plt.show()

plt.figure(10)
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
