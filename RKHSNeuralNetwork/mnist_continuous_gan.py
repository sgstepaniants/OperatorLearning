import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms

from NetworkExamples import DiscriminatorNetwork, GeneratorNetwork

##############################################################################
#   Helper Functions
##############################################################################

def noise(noise_mesh, samples=1):
    return torch.randn(noise_mesh[0].shape + tuple([samples]), dtype=torch.float64)

###############################################################################
#   Load MNIST Dataset
###############################################################################

data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)])
mnist_trainset = datasets.MNIST(root='./data', train=True,    
                           download=True, transform=data_transforms)

num_train = 1000
mnist_trainset = torch.utils.data.Subset(mnist_trainset, list(range(num_train)))
batch_size = 100
dataloader_mnist_train = t_data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

# obtain all the MNIST data in one large matrix
lst = []
for imgs, _ in dataloader_mnist_train:
    lst.append(imgs)
all_imgs = torch.cat(lst, dim=0)
height = all_imgs.shape[2]
width = all_imgs.shape[3]
all_imgs = torch.transpose(all_imgs.squeeze(1).flatten(start_dim=1), 0, 1)

###############################################################################
#   Train PyTorch Model For a Functional Neural Network
###############################################################################

# create image and weight meshes
image_mesh = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
weight_mesh = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))

# create generator noise meshes
noise_size = 50
noise_mesh = torch.meshgrid(torch.linspace(0, 1, noise_size))

discriminator_layer_meshes = [image_mesh, image_mesh, image_mesh, (torch.tensor([0]),)]
discriminator_weight_meshes = [weight_mesh, weight_mesh, (torch.tensor([0]),)]
generator_layer_meshes = [noise_mesh, image_mesh, image_mesh, image_mesh]
generator_weight_meshes = [weight_mesh, weight_mesh, weight_mesh]

# create functional neural networks
num_refs = 100
discriminator = DiscriminatorNetwork(discriminator_layer_meshes,
                                      discriminator_weight_meshes,
                                      discriminator_layer_meshes[:-1])


generator = GeneratorNetwork(generator_layer_meshes,
                              generator_weight_meshes,
                              generator_layer_meshes[:-1])

# loss functions
bceloss = nn.BCELoss()

# optimize using Adam
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-2, amsgrad=True)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=1e-2, amsgrad=True)

# regularization weights
#lambdas = 1e-6 * torch.ones(1)

example = noise(noise_mesh)

# iterate to find the optimal network parameters
epochs = 1000
discriminator_steps = 1
generator_steps = 1
disc_losses = []
gen_losses = []
for epoch in range(epochs):
    for imgs, _ in dataloader_mnist_train:
        imgs = torch.transpose(imgs.squeeze(1).flatten(start_dim=1), 0, 1)
        
        # train the discriminator to classify correctly
        disc_loss_ave = 0
        for s in range(discriminator_steps):
            print('disc')
            real_probs = discriminator(imgs)
            ones = Variable(torch.ones(1, batch_size, dtype=torch.float64))
            real_loss = bceloss(real_probs, ones)
            #print(real_probs.mean())
            
            fake_imgs = generator(noise(noise_mesh, batch_size)).detach()
            fake_probs = discriminator(fake_imgs)
            zeros = Variable(torch.zeros(1, batch_size, dtype=torch.float64))
            fake_loss = bceloss(fake_probs, zeros)
            #print(fake_probs.mean())
            #print(fake_probs)
            
            disc_loss = real_loss + fake_loss
            optimizer_disc.zero_grad()
            disc_loss.backward()
            
            optimizer_disc.step()
            disc_loss_ave += disc_loss
        
        # train the generator to trick the discriminator
        gen_loss_ave = 0
        for s in range(generator_steps):
            print('gen')
            fake_imgs = generator(noise(noise_mesh, batch_size))
            pred_probs = discriminator(fake_imgs)
            ones = Variable(torch.ones(1, batch_size, dtype=torch.float64))
            gen_loss = bceloss(pred_probs, ones)
            optimizer_gen.zero_grad()
            gen_loss.backward()
            
            optimizer_gen.step()
            gen_loss_ave += gen_loss
    
    disc_loss_ave /= (discriminator_steps * batch_size)
    disc_losses.append(disc_loss_ave.item())
    gen_loss_ave /= (generator_steps * batch_size)
    gen_losses.append(gen_loss_ave.item())
    
    if epoch % 1 == 0:
        plt.pcolormesh(generator(example).detach().numpy().reshape([height, width]))
        plt.colorbar()
        plt.show()
    
    print('Epoch {} Disc Loss {} Gen Loss {}'.format(epoch, disc_loss_ave, gen_loss_ave))

# plot loss over iterations
plt.figure(2)
plt.plot(range(len(disc_losses)), disc_losses)
plt.plot(range(len(gen_losses)), gen_losses)

plt.figure(3)
plt.pcolormesh(generator(noise(noise_mesh)).detach().numpy().reshape([height, width]))
plt.colorbar()

# indicate that we have finished training our model
#discriminator.eval()
#generator.eval()
