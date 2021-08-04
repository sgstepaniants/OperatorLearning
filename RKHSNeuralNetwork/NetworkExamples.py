import math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import cvxopt
from fft_conv import fft_conv
from scipy.optimize import least_squares
from scipy.linalg import svd

from abc import abstractmethod
import copy
import matplotlib.pyplot as plt

from RKHSNetworkModel import *
from ProdRKHSLayer import *
from BasisRKHSLayer import *
from ConvRKHSLayer import *


##############################################################################
#   Network Examples
##############################################################################

class FullyConnectedBasisNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        in_basis_shape = (10,) #(layer_meshes[0][0].shape[0] // 2 + 1,)
        out_basis_shape = (10,) #(layer_meshes[0][0].shape[0] // 2 + 1,)
        
        # create FFT basis layers
        fftlayer1 = BasisRKHSLayer(layer_meshes[1], layer_meshes[0],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        fftlayer2 = BasisRKHSLayer(layer_meshes[2], layer_meshes[1],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        fftlayer3 = BasisRKHSLayer(layer_meshes[3], layer_meshes[2],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        fftlayer4 = BasisRKHSLayer(layer_meshes[4], layer_meshes[3],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        fftlayer5 = BasisRKHSLayer(layer_meshes[5], layer_meshes[4],
                                    rfft, irfft,
                                    in_basis_shape=in_basis_shape, out_basis_shape=out_basis_shape,
                                    bias=True)
        
        # activation functions
        relu1 = nn.ReLU()
        relu2 = nn.ReLU()
        relu3 = nn.ReLU()
        relu4 = nn.ReLU()
        
        # list all of the layers in this network
        layers = nn.ModuleList([
            fftlayer1,
            relu1,
            fftlayer2,
            #relu2,
            #fftlayer3,
            #relu3,
            #fftlayer4,
            #relu4,
            #fftlayer5
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)


class FullyConnectedGaussianNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        sigma = 1e-3
        
        # create Gaussian RKHS layers
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        gaussianRKHS1 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        gaussianRKHS2 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[2], layer_meshes[1],
                                               weight_meshes[1], train_meshes[1],
                                               bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        gaussianRKHS3 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[3], layer_meshes[2],
                                               weight_meshes[2], train_meshes[2],
                                               bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        gaussianRKHS4 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[4], layer_meshes[3],
                                               weight_meshes[3], train_meshes[3],
                                               bias=True)
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        gaussianRKHS5 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[5], layer_meshes[4],
                                               weight_meshes[4], train_meshes[4],
                                               bias=True)
        
        # activation functions
        relu1 = nn.ReLU()
        relu2 = nn.ReLU()
        relu3 = nn.ReLU()
        relu4 = nn.ReLU()
        
        #relu1 = nn.LeakyReLU()
        #relu2 = nn.LeakyReLU()
        #relu3 = nn.LeakyReLU()
        #relu4 = nn.LeakyReLU()
        
        # list all of the layers in this network
        layers = nn.ModuleList([
            gaussianRKHS1,
            relu1,
            gaussianRKHS2,
            relu2,
            gaussianRKHS3#,
            #relu3,
            #gaussianRKHS4,
            #relu4,
            #gaussianRKHS5
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)

class FullyConnectedSobolevNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        in_basis_shape = (100,)
        out_basis_shape = (100,)
        
        # create Sobolev RKHS layers
        sobolevRKHS1 = SobolevRKHSLayer_1D(layer_meshes[1], layer_meshes[0],
                                       weight_meshes[0],
                                       in_basis_shape, out_basis_shape,
                                       bias=True)
        
        sobolevRKHS2 = SobolevRKHSLayer_1D(layer_meshes[1], layer_meshes[0],
                                       weight_meshes[0],
                                       in_basis_shape, out_basis_shape,
                                       bias=True)
        
        sobolevRKHS3 = SobolevRKHSLayer_1D(layer_meshes[1], layer_meshes[0],
                                       weight_meshes[0],
                                       in_basis_shape, out_basis_shape,
                                       bias=True)
        
        sobolevRKHS4 = SobolevRKHSLayer_1D(layer_meshes[1], layer_meshes[0],
                                       weight_meshes[0],
                                       in_basis_shape, out_basis_shape,
                                       bias=True)
        
        sobolevRKHS5 = SobolevRKHSLayer_1D(layer_meshes[1], layer_meshes[0],
                                       weight_meshes[0],
                                       in_basis_shape, out_basis_shape,
                                       bias=True)
        
        # activation functions
        #relu1 = nn.ReLU()
        #relu2 = nn.ReLU()
        #relu3 = nn.ReLU()
        #relu4 = nn.ReLU()
        
        relu1 = nn.LeakyReLU()
        relu2 = nn.LeakyReLU()
        relu3 = nn.LeakyReLU()
        relu4 = nn.LeakyReLU()
        
        # list all of the layers in this network
        layers = nn.ModuleList([
            sobolevRKHS1,
            relu1,
            sobolevRKHS2,
            relu2,
            sobolevRKHS3,
            relu3,
            sobolevRKHS4,
            relu4,
            sobolevRKHS5
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)

class FullyConnectedGaussianNetwork2D(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        sigma = 1e-3
        
        # create Gaussian RKHS layers
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        gaussianRKHS1 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        gaussianRKHS2 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[2], layer_meshes[1],
                                               weight_meshes[1], train_meshes[1],
                                               bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        gaussianRKHS3 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[3], layer_meshes[2],
                                               weight_meshes[2], train_meshes[2],
                                               bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        gaussianRKHS4 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[4], layer_meshes[3],
                                               weight_meshes[3], train_meshes[3],
                                               bias=True)
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        gaussianRKHS5 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[5], layer_meshes[4],
                                               weight_meshes[4], train_meshes[4],
                                               bias=True)
        
        # activation functions
        relu1 = nn.ReLU()
        relu2 = nn.ReLU()
        relu3 = nn.ReLU()
        relu4 = nn.ReLU()
        
        # list all of the layers in this network
        layers = nn.ModuleList([
            gaussianRKHS1#,
            #relu1,
            #gaussianRKHS2,
            #relu2,
            #gaussianRKHS3,
            #relu3,
            #gaussianRKHS4,
            #relu4,
            #gaussianRKHS5
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)


class GreensKernelNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        sigma = 1e-3
        
        # create Gaussian RKHS layers
        Sigma1 = torch.tensor([sigma])
        Sigma2 = torch.tensor([sigma])
        gaussianRKHS = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               bias=True)#,
                                               #symmetries=[(0, 1)])#,
                                               #causal_pair = (0, 1))
        
        # list all of the layers in this network
        layers = nn.ModuleList([gaussianRKHS])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)


class SobolevGreensFunction(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        in_basis_shape = (100,)
        out_basis_shape = (100,)
        
        # create Sobolev RKHS layers
        sobolevRKHS = SobolevRKHSLayer_1D(layer_meshes[1], layer_meshes[0],
                                       weight_meshes[0],
                                       in_basis_shape, out_basis_shape,
                                       bias=True)
        
        # list all of the layers in this network
        layers = nn.ModuleList([sobolevRKHS])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)

class HeatKernelNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        sigma = 1e-3
        
        # create Gaussian RKHS layers
        Sigma1 = sigma * torch.eye(2)
        Sigma2 = sigma * torch.eye(2)
        gaussianRKHS = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                                layer_meshes[1], layer_meshes[0],
                                                weight_meshes[0], train_meshes[0],
                                                bias=False,
                                                symmetries=[(0, 2)],
                                                causal_pair = (1, 3))
        
        # in_basis_shape = (20,20)
        # out_basis_shape = (20,20)
        # sobolevRKHS = SobolevRKHSLayer(layer_meshes[1], layer_meshes[0],
        #                                 in_basis_shape, out_basis_shape,
        #                                 bias=False,
        #                                 symmetries=[(0, 2)])#,
        #                                 #causal_pair = (1, 3))
        
        # list all of the layers in this network
        layers = nn.ModuleList([gaussianRKHS])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)

class FundamentalSolutionNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        sigma_x = 5e-2
        sigma_t = 1e-3
        
        # create Gaussian RKHS layers
        Sigma1 = torch.diag(torch.tensor([sigma_x, sigma_t]))
        Sigma2 = torch.tensor([sigma_x])
        gaussianRKHS = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                            layer_meshes[1], layer_meshes[0],
                                            weight_meshes[0], train_meshes[0])
        
        # list all of the layers in this network
        layers = nn.ModuleList([gaussianRKHS])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)

class BoundaryIntegralNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        sigma_x = 5e-4
        sigma_s = 5e-4
        
        # create Gaussian RKHS layers
        Sigma1 = torch.diag(torch.tensor([sigma_x, sigma_x]))
        Sigma2 = torch.tensor([sigma_s])
        gaussianRKHS = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0])
        
        # list all of the layers in this network
        layers = nn.ModuleList([gaussianRKHS])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)

class ConvKernelNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        sigma = 1e-3
        bounds = [math.inf]
        
        # create Gaussian RKHS layers
        Sigma = torch.tensor([sigma])
        gaussianConvRKHS = GaussianConvRKHSLayer(Sigma,
                                            layer_meshes[1], layer_meshes[0],
                                            bounds, bias=False)#,
                                            #symmetries=[(0, 1)])#,
                                            #causal_pair = (0, 1))
        
        # list all of the layers in this network
        layers = nn.ModuleList([gaussianConvRKHS])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)


class DiscriminatorNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        bounds = [0.3, 0.3]
        
        # create Gaussian RKHS layers
        Sigma = 1e-3 * torch.eye(2)
        gaussianRKHS1 = GaussianConvRKHSLayer(Sigma,
                                               layer_meshes[1], train_meshes[0],
                                               bounds, bias=True)
        Sigma = 1e-3 * torch.eye(2)
        gaussianRKHS2 = GaussianConvRKHSLayer(Sigma,
                                               layer_meshes[2], train_meshes[1],
                                               bounds, bias=True)
        Sigma1 = torch.tensor([1])
        Sigma2 = 1e-3 * torch.eye(2)
        gaussianRKHS3 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[3], layer_meshes[2],
                                               weight_meshes[2], train_meshes[2],
                                               bias=True)
        
        # activation functions
        relu1 = nn.LeakyReLU()
        relu2 = nn.LeakyReLU()
        sigmoid = nn.Sigmoid()
        
        #relu1 = nn.LeakyReLU()
        #relu2 = nn.LeakyReLU()
        
        # list all of the layers in this network
        layers = nn.ModuleList([
            gaussianRKHS1,
            relu1,
            gaussianRKHS2,
            relu2,
            gaussianRKHS3,
            sigmoid
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)


class GeneratorNetwork(FunctionalNetwork):
    def __init__(self, layer_meshes, weight_meshes, train_meshes):
        # important line that pytorch requires to initialize a network layer
        nn.Module.__init__(self)
        
        bounds = [0.3, 0.3]
        
        # create Gaussian RKHS layers
        Sigma1 = 1e-3 * torch.eye(2)
        Sigma2 = torch.tensor([1])
        gaussianRKHS1 = GaussianProdRKHSLayer(Sigma1, Sigma2,
                                               layer_meshes[1], layer_meshes[0],
                                               weight_meshes[0], train_meshes[0],
                                               bias=True)
        Sigma = 1e-3 * torch.eye(2)
        gaussianRKHS2 = GaussianConvRKHSLayer(Sigma,
                                               layer_meshes[2], train_meshes[1],
                                               bounds, bias=True)
        Sigma = 1e-3 * torch.eye(2)
        gaussianRKHS3 = GaussianConvRKHSLayer(Sigma,
                                               layer_meshes[3], train_meshes[2],
                                               bounds, bias=True)
        
        # activation functions
        relu1 = nn.LeakyReLU()
        relu2 = nn.LeakyReLU()
        tanh = nn.Tanh()
        
        # list all of the layers in this network
        layers = nn.ModuleList([
            gaussianRKHS1,
            relu1,
            gaussianRKHS2,
            relu2,
            gaussianRKHS3,
            tanh
            ])
        
        # important line that initializes the network with these attributes
        super().__init__(layers)
