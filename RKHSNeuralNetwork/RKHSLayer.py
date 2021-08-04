import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from helper_functions import *


##############################################################################
#   Parent RKHS Layer Classes
##############################################################################

class RKHSWeight(ABC, nn.Module):
    @abstractmethod
    def set_resolution(self):
        pass
    
    @abstractmethod
    def make_symmetric(self):
        pass
    
    @abstractmethod
    def to_matrix(self):
        pass
    
    @abstractmethod
    def nullspan(self):
        pass
    
    @abstractmethod
    def rkhs_norm(self):
        pass


class RKHSBias(ABC, nn.Module):
    @abstractmethod
    def set_resolution(self):
        pass
    
    @abstractmethod
    def make_symmetric(self):
        pass
    
    @abstractmethod
    def to_vector(self):
        pass
    
    @abstractmethod
    def nullspan(self):
        pass
    
    @abstractmethod
    def rkhs_norm(self):
        pass


class RKHSLayer(nn.Module):
    def __init__(self):
        #super().__init__()
        
        # initialize RKHS weight and bias term
        self.weight = None
        self.bias = None
    
    def forward(self, f):
        output = self.weight(f)
        if self.bias is not None:
            output += self.bias.to_vector()[:, None]
        
        return output
    
    def set_resolution(self, x_mesh, y_mesh):
        self.weight.set_resolution(x_mesh, y_mesh)
        self.bias.set_resolution(x_mesh)
