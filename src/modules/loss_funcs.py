import torch
import numpy as np

# ======================================================================== 
# LOSS FUNCTIONS
# ======================================================================== 

class L2_like_loss(torch.nn.Module):
    '''takes [x, y, z, r_1, r_2, r_3, E] as input and calculates a customized loss
    '''
    def __init__(self):
        super(L2_like_loss,self).__init__()
        
    def forward(self,x,y):

        alpha, beta = 1.0, 1.0

        e_pos = (x[:, 0]-y[:, 0])**2 + (x[:, 1]-y[:, 1])**2 + (x[:, 2]-y[:, 2])**2
        
        e_dir = (1.0 - (x[:, 3]*y[:, 3] + x[:, 4]*y[:, 4] + x[:, 5]*y[:, 5])/(x[:, 3]**2 + x[:, 4]**2 + x[:, 5]**2)**0.5)**2

        e_E = (x[:, 6]-y[:, 6])**2

        totloss = torch.mean(e_pos + alpha*e_dir + beta*e_E)
        return totloss

class dir_reg_L1_like_loss(torch.nn.Module):
    '''takes [r_1, r_2, r_3] as input and calculates a customized loss
    '''
    def __init__(self):
        super(dir_reg_L1_like_loss,self).__init__()
        
    def forward(self, x, y):
        print(x.shape)
        print(y.shape)
        alpha, beta = 1.0, 1.0
        
        e_dir = torch.acos((x[:, 0]*y[:, 0] + x[:, 1]*y[:, 1] + x[:, 2]*y[:, 2])/((x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5))

        totloss = torch.mean(e_dir)
        return totloss

class angle_loss(torch.nn.Module):
    '''takes two tensors with shapes (B, 3) as input and calculates the angular error. Adds 1e-8 to denominator to avoid division with zero.
    '''
    def __init__(self):
        super(angle_loss,self).__init__()
        
    def forward(self, x, y):
        dot_prods = torch.sum(x*y, dim=1)
        len_x = torch.sqrt(torch.sum(x*x, dim=1))
        len_y = torch.sqrt(torch.sum(y*y, dim=1))
        err = dot_prods/(len_x*len_y + 1e-6)
        err = torch.acos(err)
        err = torch.mean(err)
        return err

def get_loss_func(name):
    if name == 'torch.nn.L1Loss': 
        return torch.nn.L1Loss()
    elif name == 'angle_loss':
        return angle_loss()
    else:
        raise ValueError('Unknown loss function requested!')
