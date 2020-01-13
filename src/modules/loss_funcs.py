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
    '''takes two tensors with shapes (B, 3) as input and calculates the angular error. Adds and multiplies denominator with 1e-7 and 1+1e-7 to avoid division with zero.
    '''
    def __init__(self):
        super(angle_loss,self).__init__()
        
    def forward(self, x, y):
        dot_prods = torch.sum(x*y, dim=-1)
        len_x = torch.sqrt(torch.sum(x*x, dim=-1))
        len_y = torch.sqrt(torch.sum(y*y, dim=-1))
        err = dot_prods/(len_x*len_y*(1+1e-7) + 1e-7)
        err = torch.acos(err)
        err = torch.mean(err)
        return err

class angle_squared_loss(torch.nn.Module):
    '''takes two tensors with shapes (B, 3) as input and calculates the angular error. Adds and multiplies denominator with 1e-7 and 1+1e-7 to avoid division with zero.
    '''    
    def __init__(self):
        super(angle_squared_loss, self).__init__()
    
    def forward(self, x, y, eps=1e-7):
        dot_prods = torch.sum(x*y, dim=-1)
        len_x = torch.sqrt(torch.sum(x*x, dim=-1))
        len_y = torch.sqrt(torch.sum(y*y, dim=-1))
        cos = dot_prods/(len_x*len_y*(1+eps) + eps)
        loss = torch.acos(cos)**2
        loss_mean = torch.mean(loss)

        return loss_mean
    
class angle_squared_loss_with_L2(torch.nn.Module):
    '''takes two tensors with shapes (B, 3) as input and calculates the angular error plus a bit of L2. Adds 1e-7 to squareroots to avoid inf gradients and multiplies denominator with 1+1e-7 to avoid division with zero. The L2 is useful to force unit vector predictions.

    Furthermore, each batch is checked for zero-vectors - these produce nan's. If any are found, they are set to equal y --> makes them not count.
    '''    
    def __init__(self):
        super(angle_squared_loss_with_L2, self).__init__()
    
    def forward(self, x, y, eps=1e-7, alpha=0.1):
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        batch_size = x.shape[0]
        zeros, ones = torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device)
        
        # * Check for zero-length vectors --> neutralize them by adding the target. Makes gradient 0 (which is what matters) and make other entries in batch count more. 
        len_x_checker = torch.sqrt(torch.sum(x*x, dim=-1))
        zero_vectors = torch.where(len_x_checker == 0.0, ones, zeros)
        n_zeros = torch.sum(zero_vectors)
        mult_factor = batch_size/(batch_size-n_zeros)
        x = x + zero_vectors.unsqueeze(1)*y
        
        # * Add eps to avoid infinite gradient.
        len_x = torch.sqrt(eps+torch.sum(x*x, dim=-1)) + zero_vectors
        len_y = torch.sqrt(torch.sum(y*y, dim=-1))
        dot_prods = torch.sum(x*y, dim=-1)
        cos = dot_prods/(len_x*len_y*(1+eps))
        loss = torch.acos(cos)**2
        loss_mean = torch.mean(loss)

        L2 = torch.mean((x-y)**2, dim=-1)
        L2_mean = torch.mean(L2)

        return mult_factor*(loss_mean + alpha*L2_mean)    

def get_loss_func(name):
    if name == 'L1': 
        return torch.nn.L1Loss()
    elif name == 'angle_loss':
        return angle_loss()
    elif name == 'Huber':
        return torch.nn.SmoothL1Loss()
    elif name == 'L2':
        return torch.nn.MSELoss()
    elif name == 'angle_squared_loss':
        return angle_squared_loss()
    elif name == 'angle_squared_loss_with_L2':
        return angle_squared_loss_with_L2()
    else:
        raise ValueError('Unknown loss function requested!')
