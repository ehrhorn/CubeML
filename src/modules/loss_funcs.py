import torch
import numpy as np

#* ======================================================================== 
#* LOSS FUNCTIONS
#* ======================================================================== 

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

    Furthermore, the gradients of x are clipped --> we want to avoid nans.
    '''    
    def __init__(self, eps=1e-7, alpha=0.1, clip_val=10.0):
        super(angle_squared_loss_with_L2, self).__init__()
        self._clip_val = clip_val
        self._eps = eps
        self._alpha = alpha

    def _clip_x(self, grad):
        return torch.clamp(grad, min=-self._clip_val, max=self._clip_val)

    def forward(self, x, y):
        # * Register a hook, which clips the gradient values to try to avoid producing nans. 
        # * This can only be done during forward pass - therefore try, except.
        try:
            x.register_hook(self._clip_x)
        except RuntimeError:
            pass

        # * Unpack into targets and weights
        targets, weights = y

        # * Add eps to avoid infinite gradient if x is the zero-vector
        len_x = torch.sqrt(self._eps**2+torch.sum(x*x, dim=-1))
        len_y = torch.sqrt(torch.sum(targets*targets, dim=-1))
        dot_prods = torch.sum(x*targets, dim=-1)
        
        # * Multiply by a number slightly larger than one to avoid the derivative of acos becoming infinite.
        cos = dot_prods/(len_x*len_y*(1+self._eps))
        loss = torch.acos(cos)**2

        # * Now weigh it and mean over the batch
        loss_weighted = weights*loss
        loss_mean = torch.mean(loss_weighted)

        # * Calculate L2 loss aswell - same way
        L2 = torch.mean((x-targets)*(x-targets), dim=-1)

        L2_weighted = L2*weights
        L2_mean = torch.mean(L2)

        return (1-self._alpha)*loss_mean + self._alpha*L2_mean 

class L2(torch.nn.Module):
 
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y):
        """Computes L2-loss with weights
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, F), where F is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, F) and weights of shape (F)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
        
        # * Unpack into targets and weights
        targets, weights = y

        # * Calculate actual loss
        L2 = torch.mean((x-targets)*(x-targets), dim=-1)

        # * Now weigh it
        L2_weighted = L2*weights
        
        # * Mean over the batch
        L2_mean = torch.mean(L2_weighted)

        return L2_mean

class logcosh(torch.nn.Module):
    def __init__(self):
        super(logcosh, self).__init__()

    def forward(self, x, y, predict=False):
        """Computes logcosh-loss with weights
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, F), where F is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, F) and weights of shape (F)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
          
        # * Unpack into targets and weights
        targets, weights = y

        # * Calculate actual loss
        ave_logcosh = torch.mean(torch.log(torch.cosh(x-targets)), dim=-1)

        # * Now weigh it
        loss_weighted = weights*ave_logcosh
        
        # * Mean over the batch
        if not predict:
            loss = torch.mean(loss_weighted)
        else:
            loss = loss_weighted

        return loss

class logcosh_full_weighted(torch.nn.Module):
    def __init__(self, weights=[], device=None):
        super(logcosh_full_weighted, self).__init__()

        if not device:
            raise ValueError('A device must be supplied to the loss function'
            'logcosh_full_weigthed')
        weights_normed = np.array(weights)/np.sum(weights)
        self._weights = torch.tensor(weights_normed, device=device)
    
    def forward(self, x, y, predict=False):
        """Computes logcosh-loss with weights
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, F), where F is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, F) and weights of shape (F)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
          
        # * Unpack into targets and weights
        targets, weights = y

        # * Calculate actual loss
        logcosh = torch.log(torch.cosh(x-targets))

        # * weight to control contributions

        # * Now weigh it
        loss_weighted = torch.sum(self._weights*logcosh, dim=-1)
        
        # * Mean over the batch
        if not predict:
            loss = torch.mean(loss_weighted)
        else:
            loss = loss_weighted

        return loss

class logscore(torch.nn.Module):
    def __init__(self, weights=[], device=None):
        super(logscore, self).__init__()

        if not device:
            raise ValueError('A device must be supplied to the loss function'
            'logscore')
        weights_normed = np.array(weights)/np.sum(weights)
        self._weights = torch.tensor(weights_normed, device=device)

    def _normal(self, x, mean, sigma):
        const = 1.0/(sigma*torch.sqrt(2*3.14159))
        exponent = -0.5*((x-mean)/sigma)*((x-mean)/sigma)

        return const*torch.exp(exponent)
    
    def forward(self, x, y, predict=False):
        """Computes logcosh-loss with weights
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, F), where F is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, F) and weights of shape (F)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
          
        # * Unpack into targets and weights
        targets, weights = y

        # * Calculate negative score - this is what we want to minimize
        neg_score = -torch.log(self._normal())

        # * weight to control contributions

        # * Now weigh it
        loss_weighted = torch.sum(self._weights*logcosh, dim=-1)
        
        # * Mean over the batch
        if not predict:
            loss = torch.mean(loss_weighted)
        else:
            loss = loss_weighted

        return loss

def get_loss_func(name, weights=None, device=None):
    if name == 'L1': 
        return torch.nn.L1Loss()
    elif name == 'angle_loss':
        return angle_loss()
    elif name == 'Huber':
        return torch.nn.SmoothL1Loss()
    elif name == 'L2':
        return L2()
    elif name == 'logcosh':
        return logcosh()
    elif name == 'logcosh_full_weighted':
        return logcosh_full_weighted(weights=weights, device=device)
    elif name == 'angle_squared_loss':
        return angle_squared_loss()
    elif name == 'angle_squared_loss_with_L2':
        return angle_squared_loss_with_L2()
    else:
        raise ValueError('Unknown loss function requested!')
