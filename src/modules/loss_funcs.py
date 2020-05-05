import torch
import numpy as np

PROBABILISTIC_LOSS_FUNCS = [
    'logscore'
]
POINTLIKE_LOSS_FUNCS = [
    'cosine_loss', 
    'logcosh', 
    'logcosh_full_weighted', 
    'L1',
    'L1_UnitVectorPenalty'
]

#* ======================================================================== 
#* LOSS FUNCTIONS
#* ======================================================================== 

class cosine_loss(torch.nn.Module):
    '''takes two tensors with shapes (B, 3) as input and calculates the angular error. Adds and multiplies denominator with 1e-7 and 1+1e-7 to avoid division with zero.
    '''
    def __init__(self):
        super(cosine_loss, self).__init__()
        
    def forward(self, x, y, predict=False):
        """Computes cosine loss with weights
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, 3), where 3 is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, 3) and weights of shape (B,)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
          
        # Unpack into targets and weights
        targets, weights = y
        neg_cos = 1 - torch.sum(
            x*targets, dim=-1
        )
        
        # Mean over the batch
        if not predict:
            neg_cos_weighted = torch.mean(weights*neg_cos)
        else:
            neg_cos_weighted = weights*neg_cos

        return neg_cos_weighted

class angle_loss_old(torch.nn.Module):
    '''takes two tensors with shapes (B, 3) as input and calculates the angular error. Adds and multiplies denominator with 1e-7 and 1+1e-7 to avoid division with zero.
    '''
    def __init__(self):
        super(self).__init__()
        
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
        # Register a hook, which clips the gradient values to try to avoid producing nans. 
        # This can only be done during forward pass - therefore try, except.
        try:
            x.register_hook(self._clip_x)
        except RuntimeError:
            pass

        # Unpack into targets and weights
        targets, weights = y

        # Add eps to avoid infinite gradient if x is the zero-vector
        len_x = torch.sqrt(self._eps**2+torch.sum(x*x, dim=-1))
        len_y = torch.sqrt(torch.sum(targets*targets, dim=-1))
        dot_prods = torch.sum(x*targets, dim=-1)
        
        # Multiply by a number slightly larger than one to avoid the derivative of acos becoming infinite.
        cos = dot_prods/(len_x*len_y*(1+self._eps))
        loss = torch.acos(cos)**2

        # Now weigh it and mean over the batch
        loss_weighted = weights*loss
        loss_mean = torch.mean(loss_weighted)

        # Calculate L2 loss aswell - same way
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
        
        # Unpack into targets and weights
        targets, weights = y

        # Calculate actual loss
        L2 = torch.mean((x-targets)*(x-targets), dim=-1)

        # Now weigh it
        L2_weighted = L2*weights
        
        # Mean over the batch
        L2_mean = torch.mean(L2_weighted)

        return L2_mean

class L1(torch.nn.Module):
 
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, x, y, predict=False):
        """Computes L1-loss with weights
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, F), where F is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, F) and weights of shape (F)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
        
        # Unpack into targets and weights
        targets, weights = y

        # Calculate actual loss
        L1 = torch.mean(torch.abs(x-targets), dim=-1)

        # Now weigh it
        L1_weighted = L1*weights
        
        # Mean over the batch
        # Mean over the batch
        if not predict:
            loss = torch.mean(L1_weighted)

        else:
            loss = L1_weighted

        return loss

class L1_UnitVectorPenalty(torch.nn.Module):
 
    def __init__(self, eps=1e-3):
        super(L1_UnitVectorPenalty, self).__init__()
        self.eps = eps

    def forward(self, x, y, predict=False):
        """Computes L1-loss with weights
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, F), where F is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, F) and weights of shape (F)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
        
        # Unpack into targets and weights
        targets, weights = y

        # Calculate L1-loss
        L1 = torch.mean(torch.abs(x-targets), dim=-1)

        # Calculate penalty for predictions not having unit length
        length = torch.sqrt(
            torch.sum(x*x, dim=-1) + self.eps*self.eps
        )
        pen = (1.0 - length) * (1.0 - length)

        # Now weigh it
        L1_weighted = (L1 + pen) * weights
        
        # Mean over the batch
        if not predict:
            loss = torch.mean(L1_weighted)

        else:
            loss = L1_weighted

        return loss

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
          
        # Unpack into targets and weights
        targets, weights = y

        # Calculate actual loss
        ave_logcosh = torch.mean(torch.log(torch.cosh(x-targets)), dim=-1)

        # Now weigh it
        loss_weighted = weights*ave_logcosh
        
        # Mean over the batch
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
          
        # Unpack into targets and weights
        targets, weights = y

        # Calculate actual loss
        logcosh = torch.log(torch.cosh(x-targets))

        # weight to control contributions

        # Now weigh it
        loss_weighted = torch.sum(self._weights*logcosh, dim=-1)
        
        # Mean over the batch
        if not predict:
            loss = torch.mean(loss_weighted)
        else:
            loss = loss_weighted

        return loss

class logscore(torch.nn.Module):
    def __init__(self, weights=None, device=None):
        super(logscore, self).__init__()

        if not device:
            raise ValueError('A device must be supplied to the loss function'
            'logscore')

        if not weights:
            raise ValueError('Weights must be specified for logscore loss')

        weights_normed = np.array(weights)/np.sum(weights)
        self._weights = torch.tensor(weights_normed, device=device)

    def forward(self, x, y, predict=False):
        """Computes the empirical expected score using a normal distribution.

        If targets is of shape (B, F), x must be of shape (B, 2F). The last 
        bottom F features are used as standard deviations.

        To ensure positivity of sigma, the sigma-predictions are passed through 
        the softplus-function (see https://pytorch.org/docs/stable/nn.html).
        
        Arguments:
            x {torch.Tensor} -- Predictions of shape (B, 2F), where F is number of targets
            y {tuple} -- (targets, weights), where targets is of shape (B, F) and weights of shape (F)
        
        Returns:
            [torch.Tensor] -- Averaged, weighted loss over batch.
        """      
          
        # Unpack into targets and weights
        targets, weights = y
        pred_shape = x.shape
        n_features = targets.shape[-1]

        if 2*n_features != x.shape[-1]:
            raise ValueError(
                'Predictions.shape[-1] (%d) must equal' 
                '2*targets.shape[-1] (%d)'%(pred_shape[-1], 2*n_features)
        )

        # Calculate negative score - this is what we want to minimize.
        # log(normal) = c1/sigma + c2*((x-mu)/sigma)^2
        mean = x[:, :n_features]
        sigma = x[:, n_features:]
        neg_score = torch.log(sigma*np.sqrt(2*3.14159)) + 0.5*((targets-mean)/sigma)*((targets-mean)/sigma)
        
        if not predict:
            loss = torch.mean(neg_score, dim=0)
        else:
            loss = neg_score
        
        # Now weigh it
        loss_weighted = torch.sum(self._weights*loss, dim=-1)

        return loss_weighted

def get_loss_func(name, weights=None, device=None):
    if name == 'L1': 
        return L1()
    elif name == 'L1_UnitVectorPenalty':
        return L1_UnitVectorPenalty()
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
    elif name == 'logscore':
        return logscore(weights=weights, device=device)
    elif name == 'cosine_loss':
        return cosine_loss()
    else:
        raise ValueError('Unknown loss function requested!')

def is_probabilistic(loss_name):
    if loss_name in PROBABILISTIC_LOSS_FUNCS:
        PROBABILISTIC_REGRESSION = True
    elif loss_name in POINTLIKE_LOSS_FUNCS:
        PROBABILISTIC_REGRESSION = False
    else:
        raise ValueError(
        'Unknown loss function-kind encountered(%s)'%(loss_name))
    
    return PROBABILISTIC_REGRESSION