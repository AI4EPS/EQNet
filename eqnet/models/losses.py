import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal as MVN


def cross_entropy_loss(inputs, targets, mask):
    inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
    targets = targets.float()
    num=mask.sum()
    loss = (F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * mask).sum() / (num + 1e-4)
        
    return loss


def focal_loss(out, target, mask=None):
    '''
    Arguments:
      out, target: B x Time x Station
      mask: B x Time x Station
    '''
    out = out.float()
    out = torch.clamp(torch.sigmoid(out), min=1e-4, max=1-1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    if mask is not None:
        pos_inds = pos_inds * mask
        neg_inds = neg_inds * mask
    
    neg_weights = torch.pow(1 - target, 4)
    pos_loss = torch.log(out) * torch.pow(1 - out, 2) * pos_inds
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * neg_weights * neg_inds
    
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        return - neg_loss
    else:
        return - (pos_loss + neg_loss) / num_pos
    
    
def smooth_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                      reduction='mean'):
    """Smooth Focal Loss proposed in Circular Smooth Label (CSL).

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The reduction method used to
            override the original reduction method of the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".

    Returns:
        torch.Tensor: The calculated loss
    """
    pred_sigmoid = pred.float().sigmoid()
    target = target.float()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        num = weight.float().sum()
        weight = weight.unsqueeze(1).float()
        loss = loss * weight
        loss = loss.sum()/(num+1e-4) if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss
    
    return loss

'''
def smoothl1_reg_loss(output, target, ind):
    # Smooth L1 regression loss
    #Arguments:
    #    outout: B x Channel x Time x Station
    #    target: B x Channel x Time x Station
    #    ind: B x Station
    #
    output = output.float()
    pred = _transpose_and_gather_feat(output, ind)
    ground_truth = _transpose_and_gather_feat(target, ind)
    num = pred.shape[-1]

    regr_loss = F.smooth_l1_loss(pred, ground_truth, reduction='sum')
    regr_loss = regr_loss / num
    return regr_loss
'''

def smoothl1_reg_loss(output, target, mask):
    ''' Smooth L1 regression loss
    Arguments:
        outout: B x Channel x Time x Station
        target: B x Channel x Time x Station
        mask: B x Time x Station
    '''
    output = output.float()
    target = target.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask

    regr_loss = F.smooth_l1_loss(pred, ground_truth, reduction='sum')
    return regr_loss / (num + 1e-4)


def l1_reg_loss(output, target, mask):
    output = output.float()
    target = target.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    
    regr_loss = F.l1_loss(pred, ground_truth, reduction='sum')
    return regr_loss / (num + 1e-4)


def norml1_reg_loss(output, target, mask):
    output = output.float()
    target = target.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    
    pred = pred / (ground_truth + 1e-4)
    ground_truth = ground_truth * 0 + 1

    regr_loss = F.l1_loss(pred, ground_truth, reduction='sum')
    return regr_loss / (num + 1e-4)


def weighted_l1_reg_loss(output, target, mask, weights=None, type="sl1"):
    '''
    Arguments:
        weights: channel
    '''
    output = output.float()
    target = target.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    if weights is None:
        weights = torch.ones(ground_truth.shape[1]).to(ground_truth.device)
    else:
        weights = torch.tensor(weights).to(ground_truth.device)

    if type == "l1":
        regr_loss = F.l1_loss(pred, ground_truth, reduction='none')
    elif type == "sl1":
        regr_loss = F.smooth_l1_loss(pred, ground_truth, reduction='none')
    regr_loss = (regr_loss * weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).sum()

    return regr_loss / (num + 1e-4)


def vector_l1_reg_loss(output, target, mask, type="sl1"):
    '''
    return the channel-wise loss
    '''
    output = output.float()
    target = target.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    
    if type == "l1":
        regr_loss = F.l1_loss(pred, ground_truth, reduction='none')
    elif type == "sl1":
        regr_loss = F.smooth_l1_loss(pred, ground_truth, reduction='none')
    regr_loss = regr_loss.sum(dim=0).sum(dim=-1).sum(dim=-1) / (num + 1e-4)
    
    return regr_loss

def modulus_l1_reg_loss(output, target, mask, type="sl1"):
    '''
    return the modulus of channel-wise loss vector 
    '''
    output = output.float()
    target = target.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    
    if type == "l1":
        regr_loss = F.l1_loss(pred, ground_truth, reduction='none')
    elif type == "sl1":
        regr_loss = F.smooth_l1_loss(pred, ground_truth, reduction='none')
    #regr_loss = F.mse_loss(pred, ground_truth, reduction='none')
    #regr_loss = torch.sqrt(regr_loss.sum(dim=1)).sum()/ (num + 1e-4)
    
    #regr_loss = torch.sqrt((regr_loss ** 2).sum(dim=1)).sum()/ (num + 1e-4) # TODO: NaN
    regr_loss = regr_loss.sum(dim=0).sum(dim=-1).sum(dim=-1) / (num + 1e-4)
    regr_loss = torch.sqrt((regr_loss ** 2).sum())
    
    return regr_loss

def balanced_l1_reg_loss(output, target, mask, scale, shape, loc, base=1.0, weight=1.0, type="sl1"):
    '''
    The ground truth is assumed to be log-normal distributed, 
    which means ln(gt - loc) ~ N(ln(scale), shape^2)
    the extra weight of loss is how many times std the ground truth is away from the mean,
    namely, weight = abs(ln(gt - loc) - ln(scale))/shape + 1
    '''
    output = output.float()
    target = target.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    
    if type == "l1":
        regr_loss = F.l1_loss(pred, ground_truth, reduction='none')
    elif type == "sl1":
        regr_loss = F.smooth_l1_loss(pred, ground_truth, reduction='none')
    
    regr_loss = (regr_loss * \
        (torch.abs(torch.log(torch.clamp(ground_truth-loc, min=0.1)) - torch.log(torch.tensor(scale)))/shape \
            * weight + base).detach() * mask\
            ).sum() / (num + 1e-4)
    
    return regr_loss


def bmc_loss(pred, target, mask, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    
    `Balanced MSE for Imbalanced Visual Regression
    <https://arxiv.org/abs/2203.16427>`_ .
    
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.unsqueeze(1)
    pred = pred[mask].unsqueeze(1)
    target = target[mask].unsqueeze(1)
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1])
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    
    return loss


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_class, train_cutpoints=False, scale=20.0, logit_scale=1.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
        if train_cutpoints:
            self.cutpoints = nn.Parameter(self.cutpoints[None, :, None, None])
        else:
            self.cutpoints = self.cutpoints[None, :, None, None].detach()
        self.logit_scale = logit_scale

    def forward(self, pred, mask, label):
        sigmoids = torch.sigmoid(self.cutpoints - pred*self.logit_scale)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)
        
        if mask is None:
            mask = torch.ones_like(sigmoids).float()
        else:
            mask = mask.unsqueeze(1).float()
        num = mask.sum() + 1e-5

        if label is not None:
            loss = -torch.sum(label.float() * torch.log(likelihoods), dim=1).sum() / num
        else:
            loss = None
            
        return loss, likelihoods