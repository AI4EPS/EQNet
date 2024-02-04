import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(inputs, targets, mask):
    inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
    num=mask.sum()
    loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=mask, 
                     reduction='sum')/(num*targets.shape[1])
        
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

def balanced_l1_reg_loss(output, target, mask, center, scale=1.0, type="sl1"):
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
        
    regr_loss = (regr_loss * (torch.abs(ground_truth - center)*scale + 1)).sum() / (num + 1e-4)
    
    return regr_loss