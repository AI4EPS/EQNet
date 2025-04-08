import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal as MVN


def cross_entropy_loss(inputs, targets, mask):
    inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
    targets = targets.float()
    num=mask.sum()
    min_loss = -(targets * torch.nan_to_num(torch.log(targets)) + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
    loss = ((F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') - min_loss) * mask).sum() / (num + 1e-4)
        
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

def sigmoid_focal_loss(
    inputs,
    targets,
    mask=None,
    num_objects=None,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    if mask is not None:
        mask = mask.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        else:
            raise ValueError("Mask dimension not supported")
    else:
        mask = 1.0
    if num_objects is None:
        if mask is None or isinstance(mask, float): 
            num_objects = inputs.shape[0] * inputs.shape[1] * inputs.shape[3]
        else:
            num_objects = mask.sum() * inputs.shape[1]
    inputs = inputs.float()
    prob = inputs.sigmoid() * mask
    targets = targets.float() * mask
    min_loss = -(targets * torch.nan_to_num(torch.log(targets)) + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
    ce_loss = (F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")-min_loss) * mask
    #ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") * mask
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects

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
    
def dice_loss(inputs, targets, mask=None, num_objects=None, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    if mask is not None:
        mask = mask.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        else:
            raise ValueError("Mask dimension not supported")
    else:
        mask = 1.0
    if num_objects is None:
        num_objects = inputs.shape[0] * inputs.shape[1]
    ## TEMP
    inputs = inputs.sigmoid()
    targets = targets > 0.0
    
    inputs = inputs.float() * mask
    targets = targets.float() * mask
    
    # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
    # flatten spatial dimension while keeping multimask channel dimension
    inputs = inputs.flatten(2)
    targets = targets.flatten(2)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1e-5) / (denominator + 1e-5)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def tversky_loss(inputs, targets, mask=None, num_objects=None, loss_on_multimask=False, alpha=0.7, beta=0.3, smooth=1e-3):
    """
    Compute the Tversky loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Tversky loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    if mask is not None:
        mask = mask.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        else:
            raise ValueError("Mask dimension not supported")
    else:
        mask = 1.0
    if num_objects is None:
        num_objects = inputs.shape[0] * inputs.shape[1]
    ## TEMP
    inputs = inputs.sigmoid()
    targets = targets > 0.0
    
    inputs = inputs.float() * mask
    targets = targets.float() * mask
    
    # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
    # flatten spatial dimension while keeping multimask channel dimension
    inputs = inputs.flatten(2)
    targets = targets.flatten(2)
    tp = (inputs * targets).sum(-1)
    fp = (inputs * (1 - targets)).sum(-1)
    fn = ((1 - inputs) * targets).sum(-1)
    loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, mask=None, num_objects=None, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    if mask is not None:
        mask = mask.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        else:
            raise ValueError("Mask dimension not supported")
    else:
        mask = 1.0
    inputs = inputs.float().sigmoid() * mask
    targets = targets.float() * mask
    pred_ious = pred_ious.float()
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    if num_objects is None:
        assert pred_ious.shape[1] == 1, f"pred_ious should be [N, 1]"
        num_objects = pred_ious.shape[0] * pred_ious.shape[1]

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects