import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Dict


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, dilation=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv1d(inp_dim, out_dim, kernel_size=k, padding=pad, stride=stride, dilation=dilation, bias=not with_bn)
        self.bn   = nn.BatchNorm1d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

def make_kp_layer(cnv_dim, curr_dim, out_dim, kernel_size=3, stride=1, dilation=1, with_bn=False):
    return nn.Sequential(
        convolution(kernel_size, cnv_dim, curr_dim, stride, dilation, with_bn=with_bn),
        nn.Conv1d(curr_dim, out_dim, 1)
    )


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def focal_loss(out, target):
    '''
    Arguments:
      out, target: B x Time x Station
    '''
    out = out.float()
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
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
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask

    regr_loss = F.smooth_l1_loss(pred, ground_truth, reduction='sum')
    return regr_loss / (num + 1e-4)


def l1_reg_loss(output, target, mask):
    output = output.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    
    regr_loss = F.l1_loss(pred, ground_truth, reduction='sum')
    return regr_loss / (num + 1e-4)


def norml1_reg_loss(output, target, mask):
    output = output.float()
    num = mask.float().sum()
    mask = mask.unsqueeze(1).float()
    pred = output * mask
    ground_truth = target * mask
    
    pred = pred / (ground_truth + 1e-4)
    ground_truth = ground_truth * 0 + 1

    regr_loss = F.l1_loss(pred, ground_truth, reduction='sum')
    return regr_loss / (num + 1e-4)


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask,  reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterNetHead(nn.Module):
    def __init__(self, channels=[768, 256, 64], bn=False, kernel_size=3, hm_loss="focal", reg_loss="sl1", weights=[1, 0.02, 0.02]):
        super().__init__()
        self.curr_dim = channels[0]
        self.cnv_dim = channels[1]
        self.hid_dim = channels[2]
        self.bn = bn
        self.weights = weights
        self.hm_loss = hm_loss
        self.reg_loss = reg_loss
        
        self.upsample = nn.Conv1d(self.curr_dim, self.curr_dim, kernel_size=1, bias=False)
        self.conv = convolution(k=5, inp_dim=self.curr_dim, out_dim=self.cnv_dim, with_bn=True)
        # event_center
        self.heatmap = make_kp_layer(self.cnv_dim, self.hid_dim, out_dim=1, kernel_size=kernel_size)
        self.heatmap[-1].bias.data.fill_(-2.19)
        # phase_pick and offset
        self.width = make_kp_layer(self.cnv_dim, self.hid_dim, out_dim=2, kernel_size=kernel_size)
        # event_loaction
        self.reg = make_kp_layer(self.cnv_dim, self.hid_dim, out_dim=4, kernel_size=kernel_size)
        
        self.hm_loss = F.mse_loss if hm_loss=="mse" else focal_loss
        self.hw_loss = l1_reg_loss if reg_loss=="l1" else smoothl1_reg_loss
        self.reg_loss = l1_reg_loss if reg_loss=="l1" else smoothl1_reg_loss
            
        
    def forward(self, features, event_center, event_location, event_location_mask):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["out"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)    
        x = self.upsample(x)
        x = self.conv(x)
        
        heatmap = self.heatmap(x)
        if self.hm_loss!="mse":
            heatmap = torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1-1e-4)
        heatmap = heatmap.view(bt, st, heatmap.shape[2])  # chn = 1
        heatmap = heatmap.permute(0, 2, 1) # batch, time, station
        
        width = self.width(x)
        width = width.view(bt, st, width.shape[1], width.shape[2]) # batch, station, 2, time
        width = width.permute(0, 2, 3, 1) # batch, 2, time, station
        
        reg = self.reg(x)
        reg = reg.view(bt, st, reg.shape[1], reg.shape[2]) # batch, station, 4, time
        reg = reg.permute(0, 2, 3, 1) # batch, 4, time, station
        
        if self.training:            
            return None, self.losses({"event": heatmap, "offset": width, "hypocenter": reg}, event_center, event_location, event_location_mask)
        elif event_center is not None:
            return {"event": heatmap, "offset": width, "hypocenter": reg}, \
                self.losses({"event": heatmap, "offset": width, "hypocenter": reg}, event_center, event_location, event_location_mask)
        else:
            return {"event": heatmap, "offset": width, "hypocenter": reg}, {}
        
        
    def losses(self, outputs, event_center, event_location, event_location_mask):
        hm_loss = self.hm_loss(outputs["event"], event_center)
        
        outputs["offset"][:,0,:,:] = outputs["offset"][:,0,:,:] * 10 # emphasize the event_center offset loss
        event_location[:,4,:,:] = event_location[:,4,:,:] * 10 # emphasize the event_center offset loss
        hw_loss = self.hw_loss(outputs["offset"], event_location[:, 4:, :, :], event_location_mask)
        
        reg_loss = self.reg_loss(outputs["hypocenter"], event_location[:, :4, :, :], event_location_mask)
        
        loss = hm_loss * self.weights[0] + hw_loss * self.weights[1] + reg_loss * self.weights[2]
        # print(f"loss {loss}, hm_loss {hm_loss}, hw_loss {hw_loss}, reg_loss {reg_loss}", force=True)
        return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_hypocenter": reg_loss}
        
    