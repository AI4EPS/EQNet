# Based on OpenMMLab's implementation of UPerNet

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .centernet import convolution, \
    cross_entropy_loss, focal_loss, smoothl1_reg_loss, weighted_l1_reg_loss, l1_reg_loss


class Sample(nn.Module):
    def __init__(self, scale_factor=2, mode='linear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(pool_scale),
                    convolution(
                        inp_dim=self.in_channels,
                        out_dim=self.channels,
                        k=1,
                        stride=1,
                        dilation=1,
                        with_bn=True,),
                    ))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            # x shape: [batch_size, stations, channels, time], we only upsample the time dimension
            upsampled_ppm_out = F.interpolate(ppm_out, size=x.shape[-1], mode='linear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerNeck(nn.Module, metaclass=ABCMeta):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 channels=512,
                 pool_scales=(1, 2, 3, 6),
                 dropout_ratio=0.1,
                 in_index=[0, 1, 2, 3],
                 input_transform="multiple_select",
                 align_corners=False,
                 ):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index

        self.align_corners = align_corners

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
            #TODO: or nn.Dropout1d?
        else:
            self.dropout = None
        
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners)
        self.bottleneck = convolution(
            inp_dim=self.in_channels[-1] + len(pool_scales) * self.channels,
            out_dim=self.channels,
            k=3,)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = convolution(
                inp_dim=in_channels,
                out_dim=self.channels,
                k=1,
                inplace=False)
            fpn_conv = convolution(
                inp_dim=self.channels,
                out_dim=self.channels,
                k=3,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # FPN Fused Module
        self.fpn_bottleneck = convolution(
            inp_dim=len(self.in_channels) * self.channels,
            out_dim=self.channels,
            k=3,)
        

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps 
        Args:
            inputs (list[Tensor]): Listc of multi-level img features.

        Returns:
            [offset, heatmap, hypocenter]: List of multi-level features.
            offset: PPM output (Tensor): A tensor of shape (batch_size * nstations,
            self.channels, nt=1/32 raw nt) which is feature by PPM Module
            heatmap: FPN output[0] (Tensor): A tensor of shape (batch_size * nstations, 
            self.channels, nt=1/4 raw nt) which is feature by the last layer of FPN
            hypocenter: feats (Tensor): A tensor of shape (batch_size * nstations, 
            self.channels, nt=1/4 raw nt) which is feature map by Fusion Module
        """
        inputs = self._transform_inputs(inputs)
        outputs = {}

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-1]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                input=laterals[i],
                size=prev_shape,
                mode='linear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])
        outputs["ppm"] = fpn_outs[-1]
        #outputs["fpn"] = fpn_outs[0]

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[-1],
                mode='linear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        outputs["fusion"] = feats
        
        return outputs # [psp, feats] dim: bt*sta, chn, nt

    def forward(self, features):
        """Forward function.
        Args:
            features (list[Tensor]): List of multi-level features. Shape of each: bt, sta, chn, nt
        """
        features = [features[f"out{i}"] for i in range(len(features.keys()))]
        for i in range(len(features)):
            bt, st, ch, nt = features[i].shape
            features[i] = features[i].view(bt*st, ch, nt)   
        #assert len(features[0].shape)==3, "features should be a list of 3d tensor"
        
        features = self._forward_feature(features)
        
        for k in features.keys():
            features[k] = features[k].view(bt, st, features[k].shape[1], features[k].shape[2])
            
        return features
    
    
    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[-1],
                    mode='linear',
                    align_corners=self.align_corners) for x in inputs
                ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


class EventHead(nn.Module):
    def __init__(self, 
                 channels=512,
                 hidden_channels=[256, 128, 64, 32],
                 dilations=[1, 4, 8, 16],
                 kernel_size = 3,
                 loss_dict: dict = {"hm_loss": "focal", "hw_loss": "wl1", "reg_loss": "wl1"},
                 weights=[1, 0.015, 0.01],
                 offset_weight=[10, 1], 
                 reg_weight=[1,1,1,1],
                 ):
        super().__init__()
        
        self.channels = channels
        self.weights = weights
        self.offset_weight = offset_weight
        self.reg_weight = reg_weight
        
        #TODO: subhead layers? dilation?
        self.heatmap = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels[2],
                1,
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                padding_mode="reflect",
            )
        )
        self.heatmap[-1].bias.data.fill_(-2.19) # if use the initial value, the loss from 200 to 5 (v2)
        self.offset = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels[2],
                2,
                kernel_size=kernel_size,
                dilation=dilations[0],
                padding=((kernel_size - 1) * dilations[0] + 1) // 2,
                padding_mode="reflect",
            )
        )
        
        self.hypocenter = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels[2],
                #3,
                4,
                kernel_size=kernel_size,
                dilation=dilations[0],
                padding=((kernel_size - 1) * dilations[0] + 1) // 2,
                padding_mode="reflect",
            )
        )


        loss_factory = {"mse": F.mse_loss, "focal": focal_loss, "l1": l1_reg_loss, "sl1": smoothl1_reg_loss, "wl1": weighted_l1_reg_loss, "cross_entropy": cross_entropy_loss}
        self.weights = weights
        self.hm_loss = loss_factory[loss_dict["hm_loss"]]
        self.hw_loss = loss_factory[loss_dict["hw_loss"]]
        self.reg_loss = loss_factory[loss_dict["reg_loss"]]
        
    def forward(self, features, event_center, event_location, event_location_mask):
        x = features['ppm']
        bt, st, ch, nt = x.shape
        x = x.view(bt*st, ch, nt)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        
        # offset
        offset = self.offset(x)
        offset = offset.view(bt, st, offset.shape[1], offset.shape[2])
        offset = offset.permute(0, 2, 3, 1) # bt, ch, nt, sta
        
        # heatmap
        heatmap = self.heatmap(x)
        heatmap = heatmap.view(bt, st, heatmap.shape[2])  # chn = 1
        heatmap = heatmap.permute(0, 2, 1)
        
        # hypocenter
        hypocenter = self.hypocenter(x)
        hypocenter = hypocenter.view(bt, st, hypocenter.shape[1], hypocenter.shape[2])
        hypocenter = hypocenter.permute(0, 2, 3, 1)
        
        if self.training:            
            return None, self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
        elif event_center is not None:
            return {"event": heatmap, "offset": offset, "hypocenter": hypocenter}, \
                self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
        else:
            return {"event": heatmap, "offset": offset, "hypocenter": hypocenter}, {}
    
    def losses(self, outputs, event_center, event_location, event_location_mask):
        #hm_loss = cross_entropy_loss(outputs["event"], event_center)
        hm_loss = self.hm_loss(outputs["event"], event_center)
        hw_loss = self.hw_loss(outputs["offset"], event_location[:, 4:, :, :], event_location_mask, weights=self.offset_weight)
        
        #reg_loss = smoothl1_reg_loss(outputs["hypocenter"], event_location[:, :3, :, :], event_location_mask)
        reg_loss = self.reg_loss(outputs["hypocenter"], event_location[:, :4, :, :], event_location_mask, weights=self.reg_weight)
        
        loss = hm_loss * self.weights[0] + hw_loss * self.weights[1] + reg_loss * self.weights[2]
        # print(f"loss {loss}, hm_loss {hm_loss}, hw_loss {hw_loss}, reg_loss {reg_loss}", force=True)
        return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_hypocenter": reg_loss}
    
    
class PhaseHead(nn.Module):
    def __init__(self, 
                 channels=512,
                 hidden_channels=[256, 128, 64, 32],
                 dilations=[1, 8, 16, 32],
                 kernel_size = 3,
                 ):
        super().__init__()
        
        self.channels = channels
        
        self.phase_1 = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[1],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.ReLU(inplace=True),
        )
        
        self.phase_2 = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.ReLU(inplace=True),
        )
        
        self.conv_out = nn.Conv1d(
            hidden_channels[3],
            3,
            kernel_size=kernel_size,
            dilation=dilations[3],
            padding=((kernel_size - 1) * dilations[3] + 1) // 2,
            padding_mode="reflect",
        )
        
    def forward(self, features, targets=None):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["fusion"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        
        x = self.phase_1(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.phase_2(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.conv_out(x)
        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets)
        elif targets is not None:
            return x, self.losses(x, targets)
        else:
            return x, {}

    def losses(self, inputs, targets):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        num=targets.shape[0]*targets.shape[3]
        for i in range(targets.shape[0]):
            for j in range(targets.shape[3]):
                if torch.all(targets[i,:,:,j]==0):
                    num-=1
        loss = torch.sum(-targets * F.log_softmax(inputs, dim=1))/(num*targets.shape[2])

        return loss