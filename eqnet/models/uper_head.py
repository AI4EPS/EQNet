# Based on OpenMMLab's implementation of UPerNet

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .centernet import convolution
from .losses import cross_entropy_loss, focal_loss, smooth_focal_loss, \
        smoothl1_reg_loss, weighted_l1_reg_loss, l1_reg_loss,\
        vector_l1_reg_loss,modulus_l1_reg_loss, balanced_l1_reg_loss
from ..utils import CSLCoder, TrigonoCoder


class Sample(nn.Module):
    def __init__(self, scale_factor=2, mode='linear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class Sine(nn.Module):
    def  __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

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
                 kernel_size=7,
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
            k=kernel_size,)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in self.in_channels[:-1]:  # skip the top layer
            l_conv = convolution(
                inp_dim=in_channel,
                out_dim=self.channels,
                k=1,
                inplace=False)
            fpn_conv = convolution(
                inp_dim=self.channels,
                out_dim=self.channels,
                k=kernel_size,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # FPN Fused Module
        self.fpn_bottleneck = convolution(
            inp_dim=len(self.in_channels) * self.channels,
            out_dim=self.channels,
            k=kernel_size,)
        

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        #psp_outs = torch.cat([x]+self.psp_modules(x), dim=1)
        output = self.bottleneck(torch.cat([x]+self.psp_modules(x), dim=1))

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps 
        Args:
            inputs (list[Tensor]): Listc of multi-level wave features.
            amplitude (Tensor): A tensor of shape (batch_size*nsta, channels, nt) which is the amplitude

        Returns:
            outputs: Dict of multi-level features.
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
        #outputs["ppm"] = fpn_outs[-1]
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
            features (dict[Tensor]): Dict of multi-level features. Shape of each: bt, sta, chn, nt
            amplitude (Tensor): A tensor of shape (batch_size, channels, nt, nsta) which is the amplitude
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
                 hidden_channels=[256, 128, 64, 32, 16],
                 dilations=[1, 4, 8, 16],
                 kernel_size=7,
                 loss_dict: dict = {"hm_loss": "focal", "hw_loss": "wl1", "travel_time_loss": "sl1", "dist_loss": "sl1", "angle_loss": "sl1", "depth_loss": "bl1", "mag_loss": "bl1"},
                 weights=[1.49, 0.018, 0.055],
                 offset_weight=[1, 1], 
                 reg_weight=[0.8,2,20,1.0, 0.8],
                 angle_coder="trigono",
                 depth_center=8.0,
                 mag_center=1.5,
                 act_layer='gelu'
                 ):
        super().__init__()
        
        self.channels = channels
        reg_weight = [reg_w * weights[-1] for reg_w in reg_weight]
        weights = weights[:-1]+reg_weight
        self.weights = weights
        self.offset_weight = offset_weight
        self.reg_weight = reg_weight
        self.depth_center = depth_center
        self.mag_center = mag_center
        
        if angle_coder == "csl":
            self.omega = 5
            self.angle_dims = 360 // self.omega
            self.angle_coder = CSLCoder(self.omega)
        elif angle_coder == "trigono":
            self.angle_dims = 2
            self.angle_coder = TrigonoCoder()
        
        #TODO: subhead layers? dilation?
        self.heatmap = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            #Sample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2],
                1,
                kernel_size=1,
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
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
            #Sample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2],
                2,
                kernel_size=1,
            )
        )
        
        
        self.hypocenter = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[1],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.GELU(),
        )
        
        self.amplitude = nn.Sequential(
            nn.Conv1d(
                3,
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
        )
        
        self.angle = nn.Sequential(
            nn.Conv1d(
                4,
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
        )
        
        self.fuse = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1]+hidden_channels[1],
                hidden_channels[1],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.GELU(),
        )
        
        self.travel_time = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=1,
            ),
            #nn.LeakyReLU(inplace=True), # the origin time and the distance should be positive
        )
        self.distance = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=1,
            ),
            #nn.LeakyReLU(inplace=True), # the origin time and the distance should be positive
        )
        self.azimuth = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                self.angle_dims,
                kernel_size=1,
            ),
            Sine(),
        )
        # first channel is sin, second channel is cos
        # so the initial bias should be 0 and pi/2
        #self.angle[-2].bias.data[::2].fill_(0)
        #self.angle[-2].bias.data[1::2].fill_(1.5708)
        self.depth = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=1,
            ),
        )
        
        self.magnitude_1 = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
        )
        self.magnitude_2 = nn.Sequential(
            #Sample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2]+hidden_channels[2],
                hidden_channels[4],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[4]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[4],
                1,
                kernel_size=1,
            )
        )


        loss_factory = {"mse": F.mse_loss, "focal": focal_loss, "sfocal": smooth_focal_loss, \
            "l1": l1_reg_loss, "sl1": smoothl1_reg_loss, "wl1": weighted_l1_reg_loss, \
            "vl1": vector_l1_reg_loss, "ml1": modulus_l1_reg_loss, \
            "bl1": balanced_l1_reg_loss, "cross_entropy": cross_entropy_loss}
        self.weights = weights
        self.hm_loss = loss_factory[loss_dict["hm_loss"]]
        self.hw_loss = loss_factory[loss_dict["hw_loss"]]
        self.travel_time_loss = loss_factory[loss_dict["travel_time_loss"]]
        self.dist_loss = loss_factory[loss_dict["dist_loss"]]
        self.angle_loss = loss_factory[loss_dict["angle_loss"]]
        self.depth_loss = loss_factory[loss_dict["depth_loss"]]
        self.mag_loss = loss_factory[loss_dict["mag_loss"]]
        
    def forward(self, features, amplitude, event_center, event_location, event_location_mask):
        """
        Forward pass of the EventHead module.

        Args:
            features (dict): Dictionary of input features.
            amplitude (torch.Tensor): Tensor of amplitude values.
            event_center (torch.Tensor): Tensor of event center labels.
            event_location (torch.Tensor): Tensor of event location labels.
            event_location_mask (torch.Tensor): Tensor of event location masks.

        Returns:
            tuple: Tuple containing the predicted outputs and the loss values (if in training mode).
        """
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
        
        bt, ch, nt, st = amplitude.shape
        #amplitude = F.avg_pool2d(amplitude, kernel_size=(1024, 1), stride=(128, 1))
        #amplitude = F.interpolate(amplitude, scale_factor=(1024, 1), mode="bilinear", align_corners=False)[:, :, :nt, :nx]
        amplitude[torch.abs(amplitude)<1e-6] = 1e-6
        #horizon_amp = torch.sqrt(torch.sum(amplitude[:,:2,:,:]**2, dim=1, keepdim=True))
        horizon_amp = amplitude[:, 0:1, :, :] / amplitude[:, 1:2, :, :]
        horizon_amp[torch.abs(horizon_amp)<1e-6] = 1e-6
        #full_amp = torch.sqrt(torch.sum(amplitude[:,:,:,:]**2, dim=1, keepdim=True))
        full_amp = amplitude[:, :2, :, :] / amplitude[:, 2:, :, :]
        full_amp[torch.abs(full_amp)<2e-6] = 2e-6
        #assert torch.all(torch.isfinite(horizon_amp)), f"horizon_amp should be finite {torch.abs(amplitude).max()}"
        #assert torch.all(torch.isfinite(full_amp)), "full_amp should be finite"
        
        #angle = torch.cat([amplitude[:, :2, :, :]/horizon_amp, amplitude[:, 2:, :, :]/full_amp, horizon_amp/full_amp], dim=1) # sin, cos, sin(incident angle), cos(incident angle)
        angle = torch.cat([1/torch.sqrt(1+(1/horizon_amp)**2), 
                            1/torch.sqrt(1+horizon_amp**2),
                            1/torch.sqrt(1+torch.sum(full_amp**2, dim=1, keepdim=True)),
                            1/torch.sqrt(1+1/torch.sum(full_amp**2, dim=1, keepdim=True))], dim=1)
        #assert not torch.any(torch.isnan(angle)), "angle should not be nan"
        #assert torch.all(angle<100), "angle should be finite"
        amplitude = torch.log10(torch.abs(amplitude))
        amplitude = torch.cat([amplitude, angle], dim=1)
        assert not torch.any(torch.isnan(amplitude)), "amplitude should not be nan"
        assert torch.all(torch.isfinite(amplitude)), "amplitude should be finite"
        amplitude = amplitude.permute(0, 3, 1, 2).contiguous().view(bt*st, ch+4, nt)
        # downsample amplitude
        amplitude = F.interpolate(amplitude, size=x.shape[-1], mode='linear', align_corners=False)
        
        # hypocenter
        hypocenter = self.hypocenter(x)
        amp_feature = self.amplitude(amplitude[:, :ch, :])
        hypocenter = self.fuse(torch.cat([hypocenter, amp_feature, self.angle(amplitude[:, ch:, :])], dim=1))
        travel_time = self.travel_time(hypocenter)
        distance = self.distance(hypocenter)
        azimuth = self.azimuth(hypocenter)
        depth = self.depth(hypocenter)
        magnitude = self.magnitude_2(torch.cat([self.magnitude_1(x), amp_feature], dim=1))
        
        #print(f"amplitude scale: avg {amplitude.mean()}, abs avg {torch.abs(amplitude).mean()}, std {amplitude.std()}", force=True)
        #print(f"amplitude scale: min {amplitude.min()}, max {amplitude.max()}", force=True)
        #print(f"feature scale: avg {x.mean()}, abs avg {torch.abs(x).mean()}, std {x.std()}", force=True)
        #print(f"feature scale: min {x.min()}, max {x.max()}", force=True)
        
        hypocenter = torch.cat([travel_time, distance, azimuth, depth, magnitude], dim=1)
        hypocenter = hypocenter.view(bt, st, hypocenter.shape[1], hypocenter.shape[2])
        hypocenter = hypocenter.permute(0, 2, 3, 1)
        
        if self.training:            
            return None, self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
        else:
            loss = {}
            if event_center is not None:
                loss = self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
            hypocenter = torch.cat([hypocenter[:, :2, :, :], self.angle_coder.decode(hypocenter[:, 2:(2+self.angle_dims), :, :]),
                                    hypocenter[:, (2+self.angle_dims):, :, :]], dim=1)
            return {"event": heatmap, "offset": offset, "hypocenter": hypocenter}, loss
    
    def losses(self, outputs, event_center, event_location, event_location_mask):
        """
        Compute the loss values for the predicted outputs.

        Args:
            outputs (dict): Dictionary of predicted outputs.
            event_center (torch.Tensor): Tensor of event center labels.
            event_location (torch.Tensor): Tensor of event location labels.
            event_location_mask (torch.Tensor): Tensor of event location masks.

        Returns:
            dict: Dictionary containing the loss values.
        """
        #hm_loss = cross_entropy_loss(outputs["event"], event_center)
        #hm_loss = self.hm_loss(outputs["event"], event_center)
        station_mask = (event_location_mask.sum(axis=1) > 0).float().unsqueeze(1)
        hm_loss = self.hm_loss(outputs["event"], event_center, station_mask)
        hw_loss = self.hw_loss(outputs["offset"], event_location[:, 5:, :, :], event_location_mask, weights=self.offset_weight)
        
        travel_time_loss = self.travel_time_loss(outputs["hypocenter"][:, :1, :, :], event_location[:, :1, :, :], event_location_mask)
        dist_loss = self.dist_loss(outputs["hypocenter"][:, 1:2, :, :], event_location[:, 1:2, :, :], event_location_mask)
        angle_loss = self.angle_loss(outputs["hypocenter"][:, 2:(2+self.angle_dims), :, :],# torch.cat([torch.sin(outputs["hypocenter"][:, 2:3, :, :]), torch.cos(outputs["hypocenter"][:, 2:3, :, :]),], dim=1),
                                        #torch.cat([torch.sin(event_location[:, 2:3, :, :]), torch.cos(event_location[:, 2:3, :, :]),], dim=1),
                                        self.angle_coder.encode(event_location[:, 2:3, :, :]),
                                        event_location_mask)
        depth_loss = self.depth_loss(outputs["hypocenter"][:, (2+self.angle_dims):(2+self.angle_dims+1), :, :], event_location[:, 3:4, :, :], event_location_mask, scale=10.80, shape=0.47, loc=-3.90, base=0.5, weight=1.3)
        mag_loss = self.mag_loss(outputs["hypocenter"][:, (2+self.angle_dims+1):(2+self.angle_dims+2), :, :], event_location[:, 4:5, :, :], event_location_mask, scale=2.29, shape=0.30, loc=-1.32, base=0.3, weight=2.3)
        
        loss = hm_loss * self.weights[0] + hw_loss * self.weights[1] \
            + travel_time_loss * self.weights[2] \
            + dist_loss * self.weights[3] \
            + angle_loss * self.weights[4] \
            + depth_loss * self.weights[5] \
            + mag_loss * self.weights[6]
        # print(f"loss {loss}, hm_loss {hm_loss}, hw_loss {hw_loss}, reg_loss {reg_loss}", force=True)
        # return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_hypocenter": reg_loss}
        return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_origin_time": travel_time_loss, "loss_x": dist_loss, "loss_y": angle_loss, "loss_depth": depth_loss, "loss_magnitude": mag_loss}


class EventHead2(nn.Module):
    def __init__(self, 
                 channels=512,
                 hidden_channels=[256, 128, 64, 32, 16],
                 dilations=[1, 4, 8, 16],
                 kernel_size=7,
                 loss_dict: dict = {"hm_loss": "focal", "hw_loss": "wl1", "travel_time_loss": "sl1", "dist_loss": "sl1", "angle_loss": "sl1", "depth_loss": "bl1", "mag_loss": "bl1"},
                 weights=[1.45, 0.018, 0.055],
                 offset_weight=[1, 1], 
                 reg_weight=[0.8,2,20,1.0, 0.5],
                 angle_coder="trigono",
                 depth_center=8.0,
                 mag_center=1.5,
                 act_layer='gelu'
                 ):
        super().__init__()
        
        self.channels = channels
        reg_weight = [reg_w * weights[-1] for reg_w in reg_weight]
        weights = weights[:-1]+reg_weight
        self.weights = weights
        self.offset_weight = offset_weight
        self.reg_weight = reg_weight
        self.depth_center = depth_center
        self.mag_center = mag_center
        
        if angle_coder == "csl":
            self.omega = 5
            self.angle_dims = 360 // self.omega
            self.angle_coder = CSLCoder(self.omega)
        elif angle_coder == "trigono":
            self.angle_dims = 2
            self.angle_coder = TrigonoCoder()
        
        #TODO: subhead layers? dilation?
        self.heatmap = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            Sample(scale_factor=0.5, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2],
                1,
                kernel_size=1,
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
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
            Sample(scale_factor=0.5, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2],
                2,
                kernel_size=1,
            )
        )
        
        
        self.hypocenter = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[1],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.GELU(),
            Sample(scale_factor=0.5, mode='linear', align_corners=False),
        )
        
        self.amplitude = nn.Sequential(
            nn.Conv1d(
                3,
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
        )
        
        self.angle = nn.Sequential(
            nn.Conv1d(
                4,
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
        )
        
        self.fuse = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1]+hidden_channels[1],
                hidden_channels[1],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.GELU(),
        )
        
        self.travel_time = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=1,
            ),
            #nn.LeakyReLU(inplace=True), # the origin time and the distance should be positive
        )
        self.distance = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=1,
            ),
            #nn.LeakyReLU(inplace=True), # the origin time and the distance should be positive
        )
        self.azimuth = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                self.angle_dims,
                kernel_size=1,
            ),
            Sine(),
        )
        # first channel is sin, second channel is cos
        # so the initial bias should be 0 and pi/2
        #self.angle[-2].bias.data[::2].fill_(0)
        #self.angle[-2].bias.data[1::2].fill_(1.5708)
        self.depth = nn.Sequential(
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=1,
            ),
        )
        
        self.magnitude_1 = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
            Sample(scale_factor=0.5, mode='linear', align_corners=False),
        )
        self.magnitude_2 = nn.Sequential(
            #Sample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2]+hidden_channels[2],
                hidden_channels[4],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[4]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[4],
                1,
                kernel_size=1,
            )
        )


        loss_factory = {"mse": F.mse_loss, "focal": focal_loss, "sfocal": smooth_focal_loss, \
            "l1": l1_reg_loss, "sl1": smoothl1_reg_loss, "wl1": weighted_l1_reg_loss, \
            "vl1": vector_l1_reg_loss, "ml1": modulus_l1_reg_loss, \
            "bl1": balanced_l1_reg_loss, "cross_entropy": cross_entropy_loss}
        self.weights = weights
        self.hm_loss = loss_factory[loss_dict["hm_loss"]]
        self.hw_loss = loss_factory[loss_dict["hw_loss"]]
        self.travel_time_loss = loss_factory[loss_dict["travel_time_loss"]]
        self.dist_loss = loss_factory[loss_dict["dist_loss"]]
        self.angle_loss = loss_factory[loss_dict["angle_loss"]]
        self.depth_loss = loss_factory[loss_dict["depth_loss"]]
        self.mag_loss = loss_factory[loss_dict["mag_loss"]]
        
    def forward(self, features, amplitude, event_center, event_location, event_location_mask):
        """
        Forward pass of the EventHead module.

        Args:
            features (dict): Dictionary of input features.
            amplitude (torch.Tensor): Tensor of amplitude values.
            event_center (torch.Tensor): Tensor of event center labels.
            event_location (torch.Tensor): Tensor of event location labels.
            event_location_mask (torch.Tensor): Tensor of event location masks.

        Returns:
            tuple: Tuple containing the predicted outputs and the loss values (if in training mode).
        """
        x = features['fusion']
        bt, st, ch, nt = x.shape
        x = x.view(bt*st, ch, nt)
        # downsample feature 2x
        x = F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False)
        
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
        
        bt, ch, nt, st = amplitude.shape
        #amplitude = F.avg_pool2d(amplitude, kernel_size=(1024, 1), stride=(128, 1))
        #amplitude = F.interpolate(amplitude, scale_factor=(1024, 1), mode="bilinear", align_corners=False)[:, :, :nt, :nx]
        amplitude[torch.abs(amplitude)<1e-6] = 1e-6
        #horizon_amp = torch.sqrt(torch.sum(amplitude[:,:2,:,:]**2, dim=1, keepdim=True))
        horizon_amp = amplitude[:, 0:1, :, :] / amplitude[:, 1:2, :, :]
        horizon_amp[torch.abs(horizon_amp)<1e-6] = 1e-6
        #full_amp = torch.sqrt(torch.sum(amplitude[:,:,:,:]**2, dim=1, keepdim=True))
        full_amp = amplitude[:, :2, :, :] / amplitude[:, 2:, :, :]
        full_amp[torch.abs(full_amp)<2e-6] = 2e-6
        #assert torch.all(torch.isfinite(horizon_amp)), f"horizon_amp should be finite {torch.abs(amplitude).max()}"
        #assert torch.all(torch.isfinite(full_amp)), "full_amp should be finite"
        
        #angle = torch.cat([amplitude[:, :2, :, :]/horizon_amp, amplitude[:, 2:, :, :]/full_amp, horizon_amp/full_amp], dim=1) # sin, cos, sin(incident angle), cos(incident angle)
        angle = torch.cat([1/torch.sqrt(1+(1/horizon_amp)**2), 
                            1/torch.sqrt(1+horizon_amp**2),
                            1/torch.sqrt(1+torch.sum(full_amp**2, dim=1, keepdim=True)),
                            1/torch.sqrt(1+1/torch.sum(full_amp**2, dim=1, keepdim=True))], dim=1)
        #assert not torch.any(torch.isnan(angle)), "angle should not be nan"
        #assert torch.all(angle<100), "angle should be finite"
        amplitude = torch.log10(torch.abs(amplitude))
        amplitude = torch.cat([amplitude, angle], dim=1)
        assert not torch.any(torch.isnan(amplitude)), "amplitude should not be nan"
        assert torch.all(torch.isfinite(amplitude)), "amplitude should be finite"
        amplitude = amplitude.permute(0, 3, 1, 2).contiguous().view(bt*st, ch+4, nt)
        # downsample amplitude
        amplitude = F.interpolate(amplitude, size=hypocenter.shape[-1], mode='linear', align_corners=False)
        amp_feature = self.amplitude(amplitude[:, :ch, :])
        
        hypocenter = self.fuse(torch.cat([hypocenter, amp_feature, self.angle(amplitude[:, ch:, :])], dim=1))
        travel_time = self.travel_time(hypocenter)
        distance = self.distance(hypocenter)
        azimuth = self.azimuth(hypocenter)
        depth = self.depth(hypocenter)
        magnitude = self.magnitude_2(torch.cat([self.magnitude_1(x), amp_feature], dim=1))
        
        #print(f"amplitude scale: avg {amplitude.mean()}, abs avg {torch.abs(amplitude).mean()}, std {amplitude.std()}", force=True)
        #print(f"amplitude scale: min {amplitude.min()}, max {amplitude.max()}", force=True)
        #print(f"feature scale: avg {x.mean()}, abs avg {torch.abs(x).mean()}, std {x.std()}", force=True)
        #print(f"feature scale: min {x.min()}, max {x.max()}", force=True)
        
        hypocenter = torch.cat([travel_time, distance, azimuth, depth, magnitude], dim=1)
        hypocenter = hypocenter.view(bt, st, hypocenter.shape[1], hypocenter.shape[2])
        hypocenter = hypocenter.permute(0, 2, 3, 1)
        
        if self.training:            
            return None, self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
        else:
            loss = {}
            if event_center is not None:
                loss = self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
            hypocenter = torch.cat([hypocenter[:, :2, :, :], self.angle_coder.decode(hypocenter[:, 2:(2+self.angle_dims), :, :]),
                                    hypocenter[:, (2+self.angle_dims):, :, :]], dim=1)
            return {"event": heatmap, "offset": offset, "hypocenter": hypocenter}, loss
    
    def losses(self, outputs, event_center, event_location, event_location_mask):
        """
        Compute the loss values for the predicted outputs.

        Args:
            outputs (dict): Dictionary of predicted outputs.
            event_center (torch.Tensor): Tensor of event center labels.
            event_location (torch.Tensor): Tensor of event location labels.
            event_location_mask (torch.Tensor): Tensor of event location masks.

        Returns:
            dict: Dictionary containing the loss values.
        """
        #hm_loss = cross_entropy_loss(outputs["event"], event_center)
        #hm_loss = self.hm_loss(outputs["event"], event_center)
        station_mask = (event_location_mask.sum(axis=1) > 0).float().unsqueeze(1)
        hm_loss = self.hm_loss(outputs["event"], event_center, station_mask)
        hw_loss = self.hw_loss(outputs["offset"], event_location[:, 5:, :, :], event_location_mask, weights=self.offset_weight)
        
        travel_time_loss = self.travel_time_loss(outputs["hypocenter"][:, :1, :, :], event_location[:, :1, :, :], event_location_mask)
        dist_loss = self.dist_loss(outputs["hypocenter"][:, 1:2, :, :], event_location[:, 1:2, :, :], event_location_mask)
        angle_loss = self.angle_loss(outputs["hypocenter"][:, 2:(2+self.angle_dims), :, :],# torch.cat([torch.sin(outputs["hypocenter"][:, 2:3, :, :]), torch.cos(outputs["hypocenter"][:, 2:3, :, :]),], dim=1),
                                        #torch.cat([torch.sin(event_location[:, 2:3, :, :]), torch.cos(event_location[:, 2:3, :, :]),], dim=1),
                                        self.angle_coder.encode(event_location[:, 2:3, :, :]),
                                        event_location_mask)
        depth_loss = self.depth_loss(outputs["hypocenter"][:, (2+self.angle_dims):(2+self.angle_dims+1), :, :], event_location[:, 3:4, :, :], event_location_mask, scale=10.80, shape=0.47, loc=-3.98, base=0.2, weight=1.4) #TODO: large scale
        mag_loss = self.mag_loss(outputs["hypocenter"][:, (2+self.angle_dims+1):(2+self.angle_dims+2), :, :], event_location[:, 4:5, :, :], event_location_mask, scale=2.4, shape=0.29, loc=-0.68, base=0.2, weight=2.0)
        
        loss = hm_loss * self.weights[0] + hw_loss * self.weights[1] \
            + travel_time_loss * self.weights[2] \
            + dist_loss * self.weights[3] \
            + angle_loss * self.weights[4] \
            + depth_loss * self.weights[5] \
            + mag_loss * self.weights[6]
        # print(f"loss {loss}, hm_loss {hm_loss}, hw_loss {hw_loss}, reg_loss {reg_loss}", force=True)
        # return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_hypocenter": reg_loss}
        return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_origin_time": travel_time_loss, "loss_x": dist_loss, "loss_y": angle_loss, "loss_depth": depth_loss, "loss_magnitude": mag_loss}


class EventHead3(nn.Module):
    def __init__(self, 
                 channels=512,
                 hidden_channels=[256, 128, 64, 32, 16],
                 dilations=[1, 4, 8, 16],
                 kernel_size=7,
                 loss_dict: dict = {"hm_loss": "focal", "hw_loss": "wl1", "travel_time_loss": "sl1", "dist_loss": "sl1", "angle_loss": "sl1", "depth_loss": "bl1"},
                 weights=[1.45, 0.018, 0.055],
                 offset_weight=[1, 1], 
                 reg_weight=[0.8,2,20,1.0],
                 angle_coder="trigono",
                 depth_center=8.0,
                 act_layer='gelu'
                 ):
        super().__init__()
        
        self.channels = channels
        reg_weight = [reg_w * weights[-1] for reg_w in reg_weight]
        weights = weights[:-1]+reg_weight
        self.weights = weights
        self.offset_weight = offset_weight
        self.reg_weight = reg_weight
        self.depth_center = depth_center
        
        if angle_coder == "csl":
            self.omega = 5
            self.angle_dims = 360 // self.omega
            self.angle_coder = CSLCoder(self.omega)
        elif angle_coder == "trigono":
            self.angle_dims = 2
            self.angle_coder = TrigonoCoder()
        
        #TODO: subhead layers? dilation?
        self.heatmap = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            Sample(scale_factor=0.5, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2],
                1,
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
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
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
            Sample(scale_factor=0.5, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[2],
                2,
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
            )
        )
        
        self.hypocenter = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[1],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.GELU(),
            Sample(scale_factor=0.5, mode='linear', align_corners=False),
            nn.Conv1d(
                hidden_channels[1],
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.GELU(),
        )
        
        self.travel_time = nn.Sequential(
            nn.Conv1d(
                hidden_channels[2],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=kernel_size,
                dilation=dilations[0],
                padding=((kernel_size - 1) * dilations[0] + 1) // 2,
                #padding_mode="reflect",
            ),
            #nn.LeakyReLU(inplace=True), # the origin time and the distance should be positive
        )
        self.distance = nn.Sequential(
            nn.Conv1d(
                hidden_channels[2],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=kernel_size,
                dilation=dilations[0],
                padding=((kernel_size - 1) * dilations[0] + 1) // 2,
                #padding_mode="reflect",
            ),
            #nn.LeakyReLU(inplace=True), # the origin time and the distance should be positive
        )
        self.azimuth = nn.Sequential(
            nn.Conv1d(
                hidden_channels[2],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                self.angle_dims,
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
            ),
            Sine(),
        )
        # first channel is sin, second channel is cos
        # so the initial bias should be 0 and pi/2
        #self.angle[-2].bias.data[::2].fill_(0)
        #self.angle[-2].bias.data[1::2].fill_(1.5708)
        self.depth = nn.Sequential(
            nn.Conv1d(
                hidden_channels[2],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[3],
                padding=((kernel_size - 1) * dilations[3] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels[3],
                1,
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
            ),
        )


        loss_factory = {"mse": F.mse_loss, "focal": focal_loss, "sfocal": smooth_focal_loss, \
            "l1": l1_reg_loss, "sl1": smoothl1_reg_loss, "wl1": weighted_l1_reg_loss, \
            "vl1": vector_l1_reg_loss, "ml1": modulus_l1_reg_loss, \
            "bl1": balanced_l1_reg_loss, "cross_entropy": cross_entropy_loss}
        self.weights = weights
        self.hm_loss = loss_factory[loss_dict["hm_loss"]]
        self.hw_loss = loss_factory[loss_dict["hw_loss"]]
        self.travel_time_loss = loss_factory[loss_dict["travel_time_loss"]]
        self.dist_loss = loss_factory[loss_dict["dist_loss"]]
        self.angle_loss = loss_factory[loss_dict["angle_loss"]]
        self.depth_loss = loss_factory[loss_dict["depth_loss"]]
        
    def forward(self, features, event_center, event_location, event_location_mask):
        """
        Forward pass of the EventHead module.

        Args:
            features (dict): Dictionary of input features.
            amplitude (torch.Tensor): Tensor of amplitude values.
            event_center (torch.Tensor): Tensor of event center labels.
            event_location (torch.Tensor): Tensor of event location labels.
            event_location_mask (torch.Tensor): Tensor of event location masks.

        Returns:
            tuple: Tuple containing the predicted outputs and the loss values (if in training mode).
        """
        x = features['fusion']
        bt, st, ch, nt = x.shape
        x = x.view(bt*st, ch, nt)
        # downsample feature 2x
        x = F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False)
        
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
        travel_time = self.travel_time(hypocenter)
        distance = self.distance(hypocenter)
        azimuth = self.azimuth(hypocenter)
        depth = self.depth(hypocenter)
        
        #print(f"amplitude scale: avg {amplitude.mean()}, abs avg {torch.abs(amplitude).mean()}, std {amplitude.std()}", force=True)
        #print(f"amplitude scale: min {amplitude.min()}, max {amplitude.max()}", force=True)
        #print(f"feature scale: avg {x.mean()}, abs avg {torch.abs(x).mean()}, std {x.std()}", force=True)
        #print(f"feature scale: min {x.min()}, max {x.max()}", force=True)
        
        hypocenter = torch.cat([travel_time, distance, azimuth, depth], dim=1)
        hypocenter = hypocenter.view(bt, st, hypocenter.shape[1], hypocenter.shape[2])
        hypocenter = hypocenter.permute(0, 2, 3, 1)
        
        if self.training:
            return None, self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
        else:
            loss = {}
            if event_center is not None:
                loss = self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_location, event_location_mask)
            hypocenter = torch.cat([hypocenter[:, :2, :, :], self.angle_coder.decode(hypocenter[:, 2:(2+self.angle_dims), :, :]),
                                    hypocenter[:, (2+self.angle_dims):, :, :]], dim=1)
            return {"event": heatmap, "offset": offset, "hypocenter": hypocenter}, loss
    
    def losses(self, outputs, event_center, event_location, event_location_mask):
        """
        Compute the loss values for the predicted outputs.

        Args:
            outputs (dict): Dictionary of predicted outputs.
            event_center (torch.Tensor): Tensor of event center labels.
            event_location (torch.Tensor): Tensor of event location labels.
            event_location_mask (torch.Tensor): Tensor of event location masks.

        Returns:
            dict: Dictionary containing the loss values.
        """
        #hm_loss = cross_entropy_loss(outputs["event"], event_center)
        #hm_loss = self.hm_loss(outputs["event"], event_center)
        station_mask = (event_location_mask.sum(axis=1) > 0).float().unsqueeze(1)
        hm_loss = self.hm_loss(outputs["event"], event_center, station_mask)
        hw_loss = self.hw_loss(outputs["offset"], event_location[:, 5:, :, :], event_location_mask, weights=self.offset_weight)
        
        travel_time_loss = self.travel_time_loss(outputs["hypocenter"][:, :1, :, :], event_location[:, :1, :, :], event_location_mask)
        dist_loss = self.dist_loss(outputs["hypocenter"][:, 1:2, :, :], event_location[:, 1:2, :, :], event_location_mask)
        angle_loss = self.angle_loss(outputs["hypocenter"][:, 2:(2+self.angle_dims), :, :],# torch.cat([torch.sin(outputs["hypocenter"][:, 2:3, :, :]), torch.cos(outputs["hypocenter"][:, 2:3, :, :]),], dim=1),
                                        #torch.cat([torch.sin(event_location[:, 2:3, :, :]), torch.cos(event_location[:, 2:3, :, :]),], dim=1),
                                        self.angle_coder.encode(event_location[:, 2:3, :, :]),
                                        event_location_mask)
        depth_loss = self.depth_loss(outputs["hypocenter"][:, (2+self.angle_dims):(2+self.angle_dims+1), :, :], event_location[:, 3:4, :, :], event_location_mask, scale=10.80, shape=0.47, loc=-3.90, base=0.5, weight=1.3)
        
        loss = hm_loss * self.weights[0] + hw_loss * self.weights[1] \
            + travel_time_loss * self.weights[2] \
            + dist_loss * self.weights[3] \
            + angle_loss * self.weights[4] \
            + depth_loss * self.weights[5]
        # print(f"loss {loss}, hm_loss {hm_loss}, hw_loss {hw_loss}, reg_loss {reg_loss}", force=True)
        # return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_hypocenter": reg_loss}
        return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_origin_time": travel_time_loss, "loss_x": dist_loss, "loss_y": angle_loss, "loss_depth": depth_loss}

    
class PhaseHead(nn.Module):
    def __init__(self, 
                 channels=512,
                 hidden_channels=[256, 128, 64, 32],
                 dilations=[1, 8, 16, 32],
                 kernel_size=7,
                 weights=[0.9,],
                 ):
        super().__init__()
        
        self.channels = channels
        self.weights = weights
        
        self.phase_1 = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[1],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
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
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.ReLU(inplace=True),
        )
        
        self.conv_out = nn.Conv1d(
            hidden_channels[3],
            3,
            kernel_size=1,
        )
        
    def forward(self, features, targets=None, num_stations=None):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["fusion"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        
        # x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.phase_1(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.phase_2(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.conv_out(x)
        # x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets, num_stations)
        elif targets is not None:
            return {"phase": x}, self.losses(x, targets, num_stations)
        else:
            return {"phase": x}, {}

    def losses(self, inputs, targets, num_stations):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        num=num_stations.sum()
        #mask = num_stations.unsqueeze(1).unsqueeze(1)
        loss = torch.sum(-targets * F.log_softmax(inputs, dim=1) * num_stations.unsqueeze(1).unsqueeze(1))/(num*targets.shape[2])

        return {"loss_phase": loss}


class PhaseHead2(nn.Module):
    def __init__(self, 
                 channels=512,
                 hidden_channels=[256, 128, 64, 32, 16],
                 dilations=[1, 8, 16, 32],
                 kernel_size=7,
                 weights=[0.6,],
                 ):
        super().__init__()
        
        self.channels = channels
        self.weights = weights
        
        self.phase_0 = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
        )
        
        self.phase_1 = nn.Sequential(
            nn.Conv1d(
                hidden_channels[2],
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.ReLU(inplace=True),
        )
        
        self.phase_2 = nn.Sequential(
            nn.Conv1d(
                hidden_channels[3],
                hidden_channels[4],
                kernel_size=kernel_size,
                dilation=dilations[2],
                padding=((kernel_size - 1) * dilations[2] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[4]),
            nn.ReLU(inplace=True),
        )
        
        self.conv_out = nn.Conv1d(
            hidden_channels[4],
            3,
            kernel_size=1,
        )
        
    def forward(self, features, targets=None, num_stations=None):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["fusion"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        
        x = self.phase_0(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.phase_1(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.phase_2(x)
        x = self.conv_out(x)
        # x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets, num_stations)
        elif targets is not None:
            return {"phase": x}, self.losses(x, targets, num_stations)
        else:
            return {"phase": x}, {}

    def losses(self, inputs, targets, num_stations):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        num=num_stations.sum()
        #ps_weights = torch.tensor([1.0, 1.0, 1.1], device=inputs.device)[None, :, None, None].detach()
        #mask = num_stations.unsqueeze(1).unsqueeze(1)
        #loss = torch.sum(-targets * F.log_softmax(inputs, dim=1) * num_stations.unsqueeze(1).unsqueeze(1)*ps_weights)/(num*targets.shape[2])
        loss = torch.sum(-targets * F.log_softmax(inputs, dim=1) * num_stations.unsqueeze(1).unsqueeze(1))/(num*targets.shape[2])

        return {"loss_phase": loss}


class PolarityHead(nn.Module):
    def __init__(self, 
                 channels=512,
                 hidden_channels=[256, 128, 64, 32],
                 dilations=[1, 8, 16, 32],
                 kernel_size=7,
                 weights=[0.2],
                 stem_channels=64,
                 ordinal_regression=True,
                 num_class=3, 
                 train_cutpoints=False, 
                 scale=20.0, 
                 logit_scale=2.0
                ):
        super().__init__()
        
        self.channels = channels
        self.weights = weights
        
        self.polarity_1 = nn.Sequential(
            nn.Conv1d(
                self.channels,
                hidden_channels[2],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
        )
        
        self.polarity_2 = nn.Sequential(
            nn.Conv1d(
                hidden_channels[2]+stem_channels,
                hidden_channels[3],
                kernel_size=kernel_size,
                dilation=dilations[1],
                padding=((kernel_size - 1) * dilations[1] + 1) // 2,
                #padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels[3]),
            nn.ReLU(inplace=True),
        )
        
        self.polarity_out = nn.Conv1d(
            hidden_channels[3],
            1 if ordinal_regression else num_class,
            kernel_size=1,
        )
        if ordinal_regression:
            self.num_classes = num_class
            num_cutpoints = self.num_classes - 1
            self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
            if train_cutpoints:
                self.cutpoints = nn.Parameter(self.cutpoints[None, :, None, None])
            else:
                self.cutpoints = self.cutpoints[None, :, None, None].detach()
            self.logit_scale = logit_scale
            self.losses = self.losses_ordinal
        else:
            self.logit_scale = 1.0
            self.losses = self.losses_softmax
        
    def forward(self, features, polarity_stem, polarity_pick=None, polarity_mask=None):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["fusion"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        
        x = self.polarity_1(x)
        x = F.interpolate(torch.cat([x, polarity_stem.permute(0, 3, 1, 2).contiguous().view(bt*st, -1, nt)], dim=1), 
                          scale_factor=2, mode="linear", align_corners=False)
        x = self.polarity_2(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.polarity_out(x)
        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1) * self.logit_scale

        if self.training:
            return None, self.losses(x, polarity_pick, polarity_mask)
        elif polarity_pick is not None:
            return {"polarity": x}, self.losses(x, polarity_pick, polarity_mask)
        else:
            return {"polarity": x}, {}

    def losses_ordinal(self, inputs, polarity_pick, mask):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        polarity_pick = polarity_pick[:, [2,0,1]].float().contiguous() # NUD -> DNU
        sigmoids = torch.sigmoid(self.cutpoints.to(inputs.device) - inputs)
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

        loss = -torch.sum(polarity_pick * torch.log(likelihoods)*mask) / num
            
        return {"loss_polarity": loss}
    
    def losses_softmax(self, inputs, polarity_pick, mask):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        polarity_pick = polarity_pick.float()
        mask = mask.unsqueeze(1).float()
        num = mask.sum() + 1e-5
        loss = torch.sum(-polarity_pick * F.log_softmax(inputs, dim=1) * mask)/num
    
        return {"loss_polarity": loss}