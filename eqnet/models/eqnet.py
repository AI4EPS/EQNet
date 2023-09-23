import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Dict
from .unet import moving_normalize
from .resnet1d import ResNet, BasicBlock, Bottleneck
from .swin_transformer import SwinTransformer
from .swin_transformer_1d import SwinTransformer1D
from .centernet import CenterNetHead, CenterNetHeadV1, smoothl1_reg_loss, weighted_l1_reg_loss, cross_entropy_loss, focal_loss
from .uper_head import UPerNeck, EventHead, PhaseHead


def _log_transform(x):
    x = F.relu(x)
    return torch.log(1 + x)


def log_transform(x):
    yp = _log_transform(x)
    yn = _log_transform(-x)
    return yp - yn


class EventDetector(nn.Module):
    # the first channel should be 8 * embedding_dim in swin transformer
    def __init__(
        self, channels=[128, 64, 32, 16, 8], bn=True, dilations=[1, 2, 4, 8, 16], kernel_size=5, nonlin=nn.ReLU(), weights=[1, 0.015, 0.01], offset_weight=[10, 1], reg_weight=[1,1,1,1]
    ):
        super().__init__()
        self.channels = channels
        self.bn = bn
        self.nonlin = nonlin
        self.weights = weights
        self.offset_weight = offset_weight
        self.reg_weight = reg_weight

        if self.bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:-1]])
            conv_bias = False
        else:
            self.bn_layers = [lambda x: x for c in channels[1:-1]]
            conv_bias = True

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    padding=((kernel_size - 1) * dilations[i] + 1) // 2,
                    padding_mode="reflect",
                    bias=conv_bias,
                )
                for i in range(len(channels) - 2)
            ]
        )
        
        # event center
        self.heatmap = nn.Sequential(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernel_size,
                dilation=dilations[-2],
                padding=((kernel_size - 1) * dilations[-2] + 1) // 2,
                padding_mode="reflect",
                bias=conv_bias,
            ),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels[-1],
                1,
                kernel_size=kernel_size,
                dilation=dilations[-1],
                padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
                padding_mode="reflect",
            )
        )
        self.heatmap[-1].bias.data.fill_(-2.19) # if use the initial value, the loss from 200 to 5 (v2)
        self.offset = nn.Sequential(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernel_size,
                dilation=dilations[-2],
                padding=((kernel_size - 1) * dilations[-2] + 1) // 2,
                padding_mode="reflect",
                bias=conv_bias,
            ),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels[-1],
                2,
                kernel_size=kernel_size,
                dilation=dilations[-1],
                padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
                padding_mode="reflect",
            )
        )
        #'''
        self.hypocenter = nn.Sequential(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernel_size,
                dilation=dilations[-2],
                padding=((kernel_size - 1) * dilations[-2] + 1) // 2,
                padding_mode="reflect",
                bias=conv_bias,
            ),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels[-1],
                #3,
                5,
                kernel_size=kernel_size,
                dilation=dilations[-1],
                padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
                padding_mode="reflect",
            )
        )
        #'''

    def forward(self, features, event_center, event_location, event_location_mask):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["out3"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        # x = self.nonlin(self.bn_layers[0](x))
        # x = self.nonlin(x)
        # for conv, bn in zip(self.conv_layers, self.bn_layers[1:]):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.nonlin(bn(conv(x)))
        #x = self.heatmap(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        heatmap = self.heatmap(x)
        heatmap = heatmap.view(bt, st, heatmap.shape[2])  # chn = 1
        heatmap = heatmap.permute(0, 2, 1)
        
        offset = self.offset(x)
        offset = offset.view(bt, st, offset.shape[1], offset.shape[2])
        offset = offset.permute(0, 2, 3, 1)
        
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
        
        #if self.training:            
        #    return None, self.losses({"event": heatmap, "offset": offset}, event_center, event_location, event_location_mask)
        #elif event_center is not None:
        #    return {"event": heatmap, "offset": offset}, \
        #        self.losses({"event": heatmap, "offset": offset}, event_center, event_location, event_location_mask)
        #else:
        #    return {"event": heatmap, "offset": offset}, {}


    def losses(self, outputs, event_center, event_location, event_location_mask):
        #hm_loss = cross_entropy_loss(outputs["event"], event_center)
        hm_loss = focal_loss(outputs["event"], event_center)
        hw_loss = weighted_l1_reg_loss(outputs["offset"], event_location[:, 5:, :, :], event_location_mask, weights=self.offset_weight)
        
        #reg_loss = smoothl1_reg_loss(outputs["hypocenter"], event_location[:, :3, :, :], event_location_mask)
        reg_loss = weighted_l1_reg_loss(outputs["hypocenter"], event_location[:, :5, :, :], event_location_mask, weights=self.reg_weight)
        
        loss = hm_loss * self.weights[0] + hw_loss * self.weights[1] + reg_loss * self.weights[2]
        # print(f"loss {loss}, hm_loss {hm_loss}, hw_loss {hw_loss}, reg_loss {reg_loss}", force=True)
        return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_hypocenter": reg_loss}


class PhasePicker(nn.Module):
    # the first channel should be 8 * embedding_dim in swin transformer
    def __init__(
        self, channels=[128, 64, 32, 16, 8], bn=True, dilations=[1, 2, 4, 8, 16], kernel_size=5, nonlin=nn.ReLU()
    ):
        super().__init__()
        self.channels = channels
        self.bn = bn
        self.nonlin = nonlin

        if self.bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:]])
            conv_bias = False
        else:
            self.bn_layers = [lambda x: x for c in channels[1:]]
            conv_bias = True

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    padding=((kernel_size - 1) * dilations[i] + 1) // 2,
                    padding_mode="reflect",
                    bias=conv_bias,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.conv_out = nn.Conv1d(
            channels[-1],
            3,
            kernel_size=kernel_size,
            dilation=dilations[-1],
            padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
            padding_mode="reflect",
        )

    def forward(self, features, targets=None):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["out3"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        # x = self.nonlin(self.bn_layers[0](x))
        # x = self.nonlin(x)
        # for conv, bn in zip(self.conv_layers, self.bn_layers[1:]):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.nonlin(bn(conv(x)))
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
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


class EQNet(nn.Module):
    def __init__(self, backbone="resnet50", head="simple", use_station_location=True, moving_norm=(1024, 128),) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.head_name = head
        self.moving_norm = moving_norm
        if backbone == "resnet18":
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
            # self.backbone = ResNet(BasicBlock, [3, 4, 6, 3]) #ResNet34
        elif backbone == "resnet50":
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
        elif backbone == "swin": 
            out_indices = [0, 1, 2, 3] if head == "upernet" else [3]
            self.backbone = SwinTransformer(
                patch_size=[4, 1],
                embed_dim=96,
                #embed_dim=16,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                #num_heads=[2, 4, 8, 8],
                window_size=[7, 10],
                stochastic_depth_prob=0.2,
                block_name="SwinTransformerBlock",
                out_indices=out_indices,
                use_station_location=use_station_location,
            )
        elif backbone == "swin2":
            out_indices = [0, 1, 2, 3] if head == "upernet" else [3]
            self.backbone = SwinTransformer(
                patch_size=[4, 1],
                embed_dim=96,
                #embed_dim=16,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                #num_heads=[2, 4, 8, 8],
                window_size=[8, 10],
                stochastic_depth_prob=0.2,
                block_name="SwinTransformerBlockV2",
                out_indices=out_indices,
                use_station_location=use_station_location,
            )
        elif backbone == "swin_1D": 
            out_indices = [0, 1, 2, 3] if head == "upernet" else [3]
            self.backbone = SwinTransformer1D(
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                stochastic_depth_prob=0.2,
                block_name="SwinTransformerBlock",
                out_indices=out_indices,
            )
        elif backbone == "swin2_1D":
            out_indices = [0, 1, 2, 3] if head == "upernet" else [3]
            self.backbone = SwinTransformer1D(
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=8,
                stochastic_depth_prob=0.2,
                block_name="SwinTransformerBlockV2",
                out_indices=out_indices,
            )
        else:
            raise ValueError("backbone must be one of 'resnet' or 'swin'")

        # config for Swin_T
        # the len of channels and dilations should be 5 when using other config
        # or you need to change the structure of event_detector and phase_picker
        if "swin" in backbone:
            #channels=[768, 128, 32, 16, 8] 
            channels=[768, 256, 64, 16, 8]
            #dilations=[1, 6, 24, 48, 96] 
            dilations=[1, 4, 8, 32, 64]
            neck_channels=256 # TODO: 512 is the original setting
        elif backbone[:6] == "resnet":
            channels=[128, 64, 32, 16, 8]
            dilations=[1, 2, 4, 8, 16]
            
        self.neck = UPerNeck(channels=neck_channels) if head == "upernet" else nn.Sequential()

        if head == "simple":
            self.event_detector = EventDetector(channels=channels, bn=True, dilations=dilations)
        elif head == "centernet":
            if backbone == "swin" or backbone == "swin_1D":
                self.event_detector = CenterNetHeadV1(channels=[768, 256, 64])
            elif backbone == "swin2" or backbone == "swin2_1D":
                self.event_detector = CenterNetHead(channels=[768, 256, 64])
        elif head == "upernet":
            self.event_detector = EventHead(channels=neck_channels)
               
        self.phase_picker = PhaseHead(channels=neck_channels) if head == "upernet" else \
            PhasePicker(channels=channels, bn=True, dilations=dilations)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:
        data = batched_inputs["data"].to(self.device)
        amplitude = batched_inputs["amplitude"].to(self.device)
        assert data.shape[-1]==amplitude.shape[-1], f"data {data.shape}, amplitude {amplitude.shape}"
        data = moving_normalize(data, filter=self.moving_norm[0], stride=self.moving_norm[1])

        if self.training:
            phase_pick = batched_inputs["phase_pick"].to(self.device)
            event_center = batched_inputs["event_center"].to(self.device)
            event_location = batched_inputs["event_location"].to(self.device)
            event_location_mask = batched_inputs["event_location_mask"].to(self.device)
        elif "phase_pick" in batched_inputs.keys(): # validation
            phase_pick = batched_inputs["phase_pick"].to(self.device)
            event_center = batched_inputs["event_center"].to(self.device)
            event_location = batched_inputs["event_location"].to(self.device)
            event_location_mask = batched_inputs["event_location_mask"].to(self.device)
        else:
            phase_pick, event_center, event_location, event_location_mask = None, None, None, None

        if "swin" in self.backbone_name:
            station_location = batched_inputs["station_location"].to(self.device)
            # features = self.backbone({"data": data, "station_location": station_location})
            features = self.backbone(data, station_location)
        else:
            features = self.backbone(data)
            
        features = self.neck(features)

        output_phase, loss_phase = self.phase_picker(features, phase_pick)
        outputs_event, losses_event = self.event_detector(features, event_center, event_location, event_location_mask, amplitude)

        if self.training:
            loss = loss_phase + losses_event["loss"]
            del losses_event["loss"]
            return {"loss": loss, "loss_phase": loss_phase, **losses_event}
        elif phase_pick is not None: # validation
            #output_event = outputs_event["event"]
            loss = loss_phase + losses_event["loss"]
            del losses_event["loss"]
            return {"phase": output_phase, "loss": loss, "loss_phase": loss_phase, **outputs_event, **losses_event}
        else:
            #output_event = outputs_event["event"]
            return {"phase": output_phase, **outputs_event}


def build_model(backbone="resnet50", head="simple", use_station_location=True, **kwargs) -> EQNet:
    return EQNet(backbone=backbone, head=head, use_station_location=use_station_location)
