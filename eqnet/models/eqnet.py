import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Dict
from .resnet1d import ResNet, BasicBlock, Bottleneck
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2


def _log_transform(x):
    x = F.relu(x)
    return torch.log(1 + x)


def log_transform(x):
    yp = _log_transform(x)
    yn = _log_transform(-x)
    return yp - yn


class EventDetector(nn.Module):
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
            1,
            kernel_size=kernel_size,
            dilation=dilations[-1],
            padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
            padding_mode="reflect",
        )

    def forward(self, features, targets=None, *args, **kwargs):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["out"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        # x = self.nonlin(self.bn_layers[0](x))
        # x = self.nonlin(x)
        # for conv, bn in zip(self.conv_layers, self.bn_layers[1:]):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.nonlin(bn(conv(x)))
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = x.view(bt, st, x.shape[2])  # chn = 1
        x = x.permute(0, 2, 1)

        if self.training:
            return None, self.losses(x, targets)
        return x, {}

    def losses(self, inputs, targets):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        loss = F.binary_cross_entropy_with_logits(inputs, targets)

        return loss


class PhasePicker(nn.Module):
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
        x = features["out"]
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
        return x, {}

    def losses(self, inputs, targets):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        loss = torch.sum(-targets * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss


class EQNet(nn.Module):
    def __init__(self, backbone="resnet50") -> None:
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
            # self.backbone = ResNet(BasicBlock, [3, 4, 6, 3]) #ResNet34
        elif backbone == "resnet50":
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
        elif backbone == "swin":
            self.backbone = SwinTransformer(
                patch_size=[4, 1],
                embed_dim=16,
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 8, 8],
                window_size=[7, 10],
                stochastic_depth_prob=0.2,
            )
        elif backbone == "swin2":
            self.backbone = SwinTransformerV2(
                patch_size=[4, 1],
                embed_dim=16,
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 8, 8],
                window_size=[7, 10],
                stochastic_depth_prob=0.2,
            )
        else:
            raise ValueError("backbone must be one of 'resnet' or 'swin'")

        self.event_detector = EventDetector()
        self.phase_picker = PhasePicker()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:
        data = batched_inputs["data"].to(self.device)

        if self.training:
            phase_pick = batched_inputs["phase_pick"].to(self.device)
            event_center = batched_inputs["event_center"].to(self.device)
            event_location = batched_inputs["event_location"].to(self.device)
            event_location_mask = batched_inputs["event_location_mask"].to(self.device)
        else:
            phase_pick, event_center, event_location, event_location_mask = None, None, None, None

        if self.backbone_name == "swin2":
            station_location = batched_inputs["station_location"].to(self.device)
            # features = self.backbone({"data": data, "station_location": station_location})
            features = self.backbone(data, station_location)
        else:
            features = self.backbone(data)

        output_phase, loss_phase = self.phase_picker(features, phase_pick)
        output_event, loss_event = self.event_detector(features, event_center, event_location, event_location_mask)

        if self.training:
            return {"loss": loss_phase + loss_event, "loss_phase": loss_phase, "loss_event": loss_event}
        else:
            return {"phase": output_phase, "event": output_event}


def build_model(backbone="resnet", **kargs) -> EQNet:
    return EQNet(backbone=backbone)
