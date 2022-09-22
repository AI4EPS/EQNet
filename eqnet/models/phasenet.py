from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .resnet1d import BasicBlock, Bottleneck, ResNet
from .unet1d import UNet


class FCNHead(nn.Module):
    # class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.channels = out_channels
        self.layers = nn.Sequential(
            *[
                nn.Conv1d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.Conv1d(inter_channels, out_channels, 1),
            ]
        )

        # super(FCNHead, self).__init__(*layers)

    def forward(self, features, targets=None):

        x = features["out"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)

        x = self.layers(x)
        x = F.interpolate(x, scale_factor=32, mode="linear", align_corners=False)

        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets)
        return x, {}

    def losses(self, inputs, targets):
        inputs = inputs.float()

        if self.out_channels == 1:
            loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            loss = torch.sum(-targets.float() * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss


class DeepLabHead(nn.Module):
    # class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor=1) -> None:
        super(DeepLabHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.layers = nn.Sequential(
            *[
                # super().__init__(
                ASPP(in_channels, [12, 24, 36]),
                nn.Conv1d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, out_channels, 1),
            ]
        )

    def forward(self, features, targets=None):

        x = features["out"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)

        x = self.layers(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)

        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets)
        return x, {}

    def losses(self, inputs, targets):
        inputs = inputs.float()

        if self.out_channels == 1:
            loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            loss = torch.sum(-targets.float() * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-1]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="linear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv1d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class UNetHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, feature_names: str = "out") -> None:
        super().__init__()
        self.out_channels = out_channels
        self.layers = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.feature_names = feature_names

    def forward(self, features, targets=None):
        x = features[self.feature_names]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)

        x = self.layers(x)

        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)  # batch, channel, time,  station

        if self.training:
            return None, self.losses(x, targets)

        return x, {}

    def losses(self, inputs, targets):

        inputs = inputs.float()

        if self.out_channels == 1:
            loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            loss = torch.sum(-targets.float() * F.log_softmax(inputs, dim=1), dim=1).mean()

        # # loss = F.kl_div(F.log_softmax(inputs, dim=1), targets, reduction="mean")
        # # loss = F.mse_loss(inputs, targets, reduction="mean")
        # # loss = F.l1_loss(inputs, targets, reduction="mean")
        # # loss = torch.sum(-targets * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss


class PhaseNet(nn.Module):
    def __init__(self, backbone="resnet50") -> None:
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
            # self.backbone = ResNet(BasicBlock, [3, 4, 6, 3]) #ResNet34
        elif backbone == "resnet50":
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
        elif backbone == "unet":
            self.backbone = UNet()
        else:
            raise ValueError("backbone must be one of 'resnet' or 'swin'")

        if backbone == "unet":
            self.phase_picker = UNetHead(16, 3, feature_names="out")
            self.event_detector = UNetHead(256, 1, feature_names="res5")
        else:
            self.phase_picker = DeepLabHead(128, 3, scale_factor=32)
            self.event_detector = DeepLabHead(128, 1, scale_factor=2)
            # self.phase_picker = FCNHead(128, 3)
            # self.event_detector = FCNHead(128, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:

        waveform = batched_inputs["waveform"].to(self.device)

        if self.training:
            phase_pick = batched_inputs["phase_pick"].to(self.device)
            center_heatmap = batched_inputs["center_heatmap"].to(self.device)
            event_location = batched_inputs["event_location"].to(self.device)
            event_location_mask = batched_inputs["event_location_mask"].to(self.device)
        else:
            phase_pick, center_heatmap, event_location, event_location_mask = None, None, None, None

        if self.backbone_name == "swin2":
            station_location = batched_inputs["station_location"].to(self.device)
            # features = self.backbone({"waveform": waveform, "station_location": station_location})
            features = self.backbone(waveform, station_location)
        else:
            features = self.backbone(waveform)
        # features: (batch, station, channel, time)

        output_phase, loss_phase = self.phase_picker(features, phase_pick)
        output_event, loss_event = self.event_detector(features, center_heatmap)

        # print(f"{waveform.shape = }")
        # print(f"{phase_pick.shape = }")
        # print(f"{center_heatmap.shape = }")
        # raise
        # print(f"{output_phase.shape = }")
        # print(f"{output_event.shape = }")

        if self.training:
            return loss_phase + loss_event
            # return loss_phase
            # return loss_event
        else:
            return {"phase": output_phase, "event": output_event}

        # output_phase = self.phase_picker(features["out"].squeeze(1))
        # # output_phase: (bt, chn, time)
        # output_phase = F.interpolate(output_phase, scale_factor=32, mode="linear", align_corners=False)

        # output_event = self.event_detector(features["out"].squeeze(1))
        # # output_event: (bt, time)
        # output_event = F.interpolate(output_event, scale_factor=2, mode="linear", align_corners=False).squeeze(1)

        # if self.training:
        #     # phase_pick: (batch, channel, time, station)
        #     phase_pick = phase_pick.squeeze(-1)  # one station
        #     loss_phase = torch.sum(-phase_pick.float() * F.log_softmax(output_phase, dim=1), dim=1).mean()

        #     center_heatmap = center_heatmap.squeeze(-1)  # one station
        #     output_event = output_event.float()  # https://github.com/pytorch/pytorch/issues/48163
        #     loss_event = F.binary_cross_entropy_with_logits(output_event, center_heatmap)

        #     return loss_phase + loss_event
        # else:
        #     return {"phase": output_phase, "event": output_event}


def phasenet(
    backbone: ResNet,
) -> PhaseNet:

    return PhaseNet(backbone)
