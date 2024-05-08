from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .resnet1d import BasicBlock, Bottleneck, ResNet
from .unet import UNet

default_cfg = {
    "backbone": "unet",
    "head": "unet",
    "preprocess": {
        "moving_norm": {
            "flag": True,
            "window": 1024,
            "stride": 128,
        },
        "padding": {
            "flag": True,
            "nt": 1024,
            "nx": 1,
        },
    },
    "backbone_cfg": {
        "in_channels": 3,
    },
    "head_cfg": {
        "output_channels": 3,
    },
}


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
        x = features["phase"]
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

    def forward(self, features, targets=None, mask=None):
        x = features["phase"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)

        x = self.layers(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)

        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets, mask)
        return x, {}

    def losses(self, inputs, targets, mask=None):
        inputs = inputs.float()

        if self.out_channels == 1:
            loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=mask)
        else:
            loss = torch.sum(-targets.float() * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
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
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
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
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=(7, 1), padding=(3, 0), feature_names: str = "phase"
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_names = feature_names
        self.layers = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        )
        # self.layers = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=False
        #     ),
        #     nn.BatchNorm2d(num_features=in_channels),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0)),
        # )

    def forward(self, features, targets=None, mask=None):
        x = features[self.feature_names]
        x = self.layers(x)

        if self.training:
            return x, self.losses(x, targets, mask)
        else:
            if targets is not None:  ## for validation, but breaks for torch.compile
                return x, self.losses(x, targets, mask)
            return x, 0.0

    def losses(self, inputs, targets, mask=None):
        """
        targets: (batch, channel, time, station) or (batch, 1, time, station)
        """
        inputs = inputs.float()

        if mask is None:
            if self.out_channels == 1:
                loss = F.binary_cross_entropy_with_logits(inputs, targets)
            else:
                loss = torch.sum(-targets.float() * F.log_softmax(inputs, dim=1), dim=1).mean()
        else:
            mask_sum = mask.sum()
            if mask_sum == 0.0:
                mask_sum = 1.0
            if self.out_channels == 1:
                loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=mask, reduction="sum") / mask_sum
            else:
                loss = torch.sum(-targets.float() * F.log_softmax(inputs, dim=1) * mask) / mask_sum

        return loss


class EventHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(7, 1),
        padding=(3, 0),
        scaling=1000.0,
        feature_names: str = "event",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_names = feature_names
        self.scaling = scaling
        # self.layers = nn.Conv2d(
        #     in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        # )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
        )

    def forward(self, features, targets=None, mask=None):
        x = features[self.feature_names]
        x = self.layers(x) * self.scaling

        if self.training:
            return x, self.losses(x, targets, mask)
        else:
            if targets is not None:  ## for validation, but breaks for torch.compile
                return x, self.losses(x, targets, mask)
            return x, 0.0

    def losses(self, inputs, targets, mask=None):
        inputs = inputs.float()

        if mask is None:
            loss = F.mse_loss(inputs, targets) / self.scaling
        else:
            mask_sum = mask.sum()
            if mask_sum == 0.0:
                mask_sum = 1.0
            loss = (
                F.l1_loss(inputs * mask, targets * mask, reduction="sum") / mask_sum / self.scaling
            )  # trigger warning for mps
            # loss = torch.sum(torch.abs(inputs - targets) * mask, dim=(1, 2, 3)).mean() / mask_sum

        return loss


class PhaseNet(nn.Module):
    def __init__(
        self,
        backbone="unet",
        log_scale=True,
        add_polarity=False,
        add_event=False,
        event_center_loss_weight=1.0,
        event_time_loss_weight=1.0,
        polarity_loss_weight=1.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.add_event = add_event
        self.add_polarity = add_polarity
        self.event_center_loss_weight = event_center_loss_weight
        self.event_time_loss_weight = event_time_loss_weight
        self.polarity_loss_weight = polarity_loss_weight

        if backbone == "resnet18":
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
        elif backbone == "resnet50":
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
        elif backbone == "unet":
            self.backbone = UNet(log_scale=log_scale, add_polarity=add_polarity, add_event=add_event)
        else:
            raise ValueError("backbone only supports resnet18, resnet50, or unet")

        if backbone == "unet":
            self.phase_picker = UNetHead(16, 3, feature_names="phase")
            if self.add_event:
                self.event_detector = UNetHead(32, 1, feature_names="event")
                self.event_timer = EventHead(32, 1, feature_names="event")
            if self.add_polarity:
                self.polarity_picker = UNetHead(16, 3, feature_names="polarity")
                # self.polarity_picker = UNetHead(16, 1, feature_names="polarity")
        else:
            self.phase_picker = DeepLabHead(128, 3, scale_factor=32)
            if self.add_event:
                self.event_detector = DeepLabHead(128, 1, scale_factor=2)
                self.event_timer = EventHead(128, 1, scale_factor=2)
            if self.add_polarity:
                self.polarity_picker = DeepLabHead(128, 1, scale_factor=32)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:
        data = batched_inputs["data"].to(self.device)

        phase_pick = batched_inputs["phase_pick"].to(self.device) if "phase_pick" in batched_inputs else None
        event_center = batched_inputs["event_center"].to(self.device) if "event_center" in batched_inputs else None
        event_time = batched_inputs["event_time"].to(self.device) if "event_time" in batched_inputs else None
        event_mask = batched_inputs["event_mask"].to(self.device) if "event_mask" in batched_inputs else None
        polarity = batched_inputs["polarity"].to(self.device) if "polarity" in batched_inputs else None
        polarity_mask = batched_inputs["polarity_mask"].to(self.device) if "polarity_mask" in batched_inputs else None

        if self.backbone_name == "swin2":
            station_location = batched_inputs["station_location"].to(self.device)
            features = self.backbone(data, station_location)
        else:
            features = self.backbone(data)
        # features: (batch, station, channel, time)

        output = {"loss": 0.0}
        output_phase, loss_phase = self.phase_picker(features, phase_pick)
        output["phase"] = output_phase
        output["loss_phase"] = loss_phase
        output["loss"] += loss_phase
        if self.add_event:
            output_event_center, loss_event_center = self.event_detector(features, event_center)
            output["event_center"] = output_event_center
            output["loss_event_center"] = loss_event_center * self.event_center_loss_weight
            output["loss"] += loss_event_center * self.event_center_loss_weight
            output_event_time, loss_event_time = self.event_timer(features, event_time, mask=event_mask)
            output["event_time"] = output_event_time
            output["loss_event_time"] = loss_event_time * self.event_time_loss_weight
            output["loss"] += loss_event_time * self.event_time_loss_weight
        if self.add_polarity:
            output_polarity, loss_polarity = self.polarity_picker(features, polarity, mask=polarity_mask)
            output["polarity"] = output_polarity
            output["loss_polarity"] = loss_polarity * self.polarity_loss_weight
            output["loss"] += loss_polarity * self.polarity_loss_weight

        return output


def build_model(
    backbone="unet",
    log_scale=True,
    *args,
    **kwargs,
) -> PhaseNet:
    return PhaseNet(backbone=backbone, log_scale=log_scale)
