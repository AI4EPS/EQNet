from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from .unet import UNet
from ._utils import _SimpleSegmentationModel

default_cfgs = {
    "backbone": "unet",
    "head": "unet",
    "preprocess": {
        "local_norm": {
            "flag": True,
            "window": 2048,
            "stride": 256,
        },
    },
    "backbone_cfgs": {
        "in_channels": 3,
    },
    "head_cfgs": {
        "output_channels": 3,
    },
}


class WeightedLoss(_WeightedLoss):
    def __init__(self, weight=None):
        super().__init__(weight=weight)
        self.weight = weight

    def forward(self, input, target):

        log_pred = nn.functional.log_softmax(input, 1)

        if self.weight is not None:
            target = target * self.weight.unsqueeze(1).unsqueeze(1) / self.weight.sum()

        return -(target * log_pred).sum(dim=1).mean()


class UNetHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        kernel_size=(7, 7),
        padding=(3, 3),
        feature_names: str = "out",
        reg: float = 0.1,
    ):
        super().__init__()
        self.out_channels = out_channels
        laplace_kernel = (
            torch.tensor(
                [
                    [1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1],
                ]
            )
            .view(1, 1, 3, 3)
            .expand(-1, out_channels, -1, -1)
            .float()
        )
        self.register_buffer("laplace_kernel", laplace_kernel)
        self.reg = reg
        self.feature_names = feature_names
        self.layers = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, features, targets=None):
        x = features[self.feature_names]
        x = self.layers(x)

        if self.training:
            return None, self.losses(x, targets)

        return x, None

    def losses(self, inputs, targets):

        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        # loss = F.cross_entropy(
        #     inputs, targets, reduction="mean", ignore_index=self.ignore_value
        # )
        # loss = F.kl_div(F.log_softmax(inputs, dim=1), targets, reduction="mean")
        # loss = F.mse_loss(inputs, targets, reduction="mean")
        # loss = F.l1_loss(inputs, targets, reduction="mean")
        loss = torch.sum(-targets * F.log_softmax(inputs, dim=1), dim=1).mean()
        if self.reg > 0:
            loss_laplace = F.conv2d(torch.softmax(inputs, dim=1), self.laplace_kernel, padding=1).abs().mean()
            loss = loss + self.reg * loss_laplace

        return loss


class PhaseNetDAS(_SimpleSegmentationModel):
    pass


def phasenet_das(in_channels=1, out_channels=3, reg=0.0, *args, **kwargs) -> PhaseNetDAS:

    backbone = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=16,
        init_stride=(4, 4),
        kernel_size=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
        pad_input=(1024, 1024),
        local_norm=(2048, 256),
    )

    classifier = UNetHead(in_channels=16, out_channels=out_channels, kernel_size=(7, 7), padding=(3, 3), reg=reg)

    return PhaseNetDAS(backbone, classifier)
