from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


def normalize_local(data, filter=1024, stride=1):

    nb, nch, nt, nx = data.shape

    if nt % stride == 0:
        pad = max(filter - stride, 0)
    else:
        pad = max(filter - (nt % stride), 0)
    pad1 = pad // 2
    pad2 = pad - pad1

    with torch.no_grad():

        data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        mean = F.avg_pool2d(data_, kernel_size=(filter, 1), stride=(stride, 1))
        data -= mean

        data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        # std = (F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / ((filter * nch) ** 0.5))
        # data /= std
        std = F.avg_pool2d(torch.abs(data_), kernel_size=(filter, 1), stride=(stride, 1))
        std[std == 0.0] = 1.0
        data = data / std

        data = log_transform(data)

    return data


def pad_input(data, min_w=1024):

    nb, nch, nt, nx = data.shape
    pad_w = (min_w - nt % min_w) % min_w

    if pad_w > 0:
        with torch.no_grad():
            data = F.pad(data, (0, 0, 0, pad_w), mode="constant")

    return data


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        init_features=16,
        use_stft=False,
        kernel_size=(7, 1),
        stride=(4, 1),
        padding=(3, 0),
        use_polarity=False,
    ):
        super(UNet, self).__init__()

        features = init_features
        self.use_polarity = use_polarity
        if self.use_polarity:
            self.encoder1_polarity = self._block(
                1, features, kernel_size=kernel_size, stride=stride, padding=padding, name="enc1_polarity"
            )
            self.decoder1_polarity = self._block(
                features * 4, features, kernel_size=kernel_size, padding=padding, name="dec1_polarity"
            )
            self.upconv0_polarity = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)

        self.encoder1 = self._block(
            in_channels, features, kernel_size=kernel_size, stride=stride, padding=padding, name="enc1"
        )
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(
            features, features * 2, kernel_size=kernel_size, stride=stride, padding=padding, name="enc2"
        )
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = self._block(
            features * 2, features * 4, kernel_size=kernel_size, stride=stride, padding=padding, name="enc3"
        )
        # self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = self._block(
            features * 4, features * 8, kernel_size=kernel_size, stride=stride, padding=padding, name="enc4"
        )
        # self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = self._block(
            features * 8, features * 16, kernel_size=kernel_size, stride=stride, padding=padding, name="bottleneck"
        )

        self.upconv4 = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        # self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(
            (features * 8) * 3, features * 8, kernel_size=kernel_size, padding=padding, name="dec4"
        )
        self.upconv3 = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        # self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(
            (features * 4) * 3, features * 4, kernel_size=kernel_size, padding=padding, name="dec3"
        )
        self.upconv2 = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        # self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(
            (features * 2) * 3, features * 2, kernel_size=kernel_size, padding=padding, name="dec2"
        )
        self.upconv1 = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        # self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 3, features, kernel_size=kernel_size, padding=padding, name="dec1")

        self.upconv0 = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        # self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):

        bt, ch, nt, st = x.shape  # batch, channel, time, station
        x = pad_input(x)

        if self.use_polarity:
            enc1_polarity = self.encoder1_polarity(x[:, -1:, :, :])  ## last channel is vertical component
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)

        if self.use_polarity:
            dec1_polarity = torch.cat((dec1, enc1, enc1_polarity), dim=1)
            dec1_polarity = self.decoder1_polarity(dec1_polarity)
            dec0_polarity = self.upconv0_polarity(dec1_polarity)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec0 = self.upconv0(dec1)

        out = dec0[:, :, :nt, :]
        out_event = dec2[:, :, :nt//16, :]
        out_polarity = None if not self.use_polarity else dec0_polarity[:, :, :nt, :]

        result = {"out": out, "polarity": out_polarity, "event": out_event}

        return result

    @staticmethod
    def _block(in_channels, features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), name=""):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
