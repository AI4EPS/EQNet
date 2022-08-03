from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from functools import partial
from ._utils import _SimpleSegmentationModel

class WeightedLoss(_WeightedLoss):
    def __init__(self, weight=None):
        super().__init__(weight=weight)
        self.weight = weight

    def forward(self, input, target):

        log_pred = nn.functional.log_softmax(input, 1)

        if self.weight is not None:
            target = target * self.weight.unsqueeze(1).unsqueeze(1) / self.weight.sum()

        return -(target * log_pred).sum(dim=1).mean()


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


def normalize_local(data, filter=1024+1, stride=512):
    nb, nch, nt, nsta = data.shape
    if (nt % stride == 0):
        pad = max(filter - stride, 0)
    else:
        pad = max(filter - (nt % stride), 0)
    pad1 = pad // 2
    pad2 = pad - pad1
    with torch.no_grad(): 
        data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        mean = F.avg_pool2d(data_, kernel_size=(filter, 1), stride=(stride, 1))
        mean = F.interpolate(mean, size=(nt, nsta), mode="bilinear", align_corners=False)
        data -= mean
        data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        std = F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / (filter ** 0.5)
        std = F.interpolate(std, size=(nt, nsta), mode="bilinear", align_corners=False)
        data /= std
        data = torch.nan_to_num(data)
        data = log_transform(data)
    return data

def pad_input(data, ratio = (2,2)):
    nb, nch, nt, nsta = data.shape
    nt_ = ((nt - 1)//ratio[0]**4 + 1)*ratio[0]**4
    nsta_ = ((nsta - 1)//ratio[1]**4 + 1)*ratio[1]**4
    with torch.no_grad(): 
        data = F.pad(data, (0, nsta_-nsta, 0, nt_-nt), mode="constant")
    return data
    
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, init_features=8, 
                 encoder_kernel_size = (5, 5), decoder_kernel_size = (5, 5),
                 encoder_stride = (4, 4), decoder_stride = (4, 4),
                 encoder_padding = (2, 2), decoder_padding = (2, 2), 
                 use_deconv = True, use_skip = False,
                 **kwargs):
        
        super().__init__()

        self.encoder_stride = encoder_stride
        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.LeakyReLU(inplace=True)
        # self.activation = nn.Tanh()
        # self.activation = nn.ELU()
        # use_deconv = True
        self.use_deconv = use_deconv
        self.use_skip = use_skip
        if self.use_skip:
            channel_ratio = 2
        else:
            channel_ratio = 1

        features = init_features
        self.encoder1 = self._block(
            in_channels, features, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc1"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        self.encoder2 = self._block(
            features, features * 2, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc2"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        self.encoder3 = self._block(
            features * 2, features * 4, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        self.encoder4 = self._block(
            features * 4, features * 8, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc4"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)

        self.bottleneck = self._block(
            features * 8, features * 16, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="bottleneck"
        )

        if use_deconv:
            self.upconv4 = nn.Sequential(
                nn.ConvTranspose2d(features * 16, features * 8, kernel_size=decoder_stride, stride=decoder_stride),
                self.activation,
            )
        else:
            self.upconv4 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder4 = self._block(
            features * 8 * channel_ratio, features * 8, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec4"
        )
        if use_deconv:
            self.upconv3 = nn.Sequential(
                nn.ConvTranspose2d(features * 8, features * 4, kernel_size=decoder_stride, stride=decoder_stride),
                self.activation,
            )
        else:
            self.upconv3 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder3 = self._block(
            features * 4 * channel_ratio, features * 4, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec3"
        )
        if use_deconv:
            self.upconv2 = nn.Sequential(
                nn.ConvTranspose2d(features * 4, features * 2, kernel_size=decoder_stride, stride=decoder_stride),
                self.activation,
            )
        else:
            self.upconv2 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder2 = self._block(
            features * 2 * channel_ratio, features * 2, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec2"
        )
        if use_deconv:
            self.upconv1 = nn.Sequential(
                nn.ConvTranspose2d(features * 2, features, kernel_size=decoder_stride, stride=decoder_stride),
                self.activation,
            )
        else:
            self.upconv1 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder1 = self._block(
            features * channel_ratio, features, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec1"
        )

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        
        bt, ch, nt, nsta = x.shape
        # x = normalize_local(x)
        # x = pad_input(x, ratio=self.encoder_stride)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        if self.use_skip:
            dec4 = UNet._cat(enc4, dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        if self.use_skip:
            dec3 = UNet._cat(enc3, dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        if self.use_skip:
            dec2 = UNet._cat(enc2, dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        if self.use_skip:
            dec1 = UNet._cat(enc1, dec1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)
        return {"out": out}

    @staticmethod
    def _cat(enc, dec): # size encoder > decoder
        diffY = enc.size()[2] - dec.size()[2]
        diffX = enc.size()[3] - dec.size()[3]
        if (diffX < 0) or (diffY < 0):
            print(f"{diffY = }, {diffX = }, {enc.size() = }, {dec.size() = }")
        dec = F.pad(dec, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = torch.cat((enc, dec), dim=1)
        return x1

    @staticmethod
    def _block(in_channels, features, kernel_size=(3, 3), padding=None, activation=nn.ReLU(inplace=True), name=""):
        if padding is None:
            padding = tuple([x//2 for x in kernel_size])
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu1", activation),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu2", activation),
                ]
            )
        )

class UNetHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, targets=None):
        x = features["out"]
        if self.training:
            return None, self.losses(x, targets)
        return x, {}

    def losses(self, inputs, targets):

        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        # loss = F.mse_loss(inputs, targets, reduction="mean")
        loss = F.l1_loss(inputs, targets, reduction="mean")

        return loss

class AutoEncoder(_SimpleSegmentationModel):
    pass


def autoencoder(
    *args, **kwargs
) -> AutoEncoder:
    
    backbone = UNet(in_channels=1, out_channels=1, init_features=8, use_stft=False, 
                    encoder_kernel_size = (3, 3), decoder_kernel_size = (3, 3),
                    encoder_stride = (2, 2), decoder_stride = (2, 2),
                    encoder_padding = (1, 1), decoder_padding = (1, 1),
                    use_skip=True)
    classifier = UNetHead()

    return AutoEncoder(backbone, classifier)