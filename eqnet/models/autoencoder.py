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

def spectrogram(
    x,
    n_fft=128,
    hop_length=32,
    window_fn=torch.hann_window,
    log_transform=False,
    magnitude=False,
    phase=False,
    grad=False,
    discard_zero_freq=False,
    select_freq=False,
    **kwargs,
):
    """
    x: tensor of shape [batch, time_steps]
    n_fft: width of each FFT window (number of frequency bins is n_fft//2 + 1)
    hop_length: interval between consecutive windows
    window_fn: windowing function
    log_transform: if true, apply the function f(x) = log(1 + x) pointwise to the output of the spectrogram
    magnitude: if true, return the magnitude of the complex value in each time-frequency bin
    grad: if true, allow gradients to propagate through the spectrogram transformation
    discard_zero_freq: if true, remove the zero frequency row from the spectrogram
    """
    with torch.set_grad_enabled(grad):
        window = window_fn(n_fft).to(x.device)
        stft = torch.stft(x, n_fft=n_fft, window=window, hop_length=hop_length, center=True, return_complex=False)
        stft = stft[..., : x.shape[-1] // hop_length, :]
        if discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        if select_freq:
            dt = kwargs["dt"]
            fmax = 1 / 2 / dt
            freq = torch.linspace(0, fmax, n_fft)
            idx = torch.arange(n_fft)[(freq > kwargs["fmin"]) & (freq < kwargs["fmax"])]
            stft = stft.narrow(dim=-3, start=idx[0].item(), length=idx.numel())
        if magnitude:
            stft_mag = torch.norm(stft, dim=-1)
            if log_transform:
                stft_mag = torch.log(1 + F.relu(stft_mag))
            if phase:
                components = stft.split(1, dim=-1)
                stft_phase = torch.atan2(components[1].squeeze(-1), components[0].squeeze(-1))
                stft = torch.stack([stft_mag, stft_phase], dim=-1)
            else:
                stft = stft_mag
        else:
            if log_transform:
                stft = torch.log(1 + F.relu(stft)) - torch.log(1 + F.relu(-stft))
        return stft


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, init_features=8, use_stft=False, 
                 encoder_kernel_size = (5, 5), decoder_kernel_size = (5, 5),
                 encoder_stride = (2, 4), decoder_stride = (2, 4),
                 encoder_padding = (2, 2), decoder_padding = (2, 2), **kwargs):
        
        super().__init__()
        self._out_features = ["out"]
        self._out_feature_channels = {"out": out_channels}
        self._out_feature_strides = {"out": 1} 
        self.activation = nn.ReLU(inplace=True)
        self.encoder_stride = encoder_stride
        # self.activation = nn.LeakyReLU(0.2, inplace=True)
        # self.activation = nn.Tanh()
        # self.activation = nn.ELU()

        self.use_stft = use_stft
        if use_stft:
            self.spectrogram = partial(spectrogram, dt=1 / 40, hop_length=8, select_freq=True, fmin=1, fmax=10)

        features = init_features
        if use_stft:
            in_channels *= 2  ## real amd imagenary parts
        self.encoder1 = self._block(
            in_channels, features, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc1"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc1 = nn.Sequential(nn.Linear(kwargs["n_freq"], 1), self.activation)
        self.encoder2 = self._block(
            features, features * 2, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc2"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc2 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2, 1), self.activation)
        self.encoder3 = self._block(
            features * 2, features * 4, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc3 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2 ** 2, 1), self.activation)
        self.encoder4 = self._block(
            features * 4, features * 8, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="enc4"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc4 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2 ** 3, 1), self.activation)

        self.bottleneck = self._block(
            features * 8, features * 16, kernel_size=encoder_kernel_size, padding=encoder_padding, activation=self.activation, name="bottleneck"
        )
        if use_stft:
            self.fc5 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2 ** 4, 1), self.activation)

        # self.upconv4 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 16, features * 8, kernel_size=decoder_kernel_size, stride=decoder_stride),
        #     nn.ReLU(inplace=True),
        # )
        self.upconv4 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder4 = self._block(
            (features * 8) * 3, features * 8, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec4"
        )
        # self.upconv3 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 8, features * 4, kernel_size=decoder_kernel_size, stride=decoder_stride),
        #     nn.ReLU(inplace=True),
        # )
        self.upconv3 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder3 = self._block(
            (features * 4) * 3, features * 4, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec3"
        )
        # self.upconv2 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 4, features * 2, kernel_size=decoder_kernel_size, stride=decoder_stride),
        #     nn.ReLU(inplace=True),
        # )
        self.upconv2 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder2 = self._block(
            (features * 2) * 3, features * 2, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec2"
        )
        # self.upconv1 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 2, features, kernel_size=decoder_kernel_size, stride=decoder_stride),
        #     nn.ReLU(inplace=True),
        # )
        self.upconv1 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder1 = self._block(
            features * 3, features, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=self.activation, name="dec1"
        )

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        
        bt, ch, nt, nsta = x.shape
        # x = normalize_local(x)
        # x = pad_input(x, ratio=self.encoder_stride)

        if self.use_stft:
            sgram = torch.squeeze(x, 3)  # bt, ch, nt, 1
            sgram = self.spectrogram(sgram.view(-1, nt))  # bt*ch, nf, nt, 2
            sgram = sgram.view(bt, ch, *sgram.shape[-3:])  # bt, ch, nf, nt, 2
            components = sgram.split(1, dim=-1)
            sgram = torch.cat([components[1].squeeze(-1), components[0].squeeze(-1)], dim=1)
            sgram = sgram.transpose(-1, -2)
            x = sgram

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        if self.use_stft:
            enc1 = self.fc1(enc1)
        enc3 = self.encoder3(self.pool2(enc2))
        if self.use_stft:
            enc2 = self.fc2(enc2)
        enc4 = self.encoder4(self.pool3(enc3))
        if self.use_stft:
            enc3 = self.fc3(enc3)

        bottleneck = self.bottleneck(self.pool4(enc4))
        if self.use_stft:
            enc4 = self.fc4(enc4)
            bottleneck = self.fc5(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = UNet._cat(enc4, dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = UNet._cat(enc3, dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = UNet._cat(enc2, dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = UNet._cat(enc1, dec1)
        dec1 = self.decoder1(dec1)
        # return torch.sigmoid(self.conv(dec1))
        out = self.conv(dec1)
        result = {}
        result["out"] = F.interpolate(out, size=(nt, nsta), mode='bilinear', align_corners=False)
        if self.use_stft:
            result["sgram"] = sgram
        return result

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
        # loss = F.cross_entropy(
        #     inputs, targets, reduction="mean", ignore_index=self.ignore_value
        # )
        # loss = F.kl_div(F.log_softmax(inputs, dim=1), targets, reduction="mean")
        # loss = F.mse_loss(inputs, targets, reduction="mean")
        loss = F.l1_loss(inputs, targets, reduction="mean")
        # loss = torch.sum(-targets * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss

class AutoEncoder(_SimpleSegmentationModel):
    pass


def autoencoder(
    *args, **kwargs
) -> AutoEncoder:
    
    backbone = UNet(in_channels=1, out_channels=1, init_features=8, use_stft=False, 
                    encoder_kernel_size = (5, 5), decoder_kernel_size = (5, 5),
                    encoder_stride = (2, 4), decoder_stride = (2, 4),
                    encoder_padding = (2, 2), decoder_padding = (2, 2))
    classifier = UNetHead()

    return AutoEncoder(backbone, classifier)