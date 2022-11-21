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



# def normalize_local(data, filter=1024+1, stride=512):
#     nb, nch, nt, nsta = data.shape
#     if (nt % stride == 0):
#         pad = max(filter - stride, 0)
#     else:
#         pad = max(filter - (nt % stride), 0)
#     pad1 = pad // 2
#     pad2 = pad - pad1
#     with torch.no_grad(): 
#         data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
#         mean = F.avg_pool2d(data_, kernel_size=(filter, 1), stride=(stride, 1))
#         mean = F.interpolate(mean, size=(nt, nsta), mode="bilinear", align_corners=False)
#         data -= mean
#         data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
#         std = F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / (filter ** 0.5)
#         std = F.interpolate(std, size=(nt, nsta), mode="bilinear", align_corners=False)
#         data /= std
#         data = torch.nan_to_num(data)
#         data = log_transform(data)
#     return data

def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x

def normalize_local(data, filter=1024, stride=1):

    nb, nch, nt, nsta = data.shape

    if (nt % stride == 0):
        pad = max(filter - stride, 0)
    else:
        pad = max(filter - (nt % stride), 0)
    pad1 = pad // 2
    pad2 = pad - pad1

    with torch.no_grad():

        data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        mean = (F.avg_pool2d(data_, kernel_size=(filter, 1), stride=(stride, 1)))
        data -= mean

        data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        # std = (F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / ((filter * nch) ** 0.5))
        # data /= std
        std = (F.avg_pool2d(torch.abs(data_), kernel_size=(filter, 1), stride=(stride, 1)))
        mask = (std != 0)
        data[mask] = data[mask] / std[mask]

        data = log_transform(data)

    return data

def pad_input(data, min_w=1024, min_h=1024):

    nb, nch, nt, nsta = data.shape
    pad_w = (min_w - nt % min_w) % min_w
    pad_h = (min_h - nsta % min_h) % min_h

    if (pad_w > 0) or (pad_h > 0):
        with torch.no_grad():
            data = F.pad(data, (0, pad_h, 0, pad_w), mode="constant")

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

    def __init__(self, in_channels=1, out_channels=3, init_features=16, use_stft=False, 
                 encoder_kernel_size = (7, 7), decoder_kernel_size = (7, 7),
                 encoder_stride = (4, 4), decoder_stride = (4, 4), activation=nn.ReLU(inplace=True), **kwargs):
        
        super().__init__()
        self._out_features = ["out"]
        self._out_feature_channels = {"out": out_channels}
        self._out_feature_strides = {"out": 1} 

        encoder_padding = tuple([x//2 for x in encoder_kernel_size])
        decoder_padding = tuple([x//2 for x in decoder_kernel_size])

        self.use_stft = use_stft
        if use_stft:
            self.spectrogram = partial(spectrogram, dt=1 / 40, hop_length=8, select_freq=True, fmin=1, fmax=10)

        features = init_features
        if use_stft:
            in_channels *= 2  ## real and imagenary parts
        self.encoder1 = self._block(
            in_channels, features, kernel_size=encoder_kernel_size, stride=encoder_stride, padding=encoder_padding, activation=activation, name="enc1"
        )
        # self.pool1 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc1 = nn.Sequential(nn.Linear(kwargs["n_freq"], 1), activation)
        self.encoder2 = self._block(
            features, features * 2, kernel_size=encoder_kernel_size, stride=encoder_stride, padding=encoder_padding, activation=activation, name="enc2"
        )
        # self.pool2 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc2 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2, 1), activation)
        self.encoder3 = self._block(
            features * 2, features * 4, kernel_size=encoder_kernel_size, stride=encoder_stride, padding=encoder_padding, activation=activation, name="enc3"
        )
        # self.pool3 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc3 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2 ** 2, 1), activation)
        self.encoder4 = self._block(
            features * 4, features * 8, kernel_size=encoder_kernel_size, stride=encoder_stride, padding=encoder_padding, activation=activation, name="enc4"
        )
        # self.pool4 = nn.MaxPool2d(kernel_size=encoder_stride, stride=encoder_stride)
        if use_stft:
            self.fc4 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2 ** 3, 1), activation)

        self.bottleneck = self._block(
            features * 8, features * 16, kernel_size=encoder_kernel_size, stride=encoder_stride, padding=encoder_padding, activation=activation, name="bottleneck"
        )
        if use_stft:
            self.fc5 = nn.Sequential(nn.Linear(kwargs["n_freq"] // 2 ** 4, 1), activation)

        # self.upconv4 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 16, features * 8, kernel_size=decoder_stride, stride=decoder_stride), activation
        # )
        self.upconv4 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder4 = self._block(
            (features * 8) * 3, features * 8, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=activation, name="dec4"
        )
        # self.upconv3 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 8, features * 4, kernel_size=decoder_stride, stride=decoder_stride), activation
        # )
        self.upconv3 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder3 = self._block(
            (features * 4) * 3, features * 4, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=activation, name="dec3"
        )
        # self.upconv2 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 4, features * 2, kernel_size=decoder_stride, stride=decoder_stride), activation
        # )
        self.upconv2 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder2 = self._block(
            (features * 2) * 3, features * 2, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=activation, name="dec2"
        )
        # self.upconv1 = nn.Sequential(
        #     nn.ConvTranspose2d(features * 2, features, kernel_size=decoder_stride, stride=decoder_stride), activation
        # )
        self.upconv1 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.decoder1 = self._block(
            features * 3, features, kernel_size=decoder_kernel_size, padding=decoder_padding, activation=activation, name="dec1"
        )

        self.upconv0 = nn.Upsample(scale_factor=tuple(decoder_stride), mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=decoder_kernel_size, padding=decoder_padding)

    def forward(self, x):
        
        bt, ch, nt, nsta = x.shape
        x = normalize_local(x)
        x = pad_input(x, min_w=1024, min_h=1024)

        if self.use_stft:
            sgram = torch.squeeze(x, 3)  # bt, ch, nt, 1
            sgram = self.spectrogram(sgram.view(-1, nt))  # bt*ch, nf, nt, 2
            sgram = sgram.view(bt, ch, *sgram.shape[-3:])  # bt, ch, nf, nt, 2
            components = sgram.split(1, dim=-1)
            sgram = torch.cat([components[1].squeeze(-1), components[0].squeeze(-1)], dim=1)
            sgram = sgram.transpose(-1, -2)
            x = sgram

        enc1 = self.encoder1(x)
        # enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc1)
        if self.use_stft:
            enc1 = self.fc1(enc1)
        # enc3 = self.pool2(enc2)
        enc3 = self.encoder3(enc2)
        if self.use_stft:
            enc2 = self.fc2(enc2)
        # enc4 = self.pool3(enc3)
        enc4 = self.encoder4(enc3)
        if self.use_stft:
            enc3 = self.fc3(enc3)
        
        # bottleneck = self.pool4(bottleneck)
        bottleneck = self.bottleneck(enc4)
        if self.use_stft:
            enc4 = self.fc4(enc4)
            bottleneck = self.fc5(bottleneck)

        # print(f"{x.shape = }, {enc1.shape = }, {enc2.shape = }, {enc3.shape = }, {enc4.shape = }, {bottleneck.shape = }")

        dec4 = self.upconv4(bottleneck)
        dec4 = self._cat(enc4, dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._cat(enc3, dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._cat(enc2, dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._cat(enc1, dec1)
        dec1 = self.decoder1(dec1)

        # dec0 = F.interpolate(out, size=(nt, nsta), mode='bilinear', align_corners=False)
        dec0 = self.upconv0(dec1)
        out = self.conv(dec0)

        result = {}
        result["out"] = out[:, :, :nt, :nsta]
        if self.use_stft:
            result["sgram"] = sgram
        return result

    @staticmethod
    def _cat(enc, dec): # size encoder > decoder

        # diffY = enc.size()[2] - dec.size()[2]
        # diffX = enc.size()[3] - dec.size()[3]
        # if (diffX < 0) or (diffY < 0):
        #     print(f"{diffY = }, {diffX = }, {enc.size() = }, {dec.size() = }")
        # dec = F.pad(dec, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x1 = torch.cat((enc, dec), dim=1)

        return x1

    @staticmethod
    def _block(in_channels, features, kernel_size=(3, 3), stride=(1,1), padding=None, activation=nn.ReLU(inplace=True), name=""):
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
                            stride=stride,
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
        # loss = F.l1_loss(inputs, targets, reduction="mean")
        loss = torch.sum(-targets * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss

class PhaseNetDAS(_SimpleSegmentationModel):
    pass


def phasenet_das(
    *args, **kwargs
) -> PhaseNetDAS:
    
    backbone = UNet(in_channels=1, out_channels=3, init_features=16, use_stft=False, 
                    encoder_kernel_size = (7, 7), decoder_kernel_size = (7, 7),
                    encoder_stride = (4, 4), decoder_stride = (4, 4))

    classifier = UNetHead()

    return PhaseNetDAS(backbone, classifier)
