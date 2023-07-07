import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


class LogTransform:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log(1.0 + torch.abs(x) + self.eps)


def moving_norm(data, filter=1024, stride=128):
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
        mean = F.interpolate(mean, scale_factor=(stride, 1), mode="bilinear", align_corners=False)[:, :, :nt, :]
        data -= mean

        data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        # std = (F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / ((filter) ** 0.5))
        std = F.avg_pool2d(torch.abs(data_), kernel_size=(filter, 1), stride=(stride, 1))
        std = F.interpolate(std, scale_factor=(stride, 1), mode="bilinear", align_corners=False)[:, :, :nt, :]
        std[std == 0.0] = 1.0
        data = data / std

        data = log_transform(data)

    return data


class MovingNorm:
    def __init__(self, filter=1024, stride=128):
        self.filter = filter
        self.stride = stride

    def __call__(self, data: Tensor) -> Tensor:
        nb, nch, nt, nx = data.shape

        if nt % self.stride == 0:
            pad = max(self.filter - self.stride, 0)
        else:
            pad = max(self.filter - (nt % self.stride), 0)
        pad1 = pad // 2
        pad2 = pad - pad1

        with torch.no_grad():
            data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
            mean = F.avg_pool2d(data_, kernel_size=(self.filter, 1), stride=(self.stride, 1))
            mean = F.interpolate(mean, scale_factor=(self.stride, 1), mode="bilinear", align_corners=False)[
                :, :, :nt, :
            ]
            data -= mean

            data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
            # std = (F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / ((filter) ** 0.5))
            std = F.avg_pool2d(torch.abs(data_), kernel_size=(self.filter, 1), stride=(self.stride, 1))
            std = F.interpolate(std, scale_factor=(self.stride, 1), mode="bilinear", align_corners=False)[:, :, :nt, :]
            std[std == 0.0] = 1.0
            data = data / std

        return data


def padding(data, min_nt=1024, min_nx=None):
    nb, nch, nt, nx = data.shape
    pad_nt = (min_nt - nt % min_nt) % min_nt
    if min_nx is not None:
        pad_nx = (min_nx - nx % min_nx) % min_nx
    else:
        pad_nx = 0

    if (pad_nt > 0) or (pad_nx >= 0):
        with torch.no_grad():
            data = F.pad(data, (0, pad_nx, 0, pad_nt), mode="constant")

    return data


class Padding:
    def __init__(self, min_w=1024, min_h=None):
        self.min_nt = min_w
        self.min_nx = min_h

    def __call__(self, data: Tensor) -> Tensor:
        nb, nch, nt, nx = data.shape
        pad_w = (self.min_nt - nt % self.min_nt) % self.min_nt
        if self.min_nx is not None:
            pad_h = (self.min_nx - nx % self.min_nx) % self.min_nx
        else:
            pad_h = 0

        if (pad_w > 0) or (pad_h >= 0):
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
        stft = torch.stft(
            x,
            n_fft=n_fft,
            window=window,
            hop_length=hop_length,
            center=True,
            return_complex=False,
        )
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


class Spectrogram:
    def __init__(
        self,
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
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.log_transform = log_transform
        self.magnitude = magnitude
        self.phase = phase
        self.grad = grad
        self.discard_zero_freq = discard_zero_freq
        self.select_freq = select_freq
        self.kwargs = kwargs

    def __call__(self, x: Tensor) -> Tensor:
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
        with torch.set_grad_enabled(self.grad):
            window = self.window_fn(self.n_fft).to(x.device)
            stft = torch.stft(
                x,
                n_fft=self.n_fft,
                window=window,
                hop_length=self.hop_length,
                center=True,
                return_complex=False,
            )
            stft = stft[..., : x.shape[-1] // self.hop_length, :]
            if self.discard_zero_freq:
                stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
            if self.select_freq:
                dt = self.kwargs["dt"]
                fmax = 1 / 2 / dt
                freq = torch.linspace(0, fmax, self.n_fft)
                idx = torch.arange(self.n_fft)[(freq > self.kwargs["fmin"]) & (freq < self.kwargs["fmax"])]
                stft = stft.narrow(dim=-3, start=idx[0].item(), length=idx.numel())
            if self.magnitude:
                stft_mag = torch.norm(stft, dim=-1)
                if self.log_transform:
                    stft_mag = torch.log(1 + F.relu(stft_mag))
                if self.phase:
                    components = stft.split(1, dim=-1)
                    stft_phase = torch.atan2(components[1].squeeze(-1), components[0].squeeze(-1))
                    stft = torch.stack([stft_mag, stft_phase], dim=-1)
                else:
                    stft = stft_mag
            else:
                if self.log_transform:
                    stft = torch.log(1 + F.relu(stft)) - torch.log(1 + F.relu(-stft))
            return stft
