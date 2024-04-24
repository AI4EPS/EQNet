from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

default_cfgs = {}


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


def moving_normalize(data, filter=1024, stride=128):
    nb, nch, nt, nx = data.shape

    # if nt % stride == 0:
    #     pad = max(filter - stride, 0)
    # else:
    #     pad = max(filter - (nt % stride), 0)
    # pad1 = pad // 2
    # pad2 = pad - pad1
    padding = filter // 2

    with torch.no_grad():
        # data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        data_ = F.pad(data, (0, 0, padding, padding), mode="reflect")
        mean = F.avg_pool2d(data_, kernel_size=(filter, 1), stride=(stride, 1))
        mean = F.interpolate(mean, scale_factor=(stride, 1), mode="bilinear", align_corners=False)[:, :, :nt, :nx]
        data -= mean

        # data_ = F.pad(data, (0, 0, pad1, pad2), mode="reflect")
        data_ = F.pad(data, (0, 0, padding, padding), mode="reflect")
        # std = (F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / ((filter) ** 0.5))
        std = F.avg_pool2d(torch.abs(data_), kernel_size=(filter, 1), stride=(stride, 1))
        std = F.interpolate(std, scale_factor=(stride, 1), mode="bilinear", align_corners=False)[:, :, :nt, :nx]
        std[std == 0.0] = 1.0
        data = data / std

        # data = log_transform(data)

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


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        init_features=16,
        init_stride=(1, 1),
        kernel_size=(7, 1),
        stride=(4, 1),
        padding=(3, 0),
        moving_norm=(1024, 128),
        add_polarity=False,
        add_event=False,
        add_stft=False,
        log_scale=False,
    ):
        super(UNet, self).__init__()

        features = init_features
        self.add_polarity = add_polarity
        self.add_event = add_event
        self.add_stft = add_stft
        self.moving_norm = moving_norm
        self.log_scale = log_scale

        self.input_conv = self.encoder_block(
            in_channels, features, kernel_size=kernel_size, stride=init_stride, padding=padding, name="enc1"
        )

        #         if use_stft:
        #             self.fc1 = nn.Sequential(nn.Linear(kwargs["n_freq"], 1), activation)
        #         self.encoder2 = self._block(
        #             features,
        #             features * 2,
        #             kernel_size=encoder_kernel_size,
        #             stride=encoder_stride,
        #             padding=encoder_padding,
        #             activation=activation,
        #             name="enc2",
        #         )

        self.encoder12 = self.encoder_block(
            features, features * 2, kernel_size=kernel_size, stride=stride, padding=padding, name="enc2"
        )
        self.encoder23 = self.encoder_block(
            features * 2, features * 4, kernel_size=kernel_size, stride=stride, padding=padding, name="enc3"
        )
        self.encoder34 = self.encoder_block(
            features * 4, features * 8, kernel_size=kernel_size, stride=stride, padding=padding, name="enc4"
        )
        self.encoder45 = self.encoder_block(
            features * 8, features * 16, kernel_size=kernel_size, stride=stride, padding=padding, name="enc5"
        )

        self.upconv54 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "bottle_conv",
                        nn.ConvTranspose2d(
                            in_channels=features * 16,
                            out_channels=features * 8,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            output_padding=padding,
                            bias=False,
                        ),
                    ),
                    ("bottle_norm", nn.BatchNorm2d(num_features=features * 8)),
                    ("bottle_relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.decoder43 = self.decoder_block(
            (features * 8) * 2, features * 4, kernel_size=kernel_size, stride=stride, padding=padding, name="dec4"
        )
        self.decoder32 = self.decoder_block(
            (features * 4) * 2, features * 2, kernel_size=kernel_size, stride=stride, padding=padding, name="dec3"
        )
        self.decoder21 = self.decoder_block(
            (features * 2) * 2, features * 1, kernel_size=kernel_size, stride=stride, padding=padding, name="dec2"
        )

        self.output_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "output_conv",
                        nn.Conv2d(
                            in_channels=features * 2,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    ("output_norm", nn.BatchNorm2d(num_features=features)),
                    ("output_relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        if self.add_polarity:
            self.encoder_polarity = self.encoder_block(
                1, features, kernel_size=kernel_size, stride=(1, 1), padding=padding, name="enc1_polarity"
            )
            self.output_polarity = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "output_polarity_conv",
                            nn.Conv2d(
                                in_channels=features * 2,
                                out_channels=features,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False,
                            ),
                        ),
                        ("output_polarity_norm", nn.BatchNorm2d(num_features=features)),
                        ("output_polarity_relu", nn.ReLU(inplace=True)),
                    ]
                )
            )

        if self.add_event:
            self.output_event = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "output_event_conv",
                            nn.Conv2d(
                                in_channels=features * 4,
                                out_channels=features * 2,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False,
                            ),
                        ),
                        ("output_event_norm", nn.BatchNorm2d(num_features=features * 2)),
                        ("output_event_relu", nn.ReLU(inplace=True)),
                    ]
                )
            )

        if (init_stride[0] > 1) or (init_stride[1] > 1):
            self.output_upsample = nn.Upsample(scale_factor=init_stride, mode="bilinear", align_corners=False)
        else:
            self.output_upsample = None

    def forward(self, x):
        bt, ch, nt, nx = x.shape  # batch, channel, time, station
        x = moving_normalize(x, filter=self.moving_norm[0], stride=self.moving_norm[1])
        if self.log_scale:
            x = log_transform(x)

        #         if self.use_stft:
        #             sgram = torch.squeeze(x, 3)  # bt, ch, nt, 1
        #             sgram = self.spectrogram(sgram.view(-1, nt))  # bt*ch, nf, nt, 2
        #             sgram = sgram.view(bt, ch, *sgram.shape[-3:])  # bt, ch, nf, nt, 2
        #             components = sgram.split(1, dim=-1)
        #             sgram = torch.cat([components[1].squeeze(-1), components[0].squeeze(-1)], dim=1)
        #             sgram = sgram.transpose(-1, -2)
        #             x = sgram

        if self.add_polarity:
            enc_polarity = self.encoder_polarity(x[:, -1:, :, :])  ## last channel is vertical component

        enc1 = self.input_conv(x)
        enc2 = self.encoder12(enc1)
        enc3 = self.encoder23(enc2)
        enc4 = self.encoder34(enc3)
        enc5 = self.encoder45(enc4)

        dec4 = self.upconv54(enc5)

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.decoder43(dec4)
        if self.add_event:
            out_event = self.output_event(dec3)
            if self.output_upsample is not None:
                out_event = self.output_upsample(out_event)
                # out_event = out_event[:, :, : nt // 4, :nx]
            # else:
            #     out_event = out_event[:, :, : nt // 16, :nx]
        else:
            out_event = None
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.decoder32(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.decoder21(dec2)
        if self.add_polarity:
            dec_polarity = torch.cat((dec1, enc_polarity), dim=1)
            out_polarity = self.output_polarity(dec_polarity)
            if self.output_upsample is not None:
                out_polarity = self.output_upsample(out_polarity)
            #     out_polarity = out_polarity[:, :, :nt, :nx]
            # else:
            #     out_polarity = out_polarity[:, :, : nt // 4, :nx]
        else:
            out_polarity = None
        dec1 = torch.cat((dec1, enc1), dim=1)
        out_phase = self.output_conv(dec1)
        if self.output_upsample is not None:
            out_phase = self.output_upsample(out_phase)
        # TODO: Check AGAIN if these part is needed.
        # out_phase = out_phase[:, :, :nt, :nx]

        result = {"phase": out_phase, "polarity": out_polarity, "event": out_event}

        return result

    @staticmethod
    def encoder_block(in_channels, out_channels, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3), name=""):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def decoder_block(in_channels, out_channels, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3), name=""):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels // 2,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", nn.BatchNorm2d(num_features=in_channels // 2)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.ConvTranspose2d(
                            in_channels=in_channels // 2,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            output_padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
