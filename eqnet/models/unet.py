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
        std = torch.mean(std, dim=(1,), keepdim=True)
        std = F.interpolate(std, scale_factor=(stride, 1), mode="bilinear", align_corners=False)[:, :, :nt, :nx]
        std[std == 0.0] = 1.0
        data = data / std

        # data = log_transform(data)

    return data


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=128,
        hop_length=4,
        window_fn=torch.hann_window,
        log_transform=True,
        normalize=True,
        magnitude=True,
        phase=False,
        grad=False,
        discard_zero_freq=False,
        select_freq=False,
        **kwargs,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.log_transform = log_transform
        self.normalize = normalize
        self.magnitude = magnitude
        self.phase = phase
        self.grad = grad
        self.discard_zero_freq = discard_zero_freq
        self.select_freq = select_freq
        self.window = self.register_buffer("window", window_fn(n_fft))
        self.window_fn = window_fn
        if select_freq:
            dt = kwargs["dt"]
            fmax = 1.0 / 2.0 / dt
            freq = torch.linspace(0, fmax, n_fft)
            idx = torch.arange(n_fft)[(freq > kwargs["fmin"]) & (freq < kwargs["fmax"])]
            self.freq_start = idx[0].item()
            self.freq_length = idx.numel()

    def forward(self, x):
        # window = self.window_fn(self.n_fft).to(x.device)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            hop_length=self.hop_length,
            center=True,
            return_complex=False,
        )
        stft = stft[..., : x.shape[-1] // self.hop_length, :]  # bt* ch, nf, nt, 2
        if self.discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        if self.select_freq:
            stft = stft.narrow(dim=-3, start=self.freq_start, length=self.freq_length)
        if self.magnitude:
            stft_mag = torch.norm(stft, dim=-1, keepdim=True)  # bt* ch, nf, nt
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
        if self.normalize:
            vmax = torch.max(torch.abs(stft), dim=-3, keepdim=True)[0]
            vmax[vmax == 0.0] = 1.0
            stft = stft / vmax
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
        spectrogram=False,
    ):
        super(UNet, self).__init__()

        features = init_features
        self.add_polarity = add_polarity
        self.add_event = add_event
        self.add_stft = add_stft
        self.moving_norm = moving_norm
        self.log_scale = log_scale
        self.spectrogram = spectrogram
        if self.spectrogram:
            self.n_fft = 64
            self.stft = STFT(
                n_fft=self.n_fft,
                hop_length=stride[0],
                window_fn=torch.hann_window,
                log_transform=self.log_scale,
                magnitude=True,
                phase=False,
                discard_zero_freq=True,
            )
            self.fc_freq = nn.Sequential(nn.Linear(self.n_fft // 2, 1), nn.ReLU())
            kernel_size_tf = (kernel_size[0], 3)  # 3 for frequency
            padding_tf = (padding[0], 1)

            self.encoder12_tf = self.encoder_block(
                in_channels,
                features * 2,
                kernel_size=kernel_size_tf,
                stride=(1, 1),
                padding=padding_tf,
                name="enc2_tf",
            )
            self.encoder23_tf = self.encoder_block(
                features * 2,
                features * 4,
                kernel_size=kernel_size_tf,
                stride=stride,
                padding=padding_tf,
                name="enc3_tf",
            )
            self.encoder34_tf = self.encoder_block(
                features * 4,
                features * 8,
                kernel_size=kernel_size_tf,
                stride=stride,
                padding=padding_tf,
                name="enc4_tf",
            )
            self.encoder45_tf = self.encoder_block(
                features * 8,
                features * 16,
                kernel_size=kernel_size_tf,
                stride=stride,
                padding=padding_tf,
                name="enc5_tf",
            )
        self.input_conv = self.encoder_block(
            in_channels, features, kernel_size=kernel_size, stride=init_stride, padding=padding, name="enc1"
        )
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

        extra_features = 1 if self.spectrogram else 0
        self.upconv54 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "bottle_conv",
                        nn.ConvTranspose2d(
                            in_channels=features * 16 * (1 + extra_features),
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
            (features * 8) * (2 + extra_features),
            features * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            name="dec4",
        )
        self.decoder32 = self.decoder_block(
            (features * 4) * (2 + extra_features),
            features * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            name="dec3",
        )
        self.decoder21 = self.decoder_block(
            (features * 2) * (2 + extra_features),
            features * 1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            name="dec2",
        )
        self.output_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "output_conv1",
                        nn.Conv2d(
                            in_channels=features * 2,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    ("output_norm1", nn.BatchNorm2d(num_features=features)),
                    ("output_relu1", nn.ReLU(inplace=True)),
                    (
                        "output_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    ("output_norm2", nn.BatchNorm2d(num_features=features)),
                    ("output_relu2", nn.ReLU(inplace=True)),
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
                            "output_polarity_conv1",
                            nn.Conv2d(
                                in_channels=features * 2,
                                out_channels=features,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False,
                            ),
                        ),
                        ("output_polarity_norm1", nn.BatchNorm2d(num_features=features)),
                        ("output_polarity_relu1", nn.ReLU(inplace=True)),
                        # (
                        #     "output_polarity_conv2",
                        #     nn.Conv2d(
                        #         in_channels=features,
                        #         out_channels=features,
                        #         kernel_size=kernel_size,
                        #         padding=padding,
                        #         bias=False,
                        #     ),
                        # ),
                        # ("output_polarity_norm2", nn.BatchNorm2d(num_features=features)),
                        # ("output_polarity_relu2", nn.ReLU(inplace=True)),
                    ]
                )
            )

        if self.add_event:
            self.output_event = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "output_event_conv1",
                            nn.Conv2d(
                                in_channels=features * 4,
                                out_channels=features,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False,
                            ),
                        ),
                        ("output_event_norm1", nn.BatchNorm2d(num_features=features)),
                        ("output_event_relu1", nn.ReLU(inplace=True)),
                        # (
                        #     "output_event_conv2",
                        #     nn.Conv2d(
                        #         in_channels=features * 2,
                        #         out_channels=features,
                        #         kernel_size=kernel_size,
                        #         padding=padding,
                        #         bias=False,
                        #     ),
                        # ),
                        # ("output_event_norm2", nn.BatchNorm2d(num_features=features)),
                        # ("output_event_relu2", nn.ReLU(inplace=True)),
                    ]
                )
            )

        if (init_stride[0] > 1) or (init_stride[1] > 1):
            self.output_upsample = nn.Upsample(scale_factor=init_stride, mode="bilinear", align_corners=False)
        else:
            self.output_upsample = None

    def forward(self, x):
        bt, ch, nt, nx = x.shape  # batch, channel, time, station
        if self.spectrogram:
            assert nx == 1
            sgram = torch.squeeze(x, -1)  # bt, ch, nt, 1
            sgram = self.stft(sgram.view(-1, nt))  # bt*ch*nx, nf, nt, 2
            sgram = sgram.view(bt, ch, *sgram.shape[-3:])  # bt, ch, nf, nt, 2
            # components = sgram.split(1, dim=-1)
            # sgram = torch.cat([components[1].squeeze(-1), components[0].squeeze(-1)], dim=1)  # bt, ch*2, nf, nt
            sgram = torch.squeeze(sgram, -1)  # bt, ch, nt, nf
            sgram = sgram.transpose(-1, -2)  # bt, ch*2/ch, nt, nf
            enc2_tf = self.encoder12_tf(sgram)
            enc3_tf = self.encoder23_tf(enc2_tf)
            enc4_tf = self.encoder34_tf(enc3_tf)
            enc5_tf = self.encoder45_tf(enc4_tf)
            enc2_tf = self.fc_freq(enc2_tf)
            enc3_tf = self.fc_freq(enc3_tf)
            enc4_tf = self.fc_freq(enc4_tf)
            enc5_tf = self.fc_freq(enc5_tf)

        x = moving_normalize(x, filter=self.moving_norm[0], stride=self.moving_norm[1])
        if self.log_scale:
            x = log_transform(x)

        if self.add_polarity:
            z = x[:, -1:, :, :]  ## last channel is vertical component
            z = log_transform(z)
            enc_polarity = self.encoder_polarity(z)

        enc1 = self.input_conv(x)
        enc2 = self.encoder12(enc1)
        enc3 = self.encoder23(enc2)
        enc4 = self.encoder34(enc3)
        enc5 = self.encoder45(enc4)

        if self.spectrogram:
            enc2 = torch.cat((enc2, enc2_tf), dim=1)
            enc3 = torch.cat((enc3, enc3_tf), dim=1)
            enc4 = torch.cat((enc4, enc4_tf), dim=1)
            enc5 = torch.cat((enc5, enc5_tf), dim=1)

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
        if self.spectrogram:
            result["spectrogram"] = sgram

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
                        nn.Conv2d(
                            in_channels=in_channels // 2,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                    (name + "_upsample", nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)),
                    # (
                    #     name + "_conv2",
                    #     nn.ConvTranspose2d(
                    #         in_channels=in_channels // 2,
                    #         out_channels=out_channels,
                    #         kernel_size=kernel_size,
                    #         stride=stride,
                    #         padding=padding,
                    #         output_padding=padding,
                    #         bias=False,
                    #     ),
                    # ),
                    # (name + "_norm2", nn.BatchNorm2d(num_features=out_channels)),
                    # (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
