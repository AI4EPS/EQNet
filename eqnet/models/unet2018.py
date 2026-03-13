"""
UNet2018: PyTorch backbone for the original PhaseNet architecture (Zhu & Beroza 2019).

Mirrors the TensorFlow UNet architecture exactly.

TF input shape:  [batch, nt, nx, nc]   (channels-last)
PyTorch input:   [batch, nc, nx, nt]   (channels-first)
TF output shape: [batch, nt, nx, n_class]
PyTorch output:  {"phase": [batch, n_class, nx, nt]}  (raw logits, no softmax)

The backbone contract matches Unet: forward() returns a dict with key "phase".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet2018(nn.Module):
    """
    PyTorch UNet backbone matching the original PhaseNet (TF) architecture.

    Input:  [batch, n_channel, nx, nt]   (nx=1 for single-trace data)
    Output: {"phase": [batch, n_class, nx, nt]}  (raw logits)

    Call F.softmax(output["phase"], dim=1) to get probabilities.

    Args:
        depths: number of encoder/decoder stages (default 5)
        filters_root: base number of filters; doubled at each depth (default 8)
        kernel_size: conv kernel length along the time axis (default 7)
        pool_size: stride for strided conv / ConvTranspose along time (default 4)
        n_channel: number of input channels (default 3)
        n_class: number of output classes (default 3)
    """

    def __init__(
        self,
        depths: int = 5,
        filters_root: int = 8,
        kernel_size: int = 7,
        pool_size: int = 4,
        n_channel: int = 3,
        n_class: int = 3,
    ) -> None:
        super().__init__()

        self.depths = depths
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_channel = n_channel
        self.n_class = n_class

        K = kernel_size
        S = pool_size
        eps = 1e-3  # match TF default BN epsilon=0.001

        # ---- Input block ------------------------------------------------
        self.input_conv = nn.Conv2d(
            n_channel, filters_root,
            kernel_size=(1, K), padding=(0, K // 2), bias=True)
        self.input_bn = nn.BatchNorm2d(filters_root, eps=eps)

        # ---- Down path --------------------------------------------------
        # down_conv1: main conv at each depth (saved as skip connections)
        # down_conv3: strided conv for downsampling (no padding, applied manually)
        self.down_conv1 = nn.ModuleList()
        self.down_bn1 = nn.ModuleList()
        self.down_conv3 = nn.ModuleList()   # depths-1 entries
        self.down_bn3 = nn.ModuleList()     # depths-1 entries

        in_ch = filters_root
        for depth in range(depths):
            filters = int(2 ** depth * filters_root)
            self.down_conv1.append(nn.Conv2d(
                in_ch, filters,
                kernel_size=(1, K), padding=(0, K // 2), bias=False))
            self.down_bn1.append(nn.BatchNorm2d(filters, eps=eps))

            if depth < depths - 1:
                # padding=0: asymmetric padding applied manually in forward()
                # to match TF Conv2d stride='same' behavior along the nt axis.
                self.down_conv3.append(nn.Conv2d(
                    filters, filters,
                    kernel_size=(1, K), stride=(1, S),
                    padding=0, bias=False))
                self.down_bn3.append(nn.BatchNorm2d(filters, eps=eps))
                in_ch = filters

        # ---- Up path ----------------------------------------------------
        # up_conv0: transposed conv (upsampling along nt)
        # up_conv1: regular conv after skip-concat
        self.up_conv0 = nn.ModuleList()
        self.up_bn0 = nn.ModuleList()
        self.up_conv1 = nn.ModuleList()
        self.up_bn1 = nn.ModuleList()

        # Traverse depths from depths-2 down to 0 (same order as TF)
        in_ch = int(2 ** (depths - 1) * filters_root)
        for depth in range(depths - 2, -1, -1):
            filters = int(2 ** depth * filters_root)
            # Transposed conv: in_ch → filters, stride=(1, pool_size)
            # padding=0; we crop manually to match TF 'same' semantics
            self.up_conv0.append(nn.ConvTranspose2d(
                in_ch, filters,
                kernel_size=(1, K), stride=(1, S),
                padding=0, bias=False))
            self.up_bn0.append(nn.BatchNorm2d(filters, eps=eps))

            # After concat: filters (decoder) + filters (encoder skip) = 2*filters
            self.up_conv1.append(nn.Conv2d(
                2 * filters, filters,
                kernel_size=(1, K), padding=(0, K // 2), bias=False))
            self.up_bn1.append(nn.BatchNorm2d(filters, eps=eps))

            in_ch = filters

        # ---- Output layer -----------------------------------------------
        # kernel (1,1), has bias; returns logits (no softmax)
        self.output_conv = nn.Conv2d(
            filters_root, n_class,
            kernel_size=(1, 1), bias=True)

    # ------------------------------------------------------------------
    def _tf_same_conv_pad(self, x):
        """
        Asymmetric padding along nt to replicate TF Conv2d stride='same'.

        TF 'same' for strided conv with stride S and kernel K along nt:
            if T % S == 0:  pad_total = max(K - S, 0)
            else:           pad_total = max(K - T % S, 0)
            left_pad  = pad_total // 2
            right_pad = pad_total - left_pad
        """
        T = x.shape[3]
        K = self.kernel_size
        S = self.pool_size
        r = T % S
        pad_total = (K - S) if r == 0 else max(K - r, 0)
        left = pad_total // 2
        right = pad_total - left
        if left == 0 and right == 0:
            return x
        return F.pad(x, (left, right))

    def _tf_same_deconv_crop(self, x, in_t):
        """
        Crop ConvTranspose output to replicate TF conv2d_transpose padding='same'.

        TF crops the full output (size (T-1)*S+K) to T*S by removing
        floor((K-S)/2) from the left along the nt axis.

        Args:
            x:    ConvTranspose output, shape [B, C, nx, (T-1)*S+K]
            in_t: nt length T before ConvTranspose
        Returns:
            Cropped tensor of nt length T*S
        """
        S = self.pool_size
        K = self.kernel_size
        target = in_t * S
        crop_left = (K - S) // 2
        return x[:, :, :, crop_left: crop_left + target]

    def _crop_and_concat(self, skip, x):
        """
        Crop x (decoder) along nt to match skip (encoder), then concat on channel dim.
        Mirrors TF's crop_and_concat: crops floor(extra/2) from the left.
        """
        t_skip = skip.shape[3]
        t_x = x.shape[3]
        extra = t_x - t_skip
        if extra < 0:
            raise ValueError(f"Decoder nt {t_x} < encoder nt {t_skip}")
        crop_left = extra // 2
        x_crop = x[:, :, :, crop_left: crop_left + t_skip]
        return torch.cat([skip, x_crop], dim=1)

    def forward(self, x: torch.Tensor, drop_rate: float = 0.0) -> dict:
        """
        Args:
            x: [batch, n_channel, nx, nt]
            drop_rate: dropout probability (0 for inference)
        Returns:
            {"phase": [batch, n_class, nx, nt]}  raw logits
        """
        # ---- Input block ------------------------------------------------
        net = self.input_conv(x)
        net = self.input_bn(net)
        net = F.relu(net)
        net = F.dropout(net, p=drop_rate, training=self.training)

        # ---- Down path --------------------------------------------------
        convs = [None] * self.depths
        for depth in range(self.depths):
            net = self.down_conv1[depth](net)
            net = self.down_bn1[depth](net)
            net = F.relu(net)
            net = F.dropout(net, p=drop_rate, training=self.training)

            convs[depth] = net  # save skip connection

            if depth < self.depths - 1:
                net = self._tf_same_conv_pad(net)
                net = self.down_conv3[depth](net)
                net = self.down_bn3[depth](net)
                net = F.relu(net)
                net = F.dropout(net, p=drop_rate, training=self.training)

        # ---- Up path ----------------------------------------------------
        for i, depth in enumerate(range(self.depths - 2, -1, -1)):
            in_t = net.shape[3]

            net = self.up_conv0[i](net)
            net = self._tf_same_deconv_crop(net, in_t)
            net = self.up_bn0[i](net)
            net = F.relu(net)
            net = F.dropout(net, p=drop_rate, training=self.training)

            net = self._crop_and_concat(convs[depth], net)

            net = self.up_conv1[i](net)
            net = self.up_bn1[i](net)
            net = F.relu(net)
            net = F.dropout(net, p=drop_rate, training=self.training)

        # ---- Output -----------------------------------------------------
        logits = self.output_conv(net)
        return {"phase": logits}

    @classmethod
    def from_pretrained(cls, pt_path: str, **kwargs) -> "UNet2018":
        """Load a UNet2018 model from a PyTorch state dict (.pt file).

        Args:
            pt_path: path to the .pt state dict file
            **kwargs: override default architecture hyperparameters

        Returns:
            Loaded model in eval mode
        """
        model = cls(**kwargs)
        model.load_weights_from_pt(pt_path)
        model.eval()
        return model

    def load_weights_from_pt(self, pt_path: str) -> None:
        """Load weights from a pre-converted PyTorch state dict (.pt file)."""
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
        print(f"Loaded weights from: {pt_path}")
