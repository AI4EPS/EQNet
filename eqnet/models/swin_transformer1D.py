from functools import partial
from typing import Any, Callable, List, Optional, Dict
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ._utils import _ovewrite_named_param
from torch.fx import wrap

__all__ = [
    "SwinTransformer",
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]

### from ..ops.misc import MLP, Permute
class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)

class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.
    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


### from ..ops.stochastic_depth import StochasticDepth
@torch.fx.wrap
def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``
    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s

############################################

@torch.fx.wrap
def _patch_merging_pad(x):
    H, W, _ = x.shape[-3:]
    # x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    #ATTENTION: The pad function pads the last dimension first
    #so if we only want to deal with the time dimension, we should use pad((0, 0, 0, 0, 0, H % 2))
    x = F.pad(x, (0, 0, 0, 0, 0, H % 2)) ## for seismic time series
    ## for seismic time series
    x0 = x[..., 0::2, :, :]  # ... H/2 W C, H is nt
    x1 = x[..., 1::2, :, :]  # ... H/2 W C
    x = torch.cat([x0, x1], -1)  # ... H/2 W 2*C 
    
    return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(2 * dim)

    def forward(self, meta: Dict):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W, 2*C]
        """
        x, loc = meta["x"], meta["loc"]
        x = _patch_merging_pad(x)
        x = self.norm(x)

        return {"x": x, "loc": loc}

@torch.fx.wrap
def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: int,
    num_heads: int,
    shift_size: int,
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        shift_size (int): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape # batch, nt, nsta, nhead
    # pad feature maps to multiples of window size
    #pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(input, (0, 0, 0, 0, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    # shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size >= pad_H:
        shift_size = 0
    # if window_size[1] >= pad_W:
    #     shift_size[1] = 0

    # cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(-shift_size, 0), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size) # * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size, window_size, 1, pad_W, C)#pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size * pad_W, C)  #window_size * window_size[1], C) B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4) # 3, B*nW, nH, Ws*Ws, C//nH
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=np.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1)) # B*nW, nH, Ws*Ws, Ws*Ws
    #print(f"attn.shape: {attn.shape} attn.avg: {attn.mean()} attn.std: {attn.std()} attn.max: {attn.max()} attn.min: {attn.min()}")
    # add relative position bias
    # TODO: check whether relative position bias is correctly implemented
    relative_position_bias_shape = relative_position_bias.shape # B, nH, Ws*Ws, Ws*Ws
    relative_position_bias = relative_position_bias.unsqueeze(1).repeat(1, num_windows, 1, 1, 1).view(B * num_windows, *relative_position_bias_shape[1:])
    attn = attn + relative_position_bias
    #print(f"relative_position_bias.shape: {relative_position_bias.shape} relative_position_bias.avg: {relative_position_bias.mean()} relative_position_bias.std: {relative_position_bias.std()} relative_position_bias.max: {relative_position_bias.max()} relative_position_bias.min: {relative_position_bias.min()}")

    if shift_size > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, None))
        #w_slices = ((0, -window_size[1]), (-window_size[1], -0), (-0, None))
        w_slices = ((0, -pad_W), (-pad_W, -0), (-0, None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size, window_size, 1, pad_W)#pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size * pad_W)#window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size, 1, window_size, pad_W, C)#pad_W // window_size[1], window_size, window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size, 0), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


# class ShiftedWindowAttention(nn.Module):
#     """
#     See :func:`shifted_window_attention`.
#     """
# 
#     def __init__(
#         self,
#         dim: int,
#         window_size: int,
#         shift_size: int,
#         num_heads: int,
#         qkv_bias: bool = True,
#         proj_bias: bool = True,
#         attention_dropout: float = 0.0,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         if len(window_size) != 1 or len(shift_size) != 1:
#             raise ValueError("window_size and shift_size must be of length 1")
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.num_heads = num_heads
#         self.attention_dropout = attention_dropout
#         self.dropout = dropout
#         #  grid of relative spatial position
#         self.degree2km = 111.32
#         self.grid_size = self.degree2km / 10
#         self.bin = 21
# 
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
# 
#         #self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
#         self.define_relative_position_bias_table_and_index()
# 
# 
#     def define_relative_position_bias_table_and_index(self):
#         ## we put Wh as the time dimension, Wn as the station dimension
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             # torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
#             torch.zeros((2 * self.window_size - 1) * self.bin * self.bin, self.num_heads)
#         )  # 2*Wh-1 * bin * bin, nH
#         nn.init.trunc_normal_(self.relative_position_bias_table, mean=0, std=1)
#         
#         coords_t = torch.arange(self.window_size).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1) # nt, 1, 1
#         coords_t_flatten = torch.flatten(coords_t, 0, 1) # nt, 1
#         relative_coords_t = coords_t_flatten[:, None, :] - coords_t_flatten[None, :, :] # nt, nt, 1
#         self.register_buffer("relative_coords_t", relative_coords_t)
# 
#     def get_relative_position_bias(self, loc: Tensor) -> Tensor:
#         
#         # relative_coords_t = self.relative_coords_t.repeat(loc.shape[0], 1, 1, 1) # B, nt*nx, nt*nx, 1
#         # coords_x_flatten = loc.unsqueeze(-3).repeat(1, self.window_size, 1, 1).flatten(1,2)
#         # relative_coords_sta = (coords_x_flatten[:, :, None, :] - coords_x_flatten[:, None, :, :])[:,:,:,:2] # B, nt*nx, nt*nx, 2
#         # relative_coords_sta = (relative_coords_sta/self.grid_size) + (self.bin-1)/2 # B, nx, nx, 2
#         # relative_coords_sta = torch.clamp(torch.round(relative_coords_sta).long(), 0, self.bin-1) # B, nx, nx, 2
#         # relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 3
#         # # shift to start from 0
#         # relative_coords[:, :, :, 0] += self.window_size - 1
#         # relative_coords[:, :, :, 0] *= self.bin ** 2
#         # relative_coords[:, :, :, 1] *= self.bin
#         # relative_position_index = relative_coords.sum(dim=-1).flatten(start_dim=1) # B, nt*nx*nt*nx
#         # get pair-wise relative position index for each token inside the window
#         relative_coords_t = self.relative_coords_t.repeat(loc.shape[0], loc.shape[1], loc.shape[1], 1) # B, nt*nx, nt*nx, 1
#         relative_coords_sta = (loc[:, :, None, :] - loc[:, None, :, :])[:,:,:,:2] # B, nx, nx, 2
#         relative_coords_sta = (relative_coords_sta/self.grid_size) + (self.bin-1)/2 # B, nx, nx, 2
#         relative_coords_sta = torch.clamp(torch.round(relative_coords_sta).long(), 0, self.bin-1) # B, nx, nx, 2
#         relative_coords_sta = relative_coords_sta.repeat(1, self.window_size, self.window_size, 1) # B, nt*nx, nt*nx, 2
#         relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 3
#         # shift to start from 0
#         relative_coords[:, :, :, 0] += self.window_size - 1
#         relative_coords[:, :, :, 0] *= self.bin ** 2
#         relative_coords[:, :, :, 1] *= self.bin
#         relative_position_index = relative_coords.sum(dim=-1).view(loc.shape[0], -1) # B, nt*nx*nt*nx
# 
#         N = self.window_size * loc.shape[1]
#         # B, nt*nx*nt*nx, nH 
#         relative_position_bias = self.relative_position_bias_table[relative_position_index[:]]
#         relative_position_bias = relative_position_bias.view(loc.shape[0], N, N, -1) # B, nt*nx, nt*nx, nH
#         relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous() # B, nH, nt*nx, nt*nx
#         
#         return relative_position_bias
# 
# 
#     def forward(self, x: Tensor, loc: Tensor):
#         """
#         Args:
#             x (Tensor): Tensor with layout of [B, H, W, C]
#             loc (Tensor): Station location with layout of [B, W, 3]
#         Returns:
#             Tensor with same layout as input, i.e. [B, H, W, C]
#         """
#         relative_position_bias = self.get_relative_position_bias(loc)
#         #relative_position_bias = 0
# 
#         return shifted_window_attention(
#             x,
#             self.qkv.weight,
#             self.proj.weight,
#             relative_position_bias,
#             self.window_size,
#             self.num_heads,
#             shift_size=self.shift_size,
#             attention_dropout=self.attention_dropout,
#             dropout=self.dropout,
#             qkv_bias=self.qkv.bias,
#             proj_bias=self.proj.bias,
#             training=self.training,
#             #logit_scale=self.logit_scale,
#         )

class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        shift_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        #  grid of relative spatial position
        self.degree2km = 111.32
        self.grid_size = self.degree2km / 100
        self.bin_x = 51
        self.bin_y = 51
        self.bin_z = 11

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        #self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.define_relative_position_bias_table_and_index()


    def define_relative_position_bias_table_and_index(self):
        ## we put Wh as the time dimension, Wn as the station dimension
        # define a parameter table of relative position bias
        self.relative_position_bias_table_t = nn.Parameter(
            torch.zeros((2 * self.window_size - 1), self.num_heads)
        )  # 2*Wh-1, nH
        self.relative_position_bias_table_x = nn.Parameter(
            torch.zeros(self.bin_x, self.num_heads)
        )
        self.relative_position_bias_table_y = nn.Parameter(
            torch.zeros(self.bin_y, self.num_heads)
        )
        self.relative_position_bias_table_z = nn.Parameter(
            torch.zeros(self.bin_z, self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table_t, std=0.1)
        nn.init.trunc_normal_(self.relative_position_bias_table_x, std=0.1)
        nn.init.trunc_normal_(self.relative_position_bias_table_y, std=0.1)
        nn.init.trunc_normal_(self.relative_position_bias_table_z, std=0.1)
        
        coords_t = torch.arange(self.window_size).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1) # nt, 1, 1
        coords_t_flatten = torch.flatten(coords_t, 0, 1) # nt, 1
        relative_coords_t = coords_t_flatten[:, None, :] - coords_t_flatten[None, :, :] # nt, nt, 1
        self.register_buffer("relative_coords_t", relative_coords_t)

    def get_relative_position_bias(self, loc: Tensor) -> Tensor:
        
        # get pair-wise relative position index for each token inside the window
        relative_coords_t = self.relative_coords_t.repeat(loc.shape[0], loc.shape[1], loc.shape[1], 1) # B, nt*nx, nt*nx, 1
        relative_coords_sta = loc[:, :, None, :] - loc[:, None, :, :] # B, nx, nx, 3
        relative_coords_x = (relative_coords_sta[:,:,:, 0:1]/self.grid_size) + (self.bin_x-1)/2
        relative_coords_y = (relative_coords_sta[:,:,:, 1:2]/self.grid_size) + (self.bin_y-1)/2
        relative_coords_z = (relative_coords_sta[:,:,:, 2:]/self.grid_size) + (self.bin_z-1)/2
        relative_coords_x = torch.clamp(torch.round(relative_coords_x).long(), 0, self.bin_x-1) # B, nx, nx, 1
        relative_coords_y = torch.clamp(torch.round(relative_coords_y).long(), 0, self.bin_y-1) # B, nx, nx, 1
        relative_coords_z = torch.clamp(torch.round(relative_coords_z).long(), 0, self.bin_z-1) # B, nx, nx, 1
        relative_coords_sta = torch.cat((relative_coords_x, relative_coords_y, relative_coords_z), dim=-1) # B, nx, nx, 3
        
        relative_coords_sta = relative_coords_sta.repeat(1, self.window_size, self.window_size, 1) # B, nt*nx, nt*nx, 2
        #relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 3
        # shift to start from 0
        relative_coords_t[:, :, :, 0] += self.window_size - 1
        relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1).permute(3, 0, 1, 2) # 4, B, nt*nx, nt*nx
        relative_position_index = relative_coords.view(4, loc.shape[0], -1) # 4, B, nt*nx*nt*nx

        N = self.window_size * loc.shape[1]#self.window_size[1]
        # B, nt*nx*nt*nx, nH 
        # relative_position_bias = self.relative_position_bias_table[relative_position_index[:]]
        relative_position_bias = self.relative_position_bias_table_t[relative_position_index[0], :] + \
                                self.relative_position_bias_table_x[relative_position_index[1], :] + \
                                self.relative_position_bias_table_y[relative_position_index[2], :] + \
                                self.relative_position_bias_table_z[relative_position_index[3], :]
        relative_position_bias = relative_position_bias.view(loc.shape[0], N, N, -1) # B, nt*nx, nt*nx, nH
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous() # B, nH, nt*nx, nt*nx
        
        return relative_position_bias


    def forward(self, x: Tensor, loc: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
            loc (Tensor): Station location with layout of [B, W, 3]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias(loc)

        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
            #logit_scale=self.logit_scale,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """
    See :func:`shifted_window_attention_v2`.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        shift_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        # give up us logit_scale because the length of time dimension won't be too far from the original length
        # self.logit_scale = None
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        ## https://github.com/microsoft/Swin-Transformer/blob/b720b4191588c19222ccf129860e905fb02373a7/models/swin_transformer_v2.py#L92
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(4, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()


    def define_relative_position_bias_table_and_index(self):
        coords_t = torch.arange(self.window_size).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1) # nt, nx 1
        self.register_buffer("coords_t", coords_t)

    def get_relative_position_bias(self, loc: Tensor) -> Tensor:
        
        coords_t = self.coords_t.repeat(loc.shape[0], 1, loc.shape[1], 1) #bt, nt, nx 1
        coords_x = loc.unsqueeze(-3).repeat(1, self.coords_t.shape[0], 1, 1) #bt, nt, nx, 3
        coords = torch.cat((coords_t, coords_x), dim=-1) #bt, nt, nx, 4
        coords_flatten = torch.flatten(coords, 1, 2)  # bt, nt * nx, 4
        relative_coords = coords_flatten[:, :, None, :] - coords_flatten[:, None, :, :]  # bt, Wh*Ww, Wh*Ww, 4
        relative_position_bias = self.cpb_mlp(relative_coords) #bt, Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2)
        
        return relative_position_bias
    

    def forward(self, x: Tensor, loc: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
            loc (Tensor): Station location with layout of [B, W, 3]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias(loc)
        # assert not torch.isnan(relative_position_bias).any(), "relative_position_bias contains NaN"
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )

# class ShiftedWindowAttentionRotary(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         window_size: int,
#         shift_size: int,
#         num_heads: int,
#         qkv_bias: bool = True,
#         proj_bias: bool = True,
#         attention_dropout: float = 0.0,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         if len(window_size) != 1 or len(shift_size) != 1:
#             raise ValueError("window_size and shift_size must be of length 1")
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.num_heads = num_heads
#         self.attention_dropout = attention_dropout
#         self.dropout = dropout
#         #  grid of relative spatial position
#         self.degree2km = 111.32
#         self.grid_size = self.degree2km / 100
#         self.bin_x = 51
#         self.bin_y = 51
#         self.bin_z = 11
# 
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
#     
#     
#     pass


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, meta: Dict[str, Tensor]):
        x, loc = meta["x"], meta["loc"]
        # is this a good idea to apply norm before the attention?
        # x = x + self.stochastic_depth(self.attn(self.norm1(x), loc))
        # x = x + self.stochastic_depth(self.attn(self.norm1(x), self.norm1(loc))) ???
        # x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        x = x + self.stochastic_depth(self.norm1(self.attn(x, loc)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return {"x": x, "loc": loc}


class SwinTransformerBlockV2(SwinTransformerBlock):
    """
    Swin Transformer V2 Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def forward(self, meta: Dict):
        x, loc = meta["x"], meta["loc"]
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x, loc)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return {"x": x, "loc": loc}


class SwinTransformer1D(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (int): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block_name: Optional[str] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()

        block: Optional[Callable[..., nn.Module]]
        if block_name is None or block_name == "SwinTransformerBlock":
            block = SwinTransformerBlock
        elif block_name == "SwinTransformerBlockV2":
            block = SwinTransformerBlockV2
        else:
            raise ValueError(f"Unknown Swin Transformer block: {block_name}")

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        # split image into non-overlapping patches
        self.split_patch = nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size, 1), stride=(patch_size, 1)
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=0 if i_layer % 2 == 0 else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.ModuleList(layers)

        self.num_features = [int(embed_dim * 2 ** i) for i in range(len(depths))]
        for i in out_indices:
            norm = norm_layer(self.num_features[i])
            norm_name = f"norm{i}"
            self.add_module(norm_name, norm)
        self.out_indices = tuple([i*2 for i in out_indices])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, loc):
        
        x = self.split_patch(x)
        meta={"x": x, "loc": loc}
        
        outs = OrderedDict()
        for i, feature in enumerate(self.features):
            meta = feature(meta)
            if i in self.out_indices:
                norm = getattr(self, f"norm{i//2}")
                out = norm(meta["x"])
                ## for seismic time series
                out = out.permute(0, 2, 3, 1).contiguous() ## bt, st, chn, nt
                outs[f"out{i//2}"] = out
        #assert len(outs.keys()) == len(self.out_indices)
        
        return outs


def _swin_transformer(
    patch_size: int,
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: int,
    stochastic_depth_prob: float,
    weights,
    progress: bool,
    block_name: Optional[str],
    **kwargs: Any,
) -> SwinTransformer1D:
    # if weights is not None:
    #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer1D(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        block_name=block_name,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model