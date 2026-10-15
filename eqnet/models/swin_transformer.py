import math
from functools import partial
from typing import Any, Callable, List, Optional, Dict, Union, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.fx import wrap
from torch.utils.checkpoint import checkpoint

from .ops import MLP, Permute, StochasticDepth

__all__ = [
    "SwinTransformer",
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]

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

class checkpoint_wrapper(nn.Module):
    def __init__(self, module: nn.Module, use_checkpoint: bool):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            return checkpoint(self.module, *args, **kwargs)
        else:
            return self.module(*args, **kwargs)


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        #self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
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
        #x = self.reduction(x)
        x = self.norm(x)

        return {"x": x, "loc": loc}
    

class PatchReduction(nn.Module):
    """Patch Reduction Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm, 
                 k=(8, 1), s=(4, 1)):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2*dim, kernel_size=k, stride=s, 
                                   padding=(math.ceil((k[0] - s[0]) / 2), math.ceil((k[1] - s[1]) / 2)), bias=False)
        self.norm = norm_layer(2 * dim)
        
        nn.init.trunc_normal_(self.reduction.weight, std=.02)

    def forward(self, meta: Dict):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W, 2*C]
        """
        x, loc = meta["x"], meta["loc"]
        x = self.reduction(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)

        return {"x": x, "loc": loc}


@torch.fx.wrap
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Note: you must ensure that the freq_cis can match the shape of xq and xk through broadcasting
    '''
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    #assert 0, f"xq_.shape: {xq_.shape} xk_.shape: {xk_.shape} freqs_cis.shape: {freqs_cis.shape}"
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


@torch.fx.wrap
def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size_: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    use_station_location=True,
    use_relative_position_bias=True,
    use_rope=False,
    freq_cis: Optional[Tensor] = None,
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
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape # batch, nt, nsta, ndim
    # deal with the situation that H and W are even smaller than window size
    window_size = [window_size_[0], min(window_size_[1], W)]
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    # use bias shape B, nH, Wh, nsta, Wh, nsta
    ##relative_position_bias = F.pad(relative_position_bias, (0, pad_r, 0, 0, 0, pad_r))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        ##relative_position_bias = torch.roll(relative_position_bias,
        ##                            shifts=(-shift_size[1], -shift_size[1]), dims=(3, 5))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    ##relative_position_bias = relative_position_bias.view(B, num_heads, 
    ##                                                     window_size[0], pad_W // window_size[1], window_size[1], 
    ##                                                     window_size[0], pad_W // window_size[1], window_size[1]).unsqueeze(1)
    ##relative_position_bias = torch.cat([relative_position_bias[:,:,:,:,i,:,:,i,:] for i in range(pad_W // window_size[1])], dim=1)
    ##assert relative_position_bias.shape == (B, pad_W // window_size[1], num_heads, window_size[0], window_size[1], window_size[0], window_size[1])
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4) # 3, B*nW, nH, Ws*Ws, C//nH
    q, k, v = qkv[0], qkv[1], qkv[2]
    if use_rope and freq_cis is not None:
        freq_cis = F.pad(freq_cis, (0, 0, 0, pad_r)) # B, nt, nx, nH* C//nH//2
        freq_cis = freq_cis.view(B, 1, window_size[0], pad_W // window_size[1], window_size[1], -1).repeat(1, pad_H // window_size[0], 1, 1, 1, 1).permute(0, 1, 3, 2, 4, 5)
        freq_cis = freq_cis.reshape(B * num_windows, window_size[0] * window_size[1], num_heads, -1).permute(0, 2, 1, 3)  # B*nW, nH, Ws*Ws, C//nH//2
        #assert torch.allclose(freq_cis[:,0,:,:], freq_cis[:,1,:,:], atol=1e-6)
        assert freq_cis.shape == (B * num_windows, num_heads, window_size[0] * window_size[1], C // num_heads//2)
        q, k = apply_rotary_emb(q, k, freq_cis)
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
    ##relative_position_bias = relative_position_bias.repeat(1, (pad_H // window_size[0]), 1, 1, 1, 1, 1).view(B * num_windows, num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1])
    #relative_position_bias_shape = relative_position_bias.shape # B, pad_W//Ww, nH, Ws*Ws, Ws*Ws
    #relative_position_bias = relative_position_bias.unsqueeze(1).repeat(1, num_windows, 1, 1, 1).view(B * num_windows, *relative_position_bias_shape[1:])
    if use_station_location and use_relative_position_bias:
        # use bias shape B, nH, Wh, nsta, Wh, nsta
        relative_position_bias = F.pad(relative_position_bias, (0, pad_r, 0, 0, 0, pad_r))
        if shift_size[1] > 0:
            relative_position_bias = torch.roll(relative_position_bias,
                                        shifts=(-shift_size[1], -shift_size[1]), dims=(3, 5))
        relative_position_bias = relative_position_bias.view(B, num_heads, 
                                                         window_size[0], pad_W // window_size[1], window_size[1], 
                                                         window_size[0], pad_W // window_size[1], window_size[1]).unsqueeze(1)
        relative_position_bias = torch.cat([relative_position_bias[:,:,:,:,i,:,:,i,:] for i in range(pad_W // window_size[1])], dim=1)
        assert relative_position_bias.shape == (B, pad_W // window_size[1], num_heads, window_size[0], window_size[1], window_size[0], window_size[1])
        relative_position_bias = relative_position_bias.unsqueeze(1).repeat(1, (pad_H // window_size[0]), 1, 1, 1, 1, 1, 1).view(B * num_windows, num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1])
        
    if use_relative_position_bias:    
        attn = attn + relative_position_bias
    #print(f"relative_position_bias.shape: {relative_position_bias.shape} relative_position_bias.avg: {relative_position_bias.mean()} relative_position_bias.std: {relative_position_bias.std()} relative_position_bias.max: {relative_position_bias.max()} relative_position_bias.min: {relative_position_bias.min()}")

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        #NOTE: it is easy to change the shift direction from upper left to lower right by changing below 2 lines
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        #w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        #NOTE: we don't need mask for station dimension
        w_slices = ((0, -window_size[1]), (-window_size[1], -0), (-0, None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
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
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


@torch.fx.wrap
def dynamic_shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size_: List[int],
    num_heads: int,
    shift_size: List[int],
    topk_station_indices: Tensor,
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    use_station_location=True,
    use_relative_position_bias=True,
    use_rope=False,
    freq_cis: Optional[Tensor] = None,
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
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape # batch, nt, nsta, ndim
    # deal with the situation that H and W are even smaller than window size
    window_size = [window_size_[0], min(window_size_[1], W)]
    # pad feature maps to multiples of window size
    pad_r = 0#(window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    # use bias shape B, nH, Wh, nsta, Wh, nsta
    ##relative_position_bias = F.pad(relative_position_bias, (0, pad_r, 0, 0, 0, pad_r))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if shift_size[0] > 0:
        x = torch.roll(x, shifts=(-shift_size[0], 0), dims=(1, 2))
        ##relative_position_bias = torch.roll(relative_position_bias,
        ##                            shifts=(-shift_size[1], -shift_size[1]), dims=(3, 5))

    # partition windows
    num_windows = (pad_H // window_size[0])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W, C)
    x = x.view(B * num_windows, window_size[0]*pad_W, C)  # B*nW, Wt, nsta, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4) # 3, B*nW, nH, Wt*nsta, C//nH
    q, k, v = qkv[0], qkv[1], qkv[2]
    if use_rope and freq_cis is not None:
        freq_cis = F.pad(freq_cis, (0, 0, 0, pad_r))
        freq_cis = freq_cis.view(B, 1, window_size[0], pad_W, -1).repeat(1, pad_H // window_size[0], 1, 1, 1)
        freq_cis = freq_cis.reshape(B * num_windows, window_size[0] * pad_W, num_heads, -1).permute(0, 2, 1, 3)  # B*nW, nH, Wt*nsta, C//nH//2
        #assert torch.allclose(freq_cis[:,0,:,:], freq_cis[:,1,:,:], atol=1e-6)
        assert freq_cis.shape == (B * num_windows, num_heads, window_size[0] * pad_W, C // num_heads//2)
        q, k = apply_rotary_emb(q, k, freq_cis)
    # B*nW, nH, Wt, nx, C//nH
    q, k, v = q.reshape(B * num_windows, num_heads, window_size[0], pad_W, C//num_heads), k.reshape(B * num_windows, num_heads, window_size[0], pad_W, C//num_heads), v.reshape(B * num_windows, num_heads, window_size[0], pad_W, C//num_heads)
    assert topk_station_indices.shape[-1] == window_size[1], f"K nearest station size doesn't match window size: {topk_station_indices.shape = }, {window_size = }"
    topk_station_indices = topk_station_indices.repeat_interleave(num_windows, dim=0).view(-1, 1, pad_W, window_size[1], 1, 1) # B, nx, topk -> B, 1, 1, nx, topk, 1
    topk_station_indices = topk_station_indices.expand(-1, num_heads, pad_W, window_size[1], window_size[0], C//num_heads) # B, nH, Wt, Wt*nx, topk, C//nH
    k = k.permute(0, 1, 3, 2, 4).unsqueeze(2).expand(-1, num_heads, pad_W, pad_W, window_size[0], C//num_heads) # B, nH, Wt, Wt*nx, nx, C//nH
    v = v.permute(0, 1, 3, 2, 4).unsqueeze(2).expand(-1, num_heads, pad_W, pad_W, window_size[0], C//num_heads)
    k = torch.gather(k, dim=3, index=topk_station_indices) # B, nH, Wt, nx, topk, C//nH
    #v = torch.gather(v, dim=3, index=topk_station_indices)
    
    #q = q.reshape(B * num_windows, num_heads, window_size[0]*pad_W, C//num_heads)
    q = q.permute(0, 1, 3, 2, 4)
    
    if logit_scale is not None:
        # cosine attention
        #attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = torch.einsum("bhntd,bhnkud->bhntku", F.normalize(q, dim=-1), F.normalize(k, dim=-1))
        attn = attn.permute(0, 1, 3, 2, 5, 4).reshape(B * num_windows, num_heads, window_size[0] * pad_W, window_size[0] * window_size[1])
        logit_scale = torch.clamp(logit_scale, max=np.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q *= (C // num_heads) ** -0.5
        #attn = q.matmul(k.transpose(-2, -1)) # B*nW, nH, Ws*Ws, Ws*Ws
        attn = torch.einsum("bhntd,bhnkud->bhntku", q, k) # B*nW, nH, Ws*Ws, Ws*Ws
        attn = attn.permute(0, 1, 3, 2, 5, 4).reshape(B * num_windows, num_heads, window_size[0] * pad_W, window_size[0] * window_size[1])
    #print(f"attn.shape: {attn.shape} attn.avg: {attn.mean()} attn.std: {attn.std()} attn.max: {attn.max()} attn.min: {attn.min()}")
    # add relative position bias
    # TODO: check whether relative position bias is correctly implemented
    ##relative_position_bias = relative_position_bias.repeat(1, (pad_H // window_size[0]), 1, 1, 1, 1, 1).view(B * num_windows, num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1])
    #relative_position_bias_shape = relative_position_bias.shape # B, pad_W//Ww, nH, Ws*Ws, Ws*Ws
    #relative_position_bias = relative_position_bias.unsqueeze(1).repeat(1, num_windows, 1, 1, 1).view(B * num_windows, *relative_position_bias_shape[1:])
    if use_station_location and use_relative_position_bias:
        ## use bias shape B, nH, Wh, nsta, Wh, nsta
        #relative_position_bias = F.pad(relative_position_bias, (0, pad_r, 0, 0, 0, pad_r))
        #if shift_size[1] > 0:
        #    relative_position_bias = torch.roll(relative_position_bias,
        #                                shifts=(-shift_size[1], -shift_size[1]), dims=(3, 5))
        #relative_position_bias = relative_position_bias.view(B, num_heads, 
        #                                                 window_size[0], pad_W // window_size[1], window_size[1], 
        #                                                 window_size[0], pad_W // window_size[1], window_size[1]).unsqueeze(1)
        #relative_position_bias = torch.cat([relative_position_bias[:,:,:,:,i,:,:,i,:] for i in range(pad_W // window_size[1])], dim=1)
        assert relative_position_bias.shape == (B, num_heads, window_size[0], pad_W, window_size[0], window_size[1])
        relative_position_bias = relative_position_bias.unsqueeze(1).repeat(1, (pad_H // window_size[0]), 1, 1, 1, 1, 1).view(B * num_windows, num_heads, window_size[0] * pad_W, window_size[0] * window_size[1])
        
    if use_relative_position_bias:    
        attn = attn + relative_position_bias
    #print(f"relative_position_bias.shape: {relative_position_bias.shape} relative_position_bias.avg: {relative_position_bias.mean()} relative_position_bias.std: {relative_position_bias.std()} relative_position_bias.max: {relative_position_bias.max()} relative_position_bias.min: {relative_position_bias.min()}")

    if shift_size[0] > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, 1))
        #NOTE: it is easy to change the shift direction from upper left to lower right by changing below 2 lines
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        #w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        #NOTE: we don't need mask for station dimension
        count = 0
        for h in h_slices:
            attn_mask[h[0] : h[1], :] = count
            count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], 1)
        attn_mask_1 = attn_mask.repeat(1, 1, window_size[1]).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask_2 = attn_mask.repeat(1, 1, pad_W).reshape(num_windows, window_size[0] * pad_W)
        attn_mask = attn_mask_1.unsqueeze(1) - attn_mask_2.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.contiguous().view(x.size(0) // num_windows, num_windows, num_heads, window_size[0] * pad_W, window_size[0] * window_size[1])
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, window_size[0] * pad_W, window_size[0] * window_size[1])

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    #x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    v = torch.gather(v, dim=3, index=topk_station_indices)
    x = torch.einsum('bhntku,bhnkud->bhntd', attn.view(B * num_windows, num_heads, window_size[0], pad_W, window_size[0], window_size[1]).permute(0,1,3,2,5,4), v)  # B*nW, nH, Wt, nx, C//nH
    x = x.permute(0,1,3,2,4).reshape(B * num_windows, num_heads, window_size[0]*pad_W, C//num_heads).transpose(1, 2).reshape(x.size(0), window_size[0]*pad_W, C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W, C)
    x = x.reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if shift_size[0] > 0:
        x = torch.roll(x, shifts=(shift_size[0], 0), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x

# class ShiftedWindowAttention(nn.Module):
#     """
#     See :func:`shifted_window_attention`.
#     large encoding 
#     """
# 
#     def __init__(
#         self,
#         dim: int,
#         window_size: List[int],
#         shift_size: List[int],
#         num_heads: int,
#         qkv_bias: bool = True,
#         proj_bias: bool = True,
#         attention_dropout: float = 0.0,
#         dropout: float = 0.0,
#         use_station_location: bool = True,
#     ):
#         super().__init__()
#         if len(window_size) != 2 or len(shift_size) != 2:
#             raise ValueError("window_size and shift_size must be of length 2")
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.num_heads = num_heads
#         self.attention_dropout = attention_dropout
#         self.dropout = dropout
#         #  grid of relative spatial position
#         self.degree2km = 111.32
#         self.grid_size = self.degree2km / 20
#         self.grid_size_z = self.degree2km / 100
#         self.bin = 21
#         self.bin_z = 11
# 
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
# 
#         self.use_station_location = use_station_location
#         #self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
#         self.define_relative_position_bias_table_and_index()
# 
# 
#     def define_relative_position_bias_table_and_index(self):
#         ## we put Wh as the time dimension, Wn as the station dimension
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             # torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
#             torch.zeros((2 * self.window_size[0] - 1) * self.bin * self.bin, self.num_heads)
#         )  # 2*Wh-1 * bin * bin, nH
#         # self.relative_position_bias_table_z = nn.Parameter(
#         #     torch.zeros(self.bin_z, self.num_heads)
#         # )
#         nn.init.trunc_normal_(self.relative_position_bias_table, mean=0, std=0.02)
#         nn.init.trunc_normal_(self.relative_position_bias_table_z, mean=0, std=0.02)
#         
#         coords_t = torch.arange(self.window_size[0]).unsqueeze(-1) # .unsqueeze(-1).repeat(1, self.window_size[1], 1) # nt, nx 1
#         # coords_t_flatten = torch.flatten(coords_t, 0, 1) # nt*nx, 1
#         relative_coords_t = coords_t[:, None, :] - coords_t[None, :, :] # nt, nt, 1
#         self.register_buffer("relative_coords_t", relative_coords_t)
# 
#     def get_relative_position_bias(self, loc: Union[Tensor, None]) -> Tensor:
#         # get pair-wise relative position index for each token inside the window
#         relative_coords_t = self.relative_coords_t.repeat(loc.shape[0], loc.shape[1], loc.shape[1], 1) # B, nt*nx, nt*nx, 1
#         relative_coords_sta = (loc[:, :, None, :] - loc[:, None, :, :])[:,:,:,:2] # B, nx, nx, 2
#         # relative_coords_z = (loc[:, :, None, 2:] - loc[:, None, :, 2:]) # B, nx, nx, 1
#         relative_coords_sta = (relative_coords_sta/self.grid_size) + (self.bin-1)/2 # B, nx, nx, 2
#         relative_coords_sta = torch.clamp(torch.round(relative_coords_sta).long(), 0, self.bin-1) # B, nx, nx, 2
#         # relative_coords_z = (relative_coords_z/self.grid_size_z) + (self.bin_z-1)/2 # B, nx, nx, 1
#         # relative_coords_z = torch.clamp(torch.round(relative_coords_z).long(), 0, self.bin_z-1) # B, nx, nx, 1
#         relative_coords_sta = relative_coords_sta.repeat(1, self.window_size[0], self.window_size[0], 1) # B, nt*nx, nt*nx, 2
#         # relative_coords_z = relative_coords_z.repeat(1, self.window_size[0], self.window_size[0], 1) # B, nt*nx, nt*nx, 1
#         relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 3
#         # shift to start from 0
#         relative_coords[:, :, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, :, 0] *= self.bin ** 2
#         relative_coords[:, :, :, 1] *= self.bin
#         relative_position_index = relative_coords.sum(dim=-1).view(loc.shape[0], -1) # B, nt*nx*nt*nx
#         # relative_position_index_z = relative_coords_z.sum(dim=-1).view(loc.shape[0], -1) # B, nt*nx*nt*nx
# 
#         N = self.window_size[0] * self.window_size[1]
#         # B, nt*nx*nt*nx, nH 
#         relative_position_bias = self.relative_position_bias_table[relative_position_index[:]]# + self.relative_position_bias_table_z[relative_position_index_z[:]]
#         # relative_position_bias = relative_position_bias.view(loc.shape[0], N, N, -1) # B, nt*nx, nt*nx, nH
#         # relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous() # B, nH, nt*nx, nt*nx
#         relative_position_bias = relative_position_bias.view(loc.shape[0], self.window_size[0], self.window_size[1], self.window_size[0], self.window_size[1], -1) # B, nt, nx, nt, nx, nH
#         relative_position_bias = relative_position_bias.permute(0, 5, 1, 2, 3, 4).contiguous() # B, nH, nt, nx, nt, nx
#         
#         return relative_position_bias
# 
# 
#     def forward(self, x: Tensor, loc: Union[Tensor, None]):
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
#             use_station_location=self.use_station_location,
#             training=self.training,
#             #logit_scale=self.logit_scale,
#         )
# 
# class ShiftedWindowAttention(nn.Module):
#     """
#     See :func:`shifted_window_attention`.
#     small encoding t + x*y + z
#     """
# 
#     def __init__(
#         self,
#         dim: int,
#         window_size: List[int],
#         shift_size: List[int],
#         num_heads: int,
#         qkv_bias: bool = True,
#         proj_bias: bool = True,
#         attention_dropout: float = 0.0,
#         dropout: float = 0.0,
#         use_station_location: bool = True,
#     ):
#         super().__init__()
#         if len(window_size) != 2 or len(shift_size) != 2:
#             raise ValueError("window_size and shift_size must be of length 2")
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
#         self.use_station_location = use_station_location
#         #self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
#         self.define_relative_position_bias_table_and_index()
# 
# 
#     def define_relative_position_bias_table_and_index(self):
#         ## we put Wh as the time dimension, Wn as the station dimension
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table_t = nn.Parameter(
#             torch.zeros((2 * self.window_size[0] - 1), self.num_heads)
#         )  # 2*Wh-1, nH
#         self.relative_position_bias_table_xy = nn.Parameter(torch.zeros(self.bin_x * self.bin_y, self.num_heads)) 
#         self.relative_position_bias_table_z = nn.Parameter(
#             torch.zeros(self.bin_z, self.num_heads)
#         )
#         nn.init.trunc_normal_(self.relative_position_bias_table_t, std=0.02)
#         # nn.init.trunc_normal_(self.relative_position_bias_table_x, std=0.02)
#         # nn.init.trunc_normal_(self.relative_position_bias_table_y, std=0.02)
#         nn.init.trunc_normal_(self.relative_position_bias_table_xy, std=0.02)
#         nn.init.trunc_normal_(self.relative_position_bias_table_z, std=0.02)
#         
#         coords_t = torch.arange(self.window_size[0]).unsqueeze(-1)#.repeat(1, 1) # nt, 1
#         # coords_t_flatten = torch.flatten(coords_t, 0, 1) # nt*nx, 1
#         relative_coords_t = coords_t[:, None, :] - coords_t[None, :, :] # nt, nt, 1
#         self.register_buffer("relative_coords_t", relative_coords_t)
# 
#     def get_relative_position_bias(self, loc: Union[Tensor, None]) -> Tensor:
#         
#         # get pair-wise relative position index for each token inside the window
#         relative_coords_t = self.relative_coords_t.repeat(loc.shape[0], loc.shape[1], loc.shape[1], 1) # B, nt*nx, nt*nx, 1
#         relative_coords_sta = loc[:, :, None, :] - loc[:, None, :, :] # B, nx, nx, 3
#         relative_coords_x = (relative_coords_sta[:,:,:, 0:1]/self.grid_size) + (self.bin_x-1)/2
#         relative_coords_y = (relative_coords_sta[:,:,:, 1:2]/self.grid_size) + (self.bin_y-1)/2
#         relative_coords_z = (relative_coords_sta[:,:,:, 2:]/self.grid_size) + (self.bin_z-1)/2
#         relative_coords_x = torch.clamp(torch.round(relative_coords_x).long(), 0, self.bin_x-1) # B, nx, nx, 1
#         relative_coords_y = torch.clamp(torch.round(relative_coords_y).long(), 0, self.bin_y-1) # B, nx, nx, 1
#         relative_coords_z = torch.clamp(torch.round(relative_coords_z).long(), 0, self.bin_z-1) # B, nx, nx, 1
#         relative_coords_sta = torch.cat((relative_coords_x, relative_coords_y, relative_coords_z), dim=-1) # B, nx, nx, 3
#         
#         relative_coords_sta = relative_coords_sta.repeat(1, self.window_size[0], self.window_size[0], 1) # B, nt*nx, nt*nx, 3
#         #relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 3
#         # shift to start from 0
#         relative_coords_t[:, :, :, 0] += self.window_size[0] - 1
#         relative_coords_sta[:, :, :, 1] *= self.bin_y
#         relative_coords_sta = torch.cat([relative_coords_sta[:, :, :, 0:1]+relative_coords_sta[:, :, :, 1:2], relative_coords_sta[:, :, :, 2:]], dim=-1)
#         relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1).permute(3, 0, 1, 2) # 4, B, nt*nx, nt*nx
#         #relative_position_index = relative_coords.view(4, loc.shape[0], -1) # 4, B, nt*nx*nt*nx
#         relative_position_index = relative_coords.view(3, loc.shape[0], -1) # 3, B, nt*nx*nt*nx
# 
#         N = self.window_size[0] * self.window_size[1]
#         # B, nt*nx*nt*nx, nH 
#         # relative_position_bias = self.relative_position_bias_table[relative_position_index[:]]
#         relative_position_bias = self.relative_position_bias_table_t[relative_position_index[0], :] + \
#                                 self.relative_position_bias_table_xy[relative_position_index[1], :] + \
#                                 self.relative_position_bias_table_z[relative_position_index[2], :]
#                                 # self.relative_position_bias_table_x[relative_position_index[1], :] + \
#                                 # self.relative_position_bias_table_y[relative_position_index[2], :] + \
#                                 # self.relative_position_bias_table_z[relative_position_index[3], :]
#         # relative_position_bias = relative_position_bias.view(loc.shape[0], N, N, -1) # B, nt*nx, nt*nx, nH
#         # relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous() # B, nH, nt*nx, nt*nx
#         relative_position_bias = relative_position_bias.view(loc.shape[0], self.window_size[0], self.window_size[1], self.window_size[0], self.window_size[1], -1) # B, nt, nx, nt, nx, nH
#         relative_position_bias = relative_position_bias.permute(0, 5, 1, 2, 3, 4).contiguous() # B, nH, nt, nx, nt, nx
#         
#         return relative_position_bias
# 
# 
#     def forward(self, x: Tensor, loc: Union[Tensor, None]):
#         """
#         Args:
#             x (Tensor): Tensor with layout of [B, H, W, C]
#             loc (Tensor): Station location with layout of [B, W, 3]
#         Returns:
#             Tensor with same layout as input, i.e. [B, H, W, C]
#         """
#         relative_position_bias = self.get_relative_position_bias(loc)
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
#             use_station_location=self.use_station_location,
#             training=self.training,
#             #logit_scale=self.logit_scale,
#         )
# 

class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    small encoding t + x + y + z
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        use_station_location: bool = True,
        use_relative_position_bias: bool = True,
        relative_position_bias_dim: int = 512,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        #  grid of relative spatial position
        self.degree2km = 111.32
        self.grid_size = self.degree2km / 100
        self.bin_x = 101
        self.bin_y = 101
        self.bin_z = 11

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        
        self.use_station_location = use_station_location

        #self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.define_relative_position_bias_table_and_index()


    def define_relative_position_bias_table_and_index(self):
        if self.use_station_location:
            ## we put Wh as the time dimension, Wn as the station dimension
            # define a parameter table of relative position bias
            self.relative_position_bias_table_t = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1), self.num_heads)
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
            nn.init.trunc_normal_(self.relative_position_bias_table_t, std=0.02)
            nn.init.trunc_normal_(self.relative_position_bias_table_x, std=0.02)
            nn.init.trunc_normal_(self.relative_position_bias_table_y, std=0.02)
            nn.init.trunc_normal_(self.relative_position_bias_table_z, std=0.02)

            coords_t = torch.arange(self.window_size[0]).unsqueeze(-1)#.repeat(1, 1) # nt, 1
            relative_coords_t = coords_t[:, None, :] - coords_t[None, :, :] # nt, nt, 1
            self.register_buffer("relative_coords_t", relative_coords_t)
        else:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, loc: Union[Tensor, None]) -> Tensor:
        if self.use_station_location:
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

            relative_coords_sta = relative_coords_sta.repeat(1, self.window_size[0], self.window_size[0], 1) # B, nt*nx, nt*nx, 3
            #relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 3
            # shift to start from 0
            relative_coords_t[:, :, :, 0] += self.window_size[0] - 1
            relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1).permute(3, 0, 1, 2) # 4, B, nt*nx, nt*nx
            relative_position_index = relative_coords.view(4, loc.shape[0], -1) # 4, B, nt*nx*nt*nx

            N = self.window_size[0] * self.window_size[1]
            # B, nt*nx*nt*nx, nH 
            # relative_position_bias = self.relative_position_bias_table[relative_position_index[:]]
            relative_position_bias = self.relative_position_bias_table_t[relative_position_index[0], :] + \
                                    self.relative_position_bias_table_x[relative_position_index[1], :] + \
                                    self.relative_position_bias_table_y[relative_position_index[2], :] + \
                                    self.relative_position_bias_table_z[relative_position_index[3], :]
            # relative_position_bias = relative_position_bias.view(loc.shape[0], N, N, -1) # B, nt*nx, nt*nx, nH
            # relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous() # B, nH, nt*nx, nt*nx
            relative_position_bias = relative_position_bias.view(loc.shape[0], self.window_size[0], self.window_size[1], self.window_size[0], self.window_size[1], -1) # B, nt, nx, nt, nx, nH
            relative_position_bias = relative_position_bias.permute(0, 5, 1, 2, 3, 4).contiguous() # B, nH, nt, nx, nt, nx
        else:
            N = self.window_size[0] * self.window_size[1]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
            relative_position_bias = relative_position_bias.view(N, N, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        
        return relative_position_bias


    def forward(self, x: Tensor, loc: Union[Tensor, None]):
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
            use_station_location=self.use_station_location,
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
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        use_station_location: bool = True,
        use_relative_position_bias: bool = True,
        relative_position_bias_dim: int = 512,
        decoupled_relative_position_bias: bool = False,
        time_unit: float = 1.0,
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
            use_station_location=use_station_location,
            use_relative_position_bias=use_relative_position_bias,
        )

        self.earth_radius = 6371
        self.time_unit = time_unit
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        ## https://github.com/microsoft/Swin-Transformer/blob/b720b4191588c19222ccf129860e905fb02373a7/models/swin_transformer_v2.py#L92
        # mlp to generate continuous relative position bias
        coords_dim = 5 if self.use_station_location else 2
        self.decoupled_relative_position_bias = decoupled_relative_position_bias
        if not decoupled_relative_position_bias:
            self.cpb_mlp = nn.Sequential(nn.Linear(coords_dim, relative_position_bias_dim, bias=True),
                                         nn.ReLU(inplace=True),#checkpoint_wrapper(nn.GELU(), use_checkpoint=True),#nn.ReLU(inplace=True),
                                         nn.Linear(relative_position_bias_dim, num_heads, bias=False))
        else:
            self.cpb_mlp_time = nn.Sequential(nn.Linear(1, relative_position_bias_dim, bias=True),
                                                nn.ReLU(inplace=True),#checkpoint_wrapper(nn.GELU(), use_checkpoint=True),#nn.ReLU(inplace=True),
                                                nn.Linear(relative_position_bias_dim, num_heads, bias=False))
            self.cpb_mlp_space = nn.Sequential(nn.Linear(coords_dim-1, relative_position_bias_dim, bias=True),
                                                nn.ReLU(inplace=True),#checkpoint_wrapper(nn.GELU(), use_checkpoint=True),#nn.ReLU(inplace=True),
                                                nn.Linear(relative_position_bias_dim, num_heads, bias=False))
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()


    def define_relative_position_bias_table_and_index(self):
        if self.use_station_location:
            coords_t = torch.arange(self.window_size[0]).unsqueeze(-1)#.repeat(1, 1) # nt, 1
            relative_coords_t = coords_t[:, None, :] - coords_t[None, :, :] # nt, nt, 1
            self.register_buffer("relative_coords_t", relative_coords_t)
        else:
            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
            )
            self.register_buffer("relative_coords_table", relative_coords_table)
            
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, loc_: Union[Tensor, None]) -> Tensor:
        if self.use_station_location:
            #relative_coords_t = self.relative_coords_t.repeat(loc_.shape[0], loc_.shape[1], loc_.shape[1], 1) # B, nt*nx, nt*nx, 1 #FIXME: this is wrong, causing mismatch in relative_position_coords
            relative_coords_t = self.relative_coords_t.repeat(loc_.shape[0], 1,1,1).repeat_interleave(loc_.shape[1], dim=1).repeat_interleave(loc_.shape[1], dim=2)*self.time_unit # B, nt*nx, nt*nx, 1
            #device = loc_.device
            device = loc_.device.type
            dtype = loc_.dtype
            device = device if isinstance(device, str) and device != "mps" else "cpu"
            #TODO: IF we implement window for station here, more memory can be saved to deal with large number of stations ~300+  
            with torch.autocast(device_type=device, enabled=False):
                # keep precision in small distance
                loc = loc_.float().clone() # to avoid round error
                loc[...,:2] = torch.deg2rad(loc[...,:2]) # to radians
                # loc[:, :, None, :] station 1; loc[:, None, :, :] station 2
                # from station 1 to station 2
                relative_coords_sta = loc[:, None, :, :] - loc[:, :, None, :] # B, nx, nx, 3 (lon, lat, z)
                a = torch.sin(relative_coords_sta[:, :, :, 1]/2)**2 + torch.cos(loc[:, :, None, 1]) * torch.cos(loc[:, None, :, 1]) * torch.sin(relative_coords_sta[:, :, :, 0]/2)**2
                #NOTE: because of the precision, a may be slightly larger than 1, which will cause NaN in torch.sqrt
                #assert torch.all(a <= 1), f"a is larger than 1: {a.max()}"
                #assert torch.all(a >= 0), f"a is smaller than 0: {a.min()}"
                #a = torch.clamp(a, 0, 1) # clamp to avoid NaN
                relative_dist = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))*self.earth_radius # B, nx, nx
                # y = torch.sin(relative_coords_sta[:, :, :, 0]) * torch.cos(loc[:, None, :, 1]) # B, nx, nx, 1
                # x = torch.cos(loc[:, :, None, 1]) * torch.sin(loc[:, None, :, 1]) - torch.sin(loc[:, :, None, 1]) * torch.cos(loc[:, None, :, 1]) * torch.cos(relative_coords_sta[:, :, :, 0]) # B, nx, nx, 1
                # relative_azi = torch.atan2(y, x) # B, nx, nx
                relative_azi = torch.atan2(torch.sin(relative_coords_sta[:, :, :, 0]) * torch.cos(loc[:, None, :, 1]), 
                                           torch.cos(loc[:, :, None, 1]) * torch.sin(loc[:, None, :, 1]) - torch.sin(loc[:, :, None, 1]) * torch.cos(loc[:, None, :, 1]) * torch.cos(relative_coords_sta[:, :, :, 0])
                                           ) # B, nx, nx
                relative_coords_sta = torch.cat((relative_dist.unsqueeze(-1), 
                                                torch.sin(relative_azi).unsqueeze(-1), 
                                                torch.cos(relative_azi).unsqueeze(-1),
                                                relative_coords_sta[:, :, :, 2:]), dim=-1) # B, nx, nx, 4
            relative_coords_sta = relative_coords_sta.to(dtype) # recover to previous data type
            relative_coords_sta = relative_coords_sta.repeat(1, self.window_size[0], self.window_size[0], 1) # B, nt*nx, nt*nx, 4
            #relative_coords_sta = relative_coords_sta.repeat_interleave(self.window_size[0], dim=1).repeat_interleave(self.window_size[0], dim=2) # B, nt*nx, nt*nx, 4
            # relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 5
            # relative_position_bias = self.cpb_mlp(relative_coords) #bt, Wh*Ww, Wh*Ww, nH
            if not self.decoupled_relative_position_bias:
                relative_position_bias = self.cpb_mlp(torch.cat((relative_coords_t, relative_coords_sta), dim=-1)) #bt, Wh*Ww, Wh*Ww, nH
                #rpc = torch.cat((relative_coords_t, relative_coords_sta), dim=-1)
                #rpb_1 = self.cpb_mlp(rpc[:, :(self.window_size[0]*loc_.shape[1])//4, :,:]) #bt, Wh*Ww, Wh*Ww, nH
                #rpb_2 = self.cpb_mlp(rpc[:, (self.window_size[0]*loc_.shape[1])//4:((self.window_size[0]*loc_.shape[1])//4)*2, :, :]) #bt, Wh*Ww, Wh*Ww, nH
                #rpb_3 = self.cpb_mlp(rpc[:, ((self.window_size[0]*loc_.shape[1])//4)*2:((self.window_size[0]*loc_.shape[1])//4)*3, :, :]) #bt, Wh*Ww, Wh*Ww, nH
                #rpb_4 = self.cpb_mlp(rpc[:, ((self.window_size[0]*loc_.shape[1])//4)*3:, :, :]) #bt, Wh*Ww, Wh*Ww, nH
                #relative_position_bias = torch.cat((rpb_1, rpb_2, rpb_3, rpb_4), dim=1)
            else:
                relative_position_bias = self.cpb_mlp_time(relative_coords_t.to(dtype)) + self.cpb_mlp_space(relative_coords_sta) #bt, Wh*Ww, Wh*Ww, nH
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).view(loc.shape[0], self.num_heads, self.window_size[0], loc.shape[1], self.window_size[0], loc.shape[1]) #bt, nH, nt, nx, nt, nx
            relative_position_bias = 32 * torch.sigmoid(relative_position_bias)
        else:
            N = self.window_size[0] * self.window_size[1]
            relative_position_bias = (self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads))[self.relative_position_index]  # type: ignore[index]
            relative_position_bias = relative_position_bias.view(N, N, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        return relative_position_bias
    

    def forward(self, x: Tensor, loc: Union[Tensor, None]):
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
            use_station_location=self.use_station_location,
            use_relative_position_bias=True,
            training=self.training,
        )


class DynamicShiftedWindowAttention(ShiftedWindowAttentionV2):
    """
    See :func:`shifted_window_attention_v2`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        use_station_location: bool = True,
        use_relative_position_bias: bool = True,
        decoupled_relative_position_bias: bool = False,
        relative_position_bias_dim: int = 512,
        time_unit: float = 1.0,
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
            use_station_location=use_station_location,
            use_relative_position_bias=use_relative_position_bias,
            decoupled_relative_position_bias=decoupled_relative_position_bias,
            relative_position_bias_dim=relative_position_bias_dim,
            time_unit=time_unit,
        )

    def get_relative_position_bias(self, loc_: Union[Tensor, None]) -> Tensor:
        if self.use_station_location:
            #relative_coords_t = self.relative_coords_t.repeat(loc_.shape[0], loc_.shape[1], loc_.shape[1], 1) # B, nt*nx, nt*nx, 1 #FIXME: this is wrong, causing mismatch in relative_position_coords
            relative_coords_t = self.relative_coords_t.repeat(loc_.shape[0], 1,1,1).repeat_interleave(loc_.shape[1], dim=1).repeat_interleave(self.window_size[1], dim=2)*self.time_unit # B, nt*nx, nt*k, 1
            #device = loc_.device
            device = loc_.device.type
            dtype = loc_.dtype
            device = device if isinstance(device, str) and device != "mps" else "cpu"
            #TODO: IF we implement window for station here, more memory can be saved to deal with large number of stations ~300+  
            with torch.autocast(device_type=device, enabled=False):
                # keep precision in small distance
                loc = loc_.float().clone() # to avoid round error
                loc[...,:2] = torch.deg2rad(loc[...,:2]) # to radians
                # loc[:, :, None, :] station 1; loc[:, None, :, :] station 2
                # from station 1 to station 2
                relative_coords_sta = loc[:, None, :, :] - loc[:, :, None, :] # B, nx, nx, 3 (lon, lat, z)
                a = torch.sin(relative_coords_sta[:, :, :, 1]/2)**2 + torch.cos(loc[:, :, None, 1]) * torch.cos(loc[:, None, :, 1]) * torch.sin(relative_coords_sta[:, :, :, 0]/2)**2
                #NOTE: because of the precision, a may be slightly larger than 1, which will cause NaN in torch.sqrt
                #assert torch.all(a <= 1), f"a is larger than 1: {a.max()}"
                #assert torch.all(a >= 0), f"a is smaller than 0: {a.min()}"
                #a = torch.clamp(a, 0, 1) # clamp to avoid NaN
                relative_dist = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))*self.earth_radius # B, nx, nx
                _, knearest_station_indices = torch.topk(relative_dist, k=self.window_size[1], dim=-1, largest=False)
                # y = torch.sin(relative_coords_sta[:, :, :, 0]) * torch.cos(loc[:, None, :, 1]) # B, nx, nx, 1
                # x = torch.cos(loc[:, :, None, 1]) * torch.sin(loc[:, None, :, 1]) - torch.sin(loc[:, :, None, 1]) * torch.cos(loc[:, None, :, 1]) * torch.cos(relative_coords_sta[:, :, :, 0]) # B, nx, nx, 1
                # relative_azi = torch.atan2(y, x) # B, nx, nx
                relative_azi = torch.atan2(torch.sin(relative_coords_sta[:, :, :, 0]) * torch.cos(loc[:, None, :, 1]), 
                                           torch.cos(loc[:, :, None, 1]) * torch.sin(loc[:, None, :, 1]) - torch.sin(loc[:, :, None, 1]) * torch.cos(loc[:, None, :, 1]) * torch.cos(relative_coords_sta[:, :, :, 0])
                                           ) # B, nx, nx
                relative_coords_sta = torch.cat((relative_dist.unsqueeze(-1), 
                                                torch.sin(relative_azi).unsqueeze(-1), 
                                                torch.cos(relative_azi).unsqueeze(-1),
                                                relative_coords_sta[:, :, :, 2:]), dim=-1) # B, nx, nx, 4
                
            relative_coords_sta = relative_coords_sta.to(dtype) # recover to previous data type
            relative_coords_sta = relative_coords_sta.gather(2, knearest_station_indices.unsqueeze(-1).expand(-1, -1, -1, 4))
            relative_coords_sta = relative_coords_sta.repeat(1, self.window_size[0], self.window_size[0], 1) # B, nt*nx, nt*k, 4
            #relative_coords_sta = relative_coords_sta.repeat_interleave(self.window_size[0], dim=1).repeat_interleave(self.window_size[0], dim=2) # B, nt*nx, nt*nx, 4
            # relative_coords = torch.cat((relative_coords_t, relative_coords_sta), dim=-1) # B, nt*nx, nt*nx, 5
            # relative_position_bias = self.cpb_mlp(relative_coords) #bt, Wh*Ww, Wh*Ww, nH
            if not self.decoupled_relative_position_bias:
                relative_position_bias = self.cpb_mlp(torch.cat((relative_coords_t, relative_coords_sta), dim=-1)) #bt, Wh*Ww, Wh*Ww, nH
            else:
                relative_position_bias = self.cpb_mlp_time(relative_coords_t.to(dtype)) + self.cpb_mlp_space(relative_coords_sta) #bt, Wh*Ww, Wh*Ww, nH
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).view(loc.shape[0], self.num_heads, self.window_size[0], loc.shape[1], self.window_size[0], self.window_size[1]) #bt, nH, nt, nx, nt, k
            relative_position_bias = 32 * torch.sigmoid(relative_position_bias)
        else:
            N = self.window_size[0] * self.window_size[1]
            relative_position_bias = (self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads))[self.relative_position_index]  # type: ignore[index]
            relative_position_bias = relative_position_bias.view(N, N, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        return relative_position_bias, knearest_station_indices
    

    def forward(self, x: Tensor, loc: Union[Tensor, None]):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
            loc (Tensor): Station location with layout of [B, W, 3]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias, knearest_station_indices = self.get_relative_position_bias(loc)
        # assert not torch.isnan(relative_position_bias).any(), "relative_position_bias contains NaN"
        return dynamic_shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            topk_station_indices=knearest_station_indices,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            use_station_location=self.use_station_location,
            use_relative_position_bias=True,
            training=self.training,
        )

class RotaryAxialWindowAttention(ShiftedWindowAttentionV2):
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        theta_t: float = 10000.0,
        freq_lon: Tuple = (0, 14), 
        freq_lat: Tuple = (0, 13),
        freq_vertical: Tuple[float, float] = (-2.0, 8.0),
        use_station_location: bool = True,
        use_relative_position_bias: bool = False,
        decoupled_relative_position_bias: bool = False,
        relative_position_bias_dim: int = 512,
        time_unit: float = 1.0,
        use_cos_attention: bool = False,
        **kwargs,
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
            use_station_location=use_station_location,
            decoupled_relative_position_bias=decoupled_relative_position_bias,
            relative_position_bias_dim=relative_position_bias_dim,
            time_unit=time_unit,
        )
        
        self.dim = dim
        self.use_rope = True
        self.theta_t = theta_t
        self.freq_lon = freq_lon
        self.freq_lat = freq_lat
        self.freq_vertical = freq_vertical
        self.use_relative_position_bias = use_relative_position_bias
        if not self.use_relative_position_bias:
            if use_station_location:
                self.relative_coords_t = None
            else:
                self.relative_coords_table = None
                self.relative_position_index = None
            self.cpb_mlp = None
        if not use_cos_attention:
            self.logit_scale = None
                
        self.init_freqs()
        
    def init_freqs(self):
        dim = self.dim // self.num_heads
        rope_t_t = torch.arange(self.window_size[0], dtype=torch.float32)
        self.register_buffer("rope_t_t", rope_t_t)
        freqs_t = 1.0 / (self.theta_t ** (torch.arange(0, dim, 8)[: (dim // 8)].float() / dim))
        self.register_buffer("freqs_t", freqs_t)
        freqs_x = (1/360 * 2. ** torch.linspace(self.freq_lon[1], self.freq_lon[0], dim // 8))
        self.register_buffer("freqs_x", freqs_x)
        freqs_y = (1/360 * 2. ** torch.linspace(self.freq_lat[1], self.freq_lat[0], dim // 8))
        self.register_buffer("freqs_y", freqs_y)
        freqs_z = 2.**torch.linspace(self.freq_vertical[0], self.freq_vertical[1], dim // 8)
        self.register_buffer("freqs_z", freqs_z)
        
    def compute_cis_for_broadcast(self, loc: torch.Tensor):
        B, nsta, ch = loc.shape
        rope_t = torch.cat([self.rope_t_t[None, :, None, None].repeat(B, 1, nsta, 1),
                            loc[:, None, :, :].repeat(1, self.window_size[0], 1, 1)], dim=-1)
        #rope_t = loc[:, None, :, :2].repeat(1, self.window_size[0], 1, 1)
        device = loc.device
        device = device if isinstance(device, str) and device != "mps" else "cpu"
        # No float 16 for this range
        with torch.autocast(device_type=device, enabled=False):
            freqs_cis = []
            freqs_t = rope_t[:, :, :, 0:1] * self.freqs_t[None, None, None, :]
            freqs_x = rope_t[:, :, :, 1:2] * self.freqs_xy[None, None, None, :]
            freqs_y = rope_t[:, :, :, 2:3] * self.freqs_xy[None, None, None, :]
            freqs_z = rope_t[:, :, :, 3:4] * self.freqs_z[None, None, None, :]
            #freqs_x = rope_t[:, :, :, 0:1] * self.freqs_xy[None, None, None, :]
            #freqs_y = rope_t[:, :, :, 1:2] * self.freqs_xy[None, None, None, :]
            freqs_cis.append(torch.polar(torch.ones_like(freqs_t), freqs_t))
            freqs_cis.append(torch.polar(torch.ones_like(freqs_x), freqs_x))
            freqs_cis.append(torch.polar(torch.ones_like(freqs_y), freqs_y))
            freqs_cis.append(torch.polar(torch.ones_like(freqs_z), freqs_z))

        return torch.cat(freqs_cis, dim=-1).unsqueeze(-2).repeat(1, 1, 1, self.num_heads, 1).flatten(-2, -1)
    
    def forward(self, x: Tensor, loc: Union[Tensor, None]):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
            loc (Tensor): Station location with layout of [B, W, 3]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        if self.use_relative_position_bias:
            relative_position_bias = self.get_relative_position_bias(loc)
        else:
            relative_position_bias = None
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
            use_station_location=self.use_station_location,
            use_relative_position_bias=self.use_relative_position_bias,
            use_rope=True,
            freq_cis=self.compute_cis_for_broadcast(loc),
            training=self.training,
        )


class RotaryMixedWindowAttention(RotaryAxialWindowAttention):
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        theta_t: float = 10000.0,
        freq_lon: Tuple = (0, 14), 
        freq_lat: Tuple = (0, 13),
        freq_vertical: Tuple[float, float] = (-2.0, 8.0),
        use_station_location: bool = True,
        use_relative_position_bias: bool = False,
        decoupled_relative_position_bias: bool = False,
        relative_position_bias_dim: int = 512,
        time_unit: float = 1.0,
        use_cos_attention: bool = False,
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
            theta_t=theta_t,
            freq_lon=freq_lon,
            freq_lat=freq_lat,
            freq_vertical=freq_vertical,
            use_station_location=use_station_location,
            use_relative_position_bias=use_relative_position_bias,
            decoupled_relative_position_bias=decoupled_relative_position_bias,
            relative_position_bias_dim=relative_position_bias_dim,
            time_unit=time_unit,
            use_cos_attention=use_cos_attention,
        )
        
    def init_freqs(self, dt_s = 0.04):
        rope_t_t = torch.arange(self.window_size[0], dtype=torch.float32)*self.time_unit
        self.register_buffer("rope_t_t", rope_t_t)
        head_dim = self.dim // self.num_heads
        freqs_t = []
        freqs_x = []
        freqs_y = []
        freqs_z = []
        mag_t = 1 / (self.theta_t ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim)) / dt_s
        mag_x = (1/360 * 2. ** torch.linspace(self.freq_lon[1], self.freq_lon[0], head_dim // 4))
        mag_y = (1/180 * 2. ** torch.linspace(self.freq_lat[1], self.freq_lat[0], head_dim // 4))
        # +1 because cos(angle) will downscale the frequency
        #mag_xy = 2.**torch.linspace(self.freq_horizon[0]+1, self.freq_horizon[1]+1, head_dim // 4)
        mag_z = 2.**torch.linspace(self.freq_vertical[0]+1, self.freq_vertical[1]+1, head_dim // 4)
        
        for _ in range(self.num_heads):
            angles = torch.rand(1) * 2 * torch.pi
            ft = torch.cat([mag_t * torch.cos(angles), mag_t * torch.cos(angles + torch.pi/2)], dim=-1)
            freqs_t.append(ft)
            fx = torch.cat([mag_x * torch.cos(angles), mag_x * torch.cos(angles + torch.pi/2)], dim=-1)
            freqs_x.append(fx)
            fy = torch.cat([mag_y * torch.sin(angles), mag_y * torch.sin(angles + torch.pi/2)], dim=-1)
            freqs_y.append(fy)
            fz = torch.cat([mag_z * torch.sin(angles), mag_z * torch.sin(angles + torch.pi/2)], dim=-1)
            freqs_z.append(fz)
        freqs_t = torch.stack(freqs_t, dim=0).flatten() # nH * head_dim//2
        freqs_x = torch.stack(freqs_x, dim=0).flatten() # nH * head_dim//2
        freqs_y = torch.stack(freqs_y, dim=0).flatten() # nH * head_dim//2
        freqs_z = torch.stack(freqs_z, dim=0).flatten() # nH * head_dim//2
        freqs = torch.stack([freqs_t, freqs_x, freqs_y, freqs_z], dim=0)
        
        self.freqs = nn.Parameter(freqs, requires_grad=True)
        
    def compute_cis_for_broadcast(self, loc: torch.Tensor):
        B, nsta, ch = loc.shape
        rope_t = torch.cat([self.rope_t_t[None, :, None, None].repeat(B, 1, nsta, 1),
                            loc[:, None, :, :].repeat(1, self.window_size[0], 1, 1)], dim=-1)
        device = loc.device
        device = device if isinstance(device, str) and device != "mps" else "cpu"
        # No float 16 for this range
        with torch.autocast(device_type=device, enabled=False):
            freqs_cis = []
            freqs_t = rope_t[:, :, :, 0:1] * self.freqs[0:1, None, None, :]
            freqs_x = rope_t[:, :, :, 1:2] * self.freqs[1:2, None, None, :]
            freqs_y = rope_t[:, :, :, 2:3] * self.freqs[2:3, None, None, :]
            freqs_z = rope_t[:, :, :, 3:4] * self.freqs[3:4, None, None, :]
            freqs_cis.append(torch.polar(torch.ones_like(freqs_t), freqs_t+freqs_x+freqs_y+freqs_z))
            #freqs_x = rope_t[:, :, :, 0:1] * self.freqs[0:1, None, None, :]
            #freqs_y = rope_t[:, :, :, 1:2] * self.freqs[1:2, None, None, :]
            #freqs_cis.append(torch.polar(torch.ones_like(freqs_x), freqs_x+freqs_y))

        return torch.cat(freqs_cis, dim=-1)


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
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
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
        use_station_location: bool = True,
        use_relative_position_bias: bool = True,
        decoupled_relative_position_bias: bool = False,
        relative_position_bias_dim: int = 512,
        time_unit: float = 1.0,
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
            use_station_location=use_station_location,
            use_relative_position_bias=use_relative_position_bias,
            decoupled_relative_position_bias=decoupled_relative_position_bias,
            time_unit=time_unit,
            relative_position_bias_dim=relative_position_bias_dim,
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
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
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
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
        use_station_location: bool = True,
        use_relative_position_bias: bool = True,
        decoupled_relative_position_bias: bool = False,
        relative_position_bias_dim: int = 512,
        time_unit: float = 1.0,
        **kwargs,
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
            use_station_location=use_station_location,
            use_relative_position_bias=use_relative_position_bias,
            decoupled_relative_position_bias=decoupled_relative_position_bias,
            relative_position_bias_dim=relative_position_bias_dim,
            time_unit=time_unit,
        )

    def forward(self, meta: Dict):
        x, loc = meta["x"], meta["loc"]
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x, loc)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return {"x": x, "loc": loc}

class RotarySwinBlock(SwinTransformerBlockV2):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = RotaryAxialWindowAttention,
        use_station_location: bool = True,
        use_relative_position_bias: bool = False,
        decoupled_relative_position_bias: bool = False,
        relative_position_bias_dim: int = 512,
        time_unit: float = 1.0,
        use_rope: bool = True,
        use_rope_mixed: bool = False,
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
            attn_layer=RotaryAxialWindowAttention if not use_rope_mixed else RotaryMixedWindowAttention,
            use_station_location=use_station_location,
            use_relative_position_bias=use_relative_position_bias,
            decoupled_relative_position_bias=decoupled_relative_position_bias,
            relative_position_bias_dim=relative_position_bias_dim,
            time_unit=time_unit,
        )


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
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
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        patchify_kernel: Optional[List[int]] = [32, 1],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block_name: Optional[str] = None,
        downsample_layer: Callable[..., nn.Module] = PatchReduction,#PatchMerging,
        out_indices=(0, 1, 2, 3),
        use_station_location: bool = True,
        use_relative_position_bias: bool = True,
        use_rope: bool = False,
        use_rope_mixed: bool = False,
        decoupled_relative_position_bias: bool = False,
        relative_position_bias_dim: int = 512,
        sampling_rate: float = 100.0,
    ):
        super().__init__()

        block: Optional[Callable[..., nn.Module]]
        if block_name is None or block_name == "SwinTransformerBlock":
            block = SwinTransformerBlock
        elif block_name == "SwinTransformerBlockV2":
            block = SwinTransformerBlockV2
        elif block_name == "RotarySwinBlock":
            block = RotarySwinBlock
        else:
            raise ValueError(f"Unknown Swin Transformer block: {block_name}")

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)
            
        if patchify_kernel is None:
            patchify_kernel = patch_size.copy()
        if use_station_location:
            assert (math.ceil((patchify_kernel[1] - patch_size[1]) / 2)==0) & patchify_kernel[1]==1 & patch_size[1]==1, f"no downsample in station dimension, but got {patchify_kernel[1]} and {patch_size[1]}"
        # split image into non-overlapping patches
        self.split_patch = nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patchify_kernel[0], patchify_kernel[1]), stride=(patch_size[0], patch_size[1]), padding=(math.ceil((patchify_kernel[0] - patch_size[0]) / 2), math.ceil((patchify_kernel[1] - patch_size[1]) / 2))
                ),
                #nn.Conv2d(
                #    3, embed_dim//2, kernel_size=(3, patchify_kernel[1]), stride=(patch_size[0]//2, patch_size[1]), padding=(1, math.ceil((patchify_kernel[1] - patch_size[1]) / 2))
                #),
                #Permute([0, 2, 3, 1]),
                #norm_layer(embed_dim//2),
                #Permute([0, 3, 1, 2]),
                #nn.GELU(),
                #nn.Conv2d(
                #    embed_dim//2, embed_dim, kernel_size=(3, patchify_kernel[1]), stride=(patch_size[0]//2, patch_size[1]), padding=(1, math.ceil((patchify_kernel[1] - patch_size[1]) / 2))
                #),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        dt_s = 1.0/sampling_rate
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                #temp_window_size = [window_size[0], min(window_size[1], 4**i_stage)] if i_stage <= 1 else window_size
                temp_window_size = window_size
                #temp_window_size = [window_size[0], window_size[1]*(i_stage+1)] if i_stage <= 1 else [window_size[0], window_size[1]*8]
                print(f"Stage {i_stage}, Layer {i_layer}, Window Size: {temp_window_size}, Stochastic Depth Prob: {sd_prob:.4f}")
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=temp_window_size,# if i_stage > 1 else temp_window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],# if i_stage > 1 else [0 if i_layer % 2 == 0 else w // 2 for w in temp_window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        #attn_layer=ShiftedWindowAttentionV2 if i_stage==0 else DynamicShiftedWindowAttention,
                        use_station_location=use_station_location,
                        use_relative_position_bias=use_relative_position_bias,
                        use_rope=use_rope,
                        use_rope_mixed=use_rope_mixed,
                        decoupled_relative_position_bias=decoupled_relative_position_bias,
                        relative_position_bias_dim=relative_position_bias_dim,
                        time_unit=dt_s*(4**(i_stage+1)) if sampling_rate>0 else 1.0,
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
        for m in self.split_patch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table', 'freqs'}

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
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights,
    progress: bool,
    block_name: Optional[str],
    use_station_location: bool,
    use_relative_position_bias: bool,
    use_rope: bool = False,
    use_rope_mixed: bool = False,
    decoupled_relative_position_bias: bool = False,
    **kwargs: Any,
) -> SwinTransformer:
    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        block_name=block_name,
        use_station_location=use_station_location,
        use_relative_position_bias=use_relative_position_bias,
        use_rope=use_rope,
        use_rope_mixed=use_rope_mixed,
        decoupled_relative_position_bias=decoupled_relative_position_bias,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model



def swin_t(*, weights=None, progress: bool = True, use_station_location: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.

    Args:
        weights (optional): The pretrained weights to use. 
            By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``eqnet.models.swin_transformer.SwinTransformer``
            base class. 
    """

    return _swin_transformer(
        patch_size=[4, 1],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 10],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        block_name="SwinTransformerBlock",
        use_station_location=use_station_location,
        **kwargs,
    )


def swin_s(*, weights=None, progress: bool = True, use_station_location: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.

    Args:
        weights (optional): The pretrained weights to use. 
            By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``eqnet.models.swin_transformer.SwinTransformer``
            base class. 
    """
    
    return _swin_transformer(
        patch_size=[4, 1],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 10],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        block_name="SwinTransformerBlock",
        use_station_location=use_station_location,
        **kwargs,
    )


def swin_b(*, weights=None, progress: bool = True, use_station_location: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.

    Args:
        weights (optional): The pretrained weights to use. 
            By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``eqnet.models.swin_transformer.SwinTransformer``
            base class. 
    """
    
    return _swin_transformer(
        patch_size=[4, 1],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        block_name="SwinTransformerBlock",
        use_station_location=use_station_location,
        **kwargs,
    )


def swin_v2_t(*, weights=None, progress: bool = True, use_station_location: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_tiny architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (optional): The pretrained weights to use. 
            By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``eqnet.models.swin_transformer.SwinTransformer``
            base class. 
    """
    
    return _swin_transformer(
        patch_size=[4, 1],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 10],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        block_name="SwinTransformerBlockV2",
        use_station_location=use_station_location,
        **kwargs,
    )


def swin_v2_s(*, weights=None, progress: bool = True, use_station_location: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_small architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (optional): The pretrained weights to use. 
            By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``eqnet.models.swin_transformer.SwinTransformer``
            base class. 
    """
    
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        block_name="SwinTransformerBlockV2",
        use_station_location=use_station_location,
        **kwargs,
    )


def swin_v2_b(*, weights=None, progress: bool = True, use_station_location: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_base architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (optional): The pretrained weights to use. 
            By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``eqnet.models.swin_transformer.SwinTransformer``
            base class. 
    """
    
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        block_name="SwinTransformerBlockV2",
        use_station_location=use_station_location,
        **kwargs,
    )
