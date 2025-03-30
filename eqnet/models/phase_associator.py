# Adapted from [SAM](https://github.com/facebookresearch/segment-anything)

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import math
from typing import Any, List, Tuple, Type, Optional, Dict

from .ops import LayerNorm2d, TwoWayTransformer
from .losses import sigmoid_focal_loss, dice_loss, iou_loss


class PhaseAssociator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        prompt_embed_dim: int = 64,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 16,
        event_scale: int = 16,
        sampling_rate: float = 100.0,
        weights: dict = {'focal': 0.8, 'dice': 0.05, 'iou': 0.1},
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.prompt_embed_dim = prompt_embed_dim
        self.event_scale = event_scale
        self.sampling_rate = sampling_rate
        self.weights = weights
        
        self.feature_downscaling = nn.Sequential(
            nn.Conv2d(feature_dim, prompt_embed_dim, kernel_size=(2, 1), stride=(2, 1)),
            LayerNorm2d(prompt_embed_dim),
            nn.GELU(),
            nn.Conv2d(prompt_embed_dim, prompt_embed_dim, kernel_size=(2, 1), stride=(2, 1)),
            LayerNorm2d(prompt_embed_dim),
            nn.GELU(),
        )
        
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            event_scale=event_scale,
            sampling_rate=sampling_rate,
        )
        self.mask_decoder = MaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                num_heads=4,
                mlp_ratio=8,
                ),
            num_multimask_outputs=1,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        )
    
    def forward(self, features: Dict[str, Tensor], station_location: Tensor, prompt: Tuple[Tensor, Tensor], target: Tensor=None):
        waveform_embeddings = features['fusion']
        waveform_embeddings = waveform_embeddings.permute(0, 2, 3, 1).contiguous()  # batch, channel, time, station
        
        waveform_embeddings = self.feature_downscaling(waveform_embeddings)
        point_embeddings = self.prompt_encoder(points=prompt)
        bt, ch, nt, nsta = waveform_embeddings.shape
        waveform_pe = self.prompt_encoder.get_dense_pe(time_size=nt, station_location=station_location)
        
        associated_masks, iou_pred = self.mask_decoder(
            waveform_embeddings=waveform_embeddings,
            waveform_pe=waveform_pe,
            sparse_prompt_embeddings=point_embeddings,
        )
        
        if self.training:
            valid_stations = torch.any(target > 0, axis=1).float()
            return None, self.losses({'association': associated_masks, 'iou': iou_pred}, target, valid_stations)
        else:
            loss = {}
            if target is not None:
                valid_stations = torch.any(target > 0, axis=1).float()
                loss = self.losses({'association': associated_masks, 'iou': iou_pred}, target, valid_stations)
            return {'association': associated_masks, 'iou': iou_pred}, loss
        
    def losses(self, outputs: dict, targets: Tensor, valid_stations: Tensor):
        loss_focal = sigmoid_focal_loss(outputs['association'], targets, valid_stations)
        loss_dice = dice_loss(outputs['association'], targets, valid_stations)
        loss_iou = iou_loss(outputs['association'], targets, outputs['iou'], valid_stations)
        loss = self.weights['focal'] * loss_focal + self.weights['dice'] * loss_dice + self.weights['iou'] * loss_iou
        
        return {"loss": loss, "loss_focal": loss_focal, "loss_dice": loss_dice, "loss_iou": loss_iou}


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs#num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        #self.output_upscaling = nn.Sequential(
        #    nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        #    LayerNorm2d(transformer_dim // 4),
        #    activation(),
        #    nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        #    activation(),
        #)
        self.output_upscaling = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPHead(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPHead(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        waveform_embeddings: Tensor,
        waveform_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (Tensor): the embeddings from the image encoder
          image_pe (Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          Tensor: batched predicted masks
          Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            waveform_embeddings=waveform_embeddings,
            waveform_pe=waveform_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        waveform_embeddings: Tensor,
        waveform_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        #src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        #pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        assert waveform_embeddings.shape[0] == tokens.shape[0], f"Batchify not aligned: {waveform_embeddings.shape=}, {tokens.shape=}"
        assert waveform_pe.shape[0] == tokens.shape[0], f"Batchify not aligned: {waveform_pe.shape=}, {tokens.shape=}"
        b, c, h, w = waveform_embeddings.shape

        # Run the transformer
        hs, src = self.transformer(waveform_embeddings, waveform_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).contiguous()
        src = self.output_upscaling(src.view(b, c, h, w))
        hyper_in_list: List[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = src.shape
        masks = (hyper_in @ src.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        event_scale: int = 16,
        sampling_rate: float = 100.0,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dt_s = 1 / sampling_rate * event_scale # time unit
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2, event_scale=event_scale, sampling_rate=sampling_rate)

        #self.num_point_embeddings: int = 2  # pos/neg point
        self.num_point_embeddings: int = 1  # pos/neg point
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        #self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self, time_size: Tensor, station_location: Tensor) -> Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(time_size, station_location)

    def _embed_points(
        self,
        points: Tensor,
        labels: Tensor,
        pad: bool = False,
    ) -> Tensor:
        """Embeds point prompts."""
        #points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points)
        #point_embedding[labels == -1] = 0.0
        #point_embedding[labels == -1] += self.not_a_point_embed.weight
        #point_embedding[labels == 0] += self.point_embeddings[0].weight
        #point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 1] += self.point_embeddings[0].weight
        return point_embedding

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(Tensor, Tensor) or none): point coordinates
            and labels to embed.
          masks (Tensor or none): masks to embed

        Returns:
          Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = points.shape[0] if points is not None else 1
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords = points[:,:, :3]
            labels = points[:,:, 3]
            point_embeddings = self._embed_points(coords, labels, pad=False)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        return sparse_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None, 
                 event_scale: int = 16, sampling_rate: float = 100.0,) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )
        self.dt_s = 1 / sampling_rate * event_scale  # time unit

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        # coords = 2 * coords - 1
        # coords have shape d_1 x ... x d_n x 3 shape, (time, x, y)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    
    def forward(self, time_size: int, station_location: Tensor) -> Tensor:
        """
        Generate positional encoding for a grid of the specified size.
        Args:
            time_size (int): The time size of the image embedding, as (H, W).
            station_location (Tensor): The location of the station.
        Returns:
            Tensor: Positional encoding with shape
                1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        device: Any = self.positional_encoding_gaussian_matrix.device
        t_embed = torch.ones((1, time_size, 1), device=device, dtype=torch.float32)
        t_embed = t_embed.cumsum(dim=1) * self.dt_s
        t_embed = t_embed.repeat_interleave(station_location.shape[1], dim=1).repeat(station_location.shape[0], 1, 1)
        x_embed = station_location[..., :2].repeat(1, time_size, 1)
        grid = torch.cat([t_embed, x_embed], dim=-1).view(-1, time_size, station_location.shape[1], 3)
        
        pe = self._pe_encoding(grid)
        return pe.permute(0, 3, 1, 2).contiguous()  # B x C x T x X

    def forward_with_coords(
        self, coords_input: Tensor
    ) -> Tensor:
        """Positionally encode points that are converted to physical coordinates."""
        coords = coords_input.float()
        coords[:, :, 0] = coords[:, :, 0] * self.dt_s
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class PositionEmbeddingFourier(nn.Module):
    """
    Positional encoding using predefined spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        raise
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: Tensor, image_size: Tuple[int, int]
    ) -> Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
