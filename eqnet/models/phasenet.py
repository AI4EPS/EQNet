"""
PhaseNet: Deep learning for seismic phase picking.

This module provides the PhaseNet architecture with various heads for:
- Phase picking (P/S wave detection)
- Polarity classification
- Event detection and timing
- Prompt-based picking (SAM-style)

The architecture follows modern best practices from timm and torchvision.
"""
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .prompt import MaskDecoder, PromptEncoder, TwoWayTransformer
from .unet import Unet
from .unet2018 import UNet2018


# =============================================================================
# Minimal Seismic UNet Configuration
# =============================================================================

# Seismic config: single station, 3-channel (ENZ), 1D temporal
# channels/phase_channels match Unet defaults but kept explicit for clarity
SEISMIC_UNET_CONFIG = dict(
    dim=16,
    channels=3,
    phase_channels=3,
    # pre-configured for add_stft=True (100Hz seismic: 33 -> 16 freq bins)
    stft_n_fft=33,
    stft_dim_divisor=2,
)


# =============================================================================
# Loss Functions
# =============================================================================

def kl_divergence_loss(inputs: Tensor, targets: Tensor, mask: Tensor = None, num_classes: int = 3) -> Tensor | None:
    """KL divergence loss with optional masking.

    Uses cross-entropy minus minimum entropy for numerical stability.

    Args:
        inputs: Predictions (logits), shape (B, C, ...)
        targets: Target distributions, shape (B, C, ...)
        mask: Optional mask, shape (B, 1, ...) or (B, ...)
        num_classes: Number of output classes (1 for binary, >1 for multiclass)

    Returns:
        Scalar loss value, or None if mask is entirely zero.
    """
    inputs = inputs.float()
    log_targets = torch.nan_to_num(torch.log(targets))

    # Handle size mismatch
    if inputs.shape[-2:] != targets.shape[-2:]:
        inputs = F.interpolate(inputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    if mask is None:
        if num_classes == 1:
            # Binary cross entropy
            min_loss = -(targets * log_targets + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") - min_loss
        else:
            # Multiclass cross entropy
            min_loss = -(targets * log_targets).sum(dim=1)
            loss = F.cross_entropy(inputs, targets, reduction="none") - min_loss
        return loss.mean()

    # Masked loss — skip if mask is entirely zero (no valid labels in batch)
    mask = mask.float()
    mask_sum = mask.sum()
    if mask_sum == 0:
        return None
    mask_sum = mask_sum.clamp(min=1.0)

    if num_classes == 1:
        min_loss = -(targets * log_targets + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") - min_loss
        return (loss * mask).sum() / mask_sum
    else:
        min_loss = -(targets * log_targets).sum(dim=1)
        loss = F.cross_entropy(inputs, targets, reduction="none") - min_loss
        if mask.dim() > loss.dim():
            mask = mask.squeeze(1)
        return (loss * mask).sum() / mask_sum


def regression_loss(inputs: Tensor, targets: Tensor, mask: Tensor = None, scaling: float = 1000.0) -> Tensor | None:
    """L1 regression loss with optional masking and scaling.

    Args:
        inputs: Predictions, shape (B, C, ...)
        targets: Target values, shape (B, C, ...)
        mask: Optional mask
        scaling: Scale factor for loss normalization

    Returns:
        Scalar loss value, or None if mask is entirely zero.
    """
    inputs = inputs.float()

    if inputs.shape[-2:] != targets.shape[-2:]:
        inputs = F.interpolate(inputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    if mask is None:
        return F.mse_loss(inputs, targets) / scaling

    # Skip if mask is entirely zero (no valid labels in batch)
    mask = mask.float()
    mask_sum = mask.sum()
    if mask_sum == 0:
        return None
    mask_sum = mask_sum.clamp(min=1.0)
    return (F.l1_loss(inputs, targets, reduction="none") * mask).sum() / mask_sum / scaling


# =============================================================================
# Head Modules (Loss only - activations computed inline)
# =============================================================================

class PhaseHead(nn.Module):
    """Head for phase picking - computes loss on backbone output."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: Tensor, targets: Tensor = None, mask: Tensor = None):
        loss = kl_divergence_loss(logits, targets, mask, self.num_classes) if targets is not None else None
        return logits, loss


class PolarityHead(nn.Module):
    """Head for polarity prediction - computes loss on backbone output."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, targets: Tensor = None, mask: Tensor = None):
        loss = kl_divergence_loss(logits, targets, mask, num_classes=1) if targets is not None else None
        return logits, loss


class EventHead(nn.Module):
    """Head for event detection and timing."""

    def __init__(self, scaling: float = 1000.0):
        super().__init__()
        self.scaling = scaling

    def forward_center(self, logits: Tensor, targets: Tensor = None, mask: Tensor = None):
        loss = kl_divergence_loss(logits, targets, mask, num_classes=1) if targets is not None else None
        return logits, loss

    def forward_time(self, preds: Tensor, targets: Tensor = None, mask: Tensor = None):
        loss = regression_loss(preds, targets, mask, self.scaling) if targets is not None else None
        return preds, loss


class PromptHead(nn.Module):
    """Head for prompt-based picking (SAM-style)."""

    def __init__(self, prompt_embed_dim: int = 128, input_size=(8, 256), embedding_size=(8, 16)):
        super().__init__()
        self.prompt_embed_dim = prompt_embed_dim
        self.input_size = input_size
        self.embedding_size = embedding_size

        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=embedding_size,
            input_image_size=input_size,
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=0,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=512,
                num_heads=4,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=16,
        )

    def forward(self, features: Tensor, points: Tensor, pos: Tensor, targets: Tensor = None):
        B, S, T, _ = pos.shape

        pos = pos.view(B, S * T, 3)
        labels = torch.ones((points.shape[0], points.shape[1]), device=points.device)
        points = (points, labels)
        pos_labels = torch.ones((pos.shape[0], pos.shape[1]), device=pos.device)
        pos = (pos, pos_labels)
        image_size = (S, T * 16)
        image_embedding_size = (S, T)

        point_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=None, masks=None,
            image_size=image_size, image_embedding_size=image_embedding_size
        )
        pos_embeddings, _ = self.prompt_encoder(
            points=pos, boxes=None, masks=None,
            image_size=image_size, image_embedding_size=image_embedding_size
        )
        C = point_embeddings.shape[-1]
        pos_embeddings = pos_embeddings.permute(0, 2, 1).reshape(B, C, S, T)

        low_res_masks = []
        iou_predictions = []

        for i in range(B):
            mask, iou = self.mask_decoder(
                image_embeddings=features[i:i+1],
                image_pe=pos_embeddings[i:i+1],
                sparse_prompt_embeddings=point_embeddings[i:i+1],
                dense_prompt_embeddings=dense_embeddings[i:i+1],
                multimask_output=False,
            )
            low_res_masks.append(mask)
            iou_predictions.append(iou)

        low_res_masks = torch.cat(low_res_masks, dim=0)

        loss = None
        if targets is not None:
            # Focal loss for prompt
            prob = low_res_masks.sigmoid()
            min_loss = -(
                targets * torch.nan_to_num(torch.log(targets)) +
                (1 - targets) * torch.nan_to_num(torch.log(1 - targets))
            )
            ce_loss = F.binary_cross_entropy_with_logits(low_res_masks, targets, reduction="none") - min_loss
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = 10 * (ce_loss * ((1 - p_t) ** 2)).mean()

        return low_res_masks, loss


# =============================================================================
# PhaseNet Model
# =============================================================================

class PhaseNet(nn.Module):
    """PhaseNet: Deep learning model for seismic phase picking.

    Args:
        backbone: Backbone type ("unet", "unet64", "unet256", "unet1024")
        log_scale: Apply log transform to input
        add_stft: Add STFT features
        add_polarity: Enable polarity prediction
        add_event: Enable event detection
        add_prompt: Enable prompt-based picking
        event_center_loss_weight: Weight for event center loss
        event_time_loss_weight: Weight for event time loss
        polarity_loss_weight: Weight for polarity loss
        prompt_loss_weight: Weight for prompt loss
    """

    def __init__(
        self,
        backbone: str = "unet",
        log_scale: bool = False,
        add_stft: bool = False,
        add_polarity: bool = False,
        add_event: bool = False,
        add_prompt: bool = False,
        event_center_loss_weight: float = 1.0,
        event_time_loss_weight: float = 1.0,
        polarity_loss_weight: float = 1.0,
        prompt_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.add_stft = add_stft
        self.add_event = add_event
        self.add_polarity = add_polarity
        self.add_prompt = add_prompt
        self.event_center_loss_weight = event_center_loss_weight
        self.event_time_loss_weight = event_time_loss_weight
        self.polarity_loss_weight = polarity_loss_weight
        self.prompt_loss_weight = prompt_loss_weight

        # Build backbone with seismic-optimized defaults
        backbone_kwargs = {
            **SEISMIC_UNET_CONFIG,  # minimal seismic defaults
            "log_scale": log_scale,
            "add_stft": add_stft,
            "add_polarity": add_polarity,
            "add_event": add_event,
            "add_prompt": add_prompt,
            **kwargs,  # user overrides
        }

        if backbone == "unet":
            self.backbone = Unet(**backbone_kwargs)
        elif backbone == "unet2018":
            # Original PhaseNet architecture (Zhu & Beroza 2019).
            # Does not support add_stft / add_polarity / add_event / add_prompt.
            unet2018_keys = {"depths", "filters_root", "kernel_size", "pool_size", "n_channel", "n_class"}
            unet2018_kwargs = {k: v for k, v in kwargs.items() if k in unet2018_keys}
            self.backbone = UNet2018(**unet2018_kwargs)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'unet' or 'unet2018'.")

        # Compute embed_dim from dim_mults
        dim = backbone_kwargs.get("dim", 16)
        dim_mults = backbone_kwargs.get("dim_mults", (1, 2, 4, 8))
        embed_dim = dim * dim_mults[-1]  # mid_dim for prompt

        # Heads (loss computation only)
        self.phase_head = PhaseHead(num_classes=3)

        if self.add_polarity:
            self.polarity_head = PolarityHead()

        if self.add_event:
            self.event_head = EventHead()

        if self.add_prompt:
            self.prompt_head = PromptHead(prompt_embed_dim=embed_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = batch["data"].to(self.device)

        # Get targets
        phase_pick = batch.get("phase_pick")
        phase_mask = batch.get("phase_mask")
        event_center = batch.get("event_center")
        event_time = batch.get("event_time")
        # Event masks: separate masks for detection vs regression tasks
        # - event_center_mask: mask for center detection loss (Gaussian prediction)
        # - event_time_mask: mask for time regression loss (timing error)
        event_center_mask = batch.get("event_center_mask")
        event_time_mask = batch.get("event_time_mask")
        polarity = batch.get("polarity")
        polarity_mask = batch.get("polarity_mask")
        prompt_center = batch.get("prompt_center")

        # Move to device
        if phase_pick is not None:
            phase_pick = phase_pick.to(self.device)
        if phase_mask is not None:
            phase_mask = phase_mask.to(self.device)
        if polarity is not None:
            polarity = polarity.to(self.device)
        if polarity_mask is not None:
            polarity_mask = polarity_mask.to(self.device)
        if event_center is not None:
            event_center = event_center.to(self.device)
        if event_time is not None:
            event_time = event_time.to(self.device)
        if event_center_mask is not None:
            event_center_mask = event_center_mask.to(self.device)
        if event_time_mask is not None:
            event_time_mask = event_time_mask.to(self.device)

        # Backbone forward
        features = self.backbone(data)

        output = {"loss": 0.0}

        # Phase picking
        phase_logits, loss_phase = self.phase_head(features["phase"], phase_pick, phase_mask)
        output["phase"] = phase_logits
        if loss_phase is not None:
            output["loss_phase"] = loss_phase
            output["loss"] = output["loss"] + loss_phase

        # Polarity
        if self.add_polarity and "polarity" in features:
            polarity_logits, loss_polarity = self.polarity_head(
                features["polarity"], polarity, polarity_mask
            )
            output["polarity"] = polarity_logits
            if loss_polarity is not None:
                output["loss_polarity"] = loss_polarity * self.polarity_loss_weight
                output["loss"] = output["loss"] + loss_polarity * self.polarity_loss_weight

        # STFT spectrogram
        if self.add_stft and "spectrogram" in features:
            output["spectrogram"] = features["spectrogram"]

        # Event detection (separate heads for center detection vs time regression)
        if self.add_event and "event_center" in features:
            event_logits, loss_event_center = self.event_head.forward_center(
                features["event_center"], event_center, event_center_mask
            )
            output["event_center"] = event_logits
            if loss_event_center is not None:
                output["loss_event_center"] = loss_event_center * self.event_center_loss_weight
                output["loss"] = output["loss"] + loss_event_center * self.event_center_loss_weight

            # Event time regression (separate head, scaled)
            event_time_pred, loss_event_time = self.event_head.forward_time(
                features["event_time"] * 1000.0, event_time, event_time_mask
            )
            output["event_time"] = event_time_pred
            if loss_event_time is not None:
                output["loss_event_time"] = loss_event_time * self.event_time_loss_weight
                output["loss"] = output["loss"] + loss_event_time * self.event_time_loss_weight

        # Prompt-based picking
        if self.add_prompt and "prompt" in features:
            points = batch["prompt"].unsqueeze(1)
            pos = batch["position"]
            prompt_logits, loss_prompt = self.prompt_head(
                features["prompt"], points, pos, prompt_center
            )
            output["prompt"] = prompt_logits
            if loss_prompt is not None:
                output["prompt_center"] = torch.sigmoid(prompt_logits)
                output["loss_prompt"] = loss_prompt * self.prompt_loss_weight
                output["loss"] = output["loss"] + loss_prompt * self.prompt_loss_weight

        return output


# =============================================================================
# Factory Function
# =============================================================================

def build_model(
    backbone: str = "unet",
    log_scale: bool = True,
    add_stft: bool = False,
    add_polarity: bool = False,
    add_event: bool = False,
    **kwargs,
) -> PhaseNet:
    """Build a PhaseNet model.

    Args:
        backbone: Backbone type
        log_scale: Apply log transform
        add_stft: Add STFT features
        add_polarity: Enable polarity prediction
        add_event: Enable event detection
        **kwargs: Additional arguments passed to PhaseNet

    Returns:
        PhaseNet model instance
    """
    return PhaseNet(
        backbone=backbone,
        log_scale=log_scale,
        add_stft=add_stft,
        add_polarity=add_polarity,
        add_event=add_event,
        **kwargs,
    )
