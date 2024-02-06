import numpy as np
import torch

class CSLCoder:
    """Circular Smooth Label Coder, from mmrotate

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, omega=5, window='gaussian', radius=6):
        super().__init__()
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 360
        self.omega = omega
        self.window = window
        self.radius = radius
        self.coding_len = int(self.angle_range // omega)

    def encode(self, angle_targets, angle_dim=1):
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level in degrees.
                Has shape (..., 1, ...)
            angle_dim (int): The dimension index of original angle dimension and coding_len.

        Returns:
            list[Tensor]: The csl encoding of angle offset for each
                scale level. Has shape (..., coding_len, ...)
        """
        ori_shape = angle_targets.squeeze(angle_dim).shape
        angle_targets = angle_targets.flatten().unsqueeze(1)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.coding_len)
        angle_targets_deg = (angle_targets) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.coding_len
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = torch.exp(-torch.pow(base_radius_range, 2) /
                                     (2 * self.radius**2))

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        smooth_label = smooth_label.scatter(1, radius_range, smooth_value)
        if len(ori_shape) == 2:
            return smooth_label.view(*ori_shape, self.coding_len).permute(2, 0, 1)
        elif len(ori_shape) == 3:
            return smooth_label.view(*ori_shape, self.coding_len).permute(0, 3, 1, 2)
            

    def decode(self, angle_preds, angle_dim=1):
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset
                for each scale level.
                Has shape (B, coding_len, nt, nx)
            angle_dim (int): The dimension index of coding_len.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)
        """
        angle_cls_inds = torch.argmax(angle_preds, dim=angle_dim)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range
        return torch.deg2rad(angle_pred.unsqueeze(angle_dim))
    
class TrigonoCoder:
    '''Trigonometric function coder, use sine and cosine values to represent angle
    
    '''
    def __init__(self):
        super().__init__()
    
    def encode(self, angle_targets, angle_dim=1):
        trigono_vec = torch.cat([
            torch.sin(angle_targets),
            torch.cos(angle_targets)
        ], dim=angle_dim)
        return trigono_vec
    
    def decode(self, angle_preds, angle_dim=1):
        sin_idx = [slice(None)]*angle_preds.dim()
        sin_idx[angle_dim]=0
        cos_idx = [slice(None)]*angle_preds.dim()
        cos_idx[angle_dim]=1
        return torch.atan2(angle_preds[sin_idx], angle_preds[cos_idx]).unsqueeze(angle_dim)