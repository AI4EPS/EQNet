from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .resnet1d import BasicBlock, Bottleneck, ResNet
from .unet import UNet
from .x_unet import XUnet

default_cfg = {
    "backbone": "unet",
    "head": "unet",
    "preprocess": {
        "moving_norm": {
            "flag": True,
            "window": 1024,
            "stride": 128,
        },
        "padding": {
            "flag": True,
            "nt": 1024,
            "nx": 1,
        },
    },
    "backbone_cfg": {
        "in_channels": 3,
    },
    "head_cfg": {
        "output_channels": 3,
    },
}

from .prompt import MaskDecoder, PromptEncoder, TwoWayTransformer


class FCNHead(nn.Module):
    # class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.channels = out_channels
        self.layers = nn.Sequential(
            *[
                nn.Conv1d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.Conv1d(inter_channels, out_channels, 1),
            ]
        )
        # super(FCNHead, self).__init__(*layers)

    def forward(self, features, targets=None):
        x = features["phase"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)

        x = self.layers(x)
        x = F.interpolate(x, scale_factor=32, mode="linear", align_corners=False)

        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)  # batch, channel, time, station

        loss = None
        if targets is not None:
            loss = self.losses(x, targets)

        return x, loss

    def losses(self, inputs, targets):
        inputs = inputs.float()

        if self.out_channels == 1:
            loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            loss = -torch.sum(targets.float() * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss


class DeepLabHead(nn.Module):
    # class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor=1) -> None:
        super(DeepLabHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.layers = nn.Sequential(
            *[
                # super().__init__(
                ASPP(in_channels, [12, 24, 36]),
                nn.Conv1d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, out_channels, 1),
            ]
        )

    def forward(self, features, targets=None, mask=None):
        x = features["phase"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)

        x = self.layers(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)

        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets, mask)
        return x, {}

    def losses(self, inputs, targets, mask=None):
        inputs = inputs.float()

        if self.out_channels == 1:
            loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=mask)
        else:
            loss = torch.sum(-targets.float() * F.log_softmax(inputs, dim=1), dim=1).mean()

        return loss


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-1]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="linear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv1d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class UNetHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=(7, 1), padding=(3, 0), feature_names: str = "phase"
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_names = feature_names
        self.layers = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        )
        # self.layers = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0))
        # self.layers = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=False
        #     ),
        #     nn.BatchNorm2d(num_features=in_channels),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0)),
        # )

    def forward(self, features, targets=None, mask=None):
        x = features[self.feature_names]
        x = self.layers(x)

        loss = None
        if targets is not None:
            loss = self.losses(x, targets, mask)
        return x, loss

    def losses(self, inputs, targets, mask=None):
        """
        targets: (batch, channel, time, station) or (batch, 1, time, station)
        """
        inputs = inputs.float()
        log_targets = torch.nan_to_num(torch.log(targets))

        if mask is None:
            if self.out_channels == 1:
                min_loss = -torch.mean(targets * log_targets + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
                loss = F.binary_cross_entropy_with_logits(inputs, targets) - min_loss

                # inputs = torch.sigmoid(inputs)
                # loss = F.kl_div(inputs.log(), targets, reduction="none") + F.kl_div(
                #     (1 - inputs).log(), 1 - targets, reduction="none"
                # )
                # loss = torch.nan_to_num(loss)
                # loss = loss.mean()

            else:
                min_loss = -(targets * log_targets).sum(dim=1).mean()  # cross_entropy sum over dim=1
                loss = F.cross_entropy(inputs, targets) - min_loss

                # inputs = torch.log_softmax(inputs, dim=1)
                # loss = F.kl_div(inputs, targets, reduction="none").sum(dim=1).mean()

                # focal loss
                # ce_loss = F.cross_entropy(inputs, targets, reduction="none")
                # pt = torch.exp(-ce_loss)
                # focal_loss = (1 - pt) ** 2 * ce_loss
                # loss = focal_loss.mean()
        else:
            mask = mask.type_as(inputs)
            mask_sum = mask.sum()
            if mask_sum == 0.0:
                mask_sum = 1.0
            if self.out_channels == 1:
                min_loss = -(targets * log_targets + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
                loss = (
                    torch.sum((F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") - min_loss) * mask)
                    / mask_sum
                )

                # inputs = torch.sigmoid(inputs)
                # kl_div = F.kl_div(inputs.log(), targets, reduction="none") + F.kl_div(
                #     (1 - inputs).log(), 1 - targets, reduction="none"
                # )
                # kl_div = torch.nan_to_num(kl_div)
                # loss = torch.sum(kl_div * mask) / mask_sum

            else:
                min_loss = -targets * log_targets
                loss = (
                    torch.sum((F.cross_entropy(inputs, targets, reduction="none") - min_loss) * mask.squeeze(1))
                    / mask_sum
                )  # cross_entropy already sum over dim=1

                # inputs = torch.log_softmax(inputs, dim=1)
                # loss = torch.sum(F.kl_div(inputs, targets, reduction="none").sum(dim=1) * mask.squeeze(1)) / mask_sum

                # focal loss
                # ce_loss = F.cross_entropy(inputs, targets, reduction="none")
                # pt = torch.exp(-ce_loss)
                # focal_loss = (1 - pt) ** 5 * ce_loss
                # loss = torch.sum(focal_loss * mask.squeeze(1)) / mask_sum

        return loss


class EventHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(7, 1),
        padding=(3, 0),
        scaling=1000.0,
        feature_names: str = "event",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_names = feature_names
        self.scaling = scaling
        # self.layers = nn.Conv2d(
        #     in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
        # )
        # self.layers = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(7, 1), padding=(3, 0)),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        #     nn.LeakyReLU(),
        # )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
        )

    def forward(self, features, targets=None, mask=None):
        x = features[self.feature_names]
        x = self.layers(x) * self.scaling

        loss = None
        if targets is not None:
            loss = self.losses(x, targets, mask)
        return x, loss

    def losses(self, inputs, targets, mask=None):
        inputs = inputs.float()

        if mask is None:
            loss = F.mse_loss(inputs, targets) / self.scaling
        else:
            mask = mask.type_as(inputs)
            mask_sum = mask.sum()
            if mask_sum == 0.0:
                mask_sum = 1.0
            loss = torch.sum(F.l1_loss(inputs, targets, reduction="none") * mask) / mask_sum / self.scaling
            # loss = torch.sum(torch.abs(inputs - targets) * mask, dim=(1, 2, 3)).mean() / mask_sum

        return loss


class PhaseNet(nn.Module):
    def __init__(
        self,
        backbone="unet",
        init_features=16,
        upsample="conv_transpose",  # "interpolate", "conv_transpose"
        log_scale=True,
        spectrogram=False,
        add_polarity=False,
        add_event=False,
        add_prompt=True,
        event_center_loss_weight=1.0,
        event_time_loss_weight=1.0,
        polarity_loss_weight=1.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.spectrogram = spectrogram
        self.add_event = add_event
        self.add_polarity = add_polarity
        self.add_prompt = add_prompt
        self.event_center_loss_weight = event_center_loss_weight
        self.event_time_loss_weight = event_time_loss_weight
        self.polarity_loss_weight = polarity_loss_weight

        if backbone == "resnet18":
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
        elif backbone == "resnet50":
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
        elif backbone == "unet":
            self.backbone = UNet(
                init_features=init_features,
                upsample=upsample,
                log_scale=log_scale,
                spectrogram=spectrogram,
                add_polarity=add_polarity,
                add_event=add_event,
            )
        elif backbone == "xunet":
            self.backbone = XUnet(
                channels=3,
                dim=init_features,
                out_dim=init_features,
                use_convnext=True,
                weight_standardize=True,
                num_self_attn_per_stage=(0, 0, 1, 1),
                dim_mults=(1, 2, 4, 8),
                nested_unet_depths=(7, 4, 2, 1),  # nested unet depths, from unet-squared paper
                consolidate_upsample_fmaps=True,  # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
                log_scale=log_scale,
                add_polarity=add_polarity,
                add_event=add_event,
            )
        else:
            raise ValueError("backbone only supports resnet18, resnet50, or unet")

        if backbone in ["unet", "xunet"]:
            kernel_size = (7, 1)
            padding = (3, 0)
            self.phase_picker = UNetHead(
                init_features, 3, kernel_size=kernel_size, padding=padding, feature_names="phase"
            )
            if self.add_event:
                if self.backbone_name == "xunet":
                    self.event_detector = UNetHead(
                        init_features * 8, 1, kernel_size=kernel_size, padding=padding, feature_names="event"
                    )
                    self.event_timer = EventHead(
                        init_features * 8, 1, kernel_size=kernel_size, padding=padding, feature_names="event"
                    )
                else:
                    self.event_detector = UNetHead(
                        init_features, 1, kernel_size=kernel_size, padding=padding, feature_names="event"
                    )
                    self.event_timer = EventHead(
                        init_features, 1, kernel_size=kernel_size, padding=padding, feature_names="event"
                    )
                
                # #### FIXME: HARDCODED
                if self.add_prompt:
                    prompt_embed_dim = 16
                    image_embedding_size = [256, 8]
                    image_size = [256, 8]

                    self.prompt_encoder=PromptEncoder(
                        embed_dim=prompt_embed_dim,
                        image_embedding_size=image_embedding_size,
                        input_image_size=image_size,
                        mask_in_chans=16,
                    )
                    self.mask_decoder=MaskDecoder(
                        num_multimask_outputs=1,
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
                # ####

                
            if self.add_polarity:
                self.polarity_picker = UNetHead(
                    init_features, 1, kernel_size=kernel_size, padding=padding, feature_names="polarity"
                )
                # self.polarity_picker = UNetHead(16, 1, feature_names="polarity")
        else:
            self.phase_picker = DeepLabHead(128, 3, scale_factor=32)
            if self.add_event:
                self.event_detector = DeepLabHead(128, 1, scale_factor=2)
                self.event_timer = EventHead(128, 1, scale_factor=2)
            if self.add_polarity:
                self.polarity_picker = DeepLabHead(128, 1, scale_factor=32)


    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:
        data = batched_inputs["data"].to(self.device)

        phase_pick = batched_inputs["phase_pick"].to(self.device) if "phase_pick" in batched_inputs else None
        event_center = batched_inputs["event_center"].to(self.device) if "event_center" in batched_inputs else None
        event_time = batched_inputs["event_time"].to(self.device) if "event_time" in batched_inputs else None
        event_mask = batched_inputs["event_mask"].to(self.device) if "event_mask" in batched_inputs else None
        polarity = batched_inputs["polarity"].to(self.device) if "polarity" in batched_inputs else None
        polarity_mask = batched_inputs["polarity_mask"].to(self.device) if "polarity_mask" in batched_inputs else None

        if self.backbone_name == "swin2":
            station_location = batched_inputs["station_location"].to(self.device)
            features = self.backbone(data, station_location)
        else:
            features = self.backbone(data)

        output = {"loss": 0.0}
        output_phase, loss_phase = self.phase_picker(features, phase_pick)
        output["phase"] = output_phase
        if loss_phase is not None:
            output["loss_phase"] = loss_phase
            output["loss"] += loss_phase
        if self.add_event:
            output_event_center, loss_event_center = self.event_detector(features, event_center)
            output["event_center"] = output_event_center
            if loss_event_center is not None:
                output["loss_event_center"] = loss_event_center * self.event_center_loss_weight
                output["loss"] += loss_event_center * self.event_center_loss_weight
            output_event_time, loss_event_time = self.event_timer(features, event_time, mask=event_mask)
            output["event_time"] = output_event_time
            if loss_event_time is not None:
                output["loss_event_time"] = loss_event_time * self.event_time_loss_weight
                output["loss"] += loss_event_time * self.event_time_loss_weight


            ### FIXME: HARDCODED
            points = batched_inputs["prompt"].unsqueeze(1) # [B, 1, 3]
            pos = batched_inputs["position"] # [B, T, S, 3]
            B, T, S, _ = pos.shape
            pos = pos.view(B, T*S, 3)
            labels = torch.ones((points.shape[0], points.shape[1]))
            points = (points, labels)
            labels = torch.ones((pos.shape[0], pos.shape[1]))
            pos = (pos, labels)
            point_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=None, masks=None)
            pos_embeddings, _ = self.prompt_encoder(points=pos, boxes=None, masks=None)
            C = point_embeddings.shape[-1]
            pos_embeddings = pos_embeddings.reshape(B, C, T, S)

            low_res_masks = []
            iou_predictions = []
            image_embeddings = features["event"]
            for i in range(B): ## FIXME: Don't understand why have to use loop here
                low_res_masks_, iou_predictions_ = self.mask_decoder(
                    image_embeddings=image_embeddings[i:i+1],
                    image_pe=pos_embeddings[i:i+1],
                    sparse_prompt_embeddings=point_embeddings[i:i+1],
                    dense_prompt_embeddings=dense_embeddings[i:i+1],
                    multimask_output=False,
                )

                low_res_masks.append(low_res_masks_)
                iou_predictions.append(iou_predictions_)

            low_res_masks = torch.cat(low_res_masks, dim=0)
            iou_predictions = torch.cat(iou_predictions, dim=0)

            targets = batched_inputs["prompt_center"].float()
            mask = batched_inputs["prompt_mask"].float()
            inputs = low_res_masks.float()
            # min_loss = -(targets * torch.nan_to_num(torch.log(targets)) + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
            # loss_prompt = torch.mean(F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") - min_loss)

            # ## focal loss
            # bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            # pt = torch.exp(-bce_loss) # probability of the correct class
            # focal_loss = (1 - pt) ** 2 * bce_loss # alpha=1, gamma=2
            # loss_prompt = torch.mean(focal_loss) * 100

            prob = inputs.sigmoid()
            min_loss = -(targets * torch.nan_to_num(torch.log(targets)) + (1 - targets) * torch.nan_to_num(torch.log(1 - targets)))
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") - min_loss
            p_t = prob * targets + (1 - prob) * (1 - targets)
            gamma=2.0
            loss_prompt = ce_loss * ((1 - p_t) ** gamma)
            loss_prompt = 10 * torch.mean(loss_prompt)

            output["loss_prompt"] = loss_prompt
            output["prompt_center"] = torch.sigmoid(low_res_masks)
            output["loss"] += loss_prompt
            ####

        if self.add_polarity:
            output_polarity, loss_polarity = self.polarity_picker(features, polarity, mask=polarity_mask)
            output["polarity"] = output_polarity
            if loss_polarity is not None:
                output["loss_polarity"] = loss_polarity * self.polarity_loss_weight
                output["loss"] += loss_polarity * self.polarity_loss_weight
        if self.spectrogram:
            output["spectrogram"] = features["spectrogram"]

        return output


def build_model(
    backbone="unet",
    init_features=16,
    log_scale=False,
    *args,
    **kwargs,
) -> PhaseNet:
    return PhaseNet(
        backbone=backbone,
        init_features=init_features,
        log_scale=log_scale,
    )
