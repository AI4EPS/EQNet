from typing import Optional, Dict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .unet import moving_normalize, moving_normalize_channel
from .resnet1d import ResNet, BasicBlock, Bottleneck
from .swin_transformer import SwinTransformer
from .ops import LayerNorm2d
from .swin_transformer_1d import SwinTransformer1D
from .centernet import CenterNetHead, CenterNetHeadV1
from .losses import smoothl1_reg_loss, weighted_l1_reg_loss, cross_entropy_loss, focal_loss
from .uper_head import UPerNeck, EventHead, PhaseHead, EventHead2, PhaseHead2, PolarityHead
from .phase_associator import PhaseAssociator


def _log_transform(x):
    x = F.relu(x)
    return torch.log(1 + x)


def log_transform(x):
    yp = _log_transform(x)
    yn = _log_transform(-x)
    return yp - yn


class EventDetector(nn.Module):
    # the first channel should be 8 * embedding_dim in swin transformer
    def __init__(
        self, channels=[128, 64, 32, 16, 8], bn=True, dilations=[1, 2, 4, 8, 16], kernel_size=5, nonlin=nn.ReLU(), weights=[1, 0.015, 0.01], offset_weight=[10, 1], reg_weight=[1,1,1,1]
    ):
        super().__init__()
        self.channels = channels
        self.bn = bn
        self.nonlin = nonlin
        self.weights = weights
        self.offset_weight = offset_weight
        self.reg_weight = reg_weight

        if self.bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:-1]])
            conv_bias = False
        else:
            self.bn_layers = [lambda x: x for c in channels[1:-1]]
            conv_bias = True

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    padding=((kernel_size - 1) * dilations[i] + 1) // 2,
                    padding_mode="reflect",
                    bias=conv_bias,
                )
                for i in range(len(channels) - 2)
            ]
        )
        
        # event center
        self.heatmap = nn.Sequential(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernel_size,
                dilation=dilations[-2],
                padding=((kernel_size - 1) * dilations[-2] + 1) // 2,
                padding_mode="reflect",
                bias=conv_bias,
            ),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels[-1],
                1,
                kernel_size=kernel_size,
                dilation=dilations[-1],
                padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
                padding_mode="reflect",
            )
        )
        self.heatmap[-1].bias.data.fill_(-2.19) # if use the initial value, the loss from 200 to 5 (v2)
        self.offset = nn.Sequential(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernel_size,
                dilation=dilations[-2],
                padding=((kernel_size - 1) * dilations[-2] + 1) // 2,
                padding_mode="reflect",
                bias=conv_bias,
            ),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels[-1],
                2,
                kernel_size=kernel_size,
                dilation=dilations[-1],
                padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
                padding_mode="reflect",
            )
        )
        #'''
        self.hypocenter = nn.Sequential(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernel_size,
                dilation=dilations[-2],
                padding=((kernel_size - 1) * dilations[-2] + 1) // 2,
                padding_mode="reflect",
                bias=conv_bias,
            ),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels[-1],
                #3,
                5,
                kernel_size=kernel_size,
                dilation=dilations[-1],
                padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
                padding_mode="reflect",
            )
        )
        #'''

    def forward(self, features, event_center, event_mask, event_location, event_location_mask):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["out3"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        # x = self.nonlin(self.bn_layers[0](x))
        # x = self.nonlin(x)
        # for conv, bn in zip(self.conv_layers, self.bn_layers[1:]):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.nonlin(bn(conv(x)))
        #x = self.heatmap(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        heatmap = self.heatmap(x)
        heatmap = heatmap.view(bt, st, heatmap.shape[2])  # chn = 1
        heatmap = heatmap.permute(0, 2, 1)
        
        offset = self.offset(x)
        offset = offset.view(bt, st, offset.shape[1], offset.shape[2])
        offset = offset.permute(0, 2, 3, 1)
        
        hypocenter = self.hypocenter(x)
        hypocenter = hypocenter.view(bt, st, hypocenter.shape[1], hypocenter.shape[2])
        hypocenter = hypocenter.permute(0, 2, 3, 1)

        if self.training:            
            return None, self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_mask, event_location, event_location_mask)
        elif event_center is not None:
            return {"event": heatmap, "offset": offset, "hypocenter": hypocenter}, \
                self.losses({"event": heatmap, "offset": offset, "hypocenter": hypocenter}, event_center, event_mask, event_location, event_location_mask)
        else:
            return {"event": heatmap, "offset": offset, "hypocenter": hypocenter}, {}
        
        #if self.training:            
        #    return None, self.losses({"event": heatmap, "offset": offset}, event_center, event_location, event_location_mask)
        #elif event_center is not None:
        #    return {"event": heatmap, "offset": offset}, \
        #        self.losses({"event": heatmap, "offset": offset}, event_center, event_location, event_location_mask)
        #else:
        #    return {"event": heatmap, "offset": offset}, {}


    def losses(self, outputs, event_center, event_mask, event_location, event_location_mask):
        #hm_loss = cross_entropy_loss(outputs["event"], event_center)
        hm_loss = focal_loss(outputs["event"], event_center, event_mask)
        hw_loss = weighted_l1_reg_loss(outputs["offset"], event_location[:, 5:, :, :], event_location_mask, weights=self.offset_weight)
        
        #reg_loss = smoothl1_reg_loss(outputs["hypocenter"], event_location[:, :3, :, :], event_location_mask)
        reg_loss = weighted_l1_reg_loss(outputs["hypocenter"], event_location[:, :5, :, :], event_location_mask, weights=self.reg_weight)
        
        loss = hm_loss * self.weights[0] + hw_loss * self.weights[1] + reg_loss * self.weights[2]
        # print(f"loss {loss}, hm_loss {hm_loss}, hw_loss {hw_loss}, reg_loss {reg_loss}", force=True)
        return {"loss": loss, "loss_event": hm_loss, "loss_offset": hw_loss, "loss_hypocenter": reg_loss}


class PhasePicker(nn.Module):
    # the first channel should be 8 * embedding_dim in swin transformer
    def __init__(
        self, channels=[128, 64, 32, 16, 8], bn=True, dilations=[1, 2, 4, 8, 16], kernel_size=5, nonlin=nn.ReLU()
    ):
        super().__init__()
        self.channels = channels
        self.bn = bn
        self.nonlin = nonlin

        if self.bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:]])
            conv_bias = False
        else:
            self.bn_layers = [lambda x: x for c in channels[1:]]
            conv_bias = True

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    padding=((kernel_size - 1) * dilations[i] + 1) // 2,
                    padding_mode="reflect",
                    bias=conv_bias,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.conv_out = nn.Conv1d(
            channels[-1],
            3,
            kernel_size=kernel_size,
            dilation=dilations[-1],
            padding=((kernel_size - 1) * dilations[-1] + 1) // 2,
            padding_mode="reflect",
        )

    def forward(self, features, targets=None):
        """input shape [batch, in_channels, time_steps]
        output shape [batch, time_steps]"""
        x = features["out3"]
        bt, st, ch, nt = x.shape  # batch, station, channel, time
        x = x.view(bt * st, ch, nt)
        # x = self.nonlin(self.bn_layers[0](x))
        # x = self.nonlin(x)
        # for conv, bn in zip(self.conv_layers, self.bn_layers[1:]):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.nonlin(bn(conv(x)))
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = x.view(bt, st, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 3, 1)

        if self.training:
            return None, self.losses(x, targets)
        elif targets is not None:
            return x, self.losses(x, targets)
        else:
            return x, {}

    def losses(self, inputs, targets):
        inputs = inputs.float()  # https://github.com/pytorch/pytorch/issues/48163
        num=targets.shape[0]*targets.shape[3]
        for i in range(targets.shape[0]):
            for j in range(targets.shape[3]):
                if torch.all(targets[i,:,:,j]==0):
                    num-=1
        loss = torch.sum(-targets * F.log_softmax(inputs, dim=1))/(num*targets.shape[2])
        
        return loss


class EQNet(nn.Module):
    def __init__(self, backbone="resnet50", head="simple", use_station_location=True, moving_norm=(1024, 128),) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.head_name = head
        self.moving_norm = moving_norm
        embed_dim = 64#96
        sampling_rate = 100
        prompt_embed_dim = 96
        
        if backbone == "resnet18":
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
            # self.backbone = ResNet(BasicBlock, [3, 4, 6, 3]) #ResNet34
        elif backbone == "resnet50":
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
        elif backbone == "swin": 
            out_indices = [0, 1, 2, 3] if "upernet" in head else [3]
            self.backbone = SwinTransformer(
                patch_size=[4, 1],
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=[7, 10],
                stochastic_depth_prob=0.2,
                block_name="SwinTransformerBlock",
                out_indices=out_indices,
                use_station_location=use_station_location,
            )
        elif backbone == "swin2":
            out_indices = [0, 1, 2, 3] if "upernet" in head else [3]
            self.backbone = SwinTransformer(
                patch_size=[4, 1],
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                #num_heads=[3, 6, 12, 24],
                num_heads=[2, 4, 8, 16],
                window_size=[16, 16], # can use 32 for inference
                #stochastic_depth_prob=0.2,
                stochastic_depth_prob=0.15,
                block_name="SwinTransformerBlockV2",
                out_indices=out_indices,
                use_station_location=use_station_location,
                decoupled_relative_position_bias=False,
                sampling_rate=sampling_rate,
            )
        elif backbone == "swin_rope_axial":
            out_indices = [0, 1, 2, 3] if "upernet" in head else [3]
            self.backbone = SwinTransformer(
                patch_size=[4, 1],
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                #num_heads=[3, 6, 12, 24],
                num_heads=[2, 4, 8, 16],
                window_size=[8, 10],
                #stochastic_depth_prob=0.2,
                stochastic_depth_prob=0.15,
                block_name="RotarySwinBlock",
                out_indices=out_indices,
                use_station_location=use_station_location,
                use_relative_position_bias=True,
                use_rope=True,
                use_rope_mixed=False,
            )
        elif backbone == "swin_rope_mixed":
            out_indices = [0, 1, 2, 3] if "upernet" in head else [3]
            self.backbone = SwinTransformer(
                patch_size=[4, 1],
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                #num_heads=[3, 6, 12, 24],
                num_heads=[2, 4, 8, 16],
                window_size=[8, 10],
                #stochastic_depth_prob=0.2,
                stochastic_depth_prob=0.15,
                block_name="RotarySwinBlock",
                out_indices=out_indices,
                use_station_location=use_station_location,
                use_relative_position_bias=True,
                use_rope=True,
                use_rope_mixed=True,
            )
        elif backbone == "swin_1D": 
            out_indices = [0, 1, 2, 3] if "upernet" in head else [3]
            self.backbone = SwinTransformer1D(
                patch_size=4,
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                stochastic_depth_prob=0.2,
                block_name="SwinTransformerBlock",
                out_indices=out_indices,
            )
        elif backbone == "swin2_1D":
            out_indices = [0, 1, 2, 3] if "upernet" in head else [3]
            self.backbone = SwinTransformer1D(
                patch_size=4,
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=8,
                stochastic_depth_prob=0.2,
                block_name="SwinTransformerBlockV2",
                out_indices=out_indices,
            )
        else:
            raise ValueError("backbone must be one of 'resnet' or 'swin'")

        # config for Swin_T
        # the len of channels and dilations should be 5 when using other config
        # or you need to change the structure of event_detector and phase_picker
        if "swin" in backbone:
            channels=[embed_dim * (2**3), 128, 32, 16, 8] 
            #channels=[embed_dim * (2**3), 256, 64, 16, 8]
            #dilations=[1, 6, 24, 48, 96] 
            dilations=[1, 4, 8, 32, 64]
            #neck_channels=256 # TODO: 512 is the original setting
            neck_channels=128
        elif backbone[:6] == "resnet":
            channels=[128, 64, 32, 16, 8]
            dilations=[1, 2, 4, 8, 16]
            
        self.neck = UPerNeck(in_channels=[embed_dim * (2**i) for i in range(4)], channels=neck_channels) if "upernet" in head else nn.Sequential()
        
        self.polarity_stem = nn.Sequential(
            nn.Conv2d(3, embed_dim//2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            LayerNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            LayerNorm2d(embed_dim),
        )

        if head == "simple":
            self.event_detector = EventDetector(channels=channels, bn=True, dilations=dilations)
        elif head == "centernet":
            if backbone == "swin" or backbone == "swin_1D":
                self.event_detector = CenterNetHeadV1(channels=[768, 256, 64])
            elif backbone == "swin2" or backbone == "swin2_1D":
                self.event_detector = CenterNetHead(channels=[768, 256, 64])
        elif head == "upernet":
            self.event_detector = EventHead(channels=neck_channels)
        elif head == "upernet2":
            self.event_detector = EventHead2(channels=neck_channels)
        
        if head == "upernet":
            self.phase_picker = PhaseHead(channels=neck_channels)
        elif head == "upernet2":
            self.phase_picker = PhaseHead(channels=neck_channels)
        else:
            self.phase_picker = PhasePicker(channels=channels, bn=True, dilations=dilations)
        
        self.polarity_picker = PolarityHead(channels=neck_channels, stem_channels=embed_dim)
        
        self.phase_associator = PhaseAssociator(feature_dim=neck_channels, prompt_embed_dim=prompt_embed_dim, event_scale=16, sampling_rate=100.0)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.jit.ignore
    def no_weight_decay(self):
        keys = set()
        if hasattr(self.backbone, 'no_weight_decay'):
            keys.update({'backbone.' + i for i in self.backbone.no_weight_decay()})
        if hasattr(self.neck, 'no_weight_decay'):
            keys.update({'neck.' + i for i in self.neck.no_weight_decay()})
        if hasattr(self.event_detector, 'no_weight_decay'):
            keys.update({'event_detector.' + i for i in self.event_detector.no_weight_decay()})
        if hasattr(self.phase_picker, 'no_weight_decay'):
            keys.update({'phase_picker.' + i for i in self.phase_picker.no_weight_decay()})
        if hasattr(self.polarity_picker, 'no_weight_decay'):
            keys.update({'polarity_picker.' + i for i in self.polarity_picker.no_weight_decay()})
        return keys

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        keys = set()
        if hasattr(self.backbone, 'no_weight_decay_keywords'):
            keys.update({i for i in self.backbone.no_weight_decay_keywords()})
        if hasattr(self.neck, 'no_weight_decay_keywords'):
            keys.update({i for i in self.neck.no_weight_decay_keywords()})
        if hasattr(self.event_detector, 'no_weight_decay_keywords'):
            keys.update({i for i in self.event_detector.no_weight_decay_keywords()})
        if hasattr(self.phase_picker, 'no_weight_decay_keywords'):
            keys.update({i for i in self.phase_picker.no_weight_decay_keywords()})
        if hasattr(self.polarity_picker, 'no_weight_decay_keywords'):
            keys.update({i for i in self.polarity_picker.no_weight_decay_keywords()})
        return keys

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:
        data = batched_inputs["data"].to(self.device)
        amplitude = batched_inputs["amplitude"].to(self.device)
        #assert data.shape[-1]==amplitude.shape[-1], f"data {data.shape}, amplitude {amplitude.shape}"
        data = moving_normalize(data, filter=self.moving_norm[0], stride=self.moving_norm[1])
        #assert not torch.any(torch.isnan(data)), f"nan in data {data}"

        phase_pick = batched_inputs["phase_pick"].to(self.device) if "phase_pick" in batched_inputs else None
        polarity = batched_inputs["polarity"].to(self.device) if "polarity" in batched_inputs else None
        polarity_mask = batched_inputs["polarity_mask"].to(self.device) if "polarity_mask" in batched_inputs else None
        event_center = batched_inputs["event_center"].to(self.device) if "event_center" in batched_inputs else None
        event_mask = batched_inputs["event_mask"].to(self.device) if "event_mask" in batched_inputs else None
        event_location = batched_inputs["event_location"].to(self.device) if "event_location" in batched_inputs else None
        event_location_mask = batched_inputs["event_location_mask"].to(self.device) if "event_location_mask" in batched_inputs else None
        phase_mask = batched_inputs["phase_mask"].to(self.device) if "phase_mask" in batched_inputs else None
        association_mask = batched_inputs["association_mask"].to(self.device) if "association_mask" in batched_inputs else None
        prompt = batched_inputs["prompt"].to(self.device) if "prompt" in batched_inputs else None

        if "swin" in self.backbone_name:
            station_location = batched_inputs["station_location"].to(self.device)
            # features = self.backbone({"data": data, "station_location": station_location})
            features = self.backbone(data, station_location)
        else:
            features = self.backbone(data)
            
        features = self.neck(features)

        output_phase, loss_phase = self.phase_picker(features, phase_pick, phase_mask)
        outputs_event, losses_event = self.event_detector(features, amplitude, event_center, event_mask, event_location, event_location_mask)
        output_polarity, loss_polarity = self.polarity_picker(features, self.polarity_stem(data), polarity, polarity_mask)
        if prompt is not None:
            output_association, loss_association = self.phase_associator(features, station_location, prompt, association_mask)
        else:
            output_association, loss_association = {}, {'loss': torch.tensor(0.0, device=self.device)}

        if self.training:
            loss = loss_phase['loss_phase'] + losses_event["loss"] + loss_polarity["loss_polarity"] + loss_association["loss"]
            del losses_event["loss"], loss_association["loss"]
            return {"loss": loss, **loss_phase, **losses_event, **loss_polarity, **loss_association}
        elif phase_pick is not None: # validation
            #output_event = outputs_event["event"]
            loss = loss_phase['loss_phase'] + losses_event["loss"] + loss_polarity["loss_polarity"] + loss_association["loss"]
            del losses_event["loss"], loss_association["loss"]
            return {**output_phase, **outputs_event, **output_polarity, **output_association, "loss": loss, **loss_phase, **losses_event, **loss_polarity, **loss_association}
        else:
            #output_event = outputs_event["event"]
            return {**output_phase, **outputs_event, **output_polarity, **output_association}


class EQNetPredictor(EQNet):
    def __init__(self, backbone="resnet50", head="simple", use_station_location=True, moving_norm=(1024, 128), prompt_feature='ppm') -> None:
        super().__init__(backbone=backbone, head=head, use_station_location=use_station_location, moving_norm=moving_norm)
        if 'swin' not in backbone:
            raise ValueError("EQNetPredictor currently only supports swin backbone")
        self.prompt_feature = prompt_feature
        
    def forward(self, batched_inputs: Tensor, association_only=False) -> Dict[str, Tensor]:
        if not association_only:
            data = batched_inputs["data"].to(self.device)
            amplitude = batched_inputs["amplitude"].to(self.device)
            #assert data.shape[-1]==amplitude.shape[-1], f"data {data.shape}, amplitude {amplitude.shape}"
            data = moving_normalize(data, filter=self.moving_norm[0], stride=self.moving_norm[1])
            #assert not torch.any(torch.isnan(data)), f"nan in data {data}"
            prompt = batched_inputs["prompt"].to(self.device) if "prompt" in batched_inputs else None

            station_location = batched_inputs["station_location"].to(self.device)
            # features = self.backbone({"data": data, "station_location": station_location})
            features = self.backbone(data, station_location)

            features = self.neck(features)

            output_phase, loss_phase = self.phase_picker(features)
            outputs_event, losses_event = self.event_detector(features, amplitude)
            output_polarity, loss_polarity = self.polarity_picker(features, self.polarity_stem(data))
            
            if prompt is not None:
                output_association, loss_association = self.phase_associator(features, station_location, prompt)
            else:
                output_association, loss_association = {}, {}
            
            return {**output_phase, **outputs_event, **output_polarity, "feature": features[self.prompt_feature], **output_association}
        else:
            prompt = batched_inputs["prompt"].to(self.device)
            feature = batched_inputs["feature"].to(self.device)
            station_location = batched_inputs["station_location"].to(self.device)
            
            if feature.shape[0] != prompt.shape[0]:
                assert feature.shape[0] == 1, f"The default setting for multi-prompt query only support one waveform embedding each time, but got {feature.shape}"
                assert feature.shape[0] == station_location.shape[0], f"feature should be aligned with station_location, but got {feature.shape} and {station_location.shape}"
                feature = feature.repeat_interleave(prompt.shape[0], dim=0)
                station_location = station_location.repeat_interleave(prompt.shape[0], dim=0)
            
            output_association, loss_association = self.phase_associator({self.prompt_feature: feature}, station_location, prompt)
            
            return output_association

def build_model(backbone="resnet50", head="simple", use_station_location=True, prediction=False, **kwargs) -> EQNet:
    if prediction:
        return EQNetPredictor(backbone=backbone, head=head, use_station_location=use_station_location, **kwargs)
    else:
        return EQNet(backbone=backbone, head=head, use_station_location=use_station_location)
