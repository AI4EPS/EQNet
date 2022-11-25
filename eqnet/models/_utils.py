from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs: Tensor) -> Dict[str, Tensor]:

        # input_shape = x.shape[-2:]
        # # contract: features is a dict of tensors
        # features = self.backbone(x)

        # result = OrderedDict()
        # x = features["out"]
        # x = self.classifier(x)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        # result["out"] = x

        # if self.aux_classifier is not None:
        #     x = features["aux"]
        #     x = self.aux_classifier(x)
        #     x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        #     result["aux"] = x

        # return result


        data = batched_inputs["data"].to(self.device)
        if self.training:
            targets = batched_inputs["targets"].to(self.device)
        else:
            targets = None

        features = self.backbone(data)

        output, loss = self.classifier(features, targets)

        if self.training:
            return {"loss": loss}
        else:
            return output
            
V = TypeVar("V")

def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value