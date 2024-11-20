import torch.nn as nn

from models.base_model import Net
from models.metric_learning import get_arcface_resnet18
from models.resnet import resnet18 
from models.resnet import resnet8
from models.tinycnn import tinyCNN
from models.tinycnn1 import tinyCNN_1conv
from models.mobilenet import MobileNet
from models.mobilenetv2 import MobileNetV2

from typing import Tuple

def load_model(name: str, input_spec: Tuple[int, int, int], out_dims: int = 10, pretrained: bool = False)->Net:
    if name == "tinyCNN":
        return tinyCNN(input_spec=input_spec, out_dims=out_dims)
    elif name == "tinyCNN_1conv":
        return tinyCNN_1conv(input_spec=input_spec, out_dims=out_dims)
    elif name == "ResNet18":
        return resnet18(input_spec=input_spec, num_classes=out_dims)
    elif name == "GNResNet18":
        return resnet18(input_spec=input_spec, num_classes=out_dims, norm_layer=lambda x: nn.GroupNorm(2,x))
    elif name == "MobileNet":
        return MobileNet(input_spec=input_spec, out_dims=out_dims)
    elif name == "MobileNetV2":
        return MobileNetV2(input_spec=input_spec, out_dims=out_dims)
    elif name == "ResNet8":
        return resnet8(input_spec=input_spec, num_classes=out_dims)
    else:
        raise NotImplementedError(f"model {name} is not implemented.")

def load_arcface_model(name: str, input_spec: Tuple[int, int, int], out_dims: int = 10, pretrained: str = None, fixed_embeddings: int = 0)->Net:
    if name == "ResNet18":
        return get_arcface_resnet18(input_spec=input_spec, num_classes=out_dims, pretrained=pretrained, fixed_embeddings=fixed_embeddings)
    elif name == "GNResNet18":
        return get_arcface_resnet18(input_spec=input_spec, num_classes=out_dims, pretrained=pretrained, fixed_embeddings=fixed_embeddings, norm_layer=lambda x: nn.GroupNorm(2,x))
    else:
        raise NotImplementedError(f"model {name} is not implemented.")