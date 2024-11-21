from typing import List
import torch
import torch.nn as nn
from .base_model import Net
class InvertedResidual(Net):
    """
    MobileNetV2の基本ブロック: Inverted Residual Block
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        layers = []
        if expand_ratio != 1:
            # Pointwise (1x1 conv): Expansion
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # Depthwise Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        # Pointwise (1x1 conv): Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class MobileNetV2(Net):
    """
    MobileNetV2の実装
    """
    def __init__(self, input_spec: List, out_dims: int = 1000, width_multiplier: float = 1.0):
        super(MobileNetV2, self).__init__()
        self.in_channels, self.width, self.height = input_spec[0], input_spec[1], input_spec[2]
        self.width_multiplier = width_multiplier
        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        # Configurations for MobileNetV2 architecture
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # (expansion factor, output channels, number of blocks, stride)
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(32 * width_multiplier)
        last_channel = _make_divisible(1280 * width_multiplier)
        # First Conv Layer
        self.features = [nn.Sequential(
            nn.Conv2d(self.in_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        # Inverted Residual Blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # Last Conv Layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))
        # Combine all layers
        self.features = nn.Sequential(*self.features)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, out_dims),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.classifier(x)
        return x