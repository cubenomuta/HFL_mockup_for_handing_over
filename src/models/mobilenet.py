from math import ceil
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import Net

class MobileNet(Net):
    def __init__(
        self,
        input_spec: List,
        out_dims: int = 10,
        conv_kernel_size: int = 3,
        conv_kernel_stride: int = 1,
        pool_kernel_size: int = 2,
        pool_kernel_stride: int = 2,
        width_multiplier: float = 1.0,
    ) -> None:
        super(MobileNet, self).__init__()
        self.in_channels, self.width, self.height = input_spec[0], input_spec[1], input_spec[2]
        self.conv_kernel_size, self.conv_kernel_stride = conv_kernel_size, conv_kernel_stride
        self.pool_kernel_size, self.pool_kernel_stride = pool_kernel_size, pool_kernel_stride
        self.width_multiplier = width_multiplier
        
        def depthwise_separable_conv(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=self.conv_kernel_size, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            )
        # MobileNet starts with a standard conv layer
        self.conv1 = nn.Conv2d(self.in_channels, int(32 * self.width_multiplier), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * self.width_multiplier))
        self.relu = nn.ReLU6(inplace=True)
        # MobileNet architecture (depthwise separable conv layers)
        self.conv2 = depthwise_separable_conv(int(32 * self.width_multiplier), int(64 * self.width_multiplier), 1)
        self.conv3 = depthwise_separable_conv(int(64 * self.width_multiplier), int(128 * self.width_multiplier), 2)
        self.conv4 = depthwise_separable_conv(int(128 * self.width_multiplier), int(128 * self.width_multiplier), 1)
        self.conv5 = depthwise_separable_conv(int(128 * self.width_multiplier), int(256 * self.width_multiplier), 2)
        self.conv6 = depthwise_separable_conv(int(256 * self.width_multiplier), int(256 * self.width_multiplier), 1)
        self.conv7 = depthwise_separable_conv(int(256 * self.width_multiplier), int(512 * self.width_multiplier), 2)
        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * self.width_multiplier), out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x