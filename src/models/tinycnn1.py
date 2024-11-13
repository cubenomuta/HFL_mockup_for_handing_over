from math import ceil
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import Net

class tinyCNN_1conv(Net):
    def __init__(
        self,
        input_spec: List,
        out_dims: int = 10,
        conv_kernel_size: int = 5,
        conv_kernel_stride: int = 1,
        pool_kernel_size: int = 2,
        pool_kernel_stride: int = 2,
    ) -> None:
        super(tinyCNN_1conv, self).__init__()
        self.in_channels, self.width, self.height = input_spec[0], input_spec[1], input_spec[2]
        self.conv_kernel_size, self.conv_kernel_stride = conv_kernel_size, conv_kernel_stride
        self.pool_kernel_size, self.pool_kernel_stride = pool_kernel_size, pool_kernel_stride
        
        # 畳み込み層を1つに設定
        self.conv1 = nn.Conv2d(self.in_channels, 16, self.conv_kernel_size)
        self.pool = nn.MaxPool2d(self.pool_kernel_size, self.pool_kernel_stride)
        
        # 出力サイズに基づき、全結合層の入力サイズを更新
        self._conv_update()
        self._pool_update()
        self.fc1 = nn.Linear(16 * self.width * self.height, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1つの畳み込み層とプーリングを適用
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * self.width * self.height)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _conv_update(self) -> None:
        # 畳み込み層適用後の幅と高さを更新
        self.width = ceil((self.width - (self.conv_kernel_size - 1) - 1) / self.conv_kernel_stride + 1)
        self.height = ceil((self.height - (self.conv_kernel_size - 1) - 1) / self.conv_kernel_stride + 1)

    def _pool_update(self) -> None:
        # プーリング適用後の幅と高さを更新
        self.width = ceil((self.width - (self.pool_kernel_size - 1) - 1) / self.pool_kernel_stride + 1)
        self.height = ceil((self.height - (self.pool_kernel_size - 1) - 1) / self.pool_kernel_stride + 1)