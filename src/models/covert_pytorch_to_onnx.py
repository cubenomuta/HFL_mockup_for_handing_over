import torch
import torchvision.models as models
import onnxslim
import onnx
from math import ceil
from typing import List
import torch
import torch.nn as nn
from collections import OrderedDict
from flwr.common import NDArrays


class Net(nn.Module):
    def get_weights(self) -> NDArrays:
        """
        Get model weights as a list of NumPy ndarrays.
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: NDArrays) -> None:
        """
        Set model weights from a list of NumPy ndarrays.
        """
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)})

        self.load_state_dict(state_dict, strict=True)

class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, 3)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.fc = torch.nn.Linear(256, 10)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class MobileNet(Net):
    def __init__(self):
        super(MobileNet, self).__init__()
        # 固定の設定を定義
        self.in_channels, self.width, self.height = 3, 224, 224  # 入力チャネルと画像サイズ
        self.conv_kernel_size, self.conv_kernel_stride = 3, 1
        self.pool_kernel_size, self.pool_kernel_stride = 2, 2
        self.width_multiplier = 1.0  # 幅の倍率
        out_dims = 10  # 出力次元
        
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

class InvertedResidual(nn.Module):
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
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.in_channels = 3  # 入力チャネル数固定
        self.width_multiplier = 1.0
        out_dims = 1000  # 出力次元（ImageNet用）

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

        input_channel = _make_divisible(32 * self.width_multiplier)
        last_channel = _make_divisible(1280 * self.width_multiplier)

        # First Conv Layer
        self.features = [nn.Sequential(
            nn.Conv2d(self.in_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        # Inverted Residual Blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * self.width_multiplier)
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

# model_name = 'resnet18'
# model_name = 'swin_t'
# model_name = 'vgg16'
# model_name = 'test_model'
model_name="models.mobilenet"
model_name="MobileNetV2"

if model_name == 'resnet18':
    model = models.resnet18()
elif model_name == 'swin_t':
    model = models.swin_t()
elif model_name == 'vgg16':
    model = models.vgg16()
elif model_name == 'models.mobilenet':
    model = models.mobilenet_v2(pretrained=False)
elif model_name == 'test_model':
    model = TestModel()
elif model_name == "mobile_net":
    model = MobileNet()
elif model_name == "MobileNetV2":
    model = MobileNetV2()

model_path = model_name + '.onnx'
model_slim_path = model_name + '_slim.onnx'

print(model)
# モデルのパラメータを計算
params = sum([p.numel() for p in model.parameters()])
print("Total parameters:", params)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,                # モデルオブジェクト
    dummy_input,          # ダミー入力
    # 'efficientnet_b0.onnx',         # 出力ファイル名
    model_path,         # 出力ファイル名
    export_params=True,   # 学習済みのパラメータを含める
    opset_version=11,     # ONNXのバージョン
    do_constant_folding=True,  # 定数フォールディングの実行
    input_names=['image'],     # 入力名
    output_names=['output'],   # 出力名
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 動的軸の指定
)
model_onnx_slim = onnxslim.slim(model_path)

onnx.save(model_onnx_slim, model_slim_path)