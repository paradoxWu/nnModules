import torch.nn as nn
import torch
import torch.functional as F

"""
skip connection is the core
模型堆层数会导致误差上升（梯度消失）,
而skip connection给出了一条路径,让模型学不动时至少可以复制输入

input x
├─→ 主路径:Weight Layer → BN → ReLU → Weight Layer → BN
└─→ 短路:x (element-wise add)
→ ReLU → 输出

core module:1.basic_block 2. bottle_neck
「BasicBlock = 2×3×3 简单堆叠，≤34 层够用；
Bottleneck = 1×1↓→3×3→1×1↑，参数↓40 %，深度↑，≥50 层标配。」
"""

"""
BasicBlock = (B,C,H,W) → (B,C,H',W')
"""


def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)


def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


"""
Bottleneck =「先降维 → 3×3 小卷积 → 再升维」的残差块，用更少的参数获得更深的网络，是 ResNet-50/101/152 的标配。
核心目标：在增加深度的同时，不增加计算量。
默认经过一次Bottleneck，feature通道数量*4
(B, C, H, W) → (B, 4C, H', W')
"""


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, planes, stride=1, downsample=None):
        super().__init__()
        hidden_c = in_c // self.expansion
        self.conv1 = conv1x1(in_c,hidden_c)
        self.bn1 = nn.BatchNorm2d(hidden_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(hidden_c,hidden_c,stride=stride)
        self.bn2 = nn.BatchNorm2d(hidden_c)
        self.conv3 = conv1x1(hidden_c,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_c,planes*self.expansion),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


"""
- 初始卷积:降低分辨率,提高feature维度(3->64)
- 四个残差层(可选basicblock或bottleneck)
- 全局平均池化
- head: 下游任务(暂不包括)
"""


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], 1, block=block)
        self.layer2 = self._make_layer(128, layers[1], 2, block=block)
        self.layer3 = self._make_layer(256, layers[2], 2, block=block)
        self.layer4 = self._make_layer(512, layers[3], 2, block=block)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_c, block_num, stride=1, block=BasicBlock):
        downsample = None
        if stride != 1 or self.inplanes != out_c * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, out_c * block.expansion, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_c * block.expansion),
            )
        layers = [block(self.inplanes, out_c, stride, downsample)]
        self.inplanes = out_c * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        input:(B,C,H,W)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    image = torch.randn(8, 3, 224, 224)
    feature = torch.randn(8, 64, 56, 56)
    model = BasicBlock(64, 64)
    output = model(feature)
    print(output.shape)
    model = Bottleneck(64, 64)
    output = model(feature)
    print(output.shape)

    res_type = "50"
    layers = [2, 2, 2, 2]
    if res_type == "34" or res_type == "50":
        layers = [3, 4, 6, 3]
    elif res_type == "101":
        layers = [3, 4, 23, 3]

    if res_type == "18" or res_type == "34":
        model = ResNet(BasicBlock, layers)
    else:
        model = ResNet(Bottleneck, layers)

    output = model(image)
    print(output.shape)
