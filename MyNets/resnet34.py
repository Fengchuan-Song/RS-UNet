import torch
import torch.nn as nn
import numpy as np
from MyNets.CoordAttention import CoordAttention
from MyNets.SEAttentions import SEBlock


def Conv3x3(in_channels, out_channels, padding=1, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        padding=padding,
        stride=stride,
        groups=groups,
        dilation=dilation,
        bias=False
    )


def Conv1x1(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1, 1),
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = down_sample
        self.conv1 = Conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(in_channels=out_channels, out_channels=out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, filer_num, block_num, num_classes, r):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filer_num[0], kernel_size=(7, 7), stride=(2, 2),
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(filer_num[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(in_channels=filer_num[0], filter_num=filer_num[1], block_num=block_num[0])
        self.layer2 = self._make_layer(in_channels=filer_num[1], filter_num=filer_num[2], block_num=block_num[1],
                                       stride=2)
        self.layer3 = self._make_layer(in_channels=filer_num[2], filter_num=filer_num[3], block_num=block_num[2],
                                       stride=2)
        self.layer4 = self._make_layer(in_channels=filer_num[3], filter_num=filer_num[4], block_num=block_num[3],
                                       stride=2)

        self.attention_1 = CoordAttention(inp=filer_num[1], oup=filer_num[1], reduction=r[0])
        self.attention_2 = CoordAttention(inp=filer_num[2], oup=filer_num[2], reduction=r[1])
        self.attention_3 = CoordAttention(inp=filer_num[3], oup=filer_num[3], reduction=r[2])
        self.attention_4 = CoordAttention(inp=filer_num[4], oup=filer_num[4], reduction=r[3])

        # self.attention_1 = SEBlock(num_channels=filer_num[1], rate=r[0])
        # self.attention_2 = SEBlock(num_channels=filer_num[2], rate=r[1])
        # self.attention_3 = SEBlock(num_channels=filer_num[3], rate=r[2])
        # self.attention_4 = SEBlock(num_channels=filer_num[4], rate=r[3])

        self.averagepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=filer_num[4], out_features=num_classes)

        self._init_weight()

    def _make_layer(self, in_channels, filter_num, block_num, stride=1):
        down_sample = None
        if stride != 1:
            down_sample = nn.Sequential(
                Conv1x1(in_channels=in_channels, out_channels=filter_num, stride=stride),
                nn.BatchNorm2d(filter_num)
            )
        layers = []
        layers.append(
            BasicBlock(in_channels=in_channels, out_channels=filter_num, stride=stride, down_sample=down_sample)
        )
        for _ in range(1, block_num):
            layers.append(
                BasicBlock(in_channels=filter_num, out_channels=filter_num)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)
        x = self.maxpool(feat1)

        x = self.layer1(x)
        feat2 = self.attention_1(x)
        x = self.layer2(feat2)
        feat3 = self.attention_2(x)
        x = self.layer3(feat3)
        feat4 = self.attention_3(x)
        x = self.layer4(feat4)
        feat5 = self.attention_4(x)

        # x = self.averagepool(feat5)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return feat1, feat2, feat3, feat4, feat5

    def _init_weight(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet34(r, pretrained=False):
    """ResNet-34 with SE block"""
    model = ResNet([64, 64, 128, 256, 512], [2, 2, 2, 2], 1000, r=r)
    if pretrained:
        model_dict = model.state_dict()
        # print(model_dict.keys())
        pretrained_model = torch.load("./model_data/resnet18-f37072fd.pth")
        pretrained_dict = {k: v for k, v in pretrained_model.items() if np.shape(model_dict[k]) == np.shape(v)}
        # print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # del model.conv1, model.bn1, model.relu, model.maxpool
    del model.averagepool
    del model.fc
    return model
