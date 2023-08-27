import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction):
        super(ChannelAttention, self).__init__()
        self.mid_channels = channels // reduction
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.convs = nn.Sequential(
            nn.Linear(in_features=channels, out_features=self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.mid_channels, out_features=channels)
        )
        # self.convs = nn.Sequential(
        #     nn.Conv2d(in_channels=channels, out_channels=channels // reduction, kernel_size=(1, 1), bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=(1, 1), bias=False)
        # )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maximum = self.maxpool(x)
        avgerage = self.avgpool(x)
        maximum = maximum.transpose(1, 2).transpose(2, 3)
        avgerage = avgerage.transpose(1, 2).transpose(2, 3)
        maximum = self.convs(maximum)
        avgerage = self.convs(avgerage)
        maximum = maximum.transpose(2, 3).transpose(1, 2)
        avgerage = avgerage.transpose(2, 3).transpose(1, 2)
        output = self.sigmoid(maximum + avgerage)

        return output


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_channel, _ = torch.max(x, dim=1, keepdim=True)
        avg_channel = torch.mean(x, dim=1, keepdim=True)
        max_avg_channel = torch.cat([max_channel, avg_channel], dim=1)
        output = self.conv(max_avg_channel)
        output = self.sigmoid(output)

        return output


class CBAM(nn.Module):
    def __init__(self, channels, reduction):
        super(CBAM, self).__init__()
        self.CA = ChannelAttention(channels=channels, reduction=reduction)
        self.SA = SpatialAttention()

    def forward(self, x):
        output = self.CA(x) * x
        output = self.SA(output) * output

        return output