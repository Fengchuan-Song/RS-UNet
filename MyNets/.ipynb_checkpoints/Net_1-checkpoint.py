import torch
import torch.nn as nn
from MyNets.SEAttentions import SEBlock
from MyNets.PositionAttention import PositionAttention
from MyNets.CBAM import CBAM
from MyNets.CoordAttention import CoordAttention


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    """
    基础模块，包括1x1、3x3、5x5和7x7卷积
    """

    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.mid_channels = out_channels // 4

        self.conv_1x1 = Conv(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(1, 1))

        self.conv_3x3 = Conv(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                             stride=(1, 1), padding=1)

        self.conv_5x5_1 = Conv(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.conv_5x5_2 = Conv(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)

        self.conv_7x7_1 = Conv(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.conv_7x7_2 = Conv(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.conv_7x7_3 = Conv(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)

        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(num_features=self.mid_channels)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        conv1x1 = self.conv_1x1(x)
        # conv1x1 = self.relu(self.bn(conv1x1))

        conv3x3 = self.conv_3x3(x)
        # conv3x3 = self.relu(self.bn(conv3x3))

        conv5x5 = self.conv_5x5_1(x)
        conv5x5 = self.conv_5x5_2(conv5x5)
        # conv5x5 = self.relu(self.bn(conv5x5))

        conv7x7 = self.conv_7x7_1(x)
        conv7x7 = self.conv_7x7_2(conv7x7)
        conv7x7 = self.conv_7x7_3(conv7x7)
        # conv7x7 = self.relu(self.bn(conv7x7))

        output = [conv1x1, conv3x3, conv5x5, conv7x7]
        output = torch.cat(output, dim=1)
        output = self.relu(self.bn(output))

        return output


class Encoder(nn.Module):
    """
    编码器模块，特征提取网络（可以单独训练）
    """

    def __init__(self, in_channels, num_channels, r):
        super(Encoder, self).__init__()
        self.encoder_1 = BasicBlock(in_channels=in_channels, out_channels=num_channels[0])
        self.encoder_2 = BasicBlock(in_channels=num_channels[0], out_channels=num_channels[1])
        self.encoder_3 = BasicBlock(in_channels=num_channels[1], out_channels=num_channels[2])
        self.encoder_4 = BasicBlock(in_channels=num_channels[2], out_channels=num_channels[3])
        self.encoder_5 = BasicBlock(in_channels=num_channels[3], out_channels=num_channels[4])

        self.conv5x5 = nn.Conv2d(in_channels=num_channels[4], out_channels=num_channels[4], kernel_size=(5, 5),
                                 stride=(1, 1), padding=2)
        # self.bn = nn.BatchNorm2d(num_features=num_channels[4])
        # self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.se_block_1 = SEBlock(num_channels=num_channels[0], rate=r[0])
        self.se_block_2 = SEBlock(num_channels=num_channels[1], rate=r[1])
        self.se_block_3 = SEBlock(num_channels=num_channels[2], rate=r[2])
        self.se_block_4 = SEBlock(num_channels=num_channels[3], rate=r[3])
        self.se_block_5 = SEBlock(num_channels=num_channels[4], rate=r[4])

        # # self.se_block_1 = CBAM(channels=num_channels[0], reduction=r[0])
        # self.se_block_2 = CBAM(channels=num_channels[1], reduction=r[1])
        # self.se_block_3 = CBAM(channels=num_channels[2], reduction=r[2])
        # self.se_block_4 = CBAM(channels=num_channels[3], reduction=r[3])
        # self.se_block_5 = CBAM(channels=num_channels[4], reduction=r[4])

        # # self.se_block_1 = CoordAttention(inp=num_channels[0], oup=num_channels[0], reduction=r[0])
        # self.se_block_2 = CoordAttention(inp=num_channels[1], oup=num_channels[1], reduction=r[1])
        # self.se_block_3 = CoordAttention(inp=num_channels[2], oup=num_channels[2], reduction=r[2])
        # self.se_block_4 = CoordAttention(inp=num_channels[3], oup=num_channels[3], reduction=r[3])
        # self.se_block_5 = CoordAttention(inp=num_channels[4], oup=num_channels[4], reduction=r[4])

        # self.attention = PositionAttention(in_channels=num_channels[4])

    def forward(self, x):
        # x = self.encoder_1(x)
        # feat1 = self.se_block_1(x)
        # x = self.pool(feat1)
        # x = self.encoder_2(x)
        # feat2 = self.se_block_2(x)
        # x = self.pool(feat2)
        # x = self.encoder_3(x)
        # feat3 = self.se_block_3(x)
        # x = self.pool(feat3)
        # x = self.encoder_4(x)
        # feat4 = self.se_block_4(x)
        # x = self.pool(feat4)
        # x = self.encoder_5(x)
        # x = self.conv5x5(x)
        # feat5 = self.se_block_5(x)

        # 取消第一级SE Block
        feat1 = self.encoder_1(x)
        x = self.pool(feat1)
        x = self.encoder_2(x)
        feat2 = self.se_block_2(x)
        x = self.pool(feat2)
        x = self.encoder_3(x)
        feat3 = self.se_block_3(x)
        x = self.pool(feat3)
        x = self.encoder_4(x)
        feat4 = self.se_block_4(x)
        x = self.pool(feat4)
        x = self.encoder_5(x)
        # x = self.attention(x)
        x = self.conv5x5(x)
        # x = self.activation(self.bn(x))
        feat5 = self.se_block_5(x)
        # x = self.se_block_5(x)
        # x = self.conv5x5(x)
        # feat5 = self.activation(self.bn(x))

        # 去掉SE Block部分
        # feat1 = self.encoder_1(x)
        # x = self.pool(feat1)
        # feat2 = self.encoder_2(x)
        # x = self.pool(feat2)
        # feat3 = self.encoder_3(x)
        # x = self.pool(feat3)
        # feat4 = self.encoder_4(x)
        # x = self.pool(feat4)
        # x = self.encoder_5(x)
        # feat5 = self.conv5x5(x)

        return feat1, feat2, feat3, feat4, feat5
