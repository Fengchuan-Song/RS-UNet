import torch
import torch.nn as nn
from MyNets.resnet34 import resnet34
from MyNets.CoordAttention import CoordAttention


class BasicBlock(nn.Module):
    """
    基础模块，包括1x1、3x3、5x5和7x7卷积
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.mid_channels = middle_channels // 4

        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(1, 1))

        self.conv_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                                  stride=(1, 1), padding=1)

        self.conv_5x5_1 = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                                    stride=(1, 1), padding=1)
        self.conv_5x5_2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                                    stride=(1, 1), padding=1)

        self.conv_7x7_1 = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                                    stride=(1, 1), padding=1)
        self.conv_7x7_2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                                    stride=(1, 1), padding=1)
        self.conv_7x7_3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=(3, 3),
                                    stride=(1, 1), padding=1)

        self.integration = nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=(3, 3),
                                     stride=(2, 2), padding=(1, 1))

        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.bn_middle = nn.BatchNorm2d(num_features=middle_channels)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        
        self._init_weight()

    def forward(self, x):
        conv1x1 = self.conv_1x1(x)

        conv3x3 = self.conv_3x3(x)

        conv5x5 = self.conv_5x5_1(x)
        conv5x5 = self.conv_5x5_2(conv5x5)

        conv7x7 = self.conv_7x7_1(x)
        conv7x7 = self.conv_7x7_2(conv7x7)
        conv7x7 = self.conv_7x7_3(conv7x7)

        output = [conv1x1, conv3x3, conv5x5, conv7x7]
        output = torch.cat(output, dim=1)
        output = self.relu(self.bn_middle(output))
        output = self.integration(output)
        output = self.relu(self.bn(output))

        return output
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    """
    编码器模块，特征提取网络（可以单独训练）
    """

    def __init__(self, in_channels, num_channels, r, pretrained):
        super(Encoder, self).__init__()
        self.backbone = resnet34(r=r, pretrained=pretrained)

        # self.encoder_1 = BasicBlock(in_channels=in_channels, middle_channels=num_channels//2, out_channels=num_channels)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # # self.se_block_1 = SEBlock(num_channels=num_channels[0], rate=r[0])
        # self.se_block_2 = SEBlock(num_channels=num_channels[1], rate=r[1])
        # self.se_block_3 = SEBlock(num_channels=num_channels[2], rate=r[2])
        # self.se_block_4 = SEBlock(num_channels=num_channels[3], rate=r[3])
        # self.se_block_5 = SEBlock(num_channels=num_channels[4], rate=r[4])

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

        # # 取消第一级SE Block
        # feat1 = self.encoder_1(x)
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
        # # x = self.attention(x)
        # x = self.conv5x5(x)
        # # x = self.activation(self.bn(x))
        # feat5 = self.se_block_5(x)
        # # x = self.se_block_5(x)
        # # x = self.conv5x5(x)
        # # feat5 = self.activation(self.bn(x))

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

        # feat1 = self.encoder_1(x)
        # x = self.pool(feat1)
        feat1, feat2, feat3, feat4, feat5 = self.backbone(x)

        return feat1, feat2, feat3, feat4, feat5
