import torch
import torch.nn as nn
from MyNets.resnet34_backbone import resnet34
from MyNets.SEAttentions import SEBlock


class UpBlock(nn.Module):
    def __init__(self, in_channels, reduction):
        super(UpBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=(2 * in_channels[0]), out_channels=in_channels[0], kernel_size=(3, 3),
                                 stride=(1, 1), padding=1)
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels[1], out_channels=in_channels[0],
                                            kernel_size=(2, 2), stride=(2, 2))

        self.attention = SEBlock(num_channels=2 * in_channels[0], rate=reduction)

        self.bn = nn.BatchNorm2d(num_features=in_channels[0])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_1, input_2):
        """
        :param input_1:skip connection data
        :param input_2: up sample data
        :return:
        """
        input_2 = self.up_sample(input_2)
        output = torch.cat([input_1, input_2], dim=1)
        output = self.attention(output)
        output = self.conv3x3(output)
        # output = self.conv1x1(output)
        output = self.bn(output)
        output = self.relu(output)

        return output


class UpBlock_1(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(UpBlock_1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels,
                                 kernel_size=(1, 1), stride=(1, 1))
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels[1], out_channels=out_channels,
                                            kernel_size=(2, 2), stride=(2, 2))
        self.bn = nn.BatchNorm2d(num_features=2 * out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        # self.relu = nn.PReLU()

    def forward(self, input_1, input_2):
        """
        :param input_1:skip connection data
        :param input_2: up sample data
        :return:
        """
        input_1 = self.conv1x1(input_1)
        input_2 = self.up_sample(input_2)

        output = torch.cat([input_1, input_2], dim=1)
        output = self.bn(output)
        output = self.relu(output)

        return output


class Net(nn.Module):
    """Changed UNet module"""

    def __init__(self, num_classes, r, pretrained=False):
        super(Net, self).__init__()
        self.backbone = resnet34(pretrained=pretrained)

        self.up_block4 = UpBlock(in_channels=[256, 512], reduction=r[0])
        self.up_block3 = UpBlock(in_channels=[128, 256], reduction=r[1])
        self.up_block2 = UpBlock(in_channels=[64, 128], reduction=r[2])
        self.up_block1 = UpBlock(in_channels=[64, 64], reduction=r[3])

        self.final = nn.ConvTranspose2d(in_channels=64, out_channels=num_classes, kernel_size=(2, 2),
                                        stride=(2, 2))

        # self.final = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1, 1),
        #                        stride=(1, 1), padding=0)

        # self.up_block4 = UpBlock_1(in_channels=[256, 512])
        # self.up_block3 = UpBlock_1(in_channels=[128, 256])
        # self.up_block2 = UpBlock_1(in_channels=[64, 256])
        # self.up_block1 = UpBlock_1(in_channels=[64, 256])
        #
        # self.final = nn.ConvTranspose2d(in_channels=256, out_channels=num_classes, kernel_size=(2, 2),
        #                                 stride=(2, 2))
        # self.final = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1),
        #                        stride=(1, 1), padding=0)

        # self._init_weight()

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.backbone(x)

        up4 = self.up_block4(feat4, feat5)
        up3 = self.up_block3(feat3, up4)
        up2 = self.up_block2(feat2, up3)
        up1 = self.up_block1(feat1, up2)

        final = self.final(up1)

        return final

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

    def freeze_backbone(self):
        for params in self.backbone.parameters():
            params.requires_grad = False

    def unfreeze_backbone(self):
        for params in self.backbone.parameters():
            params.requires_grad = True
