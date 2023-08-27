import torch
import torch.nn as nn


class PositionAttention(nn.Module):
    """
    Position PositionAttention Module(Self-PositionAttention)
    """

    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        # self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=(1, 1))
        # self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=(1, 1))
        # self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                              stride=(1, 1), padding=1)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activation = nn.ReLU(inplace=True)
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        # key = self.key_conv(x).view(batch_size, -1, height * width)
        # value = self.value_conv(x).view(batch_size, -1, height * width)
        query = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        key = x.view(batch_size, -1, height * width)
        value = x.view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        output = torch.bmm(value, attention.permute(0, 2, 1))
        output = output.view(batch_size, channels, height, width)

        # output = self.gamma * output + x
        output = output + x
        # output = self.conv1x1(output)
        output = self.conv(output)
        output = self.activation(self.bn(output))
        return output