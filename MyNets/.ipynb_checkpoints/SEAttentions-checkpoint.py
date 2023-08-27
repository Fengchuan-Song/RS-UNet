import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 模块
    """

    def __init__(self, num_channels, rate):
        super(SEBlock, self).__init__()
        self.mid_channels = num_channels // rate
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=num_channels, out_features=self.mid_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=self.mid_channels, out_features=num_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        x = self.average_pool(x)
        x = x.transpose(1, 2).transpose(2, 3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.transpose(2, 3).transpose(1, 2)

        x = identity * x

        return x
