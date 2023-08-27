import torch
from thop import profile
from nets.UNet_original import UNet
from MyNets.Unet_resnet import Net
from fvcore.nn import FlopCountAnalysis, parameter_count_table

print('==> Building model..')
# model = UNet(in_channels=3, num_classes=2)
model = Net(num_classes=2, r=[8, 8, 8, 8])

tensor = (torch.rand(1, 3, 256, 256),)

flops = FlopCountAnalysis(model, tensor)

print("FLOPs: ", flops.total() / 1e9)

# 分析parameters
print(parameter_count_table(model))
# params = parameter_count_table(model)
# print('flops: %.2f M, params: %.2f M' % (flops.total() / 1e6, params / 1e6))
