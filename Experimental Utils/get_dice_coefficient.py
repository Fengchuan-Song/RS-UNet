import torch
import numpy
import os
import numpy as np
from PIL import Image


def Dice_Score(inputs, target):
    """
    @param inputs: predicted labels(Tensor with gradient)
    @param target: ground truth labels(Tensor with gradient)
    @return: dice coefficient based predicted labels and ground truch labels
    """
    smooth = 1e-5
    # return a tensor without gradient(detach())
    n, h, w = inputs.shape
    nt, ht, wt = target.shape

    inputs = np.eye(2)[inputs.reshape(n, -1)]
    inputs = inputs.reshape(n, h, w, 2)
    inputs = torch.from_numpy(inputs)
    seg_labels = np.eye(2)[target.reshape([nt, -1])]
    seg_labels = seg_labels.reshape(nt, ht, wt, 2)
    seg_labels = torch.from_numpy(seg_labels)
    ct = seg_labels.shape[-1]
    c = inputs.shape[-1]
    inputs = inputs.reshape(n, -1, c)
    target = seg_labels.reshape(nt, -1, ct)

    # 以类别为单位计算
    intersection = torch.sum(inputs * target, dim=[0, 1])
    summary = torch.sum(inputs, dim=[0, 1]) + torch.sum(target, dim=[0, 1])
    # 以图片为单位计算
    # intersection = torch.sum(inputs * target[..., :-1], dim=[1, 2])
    # summary = torch.sum(inputs, dim=[1, 2]) + torch.sum(target[..., :-1], dim=[1, 2])

    # 各个图像分割结果误差的均值
    score = torch.mean((2 * intersection) / (summary + smooth))

    return score


# gt_dir = 'E:/dataset/Port/SegmentationClass'
# pred_dir = '../result_MyNet_Port/segmentation_result'
# png_name_list_path = 'E:/dataset/Port/SegmentationClass'

gt_dir = '../VOCdevkit/VOC2007/SegmentationClass'
# pred_dir = '../result_MyNet/new_miou'
# pred_dir = '../result_Reversion/focal_0/PNG'
pred_dir = '../result_RS-UNet/PNG'
# pred_dir = 'E:/Semantic segmation/regression/results_ours(probability_L2_frac)/PNG'
png_name_list_path = '../VOCdevkit/VOC2007/ImageSets/Segmentation/trainval.txt'

# gt_dir = '../VOCdevkit/VOC2007/SegmentationClass'
# pred_dir = '../result_MyNet/new_miou'
# png_name_list_path = '../VOCdevkit/VOC2007/SegmentationClass'

# gt_dir = 'E:/ObjectDetection/Shoreline Detection/video_process/dataset/SegmentationClass'
# pred_dir = '../result_MyNet_USVInland'
# png_name_list_path = 'E:/ObjectDetection/Shoreline Detection/video_process/dataset/SegmentationClass'

num_classes = 2
name_classes = ['background', 'river']
png_name_list = []
# for each in os.listdir(png_name_list_path):
#     png_name_list.append(str(each.split('.')[0]))
with open(png_name_list_path, "r") as f:
    for each in f.readlines():
        png_name_list.append(each.split('\n')[0])

dice_score = 0
gt_imgs = [os.path.join(gt_dir, x + ".png") for x in png_name_list]
pred_imgs = [os.path.join(pred_dir, x + ".png") for x in png_name_list]
for ind in range(len(gt_imgs)):
    print(pred_imgs[ind], gt_imgs[ind])
    pred = np.expand_dims(np.array(Image.open(pred_imgs[ind])), axis=0)
    label = np.expand_dims(np.array(Image.open(gt_imgs[ind])), axis=0)
    dice_score += Dice_Score(pred, label)

print(dice_score / len(gt_imgs))
