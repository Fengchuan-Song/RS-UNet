"""
    对网络性能进行实验验证
    实验内容包括 mIoU、mPA、Accuracy
"""

from utils.utils_metrics import compute_mIoU
import os

# gt_dir = 'E:/dataset/Port/SegmentationClass'
# pred_dir = '../result_MyNet_Port/segmentation_result'
# png_name_list_path = 'E:/dataset/Port/SegmentationClass'

"Ours"
gt_dir = '../VOCdevkit/VOC2007/SegmentationClass'
# pred_dir = '../result_MyNet/new_miou'
# pred_dir = '../result_Reversion/focal_0/PNG'
pred_dir = '../result_RS-UNet/PNG'
# pred_dir = 'E:/Semantic segmation/regression/results_ours(probability_L2_frac)/PNG'
png_name_list_path = '../VOCdevkit/VOC2007/ImageSets/Segmentation/trainval.txt'
"USVInland"
# gt_dir = 'E:/dataset/Water Segmentation/dataset/SegmentationClass'
# pred_dir = 'E:/Semantic segmation/regression/results_ours(probability_L2_frac_usvinland)/PNG'
# pred_dir = '../result_usvinland_waterseg/PNG'
# png_name_list_path = 'E:/dataset/Water Segmentation/dataset/ImageSets/Segmentation/val.txt'
"VOC07+12"
# gt_dir = 'E:/dataset/VOCdevkit/VOC2007/SegmentationClass'
# pred_dir = '../result_MyNet/new_miou'
# pred_dir = 'E:/Semantic segmation/regression/results_ours(probability_L2_frac_voc_2)/PNG'
# pred_dir = 'E:/Semantic segmation/regression/results_ours(probability_L2_frac_voc_with_dire_2)/PNG'
# pred_dir = 'E:/Semantic segmation/regression/results_ours(classification_voc)/PNG'
# pred_dir = 'E:/Semantic segmation/pspnet-pytorch-master/pspnet-regression/results_ours(probability_L2_frac_voc)/PNG'
# png_name_list_path = 'E:/dataset/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt'

# gt_dir = 'E:/ObjectDetection/Shoreline Detection/video_process/dataset/SegmentationClass'
# gt_dir = '../VOCdevkit/VOC2007/SegmentationClass'
# gt_dir = '../result_deeplabv3+/mask'
# pred_dir = '../regression_results_png_2/regression_results_png_2'
# pred_dir = '../result_regression_result_wo_thre/regression_results_wo_thre_png'
# png_name_list_path = 'E:/ObjectDetection/Shoreline Detection/video_process/dataset/SegmentationClass'

# gt_dir = 'E:/dataset/Water Segmentation/dataset/SegmentationClass'
# pred_dir = '../result_usvinland_waterseg'
# pred_dir = '../regression_results_png_usvinland'
# png_name_list_path = 'E:/dataset/Water Segmentation/dataset/SegmentationClass'
# png_name_list_path = 'E:/dataset/Water Segmentation/test/test/640_320_undistorted_gt'

# num_classes = 21
# name_classes = [
#     "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#     "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]

num_classes = 2
name_classes = ["background", "river"]

png_name_list = []
# for each in os.listdir(png_name_list_path):
#     png_name_list.append(str(each.split('.')[0]))
with open(png_name_list_path, "r") as f:
    for each in f.readlines():
        png_name_list.append(each.split('\n')[0])

hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)
