import os
import PIL.Image as Image
import numpy as np
import torch

from utils.utils_measurment import shoreline_filter

# original_image_path = 'E:/ObjectDetection/Shoreline Detection/video_process/dataset/JPEGImages'
# ground_truth_path = 'E:/ObjectDetection/Shoreline Detection/video_process/dataset/SegmentationClass'
# original_image_path = '../VOCdevkit/VOC2007/JPEGImages'
# ground_truth_path = '../VOCdevkit/VOC2007/SegmentationClass'
# predicted_path = '../result_MyNet/PNG'
# save_path = '../result_MyNet/Draw_line'

# original_image_path = 'E:/dataset/Port/JPEGImages'
# ground_truth_path = 'E:/dataset/Port/SegmentationClass'
# predicted_path = '../result_MyNet/PNG'
# save_path = '../result_MyNet/Draw_line'

original_image_path = 'E:/dataset/Water Segmentation/dataset/JPEGImages'
# ground_truth_path = 'E:/dataset/Port/SegmentationClass'
predicted_path = '../result_usvinland_waterseg/PNG'
save_path = '../USVInland/Draw_line'

if not os.path.exists(save_path):
    os.makedirs(save_path)

counter = os.listdir(original_image_path)

for index in range(len(counter)):
    name = counter[index].split('.')[0]
    print(name)
    original_image = Image.open(os.path.join(original_image_path, name + ".jpg"))
    original_image = np.array(original_image)
    # ground_truth = Image.open(os.path.join(ground_truth_path, name + ".png"))
    predicted = Image.open(os.path.join(predicted_path, name + '.png'))

    # ground_truth_array = torch.from_numpy(np.array(ground_truth)).float().unsqueeze(0).unsqueeze(0)
    predicted_array = torch.from_numpy(np.array(predicted)).float().unsqueeze(0).unsqueeze(0)
    # if name == '00514':
    #     predicted_array[:, :, -30:, :] = 1
    # else:
    #     continue

    # ground_truth_line = shoreline_filter(ground_truth_array)
    predicted_line = shoreline_filter(predicted_array)

    # ground_truth_lines = ground_truth_line.detach().int().numpy()[0][0]
    predicted_lines = predicted_line.detach().int().numpy()[0][0]
    # height, width = predicted_lines.shape
    # flag = 2 * np.ones(shape=width)
    # for i in range(height-2):
    #     for j in range(width):
    #         if predicted_lines[i, j] == 1:
    #             if flag[j] == 2:
    #                 predicted_lines[i - 2, j] = 1
    #                 predicted_lines[i - 1, j] = 1
    #                 flag[j] -= 1
    #             elif flag[j] == 1:
    #                 predicted_lines[i + 1, j] = 1
    #                 predicted_lines[i + 2, j] = 1
    #                 flag[j] -= 1

    # ground_img = np.zeros((np.shape(ground_truth_lines)[0], np.shape(ground_truth_lines)[1], 3))
    # ground_img[ground_truth_lines != 0] = [255, 0, 0]
    # predicted_img = np.zeros((np.shape(predicted_lines)[0], np.shape(predicted_lines)[1], 3))
    # predicted_img[predicted_lines != 0] = [0, 255, 0]
    # predicted_img[predicted_lines != 0] = [255, 0, 0]

    # image = Image.fromarray(np.uint8(ground_img + predicted_img))
    # image = Image.fromarray(np.uint8(predicted_img))
    # image_result = Image.blend(original_image, image, 0.7)
    original_image[predicted_lines != 0] = [255, 0, 0]
    image_result = Image.fromarray(np.uint8(original_image))
    image_result.save(os.path.join(save_path, name + '.jpg'))
