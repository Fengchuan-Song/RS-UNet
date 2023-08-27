import os
import PIL.Image as Image
import numpy as np
import torch

from utils.utils_measurment import shoreline_filter

# original_image_path = '../img/test6.jpg'
# original_image_path = '../VOCdevkit/VOC2007/JPEGImages/00218.jpg'
# ground_truth_path = 'E:/dataset/Port/SegmentationClass/test6.png'
# predicted_path = '../result_MyNet/PNG/00218.png'
# save_path = '../result_MyNet/Draw_line'

original_image_path = 'E:/ObjectDetection/Shoreline Detection/changed/image/00514.jpg'
# ground_truth_path = 'E:/dataset/USVInland/SegmentationClass/01047.png'
predicted_path = 'E:/ObjectDetection/Shoreline Detection/changed/seg/00514.png'
save_path = 'E:/ObjectDetection/Shoreline Detection/changed'

name = original_image_path.split('/')[-1].split('.')[0]

original_image = Image.open(original_image_path)
# ground_truth = Image.open(ground_truth_path)
predicted = Image.open(predicted_path)

original_image = np.array(original_image)
# ground_truth_array = torch.from_numpy(np.array(ground_truth)).float().unsqueeze(0).unsqueeze(0)
predicted_array = torch.from_numpy(np.array(predicted)).float().unsqueeze(0).unsqueeze(0)
# if name == '00218':
#     predicted_array[:, :, -30:, 300:] = 1
# predicted_array[:, :, -30:, :] = 1

# ground_truth_line = shoreline_filter(ground_truth_array)
predicted_line = shoreline_filter(predicted_array)

# ground_truth_lines = ground_truth_line.detach().int().numpy()[0][0]
predicted_lines = predicted_line.detach().int().numpy()[0][0]
height, width = predicted_lines.shape
# flag = 2 * np.ones(shape=width)
for i in range(height):
    for j in range(width):
        if predicted_lines[i, j] != 0:
            predicted_lines[i - 5:i + 1, j] = 1
            # predicted_lines[i - 1, j] = 1
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
original_image[predicted_lines != 0] = [255, 0, 0]
# predicted_img[predicted_lines != 0] = [255, 0, 0]

# image = Image.fromarray(np.uint8(ground_img + predicted_img))
image = Image.fromarray(np.uint8(original_image))
# image_result = Image.blend(original_image, image, 0.7)

# image_result.save(os.path.join(save_path, name+'_process.jpg'))
image.save(os.path.join(save_path, name+'_process.jpg'))
