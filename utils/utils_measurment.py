import torch
import torch.nn as nn
import random
import numpy as np
import cv2


def shoreline_filter(image):
    """
    sobel边界滤波器
    :param image: 灰度图/单通道图
    :return: 边界图
    """
    conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, padding_mode='replicate')
    conv1.weight.data = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float()
    # conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='replicate')
    # conv2.weight.data = torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]]).float()

    # image = conv2(conv1(image))
    image = conv1(image)

    return image


def get_parallax(data_left, data_right):
    """
    根据双目图像获取视差，利用字符串进行严格比较
    :param data_left: 左目图像
    :param data_right: 右目图像
    :return: 视差
    """
    length = data_left.shape[-1]
    begin = length // 2 - 10
    data_line_left = data_left[begin: begin + 10]
    line_left = data_line_left.argmax(axis=0)
    line_right = data_right.argmax(axis=0)
    string_left = ''
    string_right = ''
    for each in line_left:
        string_left = string_left + "%04d" % each
    for each in line_right:
        string_right = string_right + "%04d" % each
    location = string_right.find(string_left) / 4
    parallax = abs(location - begin)

    return parallax


def get_parallax_gen2(data_left, data_right):
    """
    根据双目图像获取视差，设定阈值进行近似匹配
    :param data_left: 左目图像
    :param data_right: 右目图像
    :return: 视差
    """
    length = data_left.shape[-1]
    begin = length // 2 - 10
    data_line_left = data_left[begin: begin + 10]
    line_left = data_line_left.argmax(axis=0)
    line_right = data_right.argmax(axis=0)
    parallax = -1
    for i in range(length - 10):
        if all(abs(line_right[i: i + 10] - line_left) < 5):
            parallax = abs(i - begin)
            break

    return parallax


def get_start_location(data):
    """
    根据岸线检测的结果确定截取子图的位置
    :param data: 右目图像
    :return: 子图的开始位置
    """
    length = data.shape[-1]
    center_x = length // 2
    line = data.argmax(axis=0)
    center_y = line[center_x]

    return center_x, center_y


def get_average(image):
    average = np.sum(image) / (image.shape[0] * image.shape[1])

    return average


def SSDA(search_image, target_image, SSDA_TH):
    """
    序贯相似性检测算法（SSDA）
    :param SSDA_TH: SSDA算法阈值
    :param search_image: 原始图像（整体图像）
    :param target_image: 模板图像（局部图像）
    :return: 图像块左上角坐标
    """
    H = search_image.shape[0]
    W = search_image.shape[1]
    h = target_image.shape[0]
    w = target_image.shape[1]

    # 可取子图范围
    range_x = H - h + 1
    range_y = W - w + 1

    search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    target_image_average = get_average(target_image)
    R_count_MAX = 0
    absolute_error_min = 0
    locate_x = 0
    locate_y = 0

    # 遍历所有可能的子图
    for i in range(range_x):
        for j in range(range_y):
            R_count = 0
            absolute_error = 0
            sub_image = search_image[i: i + h, j: j + w]
            sub_image_average = get_average(sub_image)
            # 计算子图内随机点的绝对误差小于阈值时能容纳的像素数量
            while absolute_error < SSDA_TH:
                R_count += 1
                x = random.randint(0, h - 1)
                y = random.randint(0, w - 1)
                sub_image_pixel = sub_image[x][y]
                target_image_pixel = target_image[x][y]
                absolute_error += abs((sub_image_pixel - sub_image_average) -
                                      (target_image_pixel - target_image_average))
            # 容纳的像素数最多的子图作为最配子图
            if R_count > R_count_MAX:
                R_count_MAX = R_count
                absolute_error_min = absolute_error
                locate_x = i
                locate_y = j
            elif R_count == R_count_MAX:
                if absolute_error <= absolute_error_min:
                    absolute_error_min = absolute_error
                    locate_x = i
                    locate_y = j

    return locate_x, locate_y


def get_parallax_gen3(line_data, image_left, image_right, SSDA_TH):
    height = line_data.shape[0]

    # 截取100*300像素的长方形子图作为目标图像进行匹配
    # 截取150*width像素的条带作为候选匹配区域
    center_x, center_y = get_start_location(line_data)
    if abs(center_y - height) >= 50:
        begin_x = center_y - 50
        begin_y = center_x - 150
        begin_x_left = begin_x - 25
        sub_image_right = image_right[begin_x:begin_x + 100, begin_y:begin_y + 300, :]
        sub_image_left = image_left[begin_x_left:begin_x_left+150, :, :]
    else:
        begin_x = center_y - 50
        begin_y = center_x - 150
        begin_x_left = begin_x - 25
        sub_height = 50 + height - center_y
        sub_image_right = image_right[begin_x:begin_x + sub_height, begin_y:begin_y + 300, :]
        sub_image_left = image_left[begin_x_left:begin_x_left+sub_height, :, :]
    locate_x, locate_y = SSDA(sub_image_left, sub_image_right, SSDA_TH)
    locate_x = begin_x_left + locate_x

    # return abs(locate_y - begin_y)
    return begin_x, begin_y, locate_x, locate_y


def get_parallax_gen4(line_data, image_left, image_right, SSDA_TH):
    height = line_data.shape[0]

    # 截取100*300像素的长方形子图作为目标图像进行匹配
    # 截取150*width像素的条带作为候选匹配区域
    center_x, center_y = get_start_location(line_data)
    if abs(center_y - height) >= 50:
        begin_x = center_y - 50
        begin_y = center_x - 150
        # begin_x_left = begin_x - 10
        sub_image_right = image_right[begin_x:begin_x + 100, begin_y:begin_y + 300, :]
        sub_image_left = image_left[begin_x:begin_x+100, :, :]
    else:
        begin_x = center_y - 50
        begin_y = center_x - 150
        # begin_x_left = begin_x - 10
        sub_height = 50 + height - center_y
        sub_image_right = image_right[begin_x:begin_x + sub_height, begin_y:begin_y + 300, :]
        sub_image_left = image_left[begin_x:begin_x+sub_height, :, :]
    locate_x, locate_y = SSDA(sub_image_left, sub_image_right, SSDA_TH)
    locate_x = begin_x + locate_x

    # return abs(locate_y - begin_y)
    return begin_x, begin_y, locate_x, locate_y


if __name__ == "__main__":
    # shoreline_filter()
    pass
