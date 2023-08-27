import cv2
import numpy as np
from camera_configs import stereoCamera

camera = stereoCamera()


def cat2images(left_img, right_img):
    """
    拼接左目和右目图像
    :param left_img: 左目图像
    :param right_img: 右目图像
    :return: 拼接后图像
    """
    HEIGHT = left_img.shape[0]
    WIDTH = left_img.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 2, 3))
    imgcat[:, :WIDTH, :] = left_img
    imgcat[:, -WIDTH:, :] = right_img
    return imgcat


def correction(image):
    """
    对图像进行极线矫正
    :param image: 代矫正双目图像
    :return: 校正后双目图像
    """
    height = image.shape[0]
    width = image.shape[1] // 2
    left_image = image[:, :width, :]
    right_image = image[:, width:, :]

    camera_matrix_left = camera.cam_matrix_left
    distortion_left = camera.distortion_l

    camera_matrix_right = camera.cam_matrix_right
    distortion_right = camera.distortion_r

    R = camera.R

    T = camera.T

    (R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
        cv2.stereoRectify(camera_matrix_left, distortion_left, camera_matrix_right, distortion_right,
                          np.array([width, height]), R, T)  # 计算旋转矩阵和投影矩阵

    (map1, map2) = \
        cv2.initUndistortRectifyMap(camera_matrix_left, distortion_left, R_l, P_l, np.array([width, height]),
                                    cv2.CV_32FC1)  # 计算校正查找映射表

    rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_CUBIC)  # 重映射

    # 左右图需要分别计算校正查找映射表以及重映射
    (map1, map2) = \
        cv2.initUndistortRectifyMap(camera_matrix_right, distortion_right, R_r, P_r, np.array([width, height]),
                                    cv2.CV_32FC1)

    rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_CUBIC)

    imgcat_out = cat2images(rect_left_image, rect_right_image)
    return imgcat_out

    # return rect_left_image, rect_right_image
