import numpy as np


# import cv2
# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参（camera 1）
        self.cam_matrix_left = np.array([[3351.0765, 0, 603.5131],
                                         [0, 3342.2683, 374.2914],
                                         [0, 0, 1]])
        # 右相机内参（camera 2）
        self.cam_matrix_right = np.array([[3337.3114, 0, 625.1203],
                                          [0, 3333.9567, 258.3215],
                                          [0, 0, 1]])
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        # 径向畸变RadialDistortion（k1, k2, k3） 切向畸变TangentialDistortion（p1, p2）
        self.distortion_l = np.array([0.15282, -1.32701, 0.00773, -0.01944, 0.00000])
        self.distortion_r = np.array([0.14665, -0.41980, -0.01124, 0.01483, 0.00000])

        # 旋转矩阵（matlab得到的需要转置处理）
        self.R = np.array([[1, -0.0375, -0.0036],
                           [0.0375, 1.0000, -0.0151],
                           [0.0036, 0.0151, 1]])
        # 平移矩阵
        self.T = np.array([-203.7578, 2.3357, -0.3431])

        self.size = (1280, 720)

        # 焦距(还没改)
        self.focal_length = 480.203  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = 203.7578  # 单位：mm， 为平移向量的第一个参数（取绝对值）

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)

    def get_focal(self):
        return self.cam_matrix_left[1][1]

    def get_baseline(self):
        return self.baseline
