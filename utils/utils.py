import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def resize_image_opencv(image, size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, ic = image.shape
    h, w = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # cv2.resize中调整后的大小是W x H
    image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
    new_image = np.zeros((h, w, ic), np.uint8)
    new_image.fill(128)
    new_image[(h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw, :] = image

    return new_image, nw, nh


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image
