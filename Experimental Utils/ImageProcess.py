import PIL.Image as Image
import numpy as np
import cv2
import operator
import os

target = [0, 0, 0]
image_dir = '../result_MyNet/new'

for each in os.listdir(image_dir):
    name = each.split('.')[0]
    print(name)
    image_path = os.path.join(image_dir, each)

    image = Image.open(image_path)
    image = np.array(image)
    height, width, _ = image.shape
    for i in range(height):
        for j in range(width):
            if not operator.eq(image[i, j, :].tolist(), target):
                image[i, j, :] = [128, 0, 0]

    image = Image.fromarray(np.uint8(image))
    image.save(f'../result_MyNet/Processed_Image/{name}.png')
