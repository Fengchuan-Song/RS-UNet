import os
import cv2

image_path = '../VOCdevkit/VOC2007/JPEGImages'

list = []

for each in os.listdir(image_path):
    image = cv2.imread(os.path.join(image_path, each))
    list.append(image.shape[:2])

inedx_min = 0
index_max = 0
maximum = 0
minimum = 1000000

for each in range(len(list)):
    temp = list[each][0] * list[each][1]

    if temp > maximum:
        maximum = temp
        index_max = each
    if temp < minimum:
        minimum = temp
        inedx_min = each

print(list[index_max])
print(list[inedx_min])