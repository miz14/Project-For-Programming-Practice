import cv2
from matplotlib import pyplot as plt
import numpy as np


def cut_image(img):
    img_cut = img[31: 195, 8: 152]
    return img_cut


def detect_rectangle(image, obj):
    # image = image[25:, :]
    # image = image[:, :8]
    # image = image[:, 151:]
    image = cut_image(image)
    if obj == 'ball':
        image = image[62:158, :]
    elif obj == 'board':
        image = image[156:163, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_coords = (None, None)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        print(x, y)
        rectangle_coords = (x + width // 2, y + height // 2)
        plt.imshow(image, cmap='gray')
        plt.show()
        break
    
    return rectangle_coords


observation = cv2.imread('observation.jpg')
image_array = np.array(observation)
print(detect_rectangle(image_array, 'ball'))
plt.imshow(image_array, cmap='gray')
plt.show()