import cv2
import os


def num_images(path):
    for dir in os.listdir(path):
        for file in os.listdir(path + '/' + dir):
            img = cv2.imread(file)
            img = cv2.resize(img, (64, 64))
            cv2.imwrite(file, img)

