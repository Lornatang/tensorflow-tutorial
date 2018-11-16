"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: utils.py
# time: 2018/8/16 15:31
# license: MIT
"""

import numpy as np
import glob
import random
import cv2


# convert label
def label_to_list(label):
    label_list = [0, 0]
    label_list[label] = 1

    return label_list


class LoadDataset(object):
    def __init__(self, filepath):
        dogs = glob.glob(filepath + "/dog/*.jpg")
        cats = glob.glob(filepath + "/cat/*.jpg")
        classes = {0: cats, 1: dogs}
        self.data = []
        for item in classes:
            for filepath in classes[item]:
                self.data.append([filepath, item])

    def next_batch(self, batch_size):
        dataset = np.asarray(self.data)
        # shuffle data
        random.shuffle(dataset)
        images = []  # output
        labels = []  # output
        # batch_size
        for item in dataset[:batch_size]:
            img_raw = cv2.imread(item[0])
            img_raw = cv2.resize(img_raw, (128, 128))  # shape=(128,128,3)
            label = label_to_list(int(item[1]))  # shape=(2)
            images.append(img_raw)
            labels.append(label)

        return np.asarray(images), np.asarray(labels)

    def reads(self, batch_size, max_error_count=10):
        count = 0
        while True:
            try:
                return self.next_batch(batch_size)
            except OSError:
                count += 1
                if count >= max_error_count:
                    return None
