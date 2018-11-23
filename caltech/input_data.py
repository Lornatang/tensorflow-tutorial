# input_data.py

"""
    We need to do sample and label classification of generated images of specified size,
    obtain the get_files input by neural network,
    and meanwhile input raw_data for batch processing
    in order to facilitate network training.
"""

import os
import math
import numpy as np
import tensorflow as tf


input_dir = 'raw_data'

airplane = []
airplane_label = []

car = []
car_label = []

face = []
face_label = []

motorbike = []
motorbike_label = []


def get_files(file_dir, ratio=0.3, train=True):
    """Get all the image path names in the directory,
    store them in the corresponding list,
    label them and store them in the label list.

    Args:
        train: check is trainning.
        file_dir: train raw_data dir.
        ratio:    ratio to control the test ratio.

    Returns:
        train:
            raw_data, train_label
        test:
            val_data, val_label

    """
    for file in os.listdir(file_dir + '/airplane'):
        airplane.append(file_dir + '/airplane' + '/' + file)
        airplane_label.append(0)
    for file in os.listdir(file_dir + '/car'):
        car.append(file_dir + '/car' + '/' + file)
        car_label.append(1)
    for file in os.listdir(file_dir + '/face'):
        face.append(file_dir + '/face' + '/' + file)
        face_label.append(2)
    for file in os.listdir(file_dir + '/motorbike'):
        motorbike.append(file_dir + '/motorbike' + '/' + file)
        motorbike_label.append(3)

    # Disarrange the generated image path and label List,
    # and combine (airplane, car, face, motorbike) into a List (img and lab).
    image_list = np.hstack((airplane, car, face, motorbike))
    label_list = np.hstack(
        (airplane_label,
         car_label,
         face_label,
         motorbike_label))

    # Shuffle the order
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # Convert all img and lab to list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # all sample num
    sample_num = len(all_label_list)
    
    if train:
        train_data = all_image_list[0:sample_num]
        train_label = all_label_list[0:sample_num]
        train_label = [int(float(i)) for i in train_label]
    
        return train_data, train_label
    else:
        # Use ratio to control the test ratio
        val_num = int(math.ceil(sample_num * ratio))  # val num
        train_num = sample_num - val_num  # train num
    
        val_data = all_image_list[train_num:-1]
        val_label = all_label_list[train_num:-1]
        val_label = [int(float(i)) for i in val_label]
    
        return val_data, val_label


def train_of_batch(image, label, image_W, image_H, batch_size, capacity):
    """Set the batch size for the exercise.

    Args:
        image:      raw_data
        label:      label
        image_W:    image width
        image_H:    image height
        batch_size: 64
        capacity:   max of queue

    Returns:
        image_batch:
        label_batch

    """
    # Convert type
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # Decoding the image,
    img = tf.image.decode_jpeg(image_contents, channels=3)

    # Data preprocessing, image rotation, scaling, cutting, normalization and other operations
    # are carried out to make the calculated model more robust.
    img = tf.image.resize_image_with_crop_or_pad(img, image_W, image_H)
    data = tf.image.per_image_standardization(img)

    # generator batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int64
    image_batch, label_batch = tf.train.batch([data, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
