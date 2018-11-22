# input_data.py

"""
    We need to do sample and label classification of generated images of specified size,
    obtain the get_files input by neural network,
    and meanwhile input data for batch processing
    in order to facilitate network training.
"""

import os
import math
import numpy as np
import tensorflow as tf


input_dir = 'train_data'

airplane = []
airplane_label = []

car = []
car_label = []

face = []
face_label = []

motorbike = []
motorbike_label = []


def get_files(file_dir, ratio):
    """Get all the image path names in the directory,
    store them in the corresponding list,
    label them and store them in the label list.
    
    Args:
        file_dir: train data dir.
        ratio:

    Returns:
        train_data:
        train_label:
        val_data:
        val_label:

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

    # Use ratio to control the test ratio
    sample_num = len(all_label_list)  # all sample num
    val_num = int(math.ceil(sample_num * ratio))  # val num
    train_num = sample_num - val_num  # train num

    train_data = all_image_list[0:train_num]
    train_label = all_label_list[0:train_num]
    train_label = [int(float(i)) for i in train_label]
    
    val_data = all_image_list[train_num:-1]
    val_label = all_label_list[train_num:-1]
    val_label = [int(float(i)) for i in val_label]

    return train_data, train_label, val_data, val_label


# step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def train_of_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    
    Args:
        image:
        label:
        image_W:
        image_H:
        batch_size:
        capacity:

    Returns:

    """
    # Convert type
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

    # ========================================================================
