"""
  Simple,
  end-to-end,
  LeNet-5-like convolutional MNIST model example.
"""

from utils import *

import tensorflow as tf


# define network parameters
n_input = 784  # 输入的维度
n_classes = 10  # 标签的维度
dropout = 0.75  # Dropout 的概率


# 存储所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([192])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# 定义整个网络
def train(_x, _weights, _biases, _dropout):
    # 向量转为矩阵
    _X = tf.reshape(_x, shape=[-1, 28, 28, 1])

    # 第一层卷积
    # 卷积
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # 下采样
    pool1 = max_pool('pool1', conv1, k=2)
    # 归一化
    norm1 = norm('norm1', pool1, lsize=4)

    # 第二层卷积
    # 卷积
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # 下采样
    pool2 = max_pool('pool2', conv2, k=2)
    # 归一化
    norm2 = norm('norm2', pool2, lsize=4)

    # 第三层卷积
    # 卷积
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # 归一化
    norm3 = norm('norm3', conv3, lsize=4)

    # 第四层卷积
    # 卷积
    conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
    # 归一化
    norm4 = norm('norm4', conv4, lsize=4)

    # 第五层卷积
    # 卷积
    conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])
    # 下采样
    pool5 = max_pool('pool5', conv5, k=2)
    # 归一化
    norm5 = norm('norm5', pool5, lsize=4)

    # 全连接层1，先把特征图转为向量
    dense1 = tf.reshape(
        norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(
        tf.matmul(
            dense1,
            _weights['wd1']) +
        _biases['bd1'],
        name='fc1')
    dense1 = tf.nn.dropout(dense1, _dropout)

    # 全连接层2
    dense2 = tf.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]])
    dense2 = tf.nn.relu(
        tf.matmul(
            dense1,
            _weights['wd2']) +
        _biases['bd2'],
        name='fc2')  # Relu activation
    dense2 = tf.nn.dropout(dense2, _dropout)

    # 网络输出层
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out
