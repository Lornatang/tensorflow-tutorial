import tensorflow as tf


# 卷积操作
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
        l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[
        1, k, k, 1], padding='SAME', name=name)


# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0,
                     alpha=0.001 / 9.0, beta=0.75, name=name)


