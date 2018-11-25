import tensorflow as tf


def print_activations(t):
    print(t.op.name, f" {t.get_shape().as_list()}")


def data_type():
    return tf.float32


def inference(images, classes):
    """Build the AlexNet model.

    Args:
      images: Images Tensor
      classes: Image classes

    Returns:
      out: prediction image label.

    """
    # # conv1
    # with tf.name_scope('conv1') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [11, 11, 3, 64]), name='weights')
    #     conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[64],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     bias = tf.nn.bias_add(conv, biases)
    #     conv1 = tf.nn.relu(bias, name=scope)
    #     print_activations(conv1)
    #
    # # lrn1
    # with tf.name_scope('lrn1') as scope:
    #     lrn1 = tf.nn.lrn(conv1,
    #                      alpha=1e-4,
    #                      beta=0.75,
    #                      depth_radius=2,
    #                      bias=2.0
    #                      )
    #
    # # pool1
    # pool1 = tf.nn.max_pool(lrn1,
    #                        ksize=[1, 3, 3, 1],
    #                        strides=[1, 2, 2, 1],
    #                        padding='VALID',
    #                        name='pool1')
    # print_activations(pool1)
    #
    # # conv2
    # with tf.name_scope('conv2') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [5, 5, 64, 192]), name='weights')
    #     conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[192],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     bias = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.relu(bias, name=scope)
    # print_activations(conv2)
    #
    # # lrn2
    # with tf.name_scope('lrn2') as scope:
    #     lrn2 = tf.nn.lrn(conv2,
    #                      alpha=1e-4,
    #                      beta=0.75,
    #                      depth_radius=2,
    #                      bias=2.0)
    #
    # # pool2
    # pool2 = tf.nn.max_pool(lrn2,
    #                        ksize=[1, 3, 3, 1],
    #                        strides=[1, 2, 2, 1],
    #                        padding='VALID',
    #                        name='pool2')
    # print_activations(pool2)
    #
    # # conv3
    # with tf.name_scope('conv3') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [3, 3, 192, 384]), name='weights')
    #     conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[384],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     bias = tf.nn.bias_add(conv, biases)
    #     conv3 = tf.nn.relu(bias, name=scope)
    #     print_activations(conv3)
    #
    # # conv4
    # with tf.name_scope('conv4') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [3, 3, 384, 256]), name='weights')
    #     conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[256],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     bias = tf.nn.bias_add(conv, biases)
    #     conv4 = tf.nn.relu(bias, name=scope)
    #     print_activations(conv4)
    #
    # # conv5
    # with tf.name_scope('conv5') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [3, 3, 256, 256]), name='weights')
    #     conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[256],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     bias = tf.nn.bias_add(conv, biases)
    #     conv5 = tf.nn.relu(bias, name=scope)
    #     print_activations(conv5)
    #
    # # pool5
    # pool5 = tf.nn.max_pool(conv5,
    #                        ksize=[1, 3, 3, 1],
    #                        strides=[1, 2, 2, 1],
    #                        padding='VALID',
    #                        name='pool5')
    # print_activations(pool5)
    #
    # # fully 1
    # with tf.name_scope('fc1') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [2 * 2 * 256, 4096]), name='weights')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[4096],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     fc1 = tf.reshape(pool5, [-1, kernel.get_shape().as_list()[0]])
    #     fc1 = tf.nn.relu(tf.matmul(fc1, kernel) + biases)
    #     # dropout
    #     fc1 = tf.nn.dropout(fc1, 0.5)
    #
    # # fully 2
    # with tf.name_scope('fc2') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [4096, 4096]), name='weights')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[4096],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     fc2 = tf.reshape(fc1, [-1, kernel.get_shape().as_list()[0]])
    #     fc2 = tf.nn.relu(tf.matmul(fc2, kernel) + biases)
    #     # dropout
    #     fc2 = tf.nn.dropout(fc2, 0.5)
    #
    # # fully 3
    # with tf.name_scope('fc3') as scope:
    #     kernel = tf.Variable(tf.truncated_normal(
    #         [4096, 4]), name='weights')
    #     biases = tf.Variable(tf.constant(0.1,
    #                                      shape=[classes],
    #                                      dtype=data_type()),
    #                          name='biases')
    #     fc3 = tf.reshape(fc2, [-1, kernel.get_shape().as_list()[0]])
    #     out = tf.nn.relu(tf.matmul(fc3, kernel) + biases)
    #
    # return out
    # 64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                             name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(
            images, weights, strides=[
                1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # 池化层1
    # 3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(
            conv1, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(
            pool1,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm1')

    # 卷积层2
    # 16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                             name='biases', dtype=tf.float32)

        conv = tf.nn.conv2d(
            norm1, weights, strides=[
                1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # 池化层2
    # 3x3最大池化，步长strides为2，池化后执行lrn()操作，
    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(
            conv2,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm2')
        pool2 = tf.nn.max_pool(
            norm2, ksize=[
                1, 3, 3, 1], strides=[
                1, 1, 1, 1], padding='SAME', name='pooling2')

    # 全连接层3
    # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[64, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                             name='biases', dtype=tf.float32)

        local3 = tf.nn.relu(
            tf.matmul(
                reshape,
                weights) + biases,
            name=scope.name)

    # 全连接层4
    # 128个神经元，激活函数relu()
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                             name='biases', dtype=tf.float32)

        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # dropout层
    #    with tf.variable_scope('dropout') as scope:
    #        drop_out = tf.nn.dropout(local4, 0.8)

    # Softmax回归层
    # 将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, classes], stddev=0.005, dtype=tf.float32),
                              name='softmax_linear', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[classes]),
                             name='biases', dtype=tf.float32)

        softmax_linear = tf.add(
            tf.matmul(
                local4,
                weights),
            biases,
            name='softmax_linear')

    return softmax_linear


def losses(logits, labels):
    """Calculate the loss.

    Args:
        logits: prediction value.
        labels: real value.

    Returns:
        loss:   loss.

    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def optimization(loss, learning_rate):
    """optimizer loss value.

    Args:
        loss:           loss.
        learning_rate:  learning_rate.

    Returns:
        optimizer:      optimizer loss value.

    """
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
            loss, global_step=global_step)
    return optimizer


def evaluation(logits, labels):
    """Evaluation/accuracy calculation.

    Args:
        logits:     prediction value.
        labels:     real value.

    Returns:
        accuracy:   Average accuracy of current step.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
