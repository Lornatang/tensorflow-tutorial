import tensorflow as tf


def inference(images, batch_size, n_classes):
    """Use simple LeCun Net to test.

    Args:
        images:     data.
        batch_size: single training.
        n_classes:  (car, airplane, motorbike, face) 4.

    Returns:
        out: prediction image label.

    """
    # conv 1
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
                              name='weights1', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                             name='biases1', dtype=tf.float32)

        conv1 = tf.nn.conv2d(
            images, weights, strides=[
                1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv1, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        print_activity(conv1)

    # pool 1
    # local response normalization is beneficial to training。
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(
            conv1, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME', name='pool1')
        norm1 = tf.nn.local_response_normalization(
            pool1,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm1')
        print_activity(norm1)

    # conv 2
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
                              name='weights2', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                             name='biases2', dtype=tf.float32)

        conv2 = tf.nn.conv2d(
            norm1, weights, strides=[
                1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv2, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
        print_activity(conv2)

    # pool 2
    # local response normalization is beneficial to training。
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(
            conv2, ksize=[
                1, 3, 3, 1], strides=[
                1, 1, 1, 1], padding='SAME', name='pool2')
        norm2 = tf.nn.lrn(
            pool2,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm2')

        print_activity(norm2)

    # fully 1
    # 128 neurons, to reshape the output of a pool before layer into a line,
    # the activation function relu ()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 128],
                                                  stddev=0.005,
                                                  dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1,
                                         dtype=tf.float32,
                                         shape=[128]),
                             name='biases', dtype=tf.float32)

        local3 = tf.nn.relu(
            tf.matmul(
                reshape,
                weights) + biases,
            name=scope.name)

    # fully 2
    # 128 neurons, to reshape the output of a pool before layer into a line,
    # the activation function relu ()
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, 128],
                                                  stddev=0.005,
                                                  dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1,
                                         dtype=tf.float32,
                                         shape=[128]),
                             name='biases', dtype=tf.float32)

        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # dropout 0.8 data
    with tf.variable_scope('drop1') as scope:
        drop_out1 = tf.nn.dropout(local4, 0.8, name='drop1')

    # Softmax layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, n_classes],
                                                  stddev=0.005,
                                                  dtype=tf.float32),
                              name='weights',
                              dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1,
                                         dtype=tf.float32,
                                         shape=[n_classes]),
                             name='biases',
                             dtype=tf.float32)

        out = tf.add(
            tf.matmul(
                drop_out1,
                weights),
            biases,
            name='softmax_linear')

    return out


# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


def print_activity(t):
    print(t.op.name, f" {t.get_shape().as_list()}")
