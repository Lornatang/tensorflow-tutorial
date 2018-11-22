import tensorflow as tf


def inference(images, batch_size, n_classes):
    """Use simple LeCun Net to test.

    Args:
        images:     data.
        batch_size: single training.
        n_classes:  (car, airplane, motorbike, face) 4.

    Returns:
        out:        prediction image label.

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
    with tf.variable_scope('fc1') as scope:
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

        fc1 = tf.nn.relu(
            tf.matmul(
                reshape,
                weights) + biases,
            name=scope.name)

    # fully 2
    # 128 neurons, to reshape the output of a pool before layer into a line,
    # the activation function relu ()
    with tf.variable_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, 128],
                                                  stddev=0.005,
                                                  dtype=tf.float32),
                              name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1,
                                         dtype=tf.float32,
                                         shape=[128]),
                             name='biases', dtype=tf.float32)

        relu4 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name='fc2')

    # dropout 0.8 data
    # with tf.variable_scope('drop1') as scope:
    #     drop_out1 = tf.nn.dropout(fc2, 0.8, name='drop1')

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
                relu4,
                weights),
            biases,
            name='softmax_linear')

    return out


def losses(logits, labels):
    """Calculate the loss.
    
    Args:
        logits: prediction value.
        labels: real value.

    Returns:
        loss:   loss.

    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
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
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
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


def print_activity(t):
    print(t.op.name, f" {t.get_shape().as_list()}")
