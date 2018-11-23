import tensorflow as tf

NUM_CHANNELS = 3
SEED = 64487


def print_activations(t):
    print(t.op.name, f" {t.get_shape().as_list()}")


def data_type():
    return tf.float32


def inference(images, batch_size, n_labels):
    """Use simple LeCun Net to test.

    Args:
        images:     data.
        batch_size: single training.
        n_labels:  (car, airplane, motorbike, face) 4.

    Returns:
        out:        prediction image label.

    """

    # Conv 1
    with tf.name_scope('conv1'):
        conv1_weights = tf.Variable(tf.truncated_normal([11, 11, NUM_CHANNELS, 64],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=data_type()))
        conv1_biases = tf.Variable(
            tf.zeros([64], dtype=data_type()))
        conv1 = tf.nn.conv2d(images,
                             conv1_weights,
                             strides=[1, 4, 4, 1],
                             padding='SAME')
        # Bias and rectified linear non_linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        norm = tf.nn.local_response_normalization(relu,
                                                  depth_radius=2,
                                                  bias=2.0,
                                                  alpha=1e-4,
                                                  beta=0.75)
        print_activations(conv1)
    # Max pooling.The kernel size spec {ksize} also follows the layout.
    pool1 = tf.nn.max_pool(norm,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    print_activations(pool1)

    # Conv 2
    with tf.name_scope('conv2'):
        conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=data_type()))
        conv2_biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[192],
                dtype=data_type()))
        conv2 = tf.nn.conv2d(pool1,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        # Bias and rectified linear non_linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        norm = tf.nn.local_response_normalization(relu,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)
        print_activations(conv2)
    # Max pooling.The kernel size spec {ksize} also follows the layout.
    pool2 = tf.nn.max_pool(norm,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    print_activations(pool2)

    # Conv 3
    with tf.name_scope('conv3'):
        conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=data_type()))
        conv3_biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[384],
                dtype=data_type()))
        conv3 = tf.nn.conv2d(pool2,
                             conv3_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        # Bias and rectified linear non_linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        print_activations(conv3)

    # Conv 4
    with tf.name_scope('conv4'):
        conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=data_type()))
        conv4_biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[256],
                dtype=data_type()))

        conv4 = tf.nn.conv2d(relu,
                             conv4_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        # Bias and rectified linear non_linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        norm = tf.nn.lrn(relu,
                         4,
                         bias=1.0,
                         alpha=0.001 / 9.0,
                         beta=0.75)
        print_activations(conv4)

    # Conv 5
    with tf.name_scope('conv5'):
        conv5_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                        stddev=0.1,
                                                        seed=SEED,
                                                        dtype=data_type()))
        conv5_biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[256],
                dtype=data_type()))
        conv5 = tf.nn.conv2d(norm,
                             conv5_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        # Bias and rectified linear non_linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
        print_activations(conv5)
    # Max pooling.The kernel size spec {ksize} also follows the layout.
    pool5 = tf.nn.max_pool(relu,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    print_activations(pool5)

    # Fully 1
    with tf.name_scope('fc1'):
        img = tf.reshape(pool2, shape=[batch_size, -1])
        dim = img.get_shape()[1].value
        fc1_weights = tf.Variable(tf.truncated_normal([dim, 4096],
                                                      stddev=0.005,
                                                      seed=SEED,
                                                      dtype=data_type()))
        fc1_biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[4096],
                dtype=data_type()))

        fc1 = tf.reshape(pool5, [-1, fc1_weights.get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.matmul(fc1, fc1_weights) + fc1_biases)
        # dropout
        fc1 = tf.nn.dropout(fc1, 0.8)

    # Fully 2
    with tf.name_scope('fc2'):
        fc2_weights = tf.Variable(tf.truncated_normal([4096, 4096],
                                                      stddev=0.005,
                                                      seed=SEED,
                                                      dtype=data_type()))
        fc2_biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[4096],
                dtype=data_type()))
        fc2 = tf.reshape(fc1, [-1, fc2_weights.get_shape().as_list()[0]])
        fc2 = tf.nn.relu(tf.matmul(fc2, fc2_weights) + fc2_biases)
        # dropout
        fc2 = tf.nn.dropout(fc2, 0.8)

    # Fully 3
    with tf.name_scope('fc3'):
        fc3_weights = tf.Variable(tf.truncated_normal([4096, n_labels],
                                                      stddev=0.005,
                                                      seed=SEED,
                                                      dtype=data_type()))
        fc3_biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[n_labels],
                dtype=data_type()))
        fc3 = tf.reshape(fc2, [-1, fc3_weights.get_shape().as_list()[0]])
        out = tf.nn.relu(tf.matmul(fc3, fc3_weights) + fc3_biases)

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


def print_activity(t):
    print(t.op.name, f" {t.get_shape().as_list()}")
