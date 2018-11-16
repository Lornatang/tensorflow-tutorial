"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: validation.py
# time: 2018/9/11 19:02
# license: MIT
"""
import argparse
import gzip
import os
import urllib.request

import numpy
import sys
import tensorflow as tf
from PIL import Image

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = '../data/MNIST/MNIST/'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

FLAGS = None


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32


def download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print(f"Successfully downloaded {filename} size {size} bytes.")

    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels


def fake_data(num_images):
    data = numpy.ndarray(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
                         dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
    for image in range(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels


def error_rate(predictions, labels):
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])


def imageprepare():
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    file_name = '../data/MNIST/MNIST/test/2.jpg'  # 导入自己的图片地址
    # in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(file_name).convert('L')
    tva = list(im.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    # print(tva)
    return tva


def main(_):
    if FLAGS.self_test:
        print("Runing self-test.")
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
        test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    else:
        # Get the data
        # Get the data.
        train_data_filename = download('train-images-idx3-ubyte.gz')
        train_labels_filename = download('train-labels-idx1-ubyte.gz')
        test_data_filename = download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

        # Extract it into numpy arrays.
        train_data = extract_data(train_data_filename, 60000)
        train_labels = extract_labels(train_labels_filename, 60000)
        test_data = extract_data(test_data_filename, 10000)
        test_labels = extract_labels(test_labels_filename, 10000)

        # Generate a validation set.
        validation_data = train_data[:VALIDATION_SIZE, ...]
        validation_labels = train_labels[:VALIDATION_SIZE]

    eval_data = tf.placeholder(data_type(), shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=data_type()))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=data_type()))

    def model(data, train=False):
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in range(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    # Create a local session to run the training.
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # Run all the initializers to prepare the trainable parameters.
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, '../../models/tensorflow/MNIST/MNIST/mnist.ckpt')
        print('Initialized!')

        # Loop through training steps.
        print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
        print(eval_in_batches(validation_data, sess))
        sys.stdout.flush()

        # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        print('Test error: %.1f%%' % test_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fp16',
        default=False,
        help='Use half floats instead of full floats if True.',
        action='store_true')
    parser.add_argument(
        '--self_test',
        default=False,
        action='store_true',
        help='True if running a self test.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
