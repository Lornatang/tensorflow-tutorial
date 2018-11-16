"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: train.py
# time: 2018/8/16 16:10
# license: MIT
"""

import argparse

import tensorflow as tf
import time

import utils

# Log
tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/catdog/test',
                    help="""image path. Default='../data/catdog/test'.""")
parser.add_argument('--epochs', type=int, default=20,
                    help="""num epochs. Default=10""")
parser.add_argument('--num_classes', type=int, default=2,
                    help="""cat and dog. Default=2""")
parser.add_argument('--batch_size', type=int, default=64,
                    help="""batch size. Default=64""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--dropout', type=float, default=0.75,
                    help="""Dropout numbers. Default=0.75""")
parser.add_argument('--model_path', type=str, default='../../model/tensorflow/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='catdog.ckpt',
                    help="""Model name""")
parser.add_argument('--display_epoch', type=int, default=2)
args = parser.parse_args()

# tf input graph
X = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
y = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)  # drop(keep probability)

# Load data
train_datasets = utils.LoadDataset(args.path)


# Create model
def conv2d(image, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pooling(image, k):
    return tf.nn.max_pool(image, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, args.num_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([args.num_classes]))
}


def net(x, _weights, _biases, _dropout):
    # Layer 1
    x = tf.reshape(x, [-1, 128, 128, 3])
    conv1 = conv2d(x, _weights['wc1'], _biases['bc1'])
    conv1 = max_pooling(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, keep_prob=_dropout)
    # Layer 2
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = max_pooling(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, keep_prob=_dropout)
    # Fully Connected
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    dense1 = tf.nn.dropout(dense1, _dropout)
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])

    return out


# Construct model
pred = net(X, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(args.lr).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Save model
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, args.epochs+1):
        start = time.time()
        batch_xs, batch_ys = train_datasets.next_batch(64)
        sess.run(optimizer, feed_dict={X: batch_xs,
                                       y: batch_ys,
                                       keep_prob: args.dropout})
        if epoch % args.display_epoch == 0:
            end = time.time()
            acc = sess.run(accuracy, feed_dict={X: batch_xs,
                                                y: batch_ys,
                                                keep_prob: 1.})
            loss = sess.run(cost, feed_dict={X: batch_xs,
                                             y: batch_ys,
                                             keep_prob: 1.})

            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss:.8f}, "
                  f"Acc: {acc}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

    model_path = args.model_path + args.model_name
    saver.save(sess, model_path)

    print(f"Model save to {model_path}!")

    images, labels = train_datasets.reads(batch_size=256)
    print("Testing 256 numbers accuracy is: ", sess.run(accuracy, feed_dict={X: images,
                                                                             y: labels,
                                                                             keep_prob: 1.}))
