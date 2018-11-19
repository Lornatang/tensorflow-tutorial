from model import *

import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)


if not os.path.exists('./checkpoint_dir'):
    os.makedirs('./checkpoint_dir')

# define network hyper parameters
learning_rate = 0.001
epochs = 10
batch_size = 128

# 占位符输入
X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# 构建模型
pred = train(X, weights, biases, keep_prob)

# 定义损失函数和学习步骤
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Save model
saver = tf.train.Saver()

# 开启一个训练
with tf.Session() as sess:
    # init all global variables.
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 获取批数据
        sess.run(
            optimizer,
            feed_dict={
                X: batch_xs,
                y: batch_ys,
                keep_prob: dropout})
        # 计算精度
        acc = sess.run(
            accuracy,
            feed_dict={
                X: batch_xs,
                y: batch_ys,
                keep_prob: 1.})
        # 计算损失值
        loss = sess.run(
            cost,
            feed_dict={
                X: batch_xs,
                y: batch_ys,
                keep_prob: 1.})
        step += 1
        # formatted output values.
        print(f"Epoch [{epoch}/{epochs}]"
              f"Iter [{step * batch_size}]"
              f"Loss {loss:.6f}"
              f"Acc {acc:.8f}")
    print("Optimization Finished!")
    # 计算测试精度
    saver.save(sess, './checkpoint_dir/mnist.ckpt')
    print("Model save to 'checkpoint_dir/mnist.ckpt'")
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
