from model import *

import tensorflow as tf


# 构建模型
pred = train(x, weights, biases, keep_prob)

# 定义损失函数和学习步骤
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.initialize_all_variables()

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 获取批数据
        sess.run(
            optimizer,
            feed_dict={
                X: batch_xs,
                y: batch_ys,
                keep_prob: dropout})
        if step % display_step == 0:
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
            print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + "{:.6f}".format(
                loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    # 计算测试精度
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))