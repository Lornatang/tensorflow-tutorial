import os
import tensorflow as tf
import input_data
import model

# parameters
N_CLASSES = 4
IMG_W = 64  # img size
IMG_H = 64
BATCH_SIZE = 20
CAPACITY = 10
MAX_STEP = 4000
learning_rate = 0.0001

train_dir = 'train_data'
logs_train_dir = 'logs'


train_data, train_label, val_data, val_label = input_data.get_files(train_dir, 0.3)
# train data and label
train_batch, train_label_batch = input_data.train_of_batch(
    train_data, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# test data and label
val_batch, val_label_batch = input_data.train_of_batch(
    val_data, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# define train op
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.train(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

# test train op
test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = model.losses(test_logits, val_label_batch)
test_acc = model.evaluation(test_logits, val_label_batch)

# start log
summary_op = tf.summary.merge_all()

sess = tf.Session()
# writen to log
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
# Save model
saver = tf.train.Saver()
# init all variables
sess.run(tf.global_variables_initializer())
# queue monitor
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# train
try:
    for step in range(MAX_STEP):
        if coord.should_stop():
            break
        # start op node
        _, loss, accuracy = sess.run([train_op, train_loss, train_acc])

        # print and write to log.
        if step % 10 == 0:
            print(f"Step {step} loss {loss:.2f} accuracy {accuracy * 100.0:.2f}%")
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
            # Save model
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    print("Save!")

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()
