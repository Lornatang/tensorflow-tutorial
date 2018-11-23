import argparse
import os
import tensorflow as tf
import input_data
import model
import numpy as np
import sys

# parameters
N_LABELS = 4
IMG_W = 64  # img size
IMG_H = 64
BATCH_SIZE = 64
CAPACITY = 100
MAX_STEP = 1000
learning_rate = 0.0001

train_dir = 'data'
logs_train_dir = 'logs'

FLAGS = None


def main(_):
    if FLAGS.self_test:
        pass
        # train data and label
        # val_data, val_label = input_data.get_files(train_dir)
        #
        # val_batch, val_label_batch = input_data.train_of_batch(
        #     val_data,
        #     val_label,
        #     IMG_W,
        #     IMG_H,
        #     BATCH_SIZE,
        #     CAPACITY)
        #
        # # define val op
        # val_logits = model.inference(val_batch, BATCH_SIZE, N_LABELS)
        # val_loss = model.losses(val_logits, val_label_batch)
        # val_acc = model.evaluation(val_logits, val_label_batch)
    else:
        # train data and label
        train_data, train_label = input_data.get_files(train_dir)

        train_batch, train_label_batch = input_data.train_of_batch(
            train_data,
            train_label,
            IMG_W,
            IMG_H,
            BATCH_SIZE,
            CAPACITY)

        # define train op
        train_logits = model.inference(train_batch, BATCH_SIZE, N_LABELS)
        train_loss = model.losses(train_logits, train_label_batch)
        train_op = model.optimization(train_loss, learning_rate)
        train_acc = model.evaluation(train_logits, train_label_batch)

        # start log
        summary_op = tf.summary.merge_all()

        # Save model
        saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # writen to log
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

        # queue monitor
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # train
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                # start op node
                _, loss, accuracy = sess.run([train_op, train_loss, train_acc])
        
                # print and write to log.
                if step % 50 == 0:
                    print(
                        f"Step {step} Loss {loss:.6f} Accuracy {accuracy * 100.0:.2f}%")
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                if step % CAPACITY == 0:
                    # Save model
                    checkpoint_path = os.path.join(
                        logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
            print("Model saved!")

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            coord.request_stop()


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
