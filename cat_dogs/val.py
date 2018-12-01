import numpy as np
import tensorflow as tf
import cv2

import model
import input_data


BATCH_SIZE = 32
N_CLASSES = 2
IMG_SIZE = 224

X = tf.placeholder(tf.float32, shape=[IMG_SIZE, IMG_SIZE, 3])


def get_one_image(filepath):
    """Read image to train.
    Args:
        filepath:  raw_data dir.

    Returns:
        image:  random read images from raw_data.

    """
    # n = len(filepath)
    # ind = np.random.randint(0, n)
    # Randomly select the test images
    # file = filepath[ind]

    data = cv2.imread(filepath)
    cv2.imshow('img', data)
    cv2.waitKey(0)
    data = cv2.resize(data, (224, 224))
    image = np.array(data)
    return image


def evaluate_one_image(data):
    """
    Args:
        data: image raw_data for array

    """
    image = tf.cast(data, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 224, 224, 3])

    logit = model.inference(image, N_CLASSES)

    # logit = tf.nn.softmax(logit)

    # you need to change the directories to yours.
    logs_train_dir = 'logs'

    saver = tf.train.Saver()

    with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={X: data})
            prediction = np.argmax(prediction)
            if prediction == 0:
                print(f"This is a cat with possibility  %.6f" %
                      prediction[:, 0])
            elif prediction == 1:
                print(f'This is a dog with possibility %.6f' %
                      prediction[:, 1])


if __name__ == '__main__':
    train_dir = 'data'
    val, val_label = input_data.get_files(train_dir)
    img = get_one_image('/Users/mac/Desktop/airplane.jpg')
    evaluate_one_image(img)
