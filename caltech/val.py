import numpy as np
import tensorflow as tf
import cv2

import model
import input_data


BATCH_SIZE = 1
N_CLASSES = 4


def get_one_image(filepath):
    """Read image to train.
    Args:
        filepath:  data dir.

    Returns:
        image:  random read images from data.

    """
    n = len(train)
    ind = np.random.randint(0, n)
    # Randomly select the test images
    file = train[ind]

    data = cv2.imread(file)
    cv2.imshow('img', data)
    cv2.waitKey(0)
    data = cv2.resize(data, (224, 224))
    image = np.array(data)
    return image


def evaluate_one_image(image_array):
    """
    Args:
        image_array: image data for array

    """
    with tf.Graph().as_default():
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[224, 224, 3])

        # you need to change the directories to yours.
        logs_train_dir = 'logs'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print(f"This is a airplane with possibility  %.6f" % prediction[:, 0])
            elif max_index == 1:
                print(f'This is a face with possibility %.6f' %
                      prediction[:, 1])
            elif max_index == 2:
                print(f'This is a car with possibility %.6f' %
                      prediction[:, 2])
            else:
                print(f'This is a motorbike with possibility %.6f' %
                      prediction[:, 3])


# ------------------------------------------------------------------------

if __name__ == '__main__':
    train_dir = 'train_data'
    train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)
    img = get_one_image(val)
    evaluate_one_image(img)
