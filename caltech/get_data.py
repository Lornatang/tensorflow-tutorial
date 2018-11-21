"""Convert the original image to the required size and save it"""
import os
import tensorflow as tf
import cv2
import numpy as np
from glob import glob

input_dir = 'data/4'

output_dir = 'data'

# The type of recognition required
classes = {'airplanes', 'cars', 'faces', 'motorbikes'}


# sum of images
def num_images(path):
    num = 0
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isfile(sub_path):
            num += 1
    return num


num_examples = num_images('data/4')


# 制作TFRecords数据
def create_record():
    writer = tf.python_io.TFRecordWriter("caltech_4.tfrecords")
    for index, name in enumerate(classes):
        class_path = input_dir + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = cv2.imread(img_path)
            img = cv2.resize(
                img, (64, 64), interpolation=cv2.INTER_NEAREST)  # 设置需要转换的图片大小
            img = img.tobytes()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


# =======================================================================================
def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


if __name__ == '__main__':
    create_record()
    batch = read_and_decode('caltech_4.tfrecords')
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session() as sess:  # 开始一个会话
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_examples):
            example, lab = sess.run(batch)  # 在会话中取出image和label
            img = cv2.cvtColor(
                np.asarray(example),
                cv2.COLOR_RGB2BGR)  # 这里Image是之前提到的
            cv2.imwrite(
                output_dir +
                '/' +
                str(i) +
                'samples' +
                str(lab) +
                '.jpg',
                img)  # 存下图片;注意cwd后边加上‘/’
            print(example, lab)
        coord.request_stop()
        coord.join(threads)
        sess.close()
