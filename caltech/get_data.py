# get_data.py

"""Convert the original image to the required size and save it"""
import os
import tensorflow as tf
import cv2
import numpy as np

input_dir = 'data/4'

output_dir = 'train_data'

# The type of recognition required
classes = {'airplane', 'car', 'face', 'motorbike'}


def num_images(path):
    """sum of images.
    
    Args:
        path: input dir

    Returns:
        image num

    """
    num = 0
    for dir in os.listdir(path):
       for _ in os.listdir(path + '/' + dir):
           num += 1
    return num


num_examples = num_images(input_dir)


def create_record():
    """make TFRecords data"""
    writer = tf.python_io.TFRecordWriter("caltech_4.tfrecords")
    for index, name in enumerate(classes):
        class_path = input_dir + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = cv2.imread(img_path)
            img = cv2.resize(
                img, (64, 64), interpolation=cv2.INTER_NEAREST)  # resize img
            data = img.tobytes()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    """Read the image into the tensorFlow queue.
    
    Args:
        filename: read image to tf queue path.

    Returns:
        data: image data.
        label:image label.

    """
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # Analyze symbolic samples
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    data = features['data']
    data = tf.decode_raw(data, tf.uint8)
    data = tf.reshape(data, [64, 64, 3])
    label = tf.cast(label, tf.int32)
    return data, label


if __name__ == '__main__':
    print(f"Images total: {num_examples}.")
    print(f"Start create record!")
    create_record()
    batch = read_and_decode('caltech_4.tfrecords')
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session() as sess:
        # init all variables
        sess.run(init_op)
        print("Init all variables complete!")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_examples):
            example, lab = sess.run(batch)
            # Convert the array to an image
            img = cv2.cvtColor(
                np.asarray(example),
                cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                output_dir +
                '/' +
                str(i) +
                'samples' +
                str(lab) +
                '.jpg',
                img)
        print(f"Image written to '{output_dir}'.")
        coord.request_stop()
        coord.join(threads)
        sess.close()
