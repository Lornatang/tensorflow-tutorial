"""
    download the follow:
    mnist dataset,
    fashionMnist dataset
"""

import tensorflow as tf
import os
import subprocess

flags = tf.flags

flags.DEFINE_string(
    "dataset",
    None,
    "name of dataset to download [fashionmnist, mnist]")

FLAGS = flags.FLAGS


def download_mnist(dirpath):
    """Download the data from Yann's website, unless it's already here."""
    data_dir = os.path.join(dirpath, 'mnist')
    if os.path.exists(data_dir):
        print('Found MNIST - skip.')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base + file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir, file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)


def download_fmnist(dirpath):
    """Download the data from zalandoresearch github website, unless it's already here."""
    data_dir = os.path.join(dirpath, 'fashionmnist')
    if os.path.exists(data_dir):
        print('Found FashionMnist - skip.')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base + file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir, file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)


def prepare_data_dir(path='./data'):
    if not os.path.exists(path):
        os.mkdir(path)
        

if __name__ == '__main__':
    prepare_data_dir()

    if FLAGS.dataset == 'fashion':
        download_fmnist('./data')
    if FLAGS.dataset == 'mnist':
        download_mnist('./data')
