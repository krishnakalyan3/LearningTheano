#!/usr/bin/env python3
import numpy as np
import gzip
import os
from os.path import join

CWD = os.getcwd()
DATA_FOLDER = join(CWD,  '..', 'data')


def mnist_data():
    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = load_mnist_images(DATA_FOLDER + '/' +'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(DATA_FOLDER + '/' +'train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images(DATA_FOLDER + '/' +'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(DATA_FOLDER + '/' +'t10k-labels-idx1-ubyte.gz')

    y_train_mat = np.zeros((y_train.shape[0], 10), dtype=np.uint8)
    y_train_mat[np.arange(len(y_train)), y_train] = 1
    y_test_mat = np.zeros((y_test.shape[0], 10), dtype=np.uint8)
    y_test_mat[np.arange(len(y_test)), y_test] = 1

    return x_train, y_train, x_test, y_test




if __name__ == '__main__':
    data = mnist_data()
    print(data)