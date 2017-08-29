#!/usr/bin/env python3
import numpy as np
import gzip
import os
import pickle
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

    x_train = load_mnist_images(DATA_FOLDER + '/' + 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(DATA_FOLDER + '/' + 'train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images(DATA_FOLDER + '/' + 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(DATA_FOLDER + '/' + 't10k-labels-idx1-ubyte.gz')

    x_train = np.asarray(x_train, dtype='float32')
    y_train = np.asarray(y_train, dtype='uint8')
    x_test = np.asarray(x_test, dtype='float32')
    y_test = np.asarray(y_test, dtype='uint8')

    return x_train, y_train, x_test, y_test


def not_mnist_data():
    NOT_MNIST = join(DATA_FOLDER, 'NotMNIST_shuffled.npy')
    data = np.load(NOT_MNIST)
    X, y = zip(*data)
    X = np.asarray(X, dtype='float32')
    y = np.asarray(y, dtype='uint8')
    return X,y


def load_cifar100_dataset():
    CIFAR_TRAIN = join(DATA_FOLDER, 'cifar-100-python', 'train')
    CIFAR_TEST = join(DATA_FOLDER, 'cifar-100-python', 'test')

    all_data = pickle.load(open(CIFAR_TRAIN, 'rb'), encoding='latin1')

    imgs = all_data.get('data')
    X_train = imgs.reshape(50000, 3, 32, 32)
    x_train = X_train / np.float32(255)
    labels = all_data.get('fine_labels')
    Y_train = np.array(labels, dtype='uint8')
    y_train = Y_train.flatten()

    test_dic = pickle.load(open(CIFAR_TEST, 'rb'), encoding='latin1')
    X_test = test_dic.get('data')
    X_test = X_test.reshape(10000, 3, 32, 32)
    x_test = X_test / np.float32(255)
    y_test = test_dic.get('fine_labels')
    y_test = (np.asarray(y_test, dtype='uint8')).flatten()

    x_train = np.asarray(x_train, dtype='float32')
    x_test = np.asarray(x_test, dtype='float32')

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    data = mnist_data()
    print(data)