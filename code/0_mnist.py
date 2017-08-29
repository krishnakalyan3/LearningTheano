#!/usr/bin/env python3
import load_data as ldb
from arch import cuda_cnn_18
import theano.tensor as T
import lasagne
import theano


def build_functions(network):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = ldb.mnist_data()
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = cuda_cnn_18(input_var, num_chan=1, width=28, num_fill=[32, 32, 64], num_outputs=10)
    train_fn, val_fn = build_functions(network, input_var, target_var)