#!/usr/bin/env python3
import lasagne


def cuda_cnn_18(input_var=None, num_chan=3, width=32, num_fill=[32, 32, 64], num_outputs=10):

    network = lasagne.layers.InputLayer(shape=(None, num_chan, width, width),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network, num_filters=num_fill[0], filter_size=(5, 5),
                                         nonlinearity=lasagne.nonlinearities.rectify, W =lasagne.init.Normal(.05),
                                         pad=2, stride=1)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=2 )
    network = lasagne.layers.LocalResponseNormalization2DLayer(network,n=3, alpha=5e-5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=num_fill[1], filter_size=(5, 5), pad=2, stride=1,
                                         nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal(.05))
    network = lasagne.layers.Pool2DLayer(network, pool_size=(3, 3), stride=2, pad=0,  mode='average_exc_pad')
    network = lasagne.layers.LocalResponseNormalization2DLayer(network,n=3, alpha=5e-5)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=num_fill[2], filter_size=(5, 5), pad=2, stride=1,
        nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal(.05))
    network = lasagne.layers.Pool2DLayer(network, pool_size=(3, 3), stride=2, pad=0, mode='average_exc_pad')

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=num_outputs,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
