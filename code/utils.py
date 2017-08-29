#!/usr/bin/env python3
import numpy as np


def evaluation_train(x, y, batch_size, train_fn, epoch):
    train_loss = 0
    train_batches = 0
    train_acc = 0

    for i in range(epoch):
        for batch in iterate_minibatches(x, y, batch_size,  shuffle=True):
            inputs, targets = batch
            loss, acc = train_fn(inputs, targets)
            train_loss += loss
            train_acc += acc
            train_batches += 1
        loss_a = train_loss / train_batches
        acc_a = train_acc / train_batches
        print('epoch: {} \t train_loss: {:.6f} \t train_acc: {:.6f}'.format(i, loss_a, acc_a))

    loss_b = train_loss / train_batches
    acc_b = train_acc/ train_batches

    return loss_b, acc_b


def evaluation_test(x_test, y_test, batch_size, val_fn):
    test_losses = 0
    test_accuracies = 0
    test_batches = 0

    for batch in iterate_minibatches(x_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        loss, acc = val_fn(inputs, targets)
        test_losses += loss
        test_accuracies += acc
        test_batches += 1

    test_loss = test_losses / test_batches
    test_acc = test_accuracies / test_batches
    print('batch {} \t loss {:.6f} \t accuracy {:.6f}'.format(test_batches, test_losses, test_accuracies))

    return test_loss, test_acc


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    # Check if input rows is equal to target rows
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]