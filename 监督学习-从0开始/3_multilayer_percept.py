# coding=utf-8
from mxnet import gluon, autograd

import utils
from utils import load_data_fashion_mnist
import mxnet.ndarray as nd

def relu(X):
    return nd.maximum(X, 0)

def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output


if __name__ == '__main__':
    # load数据
    batch_size = 256
    train_data, test_data = load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28
    num_outputs = 10
    num_hidden = 1024
    weight_scale = 0.1

    W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
    b1 = nd.zeros(shape=num_hidden)
    W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
    b2 = nd.zeros(shape=num_outputs)

    params = [W1, b1, W2, b2]
    for param in params:
        param.attach_grad()

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    learning_rate = 0.5
    epochs = 10

    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.

        for data, label in train_data:
            with autograd.record():
                outputs = net(data)
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()
            utils.SGD(params, learning_rate / batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(outputs, label)
        test_acc = utils.evaluate_accuracy(test_data, net)

        print ("Epoch %d Loss: %f, Train acc: %f, Test acc: %f"
               % (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))

