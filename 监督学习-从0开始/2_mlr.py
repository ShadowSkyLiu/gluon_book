# coding=utf-8
# 线性回归
# y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise

import mxnet.autograd as ag
import mxnet.ndarray as nd
from mxnet import gluon
import matplotlib.pyplot as plt

import common


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


def text_labels(label):
    text_labelget_text_labelss = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]


def softmax(X):
    X_exp = nd.exp(X)
    partition = X_exp.sum(axis=0, keepdims=True)
    return X_exp / partition


def net(X, W_p, b_p):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W_p) + b_p)


def cross_entropy(yhat, y):
    return - nd.log(nd.pick(yhat, y))


def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()

def evaluate_accuracy(data_iterator, net, W, b):
    acc = 0.
    for data, label in data_iterator:
        output = net(data, W, b)
        acc += accuracy(output, label)
    return acc / len(data_iterator)


if __name__ == '__main__':
    X = nd.random_normal(shape=(2, 5))
    X_prob = softmax(X)

    print(X)
    print(X_prob)
    print(nd.exp(X[0][0]) / (nd.exp(X[0][0]) + nd.exp(X[1][0])))

    # 1. 数据
    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
    batch_size = 256
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    # 2. 模型（线性模型）W,b
    num_inputs = 28 * 28
    num_outputs = 10
    W = nd.random_normal(shape=(num_inputs, num_outputs))
    b = nd.random_normal(shape=num_outputs)
    params = [W, b]
    for param in params:
        param.attach_grad()

    learning_rate = 0.01
    epoch = 20
    for e in range(epoch):
        train_loss = 0.
        train_acc = 0.

        for data, label in train_data:
            with ag.record():
                output = net(data, W, b)
                loss = cross_entropy(output, label)
            loss.backward()
            common.SGD(params, learning_rate / batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)

        test_acc = evaluate_accuracy(test_data, net, W, b)
        print("Epoch %d, average train loss: %f, train acc: %f, test acc: %f" %
              (e, train_loss / len(train_data), train_acc / len(train_data), test_acc))

