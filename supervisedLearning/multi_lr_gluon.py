# coding=utf-8
import sys

from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd

sys.path.append('..')
import utils

if __name__ == '__main__':
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    # 定义神经网络
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(10))
    net.initialize()

    softmax_cross_entroy = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    for epoch in range(5):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entroy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
