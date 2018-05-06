# coding=utf-8
import mxnet.ndarray as nd
from mxnet import gluon, autograd

import utils
from utils import load_data_fashion_mnist

if __name__ == '__main__':
    # load数据
    batch_size = 256
    train_data, test_data = load_data_fashion_mnist(batch_size)

    # 模型
    # num_inputs = 28 * 28
    # num_outputs = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(10))
    net.initialize()

    # 分开定义Softmax和交叉熵会有数值不稳定性，gluon提供一个将这两个函数合起来的数值更稳定的版本
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    # 优化
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))


