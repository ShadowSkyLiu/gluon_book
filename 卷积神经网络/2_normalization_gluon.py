# coding=utf-8
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn

import utils


def getnet():
    net = nn.Sequential()
    with net.name_scope():
        # 第一层卷积
        net.add(nn.Conv2D(channels=20, kernel_size=5))
        # 添加了批量归一化层
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation(activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))

        # 第二层卷积
        net.add(nn.Conv2D(channels=50, kernel_size=3))
        # 添加了批量归一化层
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation(activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))

        net.add(nn.Flatten())
        # 第一层全连接
        net.add(nn.Dense(128, activation="relu"))
        # 第二层全连接
        net.add(nn.Dense(10))
    return net


if __name__ == '__main__':
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    ctx = utils.try_gpu()
    net = getnet()
    net.initialize(ctx=ctx)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})

    # learning_rate = 0.1
    epochs = 5
    for epoch in range(epochs):
        train_acc = 0.
        train_loss = 0.
        for data, label in train_data:
            with autograd.record():
                # data = data.flatten().reshape(
                #     (data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
