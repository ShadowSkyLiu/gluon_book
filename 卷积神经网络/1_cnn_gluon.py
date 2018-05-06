# coding=utf-8
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn

import utils


def net():
    net0 = nn.Sequential()
    with net0.name_scope():
        net0.add(
            nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(128, activation="relu"),
            nn.Dense(10)
        )
    return net0

if __name__ == '__main__':
    # 初始化
    ctx = utils.try_gpu()
    net = net()
    net.initialize(ctx=ctx)
    print('initialize weight on', ctx)

    # 获取数据
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    # 训练
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': 0.5})

    learning_rate = 0.5

    epochs = 5
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
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
            epoch, train_loss / len(train_data),
            train_acc / len(train_data), test_acc))
