# coding=utf-8
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn

import utils


def mlpconv(channels, kernel_size, padding,
            strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1,
                  activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1,
                  activation='relu'))
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out

def net():
    net = nn.Sequential()
    # add name_scope on the outer most Sequential
    with net.name_scope():
        net.add(
            mlpconv(96, 11, 0, strides=4),
            mlpconv(256, 5, 2),
            mlpconv(384, 3, 1),
            nn.Dropout(0.5),
            # 目标类为10类
            mlpconv(10, 3, 1, max_pooling=False),
            # 输入为 batch_size x 10 x 5 x 5, 通过AvgPool2D转成
            # batch_size x 10 x 1 x 1。
            # 我们可以使用 nn.AvgPool2D(pool_size=5),
            # 但更方便是使用全局池化，可以避免估算pool_size大小
            nn.GlobalAvgPool2D(),
            # 转成 batch_size x 10
            nn.Flatten()
        )
    return net

if __name__ == '__main__':

    train_data, test_data = utils.load_data_fashion_mnist(
        batch_size=64, resize=224)

    ctx = utils.try_gpu()
    net = net()
    net.initialize(ctx=ctx, init=init.Xavier())

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': 0.1})
    utils.train(train_data, test_data, net, loss,
                trainer, ctx, num_epochs=1)


