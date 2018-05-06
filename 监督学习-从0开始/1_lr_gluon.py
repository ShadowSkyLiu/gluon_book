# coding=utf-8
# 线性回归-gluon
# y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise

import mxnet.autograd as ag
import mxnet.ndarray as nd
from mxnet import gluon

import common

if __name__ == '__main__':
    # genData()
    num_inputs = 2
    num_examples = 1000
    batch_size = 10

    # 1.数据
    X, y = common.genData(num_inputs, num_examples)
    data_set = gluon.data.ArrayDataset(X, y)
    data_iter = gluon.data.DataLoader(data_set, batch_size, shuffle=True)

    # 2.定义模型
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()

    # 3.损失函数
    square_loss = gluon.loss.L2Loss()

    # 4.优化
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': 0.1})

    epoch = 10
    learning_rage = 0.001

    for e in range(epoch):
        total_loss = 0

        for data, label in data_iter:
            with ag.record():
                loss = square_loss(net(data), label)
            loss.backward()

            trainer.step(batch_size)
            total_loss += nd.sum(loss).asscalar()

        print("Epoch %d, average loss: %f" % (e, total_loss / num_examples))

    dense = net[0]
    print(dense.weight.data(), dense.bias.data())
