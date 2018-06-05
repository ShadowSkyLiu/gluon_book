# coding=utf-8
import random

from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn

class MultiNet(nn.Block):
    def __init__(self, **kwargs):
        super(MultiNet, self).__init__(**kwargs)
        # self.lat = gluon.nn.Dense(2)
        self.dense1 = gluon.nn.Dense(1)
        self.dense2 = gluon.nn.Dense(1)

    def forward(self, x):
        out1 = self.dense1(x)
        out2 = self.dense2(x)
        return out1, out2

def data_iter(X, y1, y2):
    batch_size = 10
    num_examples = 1000
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx)  # 随机排序
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y1, j), nd.take(y2, j)


if __name__ == '__main__':
    num_inputs = 2
    num_points = 5000
    # true_w2 = [4, 5.6]
    # true_b2 = 7

    X = nd.random_normal(shape=(num_points, num_inputs))

    # lat_w = nd.array([[2, -3.4], [4.5, 5.6]])
    # lat_b = nd.array([4.2, 7])
    # print(lat.shape)
    # lat = nd.dot(X, lat_w.T) + lat_b
    # lat1 = true_w[0][0] * X[:, 0] + true_w[0][1] * X[:, 1] + true_b[0]
    # lat1 += .01 * nd.random_normal(shape=lat1.shape)
    # lat2 = true_w[1][0] * X[:, 0] + true_w[1][1] * X[:, 1] + true_b[1]
    # lat2 += .01 * nd.random_normal(shape=lat2.shape)

    true_w1 = [2, -3]
    true_b1 = 4
    true_w2 = [5, 6]
    true_b2 = 7
    y1 = true_w1[0] * X[:, 0] + true_w1[1] * X[:, 1] + true_b1
    y1 += 0.01 * nd.random_normal(shape=y1.shape)

    y2 = true_w2[0] * X[:, 0] + true_w2[1] * X[:, 1] + true_b2
    y2 += 0.01 * nd.random_normal(shape=y2.shape)

    # y = lat[:, 0]
    # y2 = lat[:, 1]
    # lat = [lat1, lat2]


    batch_size = 10
    # dataset = gluon.data.ArrayDataset(X, (y, y2))
    # data_iter = gluon.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    data_iter = data_iter(X, y1, y2)
    # 定义神经网络
    # net = gluon.nn.Sequential()
    # net.add(gluon.nn.Dense(1))
    net = MultiNet()

    # 初始化
    net.initialize()

    # 损失函数（平方差）
    square_loss = gluon.loss.L2Loss()

    # 优化
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    # 训练
    epochs = 20
    batch_size = 10
    for e in range(epochs):
        total_loss = 0
        for data, label1, label2 in data_iter:
            with autograd.record():
                output1, output2 = net(data)
                loss = 0.5 * square_loss(output1, label1) + 0.5 * square_loss(output2, label2)
                print(loss)
            loss.backward()
            trainer.step(batch_size)  # 梯度下降
            total_loss = total_loss + nd.sum(loss).asscalar()
        print("Epoch %d, total_loss %f" % (e, total_loss/num_points))
    dense1 = net.dense1
    dense2 = net.dense2
    print(net)
    # print(net.lat.weight.data(), net.lat.bias.data())
    print(dense1.weight.data(), dense1.bias.data())
    print(dense2.weight.data(), dense2.bias.data())
