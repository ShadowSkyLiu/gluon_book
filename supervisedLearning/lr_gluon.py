# coding=utf-8
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd

if __name__ == '__main__':
    num_inputs = 2
    num_points = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    X = nd.random_normal(shape=(num_points, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += .01 * nd.random_normal(shape=y.shape)

    batch_size = 10
    dataset = gluon.data.ArrayDataset(X, y)
    data_iter = gluon.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # 定义神经网络
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))

    # 初始化
    net.initialize()

    # 损失函数（平方差）
    square_loss = gluon.loss.L2Loss()

    # 优化
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    # 训练
    epochs = 5
    batch_size = 10
    for e in range(epochs):
        total_loss = 0
        for data, label in data_iter:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size) # 梯度下降
            total_loss = total_loss + nd.sum(loss).asscalar()
        print ("Epoch %d, total_loss %f" % (e, total_loss/num_points))
    dense = net[0]
    print dense.weight.data(), dense.bias.data()


