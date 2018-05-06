# coding=utf-8
import mxnet.ndarray as nd
from mxnet import gluon, autograd
import matplotlib.pyplot as plt

def genData(num_train0, num_test0):
    # true_w = [1.2, -3.4, 5.6]
    # true_b = 5.0
    # x = nd.random_normal(shape=(num_train0 + num_test0, 1))
    # X = nd.concat(x, nd.power(x, 2), nd.power(x, 3))
    # y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
    # y[:] += .1 * nd.random_normal(shape=y.shape)
    num_inputs = 200
    true_w = nd.ones((num_inputs, 1)) * 0.01
    true_b = 0.05
    X = nd.random.normal(shape=(num_train + num_test, num_inputs))
    y = nd.dot(X, true_w) + true_b
    y += .01 * nd.random.normal(shape=y.shape)
    return X, y

def train(train_fea, train_label, test_fea, test_label, weight_decay):
    batch_size = 256

    dataset_train = gluon.data.ArrayDataset(train_fea, train_label)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)

    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    learning_rate = 0.005
    epochs = 10
    batch_size = min(10, train_label.shape[0])
    # weight_decay = 5  # 新参数, L2范数正则化

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': weight_decay})
    squire_loss = gluon.loss.L2Loss()

    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data0, label0 in data_iter_train:
            with autograd.record():
                output = net(data0)
                loss = squire_loss(output, label0)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(squire_loss(net(train_fea), train_label).mean().asscalar())
        test_loss.append(squire_loss(net(test_fea), test_label).mean().asscalar())

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    print (net[0].weight.data(), net[0].bias.data())


if __name__ == '__main__':
    num_train = 20
    num_test = 100
    data, label = genData(num_train, num_test)
    # train(data[:num_train, 0], label[:num_train], data[num_train:, 0], label[num_train:], 5)
    train(data[:num_train], label[:num_train], data[num_train:], label[num_train:], 0)
    train(data[:num_train], label[:num_train], data[num_train:], label[num_train:], 5)
    train(data[:num_train], label[:num_train], data[num_train:], label[num_train:], 10)
    train(data[:num_train], label[:num_train], data[num_train:], label[num_train:], 15)
    train(data[:num_train], label[:num_train], data[num_train:], label[num_train:], 20)
    print 'end'
