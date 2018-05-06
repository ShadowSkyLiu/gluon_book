import matplotlib as mpl
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


def generateData():
    num_train = 100
    num_test = 100
    true_w = [1.2, -3.4, 5.6]
    true_b = 5.0
    x = nd.random_normal(shape=(num_train + num_test, 1))
    X = nd.concat(x, nd.power(x, 2), nd.power(x, 3))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
    return x, X, y


def train(train_data, train_label, test_data, test_label):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    learnning_rate = 0.01
    epochs = 100
    batch_size = min(10, train_label.shape[0])

    dataset_train = gluon.data.ArrayDataset(train_data, train_label)
    dataset_train_iter = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learnning_rate})
    square_loss = gluon.loss.L2Loss()

    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        for data, label in dataset_train_iter:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(square_loss(net(train_data), train_label).mean().asscalar())
        test_loss.append(square_loss(net(test_data), test_label).mean().asscalar())

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    print net[0].weight.data(), net[0].bias.data()
    return ('learned weight', net[0].weight.data(),
            'learned bias', net[0].bias.data())

if __name__ == '__main__':
    x, X, y = generateData()
    # train(x_train.reshape(y_train.shape()), y_train, x_test.reshape(y_test.shape()), y_test)
    train(X[:2, :], y[:2], X[100:, :],  y[100:])
