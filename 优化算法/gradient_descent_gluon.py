import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np
import sys
sys.path.append('..')
import utils

def genData(num_examples):
    # 生成数据集。
    num_inputs = 2
    true_w = [2, -3.4]
    true_b = 4.2
    X = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += 0.01 * nd.random.normal(scale=1, shape=y.shape)
    return X, y

def net():
    # 线性回归模型。
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(1))
    return net


# 优化目标函数。
def optimize(batch_size, trainer, num_epochs, decay_epoch, log_interval, X, y,
             net):
    # num_examples = 1000
    # X, y = genData(num_examples)
    dataset = gdata.ArrayDataset(X, y)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    square_loss = gloss.L2Loss()

    y_vals = [square_loss(net(X), y).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1):
        if decay_epoch and epoch > decay_epoch:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for batch_i, (features, label) in enumerate(data_iter):
            with autograd.record():
                output = net(features)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            if batch_i * batch_size % log_interval == 0:
                y_vals.append(square_loss(net(X), y).mean().asnumpy())
    # 为了便于打印，改变输出形状并转化成numpy数组。
    print('w:', net[0].weight.data(), '\nb:', net[0].bias.data(), '\n')
    x_vals = np.linspace(0, num_epochs, len(y_vals), endpoint=True)
    utils.semilogy(x_vals, y_vals, 'epoch', 'loss')

net = net()
num_examples = 1000
X, y = genData(num_examples)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
optimize(batch_size=1, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.999})
optimize(batch_size=1000, trainer=trainer, num_epochs=3, decay_epoch=None,
         log_interval=1000, X=X, y=y, net=net)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 5})
optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.002})
optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)
