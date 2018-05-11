# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline
import random
import sys
import mxnet as mx
from mxnet import autograd, gluon, nd
import numpy as np
sys.path.append('..')
import utils

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def genData(num_examples):
    # 生成数据集。
    num_inputs = 2
    true_w = [2, -3.4]
    true_b = 4.2
    X = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += 0.01 * nd.random.normal(scale=1, shape=y.shape)
    return X, y

# 初始化模型参数。
def init_params():
    num_inputs = 2
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params

# 线性回归模型。
def linreg(X, w, b):
    return nd.dot(X, w) + b

# 平方损失函数。
def squared_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

# 遍历数据集。
def data_iter(batch_size, num_examples, X, y):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i + batch_size, num_examples)])
        yield X.take(j), y.take(j)


net = linreg
# squared_loss = squared_loss

def optimize(batch_size, lr, num_epochs, log_interval, decay_epoch):
    num_examples = 1000
    X, y = genData(num_examples)
    w, b = init_params()

    y_vals = [squared_loss(net(X, w, b), y).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1):
        # 学习率自我衰减。
        if decay_epoch and epoch > decay_epoch:
            lr *= 0.1
        for batch_i, (features, label) in enumerate(
            data_iter(batch_size, num_examples, X, y)):
            with autograd.record():
                output = net(features, w, b)
                loss = squared_loss(output, label)
            loss.backward()
            sgd([w, b], lr, batch_size)
            if batch_i * batch_size % log_interval == 0:
                y_vals.append(squared_loss(net(X, w, b), y).mean().asnumpy())
    print('w:', w, '\nb:', b, '\n')
    x_vals = np.linspace(0, num_epochs, len(y_vals), endpoint=True)
    utils.semilogy(x_vals, y_vals, 'epoch', 'loss')


optimize(batch_size=1, lr=0.2, num_epochs=3, decay_epoch=2, log_interval=10)

optimize(batch_size=1000, lr=0.999, num_epochs=3, decay_epoch=None,
         log_interval=1000)

optimize(batch_size=10, lr=0.2, num_epochs=3, decay_epoch=2, log_interval=10)
optimize(batch_size=10, lr=5, num_epochs=3, decay_epoch=2, log_interval=10)