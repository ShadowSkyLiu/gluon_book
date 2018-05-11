import mxnet as mx
from mxnet import autograd, nd
import numpy as np
import sys
sys.path.append('..')
import utils

def sgd_momentum(params, lr, batch_size, vs, mom):
    for param, v in zip(params, vs):
        v[:] = mom * v + lr * param.grad / batch_size
        param[:] -= v

def genData(num_examples):
    num_inputs = 2
    true_w = [2, -3.4]
    true_b = 4.2
    X = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += nd.random.normal(scale=0.01, shape=y.shape)
    return X, y

def init_params():
    num_inputs = 2
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    vs = []
    for param in params:
        param.attach_grad()
        # 把速度项初始化为和参数形状相同的零张量。
        vs.append(param.zeros_like())
    return params, vs

def optimize(batch_size, lr, mom, num_epochs, log_interval):
    num_examples = 1000
    X, y = genData(num_examples)
    [w, b], vs = init_params()
    y_vals = [utils.squared_loss(utils.linreg(X, w, b), y).mean().asnumpy()]

    for epoch in range(1, num_epochs + 1):
        # 学习率自我衰减。
        if epoch > 2:
            lr *= 0.1
        for batch_i, (features, label) in enumerate(
                utils.data_iter(batch_size, num_examples, X, y)):
            with autograd.record():
                output = utils.linreg(features, w, b)
                loss = utils.squared_loss(output, label)
            loss.backward()
            sgd_momentum([w, b], lr, batch_size, vs, mom)
            if batch_i * batch_size % log_interval == 0:
                y_vals.append(utils.squared_loss(utils.linreg(X, w, b), y).mean().asnumpy())
    print('w:', w, '\nb:', b, '\n')
    x_vals = np.linspace(0, num_epochs, len(y_vals), endpoint=True)
    utils.semilogy(x_vals, y_vals, 'epoch', 'loss')

optimize(batch_size=10, lr=0.2, mom=0.99, num_epochs=3, log_interval=10)
optimize(batch_size=10, lr=0.2, mom=0.9, num_epochs=3, log_interval=10)
optimize(batch_size=10, lr=0.2, mom=0.5, num_epochs=3, log_interval=10)
optimize(batch_size=10, lr=0.2, mom=0, num_epochs=3, log_interval=10)
