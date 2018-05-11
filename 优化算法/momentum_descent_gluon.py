import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn
import numpy as np
import sys
sys.path.append('..')
import utils


def geneData(num_inputs=2, num_examples=1000):

    # 生成数据集。
    true_w = [2, -3.4]
    true_b = 4.2
    X = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += nd.random.normal(scale=0.01, shape=y.shape)
    return X, y

# 线性回归模型。
net = nn.Sequential()
net.add(nn.Dense(1))

X, y = geneData()

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'momentum': 0.99})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
               log_interval=10, X=X, y=y, net=net)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'momentum': 0.9})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
               log_interval=10, X=X, y=y, net=net)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'momentum': 0.5})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
               log_interval=10, X=X, y=y, net=net)

net.initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
               log_interval=10, X=X, y=y, net=net)