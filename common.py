# coding=utf-8
import random

import mxnet.ndarray as nd


def genData(num_inputs0, num_examples0):

    true_w = [2, -3.4]
    true_b = 4.2

    X = nd.random_normal(shape=(num_examples0, num_inputs0))
    y = X[:, 0] * true_w[0] + X[:, 1] * true_w[1] + true_b
    y[:] = y + .01 * nd.random_normal(shape=y.shape)
    return X, y

def data_iter(X_p, y_p, batch_size_p):
    # 产生一个随机索引
    idx = range(y_p.size)
    random.shuffle(idx)
    for i in range(0, y_p.size, batch_size_p):
        j = nd.array(idx[i:min(i + batch_size_p, y_p.size)])
        yield nd.take(X_p, j), nd.take(y_p, j)

# 优化求解（梯度下降）
def SGD(params_p, lr_p):
    for param_p in params_p:
        # 必须是param_p[:] = ...  param = ...会重新创建新param，这个是没有attach_grad的
        param_p[:] = param_p - lr_p * param_p.grad

def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')