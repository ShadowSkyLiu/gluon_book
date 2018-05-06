# coding=utf-8
# 线性回归
# y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise

import mxnet.autograd as ag
import mxnet.ndarray as nd
import common


# 定义模型
def net(X_p, W_p, b_p):
    return nd.dot(X_p, W_p) + b_p

# 损失函数
def square_loss(yhat_p, y_p):
    return (yhat_p - y_p.reshape(yhat_p.shape)) ** 2

# 优化求解（梯度下降）
def SGD(params_p, lr_p):
    for param_p in params_p:
        # 必须是param_p[:] = ...  param = ...会重新创建新param，这个是没有attach_grad的
        param_p[:] = param_p - lr_p * param_p.grad

if __name__ == '__main__':
    # genData()
    num_inputs = 2
    num_examples = 1000
    X, y = common.genData(num_inputs, num_examples)  # 1.数据

    # 2.模型 (初始化）
    W = nd.random_normal(shape=(num_inputs, 1))
    b = nd.zeros((1,))
    params = [W, b]
    for param in params:
        param.attach_grad()

    # 3.参数 + 训练
    batch_size = 10
    learning_rate = 0.001
    epoch = 10

    niter = 0
    smoothing_constant = 0.01
    losses = []
    moving_loss = 0

    for e in range(epoch):  # 迭代次数
        total_loss = 0

        for data, label in common.data_iter(X, y, batch_size):  # 取数据
            with ag.record():
                output = net(data, W, b)
                loss = square_loss(output, label)  # 损失函数 （通过模型）
            loss.backward()  # 求导（对损失函数）
            SGD(params, learning_rate)  # 梯度下降（利用导数）

            total_loss += nd.sum(loss).asscalar()  # 每次迭代记录total_loss

        print("Epoch %d, average loss: %f" % (e, total_loss / num_examples))
    print(params)
