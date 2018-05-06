# coding=utf-8
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt
import random

# y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise
def generateData():
    num_inputs = 2
    num_examples = 1000

    true_w = [2, -3.4]
    true_b = 4.2
    X = nd.random_normal(shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += 0.01 * nd.random_normal(shape=y.shape)
    return X, y

def plotData(X, y):
    plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
    plt.show()

def data_iter(M, n):
    batch_size = 10
    num_examples = 1000
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx) # 随机排序
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(M, j), nd.take(n, j)

def rand_data_patch():
    X, y = generateData()
    for data, label in data_iter(X, y):
        print(data, label)
        break


def net(X, w, b):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

# 模型函数
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

# 绘制损失随训练次数降低的折线图，以及预测值和真实值的散点图
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :], w, b).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()



if __name__ == '__main__':
    # X, y = generateData()
    # plotData(X, y)
    # print rand_data_patch()

    epochs = 5
    learning_rate = .001
    niter = 0
    losses = []
    moving_loss = 0
    smoothing_constant = .01

    num_inputs = 2
    num_examples = 1000
    w = nd.random_normal(shape=(num_inputs, 1))
    b = nd.zeros((1,))
    params = [w, b]
    for param in params:
        param.attach_grad()

    # 训练
    X, y = generateData()
    for e in range(epochs):
        total_loss = 0

        for data, label in data_iter(X, y):
            with autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()
            SGD(params, learning_rate)
            total_loss += nd.sum(loss).asscalar()

            # 记录每读取一个数据点后，损失的移动平均值的变化；
            niter +=1
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

            # correct the bias from the moving averages
            est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

            if (niter + 1) % 100 == 0:
                losses.append(est_loss)
                print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (e, niter, est_loss, total_loss/num_examples))
                plot(losses, X)


