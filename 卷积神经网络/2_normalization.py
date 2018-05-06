# coding=utf-8
from mxnet import nd, gluon, autograd

import utils


# 卷基层: 卷积 + 激活 + 池化
def net(X, is_training=False, verbose=False):
    X = X.as_in_context(W1.context)
    # X = X.flatten().reshape((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=W1.shape[0])
    # 第一层归一化
    h1_bn = batch_norm(h1_conv, gamma1, beta1, is_training,
                       moving_mean1, moving_variance1)
    # 第一层激活函数
    h1_activation = nd.relu(h1_bn)
    # 第一层pooling
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))

    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
    h2_bn = batch_norm(h2_conv, gamma2, beta2, is_training,
                       moving_mean2, moving_variance2)
    h2_activation = nd.relu(h2_bn)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))

    h2 = nd.flatten(h2)

    # 第一层全连接
    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)
    # 第二层全连接
    h4_linear = nd.dot(h3, W4) + b4
    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear

def batch_norm(X, gamma, beta, is_training,
               moving_mean, moving_variance, moving_momentum=0.9, eps=1e-5):
    assert len(X.shape) in (2, 4)

    if len(X.shape) == 2:  # 全连接: batch_size x feature
        mean = X.mean(axis=0)
        variance = ((X - mean) ** 2).mean(axis=0)
    else:  # 2D卷积: batch_size x channel x height x width
        # 对每个通道算均值和方差，需要保持4D形状使得可以正确地广播
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        variance = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

        # 变形使得可以正确的广播
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)

    # 均一化
    if is_training:
        X_hat = (X - mean) / nd.sqrt(variance + eps)

        moving_mean[:] = moving_momentum * mean + (1 - moving_momentum) * moving_mean
        moving_variance[:] = moving_momentum * moving_variance \
                             + (1 - moving_momentum) * moving_variance
    else:
        X_hat = (X - moving_mean) / nd.sqrt(moving_variance + eps)
    # 拉升和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)


from mxnet import nd
def pure_batch_norm(X, gamma, beta, eps=1e-5):
    assert len(X.shape) in (2, 4)
    # 全连接: batch_size x feature
    if len(X.shape) == 2:
        # 每个输入维度在样本上的平均和方差
        mean = X.mean(axis=0, keepdims=True)
        variance = ((X - mean)**2).mean(axis=0, keepdims=True)
    # 2D卷积: batch_size x channel x height x width
    else:
        # 对每个通道算均值和方差，需要保持4D形状使得可以正确地广播
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)

    # 均一化
    X_hat = (X - mean) / nd.sqrt(variance + eps)
    # 拉升和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)

A = nd.arange(6).reshape((3,2))
pure_batch_norm(A, gamma=nd.array([1,1]), beta=nd.array([0,0]))

if __name__ == '__main__':
    ctx = utils.try_gpu()
    weight_scale = 0.01

    # 输出通道 = 20, 卷积核 = (5,5)
    c1 = 20
    W1 = nd.random_normal(shape=(c1, 1, 5, 5), scale=weight_scale, ctx=ctx)
    b1 = nd.random_normal(shape=c1, scale=weight_scale, ctx=ctx)

    gamma1 = nd.random_normal(shape=c1, scale=weight_scale, ctx=ctx)
    beta1 = nd.random_normal(shape=c1, scale=weight_scale, ctx=ctx)
    moving_mean1 = nd.zeros(c1, ctx=ctx)
    moving_variance1 = nd.zeros(c1, ctx=ctx)

    # 输出通道 = 50，kernel (3, 3)
    c2 = 50
    W2 = nd.random_normal(shape=(c2, c1, 3, 3), scale=weight_scale, ctx=ctx)
    b2 = nd.random_normal(shape=c2, scale=weight_scale, ctx=ctx)

    gamma2 = nd.random_normal(shape=c2, scale=weight_scale, ctx=ctx)
    beta2 = nd.random_normal(shape=c2, scale=weight_scale, ctx=ctx)
    moving_mean2 = nd.zeros(shape=c2, ctx=ctx)
    moving_variance2 = nd.zeros(shape=c2, ctx=ctx)

    # 输出维度 = 128
    o3 = 128
    W3 = nd.random.normal(shape=(1250, o3), scale=weight_scale, ctx=ctx)
    b3 = nd.zeros(o3, ctx=ctx)

    # 输出维度 = 10
    W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
    b4 = nd.zeros(W4.shape[1], ctx=ctx)

    # 注意这里moving_*是不需要更新的
    params = [W1, b1, gamma1, beta1,
              W2, b2, gamma2, beta2,
              W3, b3, W4, b4]

    for param in params:
        param.attach_grad()

    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    learning_rate = 0.5

    epochs = 5
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data, is_training=True)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            utils.SGD(params, learning_rate / batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)

        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
