# coding=utf-8
from mxnet import nd, gluon, autograd
import utils

# 丢弃法防过拟合
def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    assert 0 <= keep_probability <= 1

    if keep_probability == 0:
        return nd.zeros_like(X)

    mask = nd.random.uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability
    return mask * X * (1 / keep_probability)


# 我们的模型就是将层（全连接）和激活函数（Relu）串起来，
# 并在应用激活函数后添加丢弃层。每个丢弃层的元素丢弃概率可以分别设置。
# 一般情况下，我们推荐把更靠近输入层的元素丢弃概率设的更小一点。
# 这个试验中，我们把第一层全连接后的元素丢弃概率设为0.2，把第二层全连接后的元素丢弃概率设为0.5。
def train(X):
    drop_prob1 = 0.2
    drop_prob2 = 0.5
    X = X.reshape((-1, num_inputs))

    # 第一层全连接
    h1 = dropout(nd.relu(nd.dot(X, W1) + b1), drop_prob1)

    h2 = dropout(nd.relu(nd.dot(h1, W2) + b2), drop_prob2)

    return nd.dot(h2, W3) + b3

def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = nd.relu(nd.dot(X, W1) + b1)
    h2 = nd.relu(nd.dot(h1, W2) + b2)
    return nd.dot(h2, W3) + b3


if __name__ == '__main__':
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28
    num_outputs = 10
    num_hidden1 = 256
    num_hidden2 = 256
    weight_scale = .01

    W1 = nd.random_normal(shape=(num_inputs, num_hidden1), scale=weight_scale)
    b1 = nd.zeros(num_hidden1)

    W2 = nd.random_normal(shape=(num_hidden1, num_hidden2), scale=weight_scale)
    b2 = nd.zeros(num_hidden2)

    W3 = nd.random_normal(shape=(num_hidden2, num_outputs), scale=weight_scale)
    b3 = nd.zeros(num_outputs)

    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        param.attach_grad()

    softmax_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    learning_rate = 0.5
    epochs = 10
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = train(data)
                loss = softmax_entropy(output, label)
            loss.backward()
            utils.SGD(params, learning_rate / batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss / len(train_data),
            train_acc / len(train_data), test_acc))
