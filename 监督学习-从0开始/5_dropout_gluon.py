from mxnet import gluon, autograd, nd
from mxnet.gluon import nn

import utils

if __name__ == '__main__':
    drop_prob1 = 0.2
    drop_prob2 = 0.5

    num_hidden1 = 256
    num_hidden2 = 256

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Flatten())
        net.add(nn.Dense(num_hidden1, activation="relu"))
        net.add(nn.Dropout(drop_prob1))
        net.add(nn.Dense(num_hidden2, activation="relu"))
        net.add(nn.Dropout(drop_prob2))
        net.add(nn.Dense(10))
    net.initialize()

    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    softmax_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)

        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f"
              % (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))


