# coding=utf-8
from mxnet import gluon, autograd
import mxnet.ndarray as nd

import utils
from utils import load_data_fashion_mnist

if __name__ == '__main__':
    # load数据
    batch_size = 256
    train_data, test_data = load_data_fashion_mnist(batch_size)

    num_hiddens = 256
    num_outputs = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(num_hiddens, activation="relu"))
        # net.add(gluon.nn.Dense(num_hiddens * 2, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))
    net.initialize()

    softmax_entropy_cross = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_entropy_cross(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        test_acc = utils.evaluate_accuracy(test_data, net)

        print ('epoch %d Train loss: %f, Train acc: %f, Test acc: %f'
               % (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))

# epoch 0 Train loss: 0.711417, Train acc: 0.740071, Test acc: 0.841113
# epoch 1 Train loss: 0.465691, Train acc: 0.828474, Test acc: 0.844141
# epoch 2 Train loss: 0.410246, Train acc: 0.849706, Test acc: 0.870703
# epoch 3 Train loss: 0.378519, Train acc: 0.861375, Test acc: 0.868555
# epoch 4 Train loss: 0.357473, Train acc: 0.867276, Test acc: 0.874902
# epoch 5 Train loss: 0.340063, Train acc: 0.875360, Test acc: 0.880273
# epoch 6 Train loss: 0.322978, Train acc: 0.879721, Test acc: 0.881543
# epoch 7 Train loss: 0.313399, Train acc: 0.884724, Test acc: 0.883691
# epoch 8 Train loss: 0.299920, Train acc: 0.888514, Test acc: 0.878711
# epoch 9 Train loss: 0.294491, Train acc: 0.891390, Test acc: 0.889941