# coding=utf-8
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

square_loss = gluon.loss.L2Loss()

def get_rmse_log(net, X, y):
    num_train1 = X.shape[0]
    clipped_preds = nd.clip(net(X), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y))).asscalar() / num_train1)


def get_data():
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    all_X = pd.concat((train_data.loc[:, 'MSSubClass':'SaleCondition'],
                       test_data.loc[:, 'MSSubClass':'SaleCondition']))
    numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
    all_X[numeric_feats] = all_X[numeric_feats] \
        .apply(lambda x: (x - x.mean()) / (x.std()))

    # 离散数据点转换成数值标签
    all_X = pd.get_dummies(all_X, dummy_na=True)
    # 缺失数据用本特征的平均值估计
    all_X = all_X.fillna(all_X.mean())

    num_train = train_data.shape[0]
    X_train = all_X[:num_train].as_matrix()
    X_test = all_X[num_train:].as_matrix()
    y_train = train_data.SalePrice.as_matrix()

    X_train = nd.array(X_train)
    y_train = nd.array(y_train)
    y_train.reshape((num_train, 1))
    X_test = nd.array(X_test)
    return X_train, y_train, X_test, test_data

def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

def train(net, X_train, y_train, X_test, y_test, epochs,
          verbose_epoch, learning_rate, weight_decay):

    global cur_test_loss
    train_loss = []
    test_loss = []
    batch_size = 256

    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate, 'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)

    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
        train_loss.append(cur_train_loss)

        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)

    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train', 'test'])
    plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss

def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    global X_val_train, y_val_train
    assert k > 1
    fold_size = X_train.shape[0] // k

    train_loss_sum = 0.0
    test_loss_sum = 0.0

    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test,
            epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k

def learn(epochs, verbose_epoch, X_train, y_train, X_test, test, learning_rate,
          weight_decay):
    net = get_net()
    train(net, X_train, y_train, None, None, epochs, verbose_epoch,
          learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    train_fea, train_label, test_fea, test_data = get_data()

    k = 5
    epochs = 100
    verbose_epoch = 95
    learning_rate = 0.5
    weight_decay = 0.0

    # train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, train_fea,
    #                                            train_label, learning_rate, weight_decay)
    # print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %
    #       (k, train_loss, test_loss))

    # 损失函数定义为平方误差

    learn(epochs, verbose_epoch, train_fea, train_label, test_fea, test_data, learning_rate,
          weight_decay)


    # print all_X[numeric_feats][0:4]
    # print test.head()
