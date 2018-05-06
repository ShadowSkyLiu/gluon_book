from mxnet import nd, gluon
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

class MyDense(nn.Block):
    def forward(self, x):
        pass

    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))


if __name__ == '__main__':
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128))
        net.add(nn.Dense(128))
        net.add(CenteredLayer())

    net.initialize()
    x0 = nd.random.uniform(shape=(4, 8))
    y = net(x0)
    print y.mean()

    dense = MyDense(5, in_units=10, prefix='o_my_dense_')
    print dense.params
