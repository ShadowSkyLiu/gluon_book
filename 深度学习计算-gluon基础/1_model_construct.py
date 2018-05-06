from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)

    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))


def _indent(s_, numSpaces):
    """Indent string
    """
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [first] + [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    return s

class Sequential1(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential1, self).__init__(**kwargs)

    def add(self, block):
        self._children.append(block)

    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in enumerate(self._children)
                            if isinstance(block, nn.Block)])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)

if __name__ == '__main__':
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"),
                nn.Dense(10))
        # net.add(nn.Dense(10))
    net.initialize()
    print net
    #
    # x = nd.random.uniform(shape=(4, 20))
    # y = net(x)
    # print net
    # print net[1].weight.data()
    #
    # net2 = MLP()
    # print (net2)
    # net2.initialize()
    # x = nd.random.uniform(shape=(4, 20))
    # y = net2(x)
    # print (net2)
    # print net2.dense0.weight.data()
    # print y

    net3 = Sequential1()
    with net3.name_scope():
        net3.add(nn.Dense(256, activation="relu"))
        net3.add(nn.Dense(10))
    net3.initialize()
    print net3
    x = nd.random.uniform(shape=(4, 20))
    y = net3(x)
    print y


