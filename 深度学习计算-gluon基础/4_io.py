import numpy
from mxnet import nd

from mxnet.gluon import nn

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation="relu"))
        net.add(nn.Dense(2))
    return net

net = get_net()
net.initialize()
x = nd.random.uniform(shape=(2, 10))
print(net(x))

if __name__ == '__main__':
    x = nd.ones(3)
    y = nd.zeros(4)
    filename = "data/test1.params"
    numpy.savez(filename, [x, y])
    # nd.save(filename, [x, y])

    a, b = nd.load(filename)
    print(a, b)

    mydict = {"x": x, "y": y}
    filename = "data/test2.params"
    nd.save(filename, mydict)

    c = nd.load("data/test2.params")
    print c

    filename = "data/mlp.params"
    net.save_params(filename)
