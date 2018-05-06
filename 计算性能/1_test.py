from time import time

from mxnet import nd
from mxnet.gluon import nn


def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))


def add_str():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''

prog = evoke_str()
print(prog)
y = compile(prog, '', 'exec')
exec(y)


def get_net():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.Dense(256, activation="relu"),
            nn.Dense(128, activation="relu"),
            nn.Dense(2)
        )
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
# net = get_net()
# print(net(x))
#
# net.hybridize()
# print(net(x))

def benchmark(net, x):
    start = time()
    for i in range(1000):
        y = net(x)
    # 等待所有计算完成。
    nd.waitall()
    return time() - start

net = get_net()
print('Before hybridizing: %.4f sec' % (benchmark(net, x)))
net.hybridize()
print('After hybridizing: %.4f sec' % (benchmark(net, x)))