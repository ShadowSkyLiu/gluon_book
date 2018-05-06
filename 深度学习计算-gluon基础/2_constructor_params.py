# coding=utf-8
import sys
from mxnet.gluon import nn
from mxnet import gluon, nd
from mxnet import init

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

class MyInit(init.Initializer):
    def _init_weight(self, name, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)

    def _init(self):
        super(MyInit, self).__init__()
        self._verbose = True


if __name__ == '__main__':

    my_param = gluon.Parameter("exciting_parameter_yay", shape=(3, 3))
    my_param.initialize()
    print (my_param.data(), my_param.grad())

    x = nd.random.uniform(shape=(3, 5))
    try:
        net = get_net()
        net.initialize(MyInit())
        params = net.collect_params()
        print (net(x))
        print(params['sequential0_dense0_bias'].data())
        print(params.get('dense0_weight').data())

        # w = net[0].weight
        # b = net[0].bias
        # print('name: ', net[0].name)
        # print('weight:', w.data())
        # print('weight gradient', w.grad())
        # print('bias:', b.data())
        # print('bias gradient', b.grad())
        #
        # params = net.collect_params()
        # print(params['sequential0_dense0_bias'].data())
        # print(params.get('dense0_weight').data())
        #
        # params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
        # print(net[0].weight.data(), net[0].bias.data())
        #
        # params.initialize(init=init.One(), force_reinit=True)
        # print(net[0].weight.data(), net[0].bias.data())
    except RuntimeError as err:
        sys.stderr.write(str(err))

