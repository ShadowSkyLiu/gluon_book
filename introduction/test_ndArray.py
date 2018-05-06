from mxnet import ndarray as nd

if __name__ == '__main__':

    a = nd.zeros((3, 4))
    print (a)
    b = nd.ones((1, 2))
    print (b)
    c = nd.array([[1, 2], [2, 3]])
    print (c)

    # a = nd.arange(3).reshape((3, 1))
    # print a
    # b = nd.arange(3).reshape((1, 1, 3, 1))
    # print b
    # c = nd.arange(3).reshape((3, 1, 1, 1))
    # print c
    # d = nd.arange(3).reshape((1, 3, 1, 1))
    # print d
    # e = nd.arange(3).reshape((1, 3))
    # print e
