import matplotlib.pyplot as plt
from mxnet import image
from mxnet import nd
import sys
sys.path.append('..')
import utils

def apply(img, aug, n=3):
    # 转成float，一是因为aug需要float类型数据来方便做变化。
    # 二是这里会有一次copy操作，因为有些aug直接通过改写输入
    #（而不是新建输出）获取性能的提升
    X = [aug(img.astype('float32')) for _ in range(n*n)]
    # 有些aug不保证输入是合法值，所以做一次clip
    # 显示浮点图片时imshow要求输入在[0,1]之间
    Y = nd.stack(*X).clip(0,255)/255
    utils.show_images(Y, n, n, figsize=(16,16))

img = image.imread('../example1.jpg')
print(img)
plt.imshow(img.asnumpy())
# 以.5的概率做翻转
aug = image.HorizontalFlipAug(.5)
apply(img, aug)
