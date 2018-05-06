from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

from introduction import showbox


def detection(imagepath):
    net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
    x, img = data.transforms.presets.ssd.load_test(imagepath, short=512)
    class_IDs, scores, bounding_boxs = net(x)
    # CLASSES = (u'飞机', u'自行车', u'小鸟', u'船', u'瓶子', u'巴士', u'汽车',
    #            u'锚', u'椅子', u'牛', u'餐桌', u'狗', u'马', u'摩托车',
    #            u'人', u'盆栽植物', u'羊', u'沙发', u'列车', u'电视监视器')
    CLASSES = ('1', '2', '3', '4', '5', '6', '7',
               '8', '9', '10', '11', '12', '13', '14',
               '15', '16', '17', '18', '19', '20')
    ax = showbox.plot_bbox(img, bounding_boxs[0], scores[0],
                           class_IDs[0], class_names=CLASSES)
    plt.savefig("detection-result/" + imagepath)
    return "detection-result/" + imagepath

