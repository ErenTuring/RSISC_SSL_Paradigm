# -*- coding:utf-8 -*-
'''
Build model according to `net_num`.
'''
from .scene_net import Scene_Base
from .pretext_models import Discriminator_GANs
from torchvision import models


def build_model(num_classes, net_num='12345678', input_band=3, pretrain=False, **kwargs):
    '''
    Args:
        net_num: 6位数,
        第1位代表main model的arch:
        0 - Scene_Base
            第2位代表Backbone 的arch：
            1 - resnet34
            2 - resnet50
            3 - resnet101
            4 - vgg16
            5 - googlenet
    '''
    assert type(net_num) == str
    paras = [int(n) for n in net_num]  # str -> int
    print('\nNet_num:', net_num)

    if paras[0] == 0:  # Scene_Base
        if paras[1] == 9:
            model = Discriminator_GANs(input_band)
        else:
            arch = {
                1: 'resnet34',
                2: 'resnet50',
                3: 'resnet34_small',
                4: 'resnet101',
                5: 'vgg16',
                6: 'googlenet',
                7: 'alexnet',
                8: 'alexnet_small',
            }[paras[1]]
            model = Scene_Base(num_classes, input_band, arch, pretrained=pretrain, **kwargs)

    elif paras[0] == 9:
        assert input_band == 3
        arch = {
            1: 'resnet34',
            2: 'resnet50',
            3: 'resnet101',
            4: 'vgg16',
            5: 'googlenet',
            6: 'alexnet',
            # 8: 'alexnet_small',
        }[paras[1]]
        model = models.__dict__[arch](num_classes=num_classes, pretrained=pretrain)

    if hasattr(model, 'model_name'):
        print(model.model_name)
    return model
