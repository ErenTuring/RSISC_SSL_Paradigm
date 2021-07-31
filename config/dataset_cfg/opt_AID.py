#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Default Config about AID dataset, classification mode
self-supervised classification
'''
from config.category import AID


class Config(object):
    ''' Fixed initialization Settings '''
    # Path and file
    root = '/home/taopc/Data/Data_Lib'
    dataset_dir = {'aid': root + '/Classifiy/AID',
                   'nr': root + '/Classifiy/NWPU_RESISC45'}
    ckpt = root + '/JiQi/Model/AID'
    env = 'AID'  # visdom 环境
    ds = env

    train_val_ratio = [0.8, 0.2]
    train_scale = 1  # Scale of training set reduction
    val_scale = 0.5
    # Model related arguments
    sepoch = 1  # use to continue from a checkpoint
    ptcp = 1  # use pre-train checkpoint

    # Optimiztion related arguments
    bs = 16  # batch size
    mepoch = None  # 16-[256,256] dataset only need 8~9 epoch
    ckpt_freq = 0
    lr = 4e-4  # initial learning rate
    lr_decay = [0.98, 0.6]  # pre epoch
    lr_policy = 'step'  # pre epoch
    warmup = 0  # if warmup > 0, use warmup strategy and end at warmup
    hos = 0  # Hierarchical learning rate strategies
    weight_decay = 1e-5  # L2 loss
    optimizer = ['adam', 'sgd', 'lars'][1]
    loss = ['CrossEntropyLoss', 'NTXentloss'][0]
    loss_weight = None

    # Data related arguments
    workers = 4  # number of data loading workers
    dtype = ['RGB'][0]
    bl_dtype = [''][0]
    band_num = 0

    input_size = (224, 224)  # final input size of network(random-crop use this)
    # crop_params = [256, 256, 256]  # crop_params for val and pre

    # feature_dim = {1: 128, 2: 256, 3: 384, 4: 512}
    mean = [0.5, 0.5, 0.5]  # BGR, 此处的均值应该是0-1
    std = [0.5, 0.5, 0.5]  # [0.12283102, 0.1269429, 0.15580289]

    # Misc arguments
    print_freq = 5  # print (print_freq) times per epoch

    category = AID()
    classTabel = category.table
    num_classes = len(classTabel)
