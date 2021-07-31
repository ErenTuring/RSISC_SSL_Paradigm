#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Default Config about EuroSAT dataset, classification mode
URL: https://github.com/phelber/eurosat
'''
from config.category import EuroSAT


class Config(object):
    ''' Fixed initialization Settings '''
    # Path and file
    root = '/home/taopc/Data/Data_Lib'
    dataset_dir = root + '/Classifiy/EuroSATRGB'
    ckpt = root + '/JiQi/Model/EuroSAT'
    env = 'Euro'  # visdom 环境
    ds = env

    train_val_ratio = [0.8, 0.2]  # train & val set is fixed
    train_scale = 1  # Scale of training set reduction
    val_scale = 0.5
    # Model related arguments
    sepoch = 1  # use to continue from a checkpoint
    ptcp = 1  # use pre-train checkpoint

    # Optimiztion related arguments
    bs = 64  # batch size
    mepoch = None  # 16-[256,256] dataset only need 8~9 epoch
    ckpt_freq = 0
    lr = 1e-3  # initial learning rate
    lr_decay = [0.98, 0.6]  # pre epoch
    lr_policy = 'step'  # pre epoch
    warmup = 0  # if warmup > 0, use warmup strategy and end at warmup
    weight_decay = 1e-5  # L2 loss
    optimizer = ['adam', 'sgd', 'lars'][0]
    loss = ['CrossEntropyLoss', ][0]
    loss_weight = None

    # Data related arguments
    workers = 4  # number of data loading workers
    dtype = ['RGB']
    bl_dtype = [''][0]
    band_num = 0

    input_size = (64, 64)  # final input size of network(random-crop use this)
    # crop_params = [256, 256, 256]  # crop_params for val and pre

    # feature_dim = {1: 128, 2: 256, 3: 384, 4: 512}
    mean = [0.5]
    std = [0.5]

    # Misc arguments
    print_freq = 10  # print (print_freq) times per epoch

    category = EuroSAT()
    classTabel = category.table
    num_classes = len(classTabel)
