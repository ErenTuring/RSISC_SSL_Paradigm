# -*- coding: utf-8 -*-
'''
Model for scene classifcation.
'''
import torch
import torch.nn as nn
# import torch.nn.functional as F
from .BasicModule import BasicModule
from torchvision.models import resnet, vgg, googlenet, alexnet
import collections

feature_dim = {1: 128, 2: 256, 3: 384, 4: 512}


class Scene_Base(BasicModule):
    '''Main model: Scene classification Baseline network.
    '''

    def __init__(self, num_classes=2, in_channels=3, backbone='resnet50', out_dim=256, pretrained=False, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'Scene_Base_' + backbone
        self.out_dim = out_dim

        if backbone == 'resnet34':
            bb = resnet.resnet34(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(list(bb.named_children())))
            self.feature = bb[:8]
            self.out_dim = 512
            if in_channels != 3:
                self.feature.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'resnet50':
            bb = resnet.resnet50(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(list(bb.named_children())))
            self.feature = bb[:8]
            self.out_dim = 2048
            if in_channels != 3:
                self.feature.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'resnet34_small':
            # replace the first 7x7 Conv of stride 2 with 3x3 Conv of stride 1,
            # and also remove the first max pooling operation
            bb = []
            for name, module in resnet.resnet34(pretrained=pretrained).named_children():
                if name == 'conv1':
                    module = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    if not isinstance(module, nn.AdaptiveAvgPool2d):
                        bb.append(module)
            self.feature = nn.Sequential(*bb)
            self.out_dim = 512
        elif backbone == 'resnet101':
            bb = resnet.resnet101(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(list(bb.named_children())))
            self.feature = bb[:8]
            self.out_dim = 2048
            if in_channels != 3:
                self.feature.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif backbone == 'vgg16':
            self.feature = vgg.vgg16_bn(pretrained=pretrained).features
            self.out_dim = 512
            if in_channels != 3:
                self.feature[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        elif backbone == 'googlenet':
            bb = googlenet(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(
                list(bb.named_children())[:]
            ))
            self.feature = bb[:16]
            self.out_dim = 1024
            if in_channels != 3:
                self.feature.conv1.conv = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'alexnet':
            self.feature = alexnet(pretrained=pretrained).features
            self.out_dim = 256
            if in_channels != 3:
                self.feature[0] = nn.Conv2d(
                    in_channels, 64, kernel_size=11, stride=4, padding=2)
        elif backbone == 'alexnet_small':
            # bb = []
            # for name, module in alexnet(pretrained=pretrained).features.named_children():
            #     if not isinstance(module, nn.MaxPool2d):
            #         bb.append(module)
            # self.feature = nn.Sequential(*bb)
            self.feature = alexnet(pretrained=pretrained).features
            # replace the first 11x11 Conv of stride 2 with 3x3 Conv of stride 1,
            # and also remove the first max pooling operation
            self.feature[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.out_dim = 256

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(self.out_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(.0),
            nn.Linear(self.out_dim, num_classes)
        )

    def forward(self, x):
        # 1. Get feature - [N,C,?,?]
        x = self.feature(x)

        # 2. Globel avg pooling to get feature of per image
        feature = self.gap(x)  # [N,C,1,1]

        # 3. Get logits
        x = torch.flatten(feature, 1)  # [N,C,1,1] -> [N,C]
        logits = self.classifier(x)

        return logits


class Scene_Base_muilt(BasicModule):
    '''Main model: Scene classification Baseline network.
    It can also be the encoder for Sence_Seg.
    '''

    def __init__(self, num_classes=2, in_channels=3, backbone='resnet50', out_dim=256, pretrained=False, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'Scene_Base_' + backbone
        self.out_dim = out_dim

        if backbone == 'resnet34':
            bb = resnet.resnet34(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(list(bb.named_children())))
            self.feature = bb[:8]
            self.out_dim = 512
            if in_channels != 3:
                self.feature.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'resnet50':
            bb = resnet.resnet50(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(list(bb.named_children())))
            self.feature = bb[:8]
            self.out_dim = 2048
            if in_channels != 3:
                self.feature.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'resnet34_small':
            # replace the first 7x7 Conv of stride 2 with 3x3 Conv of stride 1,
            # and also remove the first max pooling operation
            bb = []
            for name, module in resnet.resnet34(pretrained=pretrained).named_children():
                if name == 'conv1':
                    module = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    if not isinstance(module, nn.AdaptiveAvgPool2d):
                        bb.append(module)
            self.feature = nn.Sequential(*bb)
            self.out_dim = 512
        elif backbone == 'Resnet101':
            bb = resnet.resnet101(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(list(bb.named_children())))
            self.feature = bb[:8]
            self.out_dim = 2048
            if in_channels != 3:
                self.feature.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif backbone == 'vgg16':
            self.feature = vgg.vgg16_bn(pretrained=pretrained).features
            self.out_dim = 512
            if in_channels != 3:
                self.feature[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        elif backbone == 'googlenet':
            bb = googlenet(pretrained=pretrained)
            bb = nn.Sequential(collections.OrderedDict(
                list(bb.named_children())[:]
            ))
            self.feature = bb[:16]
            self.out_dim = 1024
            if in_channels != 3:
                self.feature.conv1.conv = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'alexnet':
            self.feature = alexnet(pretrained=pretrained).features
            self.out_dim = 256
            if in_channels != 3:
                self.feature[0] = nn.Conv2d(
                    in_channels, 64, kernel_size=11, stride=4, padding=2)
        elif backbone == 'alexnet_small':
            # bb = []
            # for name, module in alexnet(pretrained=pretrained).features.named_children():
            #     if not isinstance(module, nn.MaxPool2d):
            #         bb.append(module)
            # self.feature = nn.Sequential(*bb)
            self.feature = alexnet(pretrained=pretrained).features
            # replace the first 11x11 Conv of stride 2 with 3x3 Conv of stride 1,
            # and also remove the first max pooling operation
            self.feature[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.out_dim = 256

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(self.out_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(.0, inplace=True),
            nn.Linear(self.out_dim, num_classes)
        )

    def forward(self, x):
        # 1. Get feature - [N,C,?,?]
        x = self.feature(x)

        # 2. Globel avg pooling to get feature of per image
        feature = self.gap(x)  # [N,C,1,1]

        # 3. Get logits
        x = torch.flatten(feature, 1)  # [N,C,1,1] -> [N,C]
        logits = self.classifier(x)

        return logits
