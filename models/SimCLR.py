# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .BasicModule import BasicModule


class SimCLR_cls(BasicModule):
    def __init__(self, net, feature_dim=128):
        super().__init__()
        self.model_name = 'SimCLR_cls_' + net.model_name

        # encoder
        self.feature = net.feature  # .copy()
        self.classifier = net.classifier

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = net.out_dim
        # projection head
        self.g = nn.Sequential(
            nn.Linear(self.out_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Linear(512, feature_dim, bias=True)
        )  # 论文中g的 input dim = output dim

    def feature_extra(self, x):
        x = self.feature(x)
        feature = self.gap(x)  # [N,C,1,1]

        x = torch.flatten(feature, start_dim=1)  # [NC]
        return x

    def forward(self, x):
        x = self.feature(x)
        feature = self.gap(x)  # [N,C,1,1]

        x = torch.flatten(feature, start_dim=1)  # [NC]
        out = self.g(x)
        return x, out
