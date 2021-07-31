# -*- coding: utf-8 -*-
import time
# from collections import OrderedDict

import torch
from torch import nn
# from torch.nn import functional as F
# from torchvision import models

# import math


class BasicModule(nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法

    deacy()  return l2_sp regularization loss
    """

    def __init__(self, wd_mode=0, wd_rate_a=0.1, wd_rate_b=0.01):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字
        self.wd_mode = wd_mode
        self.wd_rate_alpha = wd_rate_a
        self.wd_rate_beta = wd_rate_b
        self.sp_dict = {}  # Start point weights dict

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def backup_weights(self, block_layers=[]):
        block_layers = [block_layers] if type(block_layers) is str else block_layers
        model_dict = self.state_dict()  # Fix backup weights
        self.sp_dict = {}
        if len(block_layers) > 0:
            for k, v in model_dict.items():
                if any(bl_l in k for bl_l in block_layers):
                    continue
                self.sp_dict[k] = v.detach()
        else:
            for k, v in model_dict.items():
                self.sp_dict[k] = v.detach()

    def decay(self):
        existing_l2_reg, new_l2_reg = 0.0, 0.0
        no_l2_reg_paras = ['bn', ]
        if self.wd_mode == 1:
            # L2-SP regularization loss
            for name, w in self.named_parameters():
                # Basic fliter
                if 'weight' not in name:
                    continue
                if any(k in name for k in no_l2_reg_paras):
                    continue

                # Compute regularization loss
                if name in self.sp_dict:
                    w0 = self.sp_dict[name].data
                    existing_l2_reg += torch.pow(w-w0, 2).sum()
                else:
                    new_l2_reg += torch.pow(w, 2).sum()

        return existing_l2_reg * self.wd_rate_alpha + new_l2_reg * self.wd_rate_beta


class FeatureExtractor(nn.Module):
    def __init__(self, net, encoder_name='features', max_bs=64):
        super().__init__()
        # self.model_name = net.model_name

        self.features = getattr(net, encoder_name)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = net.out_dim
        self.max_bs = max_bs

    def forward(self, x):
        if len(x.shape) == 5:
            N, mn, c, h, w = x.shape[:]  # [N, m*n, 3, M/m, M/n]
            Fv_flat = x = x.view(N*mn, c, h, w).contiguous()
            fv = []
            for i in range(N*mn // self.max_bs + 1):
                batch_x = x[i*self.max_bs: (i+1)*self.max_bs, :]
                if batch_x.shape[0] == 0:
                    break  # empty tensor
                batch_fv = self.features(batch_x)
                batch_fv = self.gap(batch_fv).flatten(1)  # [max_bs, C', 1, 1] -> [max_bs, C']
                fv.append(batch_fv)
            Fv_flat = torch.cat(fv, dim=0)  # [N*mn, c]
            feature = Fv_flat.view(N, mn, mn, -1).permute(0, 3, 1, 2).contiguous()
            # [N*mn, C] -> [N, m, n, c] -> [N, c, m, n]
        elif len(x.shape) == 4:
            # N, c, h, w = x.shape[:]  # [N, 3, M/m, M/n]
            x = self.features(x)
            feature = self.gap(x).flatten(1)  # [N,c,1,1] -> [N,c]

        return feature


def set_parameter_requires_grad(model, feature_extracting):
    ''' If we are feature extracting and only want to compute gradients
    for the newly initialized layer then we want all of the other parameters
    to not require gradients.  '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
