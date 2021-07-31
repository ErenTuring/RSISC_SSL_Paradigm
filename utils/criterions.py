# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


def get_loss(name, cls_weight=None, **kwargs):
    if name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss
    elif name == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss
    elif name == 'MultiLabelSoftMarginLoss':
        criterion = nn.MultiLabelSoftMarginLoss
    elif name == 'MSELoss':
        criterion = nn.MSELoss

    if cls_weight is not None and name in ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MultiLabelSoftMarginLoss']:
        cls_weight = torch.Tensor(cls_weight)
        loss_func = criterion(weight=cls_weight, **kwargs)
    else:
        loss_func = criterion(**kwargs)

    return loss_func


def main():
    pass


if __name__ == '__main__':
    main()
