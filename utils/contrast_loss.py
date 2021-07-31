# -*- coding:utf-8 -*-
'''
Some custom loss functions for PyTorch.
'''
import torch
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import torch.autograd as autograd


def mask_type_transfer(mask):
    mask = mask.type(torch.bool)
    # mask = mask.type(torch.uint8)
    return mask


def get_pos_and_neg_mask(bs):
    ''' Org_NTXentLoss_mask '''
    zeros = torch.zeros((bs, bs), dtype=torch.uint8)
    eye = torch.eye(bs, dtype=torch.uint8)
    pos_mask = torch.cat([
        torch.cat([zeros, eye], dim=0), torch.cat([eye, zeros], dim=0),
    ], dim=1)
    neg_mask = (torch.ones(2*bs, 2*bs, dtype=torch.uint8) - torch.eye(
        2*bs, dtype=torch.uint8))
    pos_mask = mask_type_transfer(pos_mask)
    neg_mask = mask_type_transfer(neg_mask)
    return pos_mask, neg_mask


class NTXentLoss(nn.Module):
    """ NTXentLoss

    Args:
        tau: The temperature parameter.
    """

    def __init__(self,
                 bs,
                 tau=1,
                 cos_sim=True,
                 gpu=True,
                 eps=1e-8):
        super(NTXentLoss, self).__init__()
        self.name = 'NTXentLoss_Org'
        self.tau = tau
        self.use_cos_sim = cos_sim
        self.gpu = gpu
        self.eps = eps

        if cos_sim:
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
            self.name += '_CosSim'

        # Get pos and neg mask
        self.pos_mask, self.neg_mask = get_pos_and_neg_mask(bs)

        if gpu:
            self.pos_mask = self.pos_mask.cuda()
            self.neg_mask = self.neg_mask.cuda()
        print(self.name)

    def forward(self, input, target=None):
        '''
        input: {'zi': out_feature_1, 'zj': out_feature_2}
        target: one_hot lbl_prob_mat
        '''
        zi, zj = F.normalize(input['zi'], dim=1), F.normalize(input['zj'], dim=1)
        bs = zi.shape[0]

        z_all = torch.cat([zi, zj], dim=0)  # input1,input2: z_i,z_j
        # [2*bs, 2*bs] -  pairwise similarity
        if self.use_cos_sim:
            sim_mat = self.cosine_similarity(
                z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau  # s_(i,j)
        else:
            sim_mat = torch.mm(z_all, z_all.t().contiguous()) / self.tau  # s_(i,j)

        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).view(2*bs).clone())
        # [2*bs, 2*bs-1]
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).view(2*bs, -1).clone())

        # Compute loss
        loss = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss


def get_contrast_loss(name, **kwargs):
    if name == 'NTXentLoss':
        criterion = NTXentLoss

    return criterion(**kwargs)


def main():

    pass


if __name__ == '__main__':
    main()
