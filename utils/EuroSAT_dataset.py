#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script reads GeoTIFF files each of which is for one (or multiple) spectral
band of Sentinel-2 image patch in the EuroSATallBands Archive.

Bands and pixel resolution in meters:
B01: Coastal aerosol; 60m
B02: Blue; 10m
B03: Green; 10m
B04: Red; 10m
B05: Vegetation red edge; 20m
B06: Vegetation red edge; 20m
B07: Vegetation red edge; 20m
B08: NIR; 10m
B09: Water vapor; 60m
B11: SWIR; 20m
B12: SWIR; 20m
B8A: Narrow NIR; 20m

read_patch()  # Read a sample patch as ndarry
'''
import os
import time
import gdal  # , gdalconst
import numpy as np
import torch
import itertools
# from copy import deepcopy
from torch.utils import data
from torchvision.transforms import Compose
from . import transform as T
from .transform import get_cls_transform as get_transform

BAND_mean = [1353.036, 1116.468, 1041.475, 945.344, 1198.498, 2004.878,
             2376.699, 2303.738, 732.957, 12.092, 1818.820, 1116.271, 2602.579]
BAND_std = [65.479, 154.008, 187.997, 278.508, 228.122, 356.598, 456.035,
            531.570, 98.947, 1.188, 378.993, 303.851, 503.181]

RGB_bands = [4, 3, 2]
MS_bands = [i+1 for i in range(13)]
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
ROOT = '/project/zhangguo/Data/EuroSATallBands'


# **************************************************
# *************** BigEarthNet Basic ****************
# **************************************************
def _scale_image(img, bands):
    """ Scale image and return uint8 array with values range from 0-255. """
    img = img.astype(np.float)
    # Normalize
    for i, band in enumerate(bands):
        img[:, :, i] -= BAND_mean[band-1]
        img[:, :, i] /= BAND_std[band-1]

    # [-2, 3] mapping to [0, 255]
    img = (img + 2) * 51
    return np.clip(img, 0, 255).astype(np.uint8)


def read_patch(path, bands=RGB_bands, scale=False):
    ''' Read a sample patch, and return ndarry. '''
    # Checks the existence of patch folders and populate the list of patch folder paths
    if not os.path.exists(path):
        print('ERROR: patch "%s" is not existence! Please Check' % path)
        exit()

    # Reads spectral bands of patch whose folder name is populated before
    band_ds = gdal.Open(path, gdal.GA_ReadOnly)

    if len(bands) == 13:
        patch_data = band_ds.ReadAsArray()
        patch_data = patch_data.transpose((1, 2, 0))
    else:
        patch_data = []
        for band_ind in bands:
            raster_band = band_ds.GetRasterBand(band_ind)
            array = raster_band.ReadAsArray()
            patch_data.append(array)
        patch_data = np.stack(patch_data, axis=2)

    if scale:
        patch_data = _scale_image(patch_data, bands)

    return patch_data


# **************************************************
# **************** Dataset utils *******************
# **************************************************
class EuroSATData(data.Dataset):
    '''
    Dataset loader for EuroSAT.
    Args:
        init_data - Dict of init data = {'train': [(img_path, lbl), ..], 'val': [...]}
        split - One of ['train', 'val', 'test']
        ratio - (float) extra parameter to conctrl the num of dataset
            if ratio < 0.01, then stand for nshort mode:
                ratio = 0.00n, where n is the num of sample per category
        transform - 可以传入自定义的transform对象
    '''

    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False,
                 org_img=False):
        self.split = split
        self.ratio = ratio
        self.org_img = org_img
        self.dtype = opt.dtype[0]  # ['RGB'] or ['MS']

        self.band_list = {'RGB': RGB_bands, 'MS': MS_bands}[self.dtype]
        self.imread_func = self.imread_with_gdal
        # Collect all dataset files, divide into dict.
        tic = time.time()
        self.imgs, self.lbls = [], []
        total_num = 0
        sample_statis = ''

        for (cls, samples) in init_data[split].items():
            total_num += len(samples)
            if ratio <= 1:
                use_num = max(round(len(samples)*ratio), 1)
            else:
                use_num = min(int(ratio), len(samples))
            sample_statis += '| %s | %d |_' % (cls, use_num)
            if use_num == 0:
                continue
            for (pth, lbl) in samples[:use_num]:
                self.imgs.append(pth)
                self.lbls.append(lbl)
        self.load_data = self.load_data_from_disk
        if pin_memory:
            self.pin_imgs = [self.imread_func(pth) for pth in self.imgs]
            # self.pin_lbls = self.lbls
            self.load_data = self.load_data_from_memory

        print('%s set contains %d images, %s spectrals, a total of %d categories.' % (
              split, total_num, self.dtype, opt.num_classes))
        print('Actual number of samples used = %d, time to collect data = %.2fs.' % (
            len(self.imgs), time.time()-tic))
        if split == 'train':
            print(sample_statis)

        # Transform
        if type(transform) is str:
            self.transforms = get_transform(opt, split, transform, dtype=self.dtype)
        else:
            self.transforms = transform
        # self.basic_transforms = get_transform(opt, 'test')

    def imread_with_gdal(self, pth):
        return read_patch(pth, self.band_list, scale=True)

    def load_data_from_disk(self, index):
        img = self.imread_func(self.imgs[index])
        lbl = np.array(self.lbls[index], dtype=np.int64)
        return img, lbl

    def load_data_from_memory(self, index):
        return self.pin_imgs[index], np.array(self.lbls[index], dtype=np.int64)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data(index)
        sample = {'image': img, 'label': lbl, 'name': self.imgs[index]}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class EuroSATDataPair(EuroSATData):
    ''' Dataset loader for BigEarthNet classification in pair mode.

    Sing brand augmentation & Same localtion for two branchs.

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self, init_data, opt, split='train',
                 ratio=1, transform=None, pin_memory=False):
        super(EuroSATDataPair, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        self.transforms_a = get_transform(opt, 'test')
        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_b = get_transform(opt, 'train', opt.aug_p, tap, dtype=self.dtype)
        else:
            self.transforms_b = transform

    def __getitem__(self, index):
        img, lbl = self.load_data(index)

        img_1 = self.transforms_a({'image': img.copy()})['image']
        img_2 = self.transforms_b({'image': img.copy()})['image']

        return {'image': (img_1, img_2), 'label': lbl, 'name': self.imgs[index]}


class EuroSATDataPair3(EuroSATData):
    ''' Dataset loader for BigEarthNet classification in pair mode.

    Sing brand augmentation & Random localtion(for different branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self, init_data, opt, split='train',
                 ratio=1, transform=None, pin_memory=False):
        super(EuroSATDataPair3, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        self.transforms_a = Compose([
            T.RandomScaleAspctCrop(opt.input_size, scale=(0.5, 0.9), p=0.8, mode='cls'),
            T.ToTensor(), T.Normalizer(opt.mean, opt.std)])
        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_b = get_transform(opt, 'train', opt.aug_p, tap, dtype=self.dtype)
            # self.transforms_b = get_transform(opt, 'train', 'B', [.8, .5, .5, .5, .2, .3])
        else:
            self.transforms_b = transform

    def __getitem__(self, index):
        img, lbl = self.load_data(index)

        if self.transforms_a is not None:
            img_1 = self.transforms_a({'image': img.copy()})['image']
        if self.transforms_b is not None:
            img_2 = self.transforms_b({'image': img.copy()})['image']

        return {'image': (img_1, img_2), 'label': lbl, 'name': self.imgs[index]}


class EuroSATDataPair5(EuroSATData):
    ''' Dataset loader for scene classification in pair mode.

    Two brand augmentation & Random localtion(for two branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self, init_data, opt, split='train',
                 ratio=1, transform=None, pin_memory=False):
        super(EuroSATDataPair5, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_a = get_transform(opt, 'train', 'C', tap, dtype=self.dtype)
            self.transforms_b = get_transform(opt, 'train', 'C', tap, dtype=self.dtype)
        else:
            self.transforms_a = transform
            self.transforms_b = transform

        self.basic_transforms = get_transform(opt, 'test')

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data(index)
        # img_0, img_a, img_b = img.copy(), img.copy(), img.copy()
        # img_0 = self.basic_transforms({'image': img_0})['image']
        img_a, img_b = img.copy(), img.copy()

        if self.transforms_a is not None:
            img_a = self.transforms_a({'image': img_a})['image']
        if self.transforms_b is not None:
            img_b = self.transforms_b({'image': img_b})['image']

        # return {'image': (img_a, img_b), 'image_': img_0, 'org_img': img, 'label': lbl, 'name': self.imgs[index]}
        return {'image': (img_a, img_b), 'label': lbl, 'name': self.imgs[index]}


class EuroSATDataJigsaw(EuroSATData):
    ''' Dataset loader for scene classification in pair mode.

    Two brand augmentation & Random localtion(for two branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self, init_data, opt, split='train',
                 ratio=1, transform=None, pin_memory=False,
                 puzzle=2):
        super(EuroSATDataJigsaw, self).__init__(init_data, opt, split, ratio, None, pin_memory)
        self.puzzle = puzzle
        if puzzle == 2:
            self.cr = [0, 75]
            self.cw = [0, 75]
            resize = (155, 155)
            crop_size = (150, 150)
            self.permutations = np.array(list(itertools.permutations(list(range(4)), 4)))
        elif puzzle == 3:
            self.cr = [0, 75, 150]
            self.cw = [0, 75, 150]
            resize = (230, 230)
            crop_size = (225, 225)
            print('permutations_hamming_max_%d.npy' % opt.num_classes)
            self.permutations = np.load('data/permutations_hamming_max_%d.npy' % (opt.num_classes))
        tap = [.8, .5, .8, .0, .2, .5]
        self.transforms_img = Compose([
            T.Resize(resize, mode='cls'),
            T.CenterCrop(crop_size, mode='cls'),
            T.RandomRotateAndFlip(p=tap[1], mode='cls'),
            T.ColorAugment(hue=0.1, sat=0.4, brightness=0.4, contrast=0.4, p=tap[2], dtype=self.dtype),
            T.RandomGrayscale(p=tap[4]),
        ])
        self.transforms_patch = Compose([
            T.RandomCrop((64, 64), mode='cls'),
            T.ToTensor(),
            T.Normalizer(opt.mean, opt.std)
        ])

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, _ = self.load_data(index)
        lbl = np.random.randint(len(self.permutations))

        # Overall augmentation
        img_ = self.transforms_img({'image': img.copy()})['image']

        # Make Puzzles
        patchs = [None]*(self.puzzle**2)
        cnt = 0
        for cr in self.cr:
            for cw in self.cw:
                patch = img_[cr: cr+75, cw: cw+75, :].copy()
                patch = self.transforms_patch({'image': patch})['image']
                patchs[cnt] = patch
                cnt += 1

        # Shufflt
        order = self.permutations[lbl]
        patchs = [patchs[t] for t in order]
        data = torch.stack(patchs, 0)

        return {'image': data, 'image_': patchs, 'org_img': img,
                'label': np.array(lbl, dtype=np.int64), 'name': self.imgs[index]}


def test():
    from . import datasets_init
    from config.chaosuan.opt_EuroSAT_MS import Config
    # from config import get_opt
    # OPTS = get_opt('euroms')
    OPTS = Config()

    # Stats the pixel value of data after scale
    init_data = datasets_init('euroms', OPTS, full_train=True)
    mdataset = EuroSATData(init_data, OPTS, 'train')
    stats = np.zeros((3, 13))  # row: (min, mean, max); col: all 13 bands

    for img_pth in mdataset.imgs:
        # print(img.shape)
        img = mdataset.imread_func(img_pth)
        minValue = img.min(axis=0).min(axis=0)
        stats[0, :] += minValue
        # stats[0, :] = np.where(stats[0, :] > minValue, minValue, stats[0, :])

        meanValue = img.mean(axis=0).mean(axis=0)
        stats[1, :] += meanValue

        maxValue = img.max(axis=0).max(axis=0)
        stats[2, :] += maxValue
        # stats[2, :] = np.where(stats[2, :] < maxValue, maxValue, stats[2, :])
    stats /= mdataset.__len__()
    print(stats)
    pass


if __name__ == "__main__":
    test()
    pass
