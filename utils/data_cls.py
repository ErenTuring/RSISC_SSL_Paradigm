# -*- coding: utf-8 -*-
'''
Dataset for Scene classification.
'''
import time
import h5py
import numpy as np
import cv2
import torch
import itertools
from torch.utils import data
from torchvision.transforms import Compose
from .transform import get_cls_transform as get_transform
from . import transform as T
from osgeo import gdal


def raster2array(img_path, geoInfo=False, bands=None):
    """ Load raster image from `img_path` and transform to ndarray.
    Returns:
        if geoInfoarray is Ture: array, geotransform, geoprojection
        if geoInfoarray is Flase: array
    Note: 会将整个数据影像全部读入内存，需要注意图像大小与可用RAM
    """
    # raster = gdal.Open(img_path, options=OPEN_OPTS)
    raster = gdal.Open(img_path)
    # Get Geo-Information
    band_num = raster.RasterCount
    geotransform = list(raster.GetGeoTransform())
    geoprojection = raster.GetProjection()

    # Get array
    if band_num == 1:
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
    elif bands is None:
        array = raster.ReadAsArray()
        array = array.transpose((1, 2, 0))
    else:
        arrays = []
        for i in range(raster.RasterCount):
            if i+1 in bands:
                band = raster.GetRasterBand(i+1)
                arrays.append(band.ReadAsArray())
        array = np.stack(arrays, axis=2)

    raster = None
    if geoInfo:
        return array, geotransform, geoprojection
    else:
        return array


class ClassfyData(data.Dataset):
    '''
    Dataset loader for scene classification.
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
        self.insize = opt.input_size
        self.org_img = org_img

        if len(opt.dtype) == 1 and opt.dtype[0] == 'RGB':
            self.dtype = 'RGB'
            self.imread_func = self.imread_with_cv
        else:
            self.dtype = 'other'
            self.imread_func = self.imread_with_gdal

        # Collect all dataset files, divide into dict.
        tic = time.time()
        self.imgs, self.lbls = [], []
        total_num = 0
        sample_statis = ''
        if type(init_data) is dict:
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
        elif type(init_data) is str:
            self.imread_func = self.imread_from_h5
            self.h5File = h5py.File(init_data, mode='r', libver='latest', swmr=True)
            self.img_path = split

            total_num = self.h5File[self.img_path].shape[0]
            if ratio <= 1:
                use_num = max(round(total_num*ratio), 1)
            else:
                use_num = min(int(ratio), total_num)

            self.imgs = [(self.img_path, i) for i in range(use_num)]
            self.lbls = np.zeros((len(self.imgs)), dtype=np.int64)

        self.load_data = self.load_data_from_disk
        if pin_memory:
            if type(init_data) is str:
                self.pin_imgs = self.h5File[self.img_path][:use_num, :]
            else:
                self.pin_imgs = [self.imread_func(pth) for pth in self.imgs]
            # self.pin_lbls = self.lbls
            self.load_data = self.load_data_from_memory

        print('%s set contains %d images, a total of %d categories.' % (
              split, total_num, opt.num_classes))
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

    def imread_with_cv(self, pth):
        ''' Only load RGB images data. '''
        img = cv2.imread(pth, 1)
        img = img[:, :, ::-1]  # BGR→RGB
        return img.copy()

    def imread_with_gdal(self, pth):
        img = raster2array(pth)
        if len(img) == 2:
            img = img[:, None]
        elif img.shape[2] >= 3:
            img[:, :, :3] = img[:, :, :3][:, :, ::-1]
        return img.copy()

    def imread_from_h5(self, pth):
        h5key, ind = pth
        img = self.h5File[h5key][ind, :]  # RGB
        return img

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
        if self.org_img:
            if img.shape[2] == 2 or img.shape[2] > 4:
                new_image = np.zeros((self.insize[0], self.insize[1], img.shape[2]), dtype=img.dtype)
                for c in range(0, img.shape[2]):
                    new_image[:, :, c] = cv2.resize(img[:, :, c], self.insize[::-1],
                                                    interpolation=cv2.INTER_LINEAR)
                sample['org_img'] = new_image
            else:
                sample['org_img'] = cv2.resize(img, self.insize[::-1], interpolation=cv2.INTER_LINEAR)
        return sample


class ClassfyDataPair(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.

    Sing brand augmentation & Same localtion for two branchs.

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False):
        super(ClassfyDataPair, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        # RandomCrop for both branch
        self.transforms_0 = Compose([T.RandomScaleAspctCrop(opt.input_size, (0.5, 0.9), p=0.8, mode='cls')])

        self.transforms_a = get_transform(opt, 'test')
        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_b = get_transform(opt, 'train', transform, tap, dtype=self.dtype)
            # self.transforms_b = get_transform_fixloc(opt, 'train', 'A')
        else:
            self.transforms_b = transform

        self.basic_transforms = get_transform(opt, 'test')

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data(index)
        # img_0, img_1 = img.copy(), img.copy()
        # img_0 = self.basic_transforms({'image': img_0})['image']
        img_a, img_b = img.copy(), img.copy()

        if self.transforms_a is not None:
            img_a = self.transforms_a({'image': img_a})['image']
        if self.transforms_b is not None:
            img_b = self.transforms_b({'image': img_b})['image']

        # return {'image': (img_1, img_2), 'image_': img_0, 'org_img': img, 'label': lbl, 'name': self.imgs[index]}
        return {'image': (img_a, img_b), 'label': lbl, 'name': self.imgs[index]}


class ClassfyDataPair2(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.

    Sing brand augmentation & One branch resize and the other RandomCrop

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False):
        super(ClassfyDataPair2, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        self.transforms_a = get_transform(opt, 'test')
        if type(transform) is str:
            tap = [.8, .5, .5, .5, .2, .3]
            self.transforms_b = get_transform(opt, 'train', transform, tap, dtype=self.dtype)
        else:
            self.transforms_b = transform

        self.basic_transforms = get_transform(opt, 'test')

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data(index)
        img_0, img_1, img_2 = img.copy(), img.copy(), img.copy()
        img_0 = self.basic_transforms({'image': img_0})['image']

        img_1 = self.transforms_a({'image': img_1})['image']
        img_2 = self.transforms_b({'image': img_2})['image']

        return {'image': (img_1, img_2), 'image_': img_0, 'org_img': img, 'label': lbl, 'name': self.imgs[index]}


class ClassfyDataPair3(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.

    Sing brand augmentation & Random localtion(for different branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False):
        super(ClassfyDataPair3, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        resize_size = (opt.input_size[0] + 30, opt.input_size[0] + 30)
        self.transforms_a = Compose([
            T.Resize(resize_size, mode='cls'),  # Make AID, NR and Euro can combine together
            T.RandomScaleAspctCrop(opt.input_size, scale=(.5, 1.), p=0.8, mode='cls'),
            T.Resize(opt.input_size, mode='cls'),
            T.ToTensor(), T.Normalizer(opt.mean, opt.std)])
        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_b = get_transform(opt, 'train', transform, tap, dtype=self.dtype)
        else:
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


class ClassfyDataPair4(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.

    Two brand augmentation & Same localtion for two branchs.

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False):
        super(ClassfyDataPair4, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        # RandomCrop for both branch
        self.transforms_0 = Compose([T.RandomScaleAspctCrop(opt.input_size, (0.5, 0.9), p=0.8, mode='cls')])

        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_b = get_transform(opt, 'train', transform, tap, dtype=self.dtype)
            self.transforms_a = get_transform(opt, 'train', transform, tap, dtype=self.dtype)
        else:
            self.transforms_a = self.transforms_b = transform

        self.basic_transforms = get_transform(opt, 'test')

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data(index)
        img_0, img_1 = img.copy(), img.copy()
        img_0 = self.basic_transforms({'image': img_0})['image']

        img_1 = self.transforms_0({'image': img_1})['image']
        img_2 = img_1.copy()

        img_1 = self.transforms_a({'image': img_1})['image']
        img_2 = self.transforms_b({'image': img_2})['image']

        return {'image': (img_1, img_2), 'image_': img_0, 'org_img': img, 'label': lbl, 'name': self.imgs[index]}


class ClassfyDataPair5(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.

    Two brand augmentation & Random localtion(for two branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False):
        super(ClassfyDataPair5, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_a = get_transform(opt, 'train', transform, tap, dtype=self.dtype)
            self.transforms_b = get_transform(opt, 'train', transform, tap, dtype=self.dtype)
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


class ClassfyDataPair6(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.

    Two brand augmentation & Random localtion(for two branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False):
        super(ClassfyDataPair6, self).__init__(init_data, opt, split, ratio, None, pin_memory)

        if type(transform) is str:
            tap = [.8, .5, .8, .0, .2, .5]
            self.transforms_a = get_transform(opt, 'train', 'C', tap, dtype=self.dtype)
            self.transforms_b = get_transform(opt, 'train', 'C', tap, dtype=self.dtype)
        else:
            self.transforms_a = transform
            self.transforms_b = transform

        self.basic_transforms = Compose([
            T.RandomScaleAspctCrop(opt.input_size, scale=(.5, 1.), p=0.8, mode='cls'),
            T.RandomRotateAndFlip(p=0.5, mode='cls'),
            T.ToTensor(),
            T.Normalizer(opt.mean, opt.std),
        ])

    def __getitem__(self, index):
        ''' Return one image(and label) per time. '''
        img, lbl = self.load_data(index)
        # img_0, img_a, img_b = img.copy(), img.copy(), img.copy()
        img_a, img_b, img_1, img_2 = img.copy(), img.copy(), img.copy(), img.copy()
        img_1 = self.basic_transforms({'image': img_1})['image']
        img_2 = self.basic_transforms({'image': img_2})['image']

        if self.transforms_a is not None:
            img_a = self.transforms_a({'image': img_a})['image']
        if self.transforms_b is not None:
            img_b = self.transforms_b({'image': img_b})['image']

        # return {'image': (img_a, img_b), 'image_': img_0, 'org_img': img, 'label': lbl, 'name': self.imgs[index]}
        return {'image': (img_a, img_b), 'image_': (img_1, img_2),
                'label': lbl, 'name': self.imgs[index]}


class ClassfyDataJigsaw(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.

    Two brand augmentation & Random localtion(for two branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False,
                 puzzle=2):
        super(ClassfyDataJigsaw, self).__init__(init_data, opt, split, ratio, None, pin_memory)
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


class ClassfyDataJigsaw2(ClassfyData):
    ''' Dataset loader for scene classification in pair mode.
    The puzzle_size is bigger than ClassfyDataJigsaw
    Two brand augmentation & Random localtion(for two branchs)

    Note:
        transform: specific transform of the 2nd branch.
    '''
    def __init__(self,
                 init_data,
                 opt,
                 split='train',
                 ratio=1,
                 transform=None,
                 pin_memory=False,
                 puzzle=2):
        super(ClassfyDataJigsaw2, self).__init__(init_data, opt, split, ratio, None, pin_memory)
        self.puzzle = puzzle
        if puzzle == 2:
            self.cr = [0, 135]
            self.cw = [0, 135]
            resize = (280, 280)
            crop_size = (270, 270)
            self.permutations = np.array(list(itertools.permutations(list(range(4)), 4)))
        elif puzzle == 3:
            self.cr = [0, 135, 270]
            self.cw = [0, 135, 270]
            resize = (420, 420)
            crop_size = (405, 405)
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
            T.RandomCrop((128, 128), mode='cls'),
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
                patch = img_[cr: cr+135, cw: cw+135, :].copy()
                patch = self.transforms_patch({'image': patch})['image']
                patchs[cnt] = patch
                cnt += 1

        # Shufflt
        order = self.permutations[lbl]
        patchs = [patchs[t] for t in order]
        data = torch.stack(patchs, 0)

        return {'image': data, 'image_': patchs, 'org_img': img,
                'label': np.array(lbl, dtype=np.int64), 'name': self.imgs[index]}


class ClassfyData_patch(data.Dataset):
    '''
    Patch-level Dataset Class for scene classification,
    load all normal patches of specific size from h5 file.
    Match with the `img2patches_cls()` in `operations.py`

    Args:
        ├── root
        |   ├── train_data.h5 or val_data.h5:
                        contain all data
            where images(patches) are NHW3 uint8 ndarray in f['size/image'] (size=50/100/...)
            where labels are NC int64 ndarray in f['size/label_cls'] (size=50/100/...)
        split: 'train' or 'val
        transform: transform object or None(default) or
            'auto'(will make normal image transform according to train/eval)
    '''

    def __init__(
            self,
            opt,  # 包含了数据集路径
            split='train',
            ratio=1,
            transform=None):
        self.opt = opt
        self.root = opt.dataset_dir
        self.split = split
        self.ratio = ratio
        self.insize = opt.input_size
        self.h5File = h5py.File(self.root + '/%s_data.h5' % split, 'r')

        if transform == 'auto':
            self.transforms = get_transform(opt, split)
        else:
            self.transforms = transform

        # Load data: collect indexs of all data or indexs into lists.
        self.img_path = '%s/image' % opt.scene_size[0]
        self.lbl_path = '%s/label' % opt.scene_size[0]

        print('%s set contains %s samples(patches), a total of %d categories.' %
              (split, self.h5File[self.img_path].shape[0], len(opt.num_classes)))
        print('Actual number of samples used = %d' % int(
            self.h5File[self.img_path].shape[0] * self.ratio))

    def __len__(self):
        return int(self.h5File[self.img_path].shape[0] * self.ratio)

    def __getitem__(self, index):
        ''' Return data of one object. '''
        sample = {}
        img = self.h5File[self.img_path][index, :]
        lbl = self.h5File[self.lbl_path][index]
        sample['image'], sample['label'] = img[:, :, ::-1], lbl  # RGB

        # print(lbl, self.opt.category.names[lbl])
        # cv2.imshow('patch', img)
        # cv2.waitKey(0)
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
