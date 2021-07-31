# -*- coding:utf-8 -*-
'''
Inital (train) dataset for scene classification.

Version 1.0  2019-06-02 22:19:24 by QiJi
'''
import os
from torch.utils.data import DataLoader  # Dataset
from .distributed import DistributedSampler


def get_data_dir(ds, opt):
    if type(opt.dataset_dir) == str:
        return opt.dataset_dir
    elif type(opt.dataset_dir) == dict:
        return opt.dataset_dir[ds]


def cls_dataset(ds, opt, split=None, val_ratio=1, transform='B', distributed=False, loader=False):
    """ Initialize the classification dataset and return dataloader.

    Returns:
        if split is None: return train_loader, val_loader
        if split is specified: return train/val dataloader
    """
    from . import data_cls, EuroSAT_dataset

    if ds in ['euroms', 'euromsrgb']:
        mDataset = EuroSAT_dataset.EuroSATData
    else:
        mDataset = data_cls.ClassfyData

    init_data = datasets_init(ds, opt)

    if split is None:
        train_data = mDataset(init_data, opt, 'train', opt.train_scale, transform, opt.pin_memory)
        val_data = mDataset(init_data, opt, 'val', val_ratio, transform, opt.pin_memory)

        if loader:
            val_loader = data2loader(val_data, opt, 'val', distributed)
            train_loader = data2loader(train_data, opt, 'train', distributed)
            return train_loader, val_loader
        else:
            return train_data, val_data
    else:
        ratio = opt.train_scale if split == 'train' else val_ratio
        data_set = mDataset(init_data, opt, split, ratio, transform, opt.pin_memory)
        if loader:
            return data2loader(data_set, opt, split, distributed)
        else:
            return data_set


def data2loader(dataset, opt, split, distributed=False):
    kwdict = {'num_workers': opt.workers}
    if split == 'train':
        if len(dataset) < opt.bs:
            print('Total sample num small than batch_size, adjust the batchsize to %d.' % len(dataset))
            opt.bs = len(dataset)
        kwdict.update({'batch_size': opt.bs, 'drop_last': True, 'shuffle': True})
        if distributed:
            kwdict.update({'sampler': DistributedSampler(dataset), 'shuffle': False, 'pin_memory': True})
    else:
        if opt.bs > 1:
            kwdict.update({'batch_size': opt.bs*2})

    return DataLoader(dataset, **kwdict)


def datasets_init(ds, opt, full_train=False):
    """ Repackaged the function `classifydataset_init()` to aggregate multiple datasets.
    """
    # from . import data_utils
    from config.category import get_category

    if '+' in ds:  # multi datasets
        ds_list = ds.split('+')
        init_data = {k: {} for k in ['train', 'val', 'test']}
        for ds in ds_list:

            get_data = classifydataset_init(
                opt.dataset_dir[ds], get_category(ds), opt.train_val_ratio, full_train, 1)
            for sp in ['train', 'val', 'test']:
                for cls, v in get_data[sp].items():
                    if cls in init_data[sp]:
                        init_data[sp][cls] += get_data[sp][cls]
                    else:
                        init_data[sp][cls] = get_data[sp][cls]
    else:
        root = get_data_dir(ds, opt)
        # For h5 file data, only return the path of h5 file
        if root.endswith('.h5'):
            return root

        init_data = classifydataset_init(
            root, get_category(ds), opt.train_val_ratio, full_train, 1)

    return init_data


def split_dataset(dataset, split, ratio):
    new_dataset1 = {split: {}}
    new_dataset2 = {split: {}}

    for cls, samples in dataset[split].items():
        # total_num += len(samples)
        if ratio >= 0.01:
            use_num = max(round(len(samples)*ratio), 1)
        else:
            use_num = int(ratio*1000)
        use_num = min(use_num, len(samples))
        new_dataset1[split][cls] = samples[:use_num]
        new_dataset2[split][cls] = samples[use_num:]

    return new_dataset1, new_dataset2


def classifydataset_init(root, category, ratio=[0.5, 0.3, 0.2], full_train=False, mode=0, seed=0):
    '''
    分类数据初始化， 即获得训练(测试)集的所有文件路径及其标签
    Args:
        root - The dir of the train(test) dataset folder.
        ratio - [train, val, test] or [train, val]
            Note: 在2、3种模式下, ratio[0]默认为0,
                若为1则表示train_set包含train/val/test所有数据
        mode - 文件的组织结构不同
            1: 不同类别分文件夹放置
            2: 预测划分了train/val/test
        full_train - wether return all the images for train
    '''
    init_dataset = {k: {} for k in ['train', 'val', 'test']}
    cls_table = category.table

    if mode == 1:
        # 先遍历文件夹
        for f in sorted(os.listdir(root)):
            cls = cls_table[f]
            img_names = sorted(os.listdir(root + '/' + f))
            samples = [(root+'/'+f+'/'+x, cls) for x in img_names]
            n = len(img_names)
            train_num, val_num = int(n * ratio[0]), int(n * ratio[1])
            if full_train:
                init_dataset['train'].update({f: samples[:]})
            else:
                init_dataset['train'].update({f: samples[: train_num]})
            init_dataset['val'].update({f: samples[train_num: train_num + val_num]})
            init_dataset['test'].update({f: samples[train_num + val_num:]})

    elif mode == 2:
        for split in ['train', 'val', 'test']:
            data_dir = root + '/' + split
            if os.path.exists(data_dir):
                for f in sorted(os.listdir(data_dir)):
                    cls = cls_table[f]
                    img_names = sorted(os.listdir(data_dir + '/' + f))
                    samples = [(data_dir+'/'+f+'/'+x, cls) for x in img_names]
                    init_dataset[split].update({f: samples})
                    if full_train and split != 'train':
                        init_dataset['train'][f].append(samples)
            else:
                print('No %sing set found at: %s.' % (split, data_dir))

    return init_dataset
