import torch
# from torch.utils.data import DataLoader
# import numpy as np
from utils import (
    datasets_init,
    EuroSAT_dataset,
    data_cls,
    # tools,
)


# **************************************************
# ****************** CKPT tools ********************
# **************************************************
def sanity_check(state_dict, pretrained_ckpt):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_ckpt))
    state_dict_pre = torch.load(pretrained_ckpt, map_location="cpu")
    # state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if ('classifier' in k) or ('fc' in k):
            continue
        # name in pretrained model
        if k.startswith('module.') and not list(
                state_dict_pre.keys())[0].startswith('module.'):
            k_pre = k[len('module.'):]
        else:
            k_pre = k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def prepare_data(args, with_val=False):
    """
    Initialize the classification dataset and return dataloader for self-supervision.

    Returns:
        if split is None: return train_loader, val_loader
        if split is specified: return train/val dataloader
    """
    val_ratio = 0.002  # 3 sample per class
    print('Branch data augmentation: ', args.aug)

    init_data = datasets_init(args.ds, args, full_train=False)

    if args.ds in ['euroms', 'euromsrgb']:
        init_data = datasets_init(args.ds, args, full_train=False)
        ssDataset = {1: EuroSAT_dataset.EuroSATDataPair,
                     3: EuroSAT_dataset.EuroSATDataPair3,
                     5: EuroSAT_dataset.EuroSATDataPair5,
                     7: EuroSAT_dataset.EuroSATDataJigsaw}[args.aug]
        baseDataset = EuroSAT_dataset.EuroSATData
    else:
        ssDataset = {1: data_cls.ClassfyDataPair,
                     2: data_cls.ClassfyDataPair2,
                     3: data_cls.ClassfyDataPair3,
                     4: data_cls.ClassfyDataPair4,
                     5: data_cls.ClassfyDataPair5,
                     7: data_cls.ClassfyDataJigsaw,
                     8: data_cls.ClassfyDataJigsaw2}[args.aug]
        baseDataset = data_cls.ClassfyData

    if args.mode//(10**(len(str(args.mode))-2)) in [96, 99]:
        train_data = baseDataset(init_data, args, 'train', args.train_scale, 'D', args.pin_memory)
    elif args.mode//(10**(len(str(args.mode))-2)) == 97:
        puzzle_size = args.mode % 10
        train_data = ssDataset(init_data, args, 'train', args.train_scale, args.aug_p, args.pin_memory, puzzle_size)
    else:
        train_data = ssDataset(init_data, args, 'train', args.train_scale, args.aug_p, args.pin_memory)

    if with_val:
        val_data = ssDataset(init_data, args, 'val', val_ratio, 'x', args.pin_memory)
        return train_data, val_data

    return train_data
