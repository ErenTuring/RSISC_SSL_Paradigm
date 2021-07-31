'''
A collection of tools used in model building and operations.
'''
import os
import copy
import torch


def load_ckpt(model, ckpt, train=False, block_layers=[], map_loc=None, vis=None):
    ''' Load ckpt and setup some state. '''
    print("Trying to load:", ckpt)
    if os.path.isfile(ckpt):
        if map_loc is None:
            map_loc = None if torch.cuda.is_available() else 'cpu'
        block_layers = [block_layers] if type(block_layers) is str else block_layers
        model_dict = model.state_dict()
        model_dict_bk = copy.deepcopy(model_dict)
        pretrained_dict = torch.load(ckpt, map_location=map_loc)
        print('Load ckpt info:')
        print('\tModel parameters count: %d' % (len(list(model_dict.keys()))))
        print('\tCp parameters count: %d' % (len(list(pretrained_dict.keys()))))
        # 剔除由于 torch.nn.DataParallel 包装而增多的key
        if 'module' in list(pretrained_dict.keys())[0]:
            if 'module' not in list(model_dict.keys())[0]:
                new_dict = {}
                for k, v in pretrained_dict.items():
                    new_dict[k[len('module.'):]] = v
                pretrained_dict = new_dict
        if 'module' in list(model_dict.keys())[0]:
            if 'module' not in list(pretrained_dict.keys())[0]:
                new_dict = {}
                for k, v in pretrained_dict.items():
                    new_dict['module.'+k] = v
                pretrained_dict = new_dict

        if len(block_layers) > 0:
            filtered_dict = {}
            bl_count = 0
            for k, v in pretrained_dict.items():
                if any(bl_l in k for bl_l in block_layers):
                    # print('Block_layer:\t %s' % k)
                    bl_count += 1
                    continue
                filtered_dict[k] = v
            model_dict.update(filtered_dict)
            print('\tBlock layer names: {}\n\tBlock parameters count: {}.'.format(block_layers, bl_count))
        else:
            model_dict = pretrained_dict
        model.load_state_dict(model_dict)  # , strict=False)

        # Count update params
        update_param_num = 0
        for k, v in model_dict.items():
            if not (model_dict_bk[k] == model_dict[k]).all():
                update_param_num += 1
        print('\tNum of update params: {}'.format(update_param_num))

    else:
        if train:
            print('\tCheckpoint file (%s) is not exist, re-initializtion' % ckpt)
        else:
            raise ValueError('Failed to load model checkpoint (%s).' % ckpt)
    return model


def save_ckpt(model, ckpt='cp', filename='checkpoint'):
    ''' Save checkpoint.
    Args:
        ckpt - Dir of ckpt to save.
        filename - Only name of ckpt to save.
    '''
    # import shutil
    filepath = ckpt+'/'+filename+'.pth'
    if type(model) in [torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel]:
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    # if torch.__version__ > '1.4':
    torch.save(model_dict, filepath, _use_new_zipfile_serialization=False)
    # torch.save(model_dict, filepath)
