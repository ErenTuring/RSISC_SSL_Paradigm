import os
import time


TYPE2BAND = {'RGB': 3, 'NIR': 1, 'SAR': 1, 'TEN': 10, 'ALL': 12, 'MS': 13}  # 'ALL' for sentienl data; 'MS' for EuroSAT
MODE_NAME = {1: 'train', 2: 'val', 3: 'test', 4: 'finetune',  # 5: 'exp',
             5: '', 6: '', 7: '',
             8: '', 9: 'pretrain', 0: 'debug'}


def unify_type(param, ptype=list, repeat=1):
    ''' Unify the type of param.

    Args:
        ptype: support list or tuple
        repeat: The times of repeating param in a list or tuple type.
    '''
    if repeat == 1:
        if type(param) is not ptype:
            if ptype == list:
                param = [param]
            elif ptype == tuple:
                param = (param)
    elif repeat > 1:
        if type(param) is ptype and len(param) == repeat:
            return param
        elif type(param) is list:
            param = param * repeat
        else:
            param = [param] * repeat
            param = ptype(param)

    return param


def add_default_settings(args=None):
    ''' Add some default settings '''
    if args is not None:
        # training - opitmization
        if not hasattr(args, 'hos'):
            args.hos = 0

        # training - distributed
        if not hasattr(args, 'mp_distributed'):
            args.mp_distributed = False
        if not hasattr(args, 'world_size'):
            args.world_size = 1
        if not hasattr(args, 'dist_url'):
            args.dist_url = 'tcp://localhost:10000'
        if not hasattr(args, 'dist_backend'):
            args.dist_backend = 'nccl'
        if not hasattr(args, 'rank'):
            args.rank = 0

        # Other
        if not hasattr(args, 'pin_memory'):
            args.pin_memory = False
    return args


def get_opt(name, args=None):
    '''Get options by name and current platform, and may use args to update them.'''
    from .dataset_cfg import get_config
    opts = get_config(name)

    if args is not None:
        # Extra
        args.train_scale = args.ts
        # Use ArgumentParser object to update the default configs
        for k, v in args.__dict__.items():
            if v is not None or not hasattr(opts, k):
                setattr(opts, k, v)
    opts = add_default_settings(opts)

    # Normalize the form of some parameters
    opts.dtype = unify_type(opts.dtype, list)
    opts.input_size = unify_type(opts.input_size, tuple, 2)
    for dt in opts.dtype:
        opts.band_num += TYPE2BAND[dt]
    opts.mean = opts.mean if len(opts.mean) == opts.band_num else opts.mean * opts.band_num
    opts.std = opts.std if len(opts.std) == opts.band_num else opts.std * opts.band_num

    # Log parameterss
    opts.timestamp = time.strftime('%y%j%H%M%S', time.localtime(time.time()))
    mode_digits = len(str(opts.mode))
    opts.mode_name = MODE_NAME[opts.mode // (10**(mode_digits-1))]
    opts.net_suffix = opts.timestamp + '_' + opts.mode_name
    if opts.mode_name == 'finetune':
        opts.net_suffix = opts.expnum.split('_')[0] + '_' + opts.net_suffix
    opts.exp_name = '%s_%s_%s' % (opts.env, opts.netnum, opts.net_suffix)
    print('Current Main Mode: %d - %s\n' % (opts.mode, opts.mode_name))

    # Basic prepare
    if not os.path.exists(opts.ckpt):
        os.mkdir(opts.ckpt)
    return opts


def opts_log(opt):
    # Creat the options as strings
    option_log = '-' * 50
    option_log += '\nCurrent Main Mode: %d - %s' % (opt.mode, opt.mode_name)
    option_log += '\nConfigs:\n'
    option_log += '\tMode: %d\n' % opt.mode
    option_log += '\tRoot: %s\n' % opt.root
    option_log += '\tDataset_dir: %s\n' % opt.dataset_dir
    option_log += '\tTraining set: %s\n' % opt.ds
    option_log += '\tTrain scale: {:.3f}\n'.format(opt.train_scale)
    option_log += '\tsepoch: %d\n' % opt.sepoch
    option_log += '\tPretrain checkpoint: %d\n' % opt.ptcp
    option_log += '\tMax epoch: {}\n'.format(opt.mepoch)
    option_log += '\tBatch size: %d\n' % opt.bs
    option_log += '\tNum_workers: %d\n' % opt.workers
    option_log += '\tLearning rate: %e\n' % opt.lr
    option_log += '\tLearning decay: [{:.2f}, {:.2f}]\n'.format(*opt.lr_decay)
    option_log += '\tLearning policy (schedule): %s\n' % opt.lr_policy
    option_log += '\tHOS: %d\n' % opt.hos
    option_log += '\tPTCP: %d\n' % opt.ptcp
    option_log += '\tWarmup: %d\n' % opt.warmup
    option_log += '\tWeight decay: %e\n' % opt.weight_decay
    option_log += '\tLoss: %s\n' % opt.loss
    if opt.loss_weight is None:
        option_log += '\tLoss weight: None\n'
    else:
        for w in opt.loss_weight:
            option_log += '\t  %f\n' % w

    option_log += '\tData type:\n'
    for dt in opt.dtype:
        option_log += '\t  %s\n' % dt
    option_log += '\tInput size: %d x %d x %d\n' % (opt.input_size[0], opt.input_size[1], opt.band_num)
    option_log += '\tMean: '
    for m in opt.mean:
        option_log += '%f, ' % m
    option_log += '\n\tstd: '
    for s in opt.std:
        option_log += '%f, ' % s

    return option_log
