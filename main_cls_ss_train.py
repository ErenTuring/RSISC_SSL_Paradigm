#!/usr/bin/env python
import argparse
import builtins
import time
# import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from models import build_model, net_utils, SimCLR, pretext_models
from config import get_opt, opts_log
from utils import contrast_loss, tools  # , distributed
import utils_SS_pretexts
import utils_for_main


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--pj', default='', help='project name')
parser.add_argument('--ds', metavar='dataset', default='aid', help='dataset name')
parser.add_argument('--ts', '--train_scale', default=20, type=float, help="Scale of training set reduction, e.g. 0.1")
parser.add_argument('--dtype', default=None, type=str, nargs='+', help='dtype of dataset')
parser.add_argument('--aug', default=5, type=int, help="Mode of data augmentation for both branch")
parser.add_argument('--aug_p', default='C', type=str, help="Data augmentation strageries")
parser.add_argument('--input_size', default=None, type=int, help="Input size")

parser.add_argument('--mode', default=910, type=int, help="Current main mode")
parser.add_argument('--netnum', default='020000', help="Network number")
parser.add_argument('--encoder_cp', default='', help="Checkpoint name of Encoder")
parser.add_argument('--agent_cp', default='', help="Checkpoint name of Agent")
parser.add_argument('--expnum', default='', help="Experiment number")
parser.add_argument('--ptcp', default=0, choices=[0, 1], type=int, help="wether use pretrain cp on ImageNet")
parser.add_argument('--ckpt', default=None, help="Path to save model")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--pin-memory', action='store_true', default=False, help='Pin data into memory')
parser.add_argument('--preload', action='store_true', default=False, help='Preload next batch data')

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--sepoch', default=None, type=int, help="Start epoch for countiue learning")
parser.add_argument('--cepoch', default=0, type=int, help="Current epoch")
parser.add_argument('--mepoch', default=None, type=int, help="Max epoch for countiue learning")
parser.add_argument('--ckpt_freq', default=None, type=int, help="Save ckpt every n epoches")

parser.add_argument('--optimizer', default='sgd', help="Optimizer: 'adam', 'sgd', 'lars'")
parser.add_argument('--loss', default='NTXentLoss', help="Loss: 'NCELoss', 'NTXentLoss', ...")
parser.add_argument('--lr', default=0.03, type=float)
parser.add_argument('--lr_decay', default=[1, 1], type=float, nargs='+', help="Learning rate decay")
parser.add_argument('--lr_policy', default='cosine', help="Learning rate policy (schedule)")
parser.add_argument('--milestones', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--warmup', default=0, type=int, help="The end epoch of warmup")
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--bs', default=2, type=int, help='the total batch size of all GPUs')

parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10000', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--mp-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--vis', action='store_true')

parser.add_argument('--feature_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--tau', default=0.5, type=float,
                    help='softmax temperature (default: 0.5)')


def main():
    args = parser.parse_args()
    args = get_opt(args.ds, args)

    cudaInfo = tools.CudaDevices()
    args.ngpus_per_node = cudaInfo.avail_num

    args.distributed = args.world_size > 1 or args.mp_distributed or args.ngpus_per_node > 1
    if args.mp_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))
    else:
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.mp_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # total_bs = args.bs
    if args.mp_distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        args.bs = args.bs // args.ngpus_per_node
        args.workers = (args.workers + args.ngpus_per_node - 1) // args.ngpus_per_node
    cudnn.benchmark = True
    # print('Adjust LR from %e to %e' % (args.lr, args.lr * total_bs / 256))
    # args.lr = args.lr * total_bs / 256

    # create model
    net = build_model(args.feature_dim, args.netnum, args.band_num, args.ptcp)  # .to(DEVICE)
    train_epoch = utils_SS_pretexts.SimCLR_train
    # Pake target net into SSL model
    # **** SimCLR ****
    if args.mode == 910:  # SimCLR
        model = SimCLR.SimCLR_cls(net)   # packge
        train_epoch = utils_SS_pretexts.SimCLR_train
        # criterion = contrast_loss.get_contrast_loss(args.loss, **loss_kwdict).cuda(args.gpu)
        criterion = contrast_loss.NTXentLoss(args.bs, args.tau)
    # **** Image Inpainting ****
    elif args.mode == 96:  # Image Inpainting
        model = pretext_models.Context_Encoder(net, args.band_num)
        # model = pretext_models.Context_Encoder0(net)  # cannot work
        train_epoch = utils_SS_pretexts.inpainting_train
        criterion = torch.nn.MSELoss(reduction='none')
    # **** Predict relative position ****
    elif args.mode//(10**(len(str(args.mode))-2)) == 97:
        args.aug = 7
        if args.mode == 972:
            args.num_classes = 24
            # args.input_size = (150, 150)
        elif args.mode == 973:
            args.num_classes = 1000
            # args.input_size = (225, 225)
        model = pretext_models.Jigsawer(net, args.num_classes, args.mode % 10)
        train_epoch = utils_SS_pretexts.jigsaw_train
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.mode == 98:
        model = pretext_models.ColorizationNet(net, args.band_num)
        train_epoch = utils_SS_pretexts.colorization_train
        criterion = torch.nn.MSELoss()
    # **** GANs ****
    elif args.mode//(10**(len(str(args.mode))-2)) == 99:  # MARTR_GANs
        args.input_size = (256, 256)
        if args.mode == 990:
            net = pretext_models.Discriminator_GANs(1, args.band_num, supervised=True)
            model = pretext_models.Generator_GAN(args.band_num)
        elif args.mode == 991:
            net = pretext_models.Discriminator_GANs2(1, args.band_num, supervised=True)
            model = pretext_models.Generator_GAN2(args.band_num)

        train_epoch = utils_SS_pretexts.gan_train
        criterion = torch.nn.BCELoss()  # Binary Cross Entropy loss
        net = tools.place_model(net, args)
        optimizerD = torch.optim.Adam(
            net.parameters(), lr=args.lr, betas=(0.5, 0.999))  # weight_decay=0.9
    model = tools.place_model(model, args)

    # define loss function (criterion) and optimizer
    if args.mode//(10**(len(str(args.mode))-2)) == 99:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.5, 0.999))  # weight_decay=0.9
    else:
        optimizer = tools.set_optimizer(args, model)

    # optionally resume from a checkpoint
    if args.resume:
        cp_epoch = str(args.cepoch) if args.cepoch else 'best'
        ckpt = args.ckpt + '/%s_%s_full_%s.pth' % (args.netnum, args.expnum, cp_epoch)
        loc = 'cuda:{}'.format(args.gpu) if args.gpu is not None else None
        net_utils.load_ckpt(model, ckpt, True, map_loc=loc)

    # Data loading code
    train_dataset = utils_for_main.prepare_data(args)
    train_sampler = None
    if args.mp_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.bs, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if not args.mp_distributed or args.gpu == 0:
        print('```\n' + opts_log(args) + '\n```')

    epoch_best_loss = 1000
    print('%s\nBegain training' % args.exp_name)
    for epoch in range(args.sepoch, args.mepoch+1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        LR = tools.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        tic = time.time()
        if args.mode//(10**(len(str(args.mode))-2)) == 99:
            tools.adjust_learning_rate(optimizerD, epoch, args)
            epoch_loss = train_epoch(train_loader, model, net, criterion, optimizer, optimizerD, epoch, args)
        else:
            epoch_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

        print('\n\tEpoch: [%d | %d], lr:%e, loss: %.6f, Time: %.1fs\n' % (
              epoch, args.mepoch, LR, epoch_loss, (time.time() - tic)),
              '\t*****************')

        if not args.mp_distributed or args.gpu == 0:
            if (epoch > (args.mepoch//10)) and (epoch_loss < epoch_best_loss):
                net_utils.save_ckpt(net, args.ckpt, '%s_%s_best' % (args.netnum, args.net_suffix))
            if args.ckpt_freq and (epoch % args.ckpt_freq == 0):
                net_utils.save_ckpt(model, args.ckpt, '%s_%s_full_%d' % (
                    args.netnum, args.net_suffix, epoch))
        epoch_best_loss = min(epoch_loss, epoch_best_loss)

    if not args.mp_distributed or args.gpu == 0:
        net_utils.save_ckpt(net, args.ckpt, '%s_%s_%d' % (args.netnum, args.net_suffix, epoch))
        print('Finish! Stop at epoch %d (max epoch=%d).' % (epoch, args.mepoch))


if __name__ == '__main__':
    main()
