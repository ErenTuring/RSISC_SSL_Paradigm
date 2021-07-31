#!/usr/bin/env python
import argparse
import builtins
import time
from collections import deque
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from models import build_model, net_utils
from config import get_opt, opts_log
from utils import criterions, tools, cls_dataset
import utils_for_main


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--ds', metavar='dataset', default='aid', help='dataset name')
parser.add_argument('--dtype', default=None, type=str, nargs='+', help='dtype of dataset')
parser.add_argument('--ts', '--train_scale', default=9, type=float, help="Scale of training set reduction, e.g. 0.1")
parser.add_argument('--input_size', default=None, type=int, help="Input size")

parser.add_argument('--mode', default=2, type=int, help="Current main mode")
parser.add_argument('--netnum', default='010'+'2100'+'000', help="Network number")  # 030000
parser.add_argument('--encoder_cp', default='', help="Checkpoint name of Encoder")
parser.add_argument('--agent_cp', default='', help="Checkpoint name of Agent")
parser.add_argument('--expnum', default='', help="Experiment number")
parser.add_argument('--ptcp', default=0, choices=[0, 1], type=int, help="wether use pretrain cp on ImageNet")
parser.add_argument('--ckpt', default=None, help="Path to save model")

parser.add_argument('--pin-memory', action='store_true', default=False, help='Pin data into memory')
parser.add_argument('--preload', action='store_true', default=False, help='Preload next batch data')

parser.add_argument('--h5', action='store_true', default=False, help='Use h5 format dataset')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--sepoch', default=None, type=int, help="Start epoch for countiue learning")
parser.add_argument('--cepoch', default=0, type=int, help="Current epoch")
parser.add_argument('--mepoch', default=None, type=int, help="Max epoch for countiue learning")
parser.add_argument('--evaluate', action='store_true', default=True, help='evaluate during training')

parser.add_argument('--hos', default=0, choices=[0, 1, 9], type=int, help="Wether use hierarchical optimization strategy. (1-use)")
parser.add_argument('--optimizer', default='sgd', help="Optimizer: 'adam', 'sgd', 'lars'")
parser.add_argument('--loss', default='CrossEntropyLoss', help="Loss: 'CrossEntropyLoss', 'BCEWithLogitsLoss', ...")
parser.add_argument('--lr', default=0.15, type=float)
parser.add_argument('--lr_decay', default=[1, 1], type=float, nargs='+', help="Learning rate decay")
parser.add_argument('--lr_policy', default='cosine', help="Learning rate policy (schedule)")
parser.add_argument('--milestones', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--warmup', default=0, type=int, help="The end epoch of warmup")
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--bs', default=2, type=int, help='the total batch size of all GPUs')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

parser.add_argument('--vis', default=0, choices=[0, 1], type=int, help="Wether visualize samples.")
parser.add_argument('--tta', default=0, choices=[0, 1], type=int, help="Wether use test time argumentation. (1-use).")
best_acc = 0


def main():
    args = parser.parse_args()
    args = get_opt(args.ds, args)

    cudaInfo = tools.CudaDevices()
    args.ngpus_per_node = cudaInfo.avail_num

    args.distributed = args.world_size > 1 or args.mp_distributed or args.ngpus_per_node > 1
    main_worker = {
        'train': main_train,
        'val': main_predict,
        'test': main_predict,
        'finetune': main_train
    }[args.mode_name]
    if args.mp_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))
    else:
        main_worker(args.gpu, args)


def main_train(gpu, args):
    global best_acc
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.mp_distributed and args.gpu != 0:
        def print_pass(*args):
            pass  # suppress printing if not master
        builtins.print = print_pass

    # total_bs = args.bs
    if args.mp_distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        args.bs = int(args.bs / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    cudnn.benchmark = True
    # create model
    model = get_net(args.netnum, args)  # .to(DEVICE)

    # define loss function (criterion) and optimizer
    loss_kwdict = {}
    criterion = criterions.get_loss(args.loss, args.loss_weight, **loss_kwdict).cuda(args.gpu)
    optimizer = tools.set_optimizer(args, model)

    # Data loading code
    train_dataset, val_dataset = cls_dataset(
        args.ds, args, val_ratio=args.val_scale, transform='D')
    train_sampler, val_sampler = None, None
    if args.mp_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    args.bs = min(args.bs, train_dataset.__len__()//args.ngpus_per_node)
    # print('Adjust LR from %e to %e (Train sample num = %d)' % (
    #     args.lr, args.lr * total_bs / 256, sample_num))
    # args.lr = args.lr * total_bs / 256
    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.bs, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, args.bs, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if not args.mp_distributed or args.gpu == 0:
        print('```\n' + opts_log(args) + '\n```')

    print('Begain Train! %s' % (args.exp_name))
    for epoch in range(args.sepoch, args.mepoch+1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        LR = tools.adjust_learning_rate(optimizer, epoch, args)
        # Train for one epoch
        epoch_loss, epoch_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

        log_str = ('\n\tEpoch: [%d | %d], lr:%e, loss: %.6f' % (epoch, args.mepoch, LR, epoch_loss))
        if epoch % max(args.mepoch//500, 1) == 0:
            scores = tools.quick_val_c(model, val_loader, args)
            epoch_acc = scores['OA']
            log_str += (', Val_OA: %.4f' % scores['OA'])
            if 'Kappa' in scores:
                log_str += (', Kappa: %.4f' % scores['Kappa'])
            if not args.mp_distributed or args.gpu == 0:
                if epoch_acc > best_acc:
                    net_utils.save_ckpt(model, args.ckpt, '%s_%s_best' % (args.netnum, args.net_suffix))
                if epoch == args.sepoch and args.hos == 9:
                    utils_for_main.sanity_check(model.state_dict(), args)
            best_acc = max(epoch_acc, best_acc)
        print(log_str + '\n\t*****************')
    print('Finish training! Stop at epoch %d (max epoch=%d).' % (epoch, args.mepoch))

    # * Extra_test *
    if not args.mp_distributed or args.gpu == 0:
        net_utils.save_ckpt(model, args.ckpt, '%s_%s_%d' % (args.netnum, args.net_suffix, epoch))
    loc = 'cuda:{}'.format(args.gpu) if args.gpu is not None else None
    net_utils.load_ckpt(model, args.ckpt+'/%s_%s_best.pth' % (args.netnum, args.net_suffix), map_loc=loc)
    val_dataset = cls_dataset(args.ds, args, 'val', transform='val')  # , loader=True
    val_loader = torch.utils.data.DataLoader(val_dataset, args.bs*2, num_workers=args.workers)
    scores = tools.quick_val_c(model, val_loader, args, True)

    print('Finish!\n\n\n\n\n\n')


def main_predict(gpu, args):
    args.gpu = gpu
    if args.mp_distributed and args.gpu != 0:
        def print_pass(*args):
            pass  # suppress printing if not master
        builtins.print = print_pass
    print('```\n' + opts_log(args) + '\n```')

    if args.mp_distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        args.bs = int(args.bs / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)

    # create model
    model = get_net(args.netnum, args)  # changed batch_size
    model.eval()

    val_dataset = cls_dataset(args.ds, args, split=args.mode_name, transform='val')
    val_sampler = None
    if args.mp_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, args.bs, num_workers=args.workers, sampler=val_sampler)

    runningscore = tools.ClassifyScore(args.num_classes, args.category.names)

    print('Begain predicting...')
    with torch.no_grad():
        for ii, data in enumerate(val_loader):
            input, target = data['image'], data['label'].numpy().astype(np.int64)
            input = input.cuda(args.gpu, non_blocking=True)

            if args.tta:
                predict = tools.test_time_augment(model, input, mode='cls')
            else:
                predict = model(input)

            predict = predict.argmax(1).cpu().numpy().astype(target.dtype)
            runningscore.update(target, predict)
    score = runningscore.get_scores()
    runningscore.print_score(score)
    print('Finished!\n\n\n\n\n\n')


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    if args.hos == 9:
        model.eval()
    else:
        model.train()  # transform model mode

    batch_num = len(train_loader)
    loss_hist, acc_hist = deque(maxlen=batch_num), deque(maxlen=batch_num)

    tic = time.time()
    for i in range(batch_num):
        if i % len(train_loader) == 0:
            data_loader = tools.package_loader(train_loader, args)

        data = data_loader.next()
        if data is None:
            break

        input, target = data['image'], data['label']  # [NCHW]
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)  # [NCHW]
        else:
            input = input.cuda(non_blocking=True)  # [NCHW]
        target = target.cuda(args.gpu, non_blocking=True)  # [NCHW]
        optimizer.zero_grad()

        output = model(input)  # no softmax layer; [N,class_num]
        loss = criterion(input=output, target=target)
        # if args.weight_decay == 0:
        #     loss += model.decay()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predict = output.argmax(1)
            acc = predict.eq(target).float().mean().cpu().numpy()

        # Meters update and visualize
        loss_hist.append(loss.item())
        acc_hist.append(acc)

        if not args.mp_distributed or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                tools.train_log(
                    ' Epoch: [{} | {}] iters: {:6d} loss: {:.3f} ({:.3f}) acc: {:.3f} ({:.3f})'
                    ' Time: {:.2f}s\r'.format(epoch, args.mepoch, i, loss.item(), np.mean(loss_hist),
                                              acc, np.mean(acc_hist), time.time() - tic), '\t')
                tic = time.time()  # update time

    return np.mean(loss_hist), np.mean(acc_hist)


def get_net(net_num, args):
    # from thop import profile, clever_format
    if args.weight_decay:
        model = build_model(args.num_classes, net_num, args.band_num, args.ptcp)
    else:
        model = build_model(args.num_classes, net_num, args.band_num, args.ptcp,
                            wd_kwargs={'wd_mode': 1})

    # Optionally resume from a checkpoint
    bl_layer = []
    cp_epoch = str(args.cepoch) if args.cepoch else 'best'
    if args.mode_name == 'train':
        cp_epoch = str(args.sepoch) if args.sepoch > 1 else 'NoNo'
    elif args.mode_name == 'finetune':
        print('Finetune:\n\tExp num: %s\n\tEpoch: %s\n' % (args.expnum, cp_epoch))
        bl_layer += ['classifier', 'fc']
    ckpt = args.ckpt + '/%s_%s_%s.pth' % (net_num, args.expnum, cp_epoch)
    if args.resume:
        ckpt = args.resume
    # loc = 'cuda:{}'.format(args.gpu) if args.gpu is not None else None
    net_utils.load_ckpt(model, ckpt, (args.mode_name in ['train', 'finetune']),
                        block_layers=bl_layer, map_loc='cpu')

    if args.hos == 9:
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if ('classifier' not in name) and ('fc' not in name):
                param.requires_grad = False
        # model.classifier[1].weight.data.normal_(mean=0.0, std=0.01)
        # model.classifier[1].bias.data.zero_()

    model = tools.place_model(model, args)
    return model


if __name__ == '__main__':
    main()
