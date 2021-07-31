#!/usr/bin/env python
import argparse
import numpy as np
import torch
import joblib
import torch.utils.data
from models import build_model, net_utils
from config import get_opt, opts_log
from utils import tools, cls_dataset  # , distributed
from sklearn.svm import LinearSVC


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--ds', metavar='dataset', default='aid', help='dataset name')
parser.add_argument('--dtype', default=None, type=str, nargs='+', help='dtype of dataset')
parser.add_argument('--ts', '--train_scale', default=2, type=float, help="Scale of training set reduction, e.g. 0.1")
parser.add_argument('--input_size', default=256, type=int, help="Input size")

parser.add_argument('--mode', default=1, type=int, help="Current main mode")
parser.add_argument('--netnum', default='090000', help="Network number")  # 030000
parser.add_argument('--expnum', default='', help="Experiment number")
parser.add_argument('--ptcp', default=0, choices=[0, 1], type=int, help="wether use pretrain cp on ImageNet")
parser.add_argument('--ckpt', default=None, help="Path to save model")

parser.add_argument('--pin-memory', action='store_true', default=False, help='Pin data into memory')
parser.add_argument('--h5', action='store_true', default=False, help='Use h5 format dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--sepoch', default=None, type=int, help="Start epoch for countiue learning")
parser.add_argument('--cepoch', default=0, type=int, help="Current epoch")
parser.add_argument('--mepoch', default=None, type=int, help="Max epoch for countiue learning")
parser.add_argument('--evaluate', action='store_true', default=True, help='evaluate during training')

parser.add_argument('--hos', default=0, choices=[0, 1, 9], type=int, help="Wether use hierarchical optimization strategy. (1-use)")
parser.add_argument('--optimizer', default='sgd', help="Optimizer: 'adam', 'sgd', 'lars'")
parser.add_argument('--loss', default='CrossEntropyLoss', help="Loss: 'CrossEntropyLoss', 'BCEWithLogitsLoss', ...")
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--lr_decay', default=[1, 1], type=float, nargs='+', help="Learning rate decay")
parser.add_argument('--lr_policy', default='cosine', help="Learning rate policy (schedule)")
parser.add_argument('--milestones', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--warmup', default=0, type=int, help="The end epoch of warmup")
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--bs', default=8, type=int, help='the total batch size of all GPUs')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

parser.add_argument('--vis', default=0, choices=[0, 1], type=int, help="Wether visualize samples.")
parser.add_argument('--tta', default=0, choices=[0, 1], type=int, help="Wether use test time argumentation. (1-use).")
best_acc = 0


def main():
    args = parser.parse_args()
    args = get_opt(args.ds, args)
    args.distributed = False
    args.bs = 10  # TODO

    # create model
    model = get_net(args.netnum, args)  # .to(DEVICE)
    model.eval()
    model.supervised = True

    # Data loading code
    train_dataset, val_dataset = cls_dataset(
        args.ds, args, val_ratio=args.val_scale, transform='val')
    sample_num = train_dataset.__len__()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.bs, num_workers=args.workers)

    train_x = np.zeros((sample_num, 14336), dtype='float32')
    train_y = np.zeros((sample_num), dtype=np.int64)
    with torch.no_grad():
        for ii, data in enumerate(train_loader):
            input, target = data['image'], data['label'].numpy().astype(np.int64)
            if args.gpu is not None:
                input = input.cuda(non_blocking=True)
            else:
                input = input.cuda(non_blocking=True)
            feature, _ = model(input)
            feature = torch.nn.functional.normalize(feature, dim=1).cpu().numpy()
            # bs = feature.shape[0]

            # if ii < (len(train_loader) - 1):
            if ii == (len(train_loader) - 1):
                # special treatment for final batch
                train_x[ii * args.bs:] = feature
                train_y[ii * args.bs:] = target
            else:
                train_x[ii * args.bs: (ii + 1) * args.bs] = feature
                train_y[ii * args.bs: (ii + 1) * args.bs] = target

    print('```\n' + opts_log(args) + '\n```')

    print('Begain Train! %s' % (args.exp_name))
    clf = LinearSVC(C=1)
    clf.fit(train_x, train_y)
    # Save the SVM
    joblib.dump(clf,
                "{}/{}_{}.pkl".format(args.ckpt, args.expnum, 'MARTR_GANs'), compress=3)

    val_dataset = cls_dataset(args.ds, args, 'val')  # , loader=True
    val_loader = torch.utils.data.DataLoader(val_dataset, args.bs, num_workers=args.workers)
    sample_num = val_dataset.__len__()
    val_x = np.zeros((sample_num, 14336), dtype='float32')
    val_y = np.zeros((sample_num), dtype=np.int64)
    with torch.no_grad():
        for ii, data in enumerate(val_loader):
            input, target = data['image'], data['label'].numpy().astype(np.int64)
            if args.gpu is not None:
                input = input.cuda(non_blocking=True)
            else:
                input = input.cuda(non_blocking=True)
            feature, _ = model(input)
            feature = torch.nn.functional.normalize(feature, dim=1).cpu().numpy()
            # bs = feature.shape[0]

            if ii == (len(val_loader) - 1):
                # special treatment for final batch
                val_x[ii * args.bs:] = feature
                val_y[ii * args.bs:] = target
            else:
                val_x[ii * args.bs: (ii + 1) * args.bs] = feature
                val_y[ii * args.bs: (ii + 1) * args.bs] = target

    runningscore = tools.ClassifyScore(args.num_classes, args.category.names)
    predicts = clf.predict(val_x)
    runningscore.update(val_y, predicts)
    score = runningscore.get_scores()
    runningscore.print_score(score)
    print('Finish!\n\n\n\n\n\n')


def get_net(net_num, args):
    # from thop import profile, clever_format
    if args.weight_decay:
        model = build_model(args.num_classes, net_num, args.band_num, args.ptcp)
    else:
        model = build_model(args.num_classes, net_num, args.band_num, args.ptcp, wd_mode=1)

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

    # if args.weight_decay == 0:
    #     model.backup_weights(['classifier'])
    #     print('L2-SP regularization: wd_rate_alpha: %e, wd_rate_beta: %e' % (
    #         model.wd_rate_alpha, model.wd_rate_beta))

    model = tools.place_model(model, args)
    return model


if __name__ == '__main__':
    main()
