# -*- coding:utf-8 -*-
'''
Abstract.

Version 1.0  2020-07-06 14:32:04
by QiJi Refence:
TODO:
'''

import os
import time
from collections import deque
# from itertools import chain

import numpy as np
import torch
from PIL import Image
# import torch.nn as nn
import torch.nn.parallel
# import torch.distributed as dist
import torch.optim
import torch.utils.data
from utils import transform, tools  # , distributed

VIS_DIR = '.'


def get_erase_mask(bs, opts, erase_shape=[16, 16], erase_count=16):
    H, W = opts.input_size
    masks = torch.ones((bs, opts.band_num, H, W))
    for n in range(bs):
        for _ in range(erase_count):
            row = np.random.randint(0, H - erase_shape[0] - 1)
            col = np.random.randint(0, W - erase_shape[1] - 1)
            masks[n, :, row: row+erase_shape[0], col: col+erase_shape[1]] = 0
    return masks


def get_central_mask(bs, opts, erase_ratio=1/2):
    H, W = opts.input_size
    masks = torch.ones((bs, opts.band_num, H, W))
    eH, eW = int(H*erase_ratio), int(W*erase_ratio)
    row_st = (H - eH) // 2
    col_st = (W - eW) // 2
    masks[:, :, row_st: row_st+eH, col_st: col_st+eW] = 0
    return masks


def central_block(input, opts, overlapPred):
    pass


def inpainting_train(train_loader, model, criterion, optimizer, epoch, args,
                     overlapPred=4):
    ''' One epoch training use inpainting pretext task. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    tic = time.time()
    vis_dir = VIS_DIR + args.exp_name

    masks = get_central_mask(args.bs, args, erase_ratio=0.5)
    if args.gpu is not None:
        masks = masks.cuda(args.gpu, non_blocking=True)  # [NCHW]
    else:
        masks = masks.cuda(non_blocking=True)

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        input = data['image']
        # masks = get_erase_mask(args.bs, args, [16, 16], erase_count=16)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)  # [NCHW]
        else:
            input = input.cuda(non_blocking=True)  # [NCHW]

        output = model(input * masks)
        mse_loss = criterion(output, input)
        # mse_loss = (output - input)**2
        # mse_loss = -1*torch.nn.functional.threshold(-1*mse_loss, -2, -2)
        loss_rec = torch.sum(mse_loss*(1-masks))/torch.sum(1-masks)
        loss_con = torch.sum(mse_loss*masks)/torch.sum(masks)
        # loss = torch.sum(mse_loss*(1-masks)) / torch.sum(1-masks)
        loss = 0.99 * loss_rec + 0.01 * loss_con
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        # Meters update and visualize
        loss_hist.append(loss.item())
        if not args.mp_distributed or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                tools.train_log(
                    ' Epoch: [{} | {}] iters: {:6d} loss: {:.3f} Time: {:.2f}s\r'.format(
                        epoch, args.mepoch, i, loss.item(), time.time() - tic), '\t')
                tic = time.time()  # update time
    if args.vis and (epoch % max(args.mepoch // 10, 1) == 0) and (
            not args.mp_distributed or args.gpu == 0):
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        with torch.no_grad():
            cropped = input * masks
            input = transform.unnormalize(input, args.mean, args.std, True)
            output = transform.unnormalize(output, args.mean, args.std, True)
            cropped = transform.unnormalize(cropped, args.mean, args.std, True)

            for j, (real, crop, rec) in enumerate(zip(input, cropped, output)):
                real = Image.fromarray(real[:, :, :3], 'RGB')
                crop = Image.fromarray(crop[:, :, :3], 'RGB')
                rec = Image.fromarray(rec[:, :, :3], 'RGB')

                real.save(vis_dir+'/Real_%d.jpg' % j)
                crop.save(vis_dir+'/Croped_%d.jpg' % j)
                rec.save(vis_dir+'/Recovered_%d.jpg' % j)

                if j > 32:
                    break
    return np.mean(loss_hist)


def colorization_train(train_loader, model, criterion, optimizer, epoch, args):
    ''' One epoch training use inpainting pretext task. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    tic = time.time()
    vis_dir = VIS_DIR + args.exp_name

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        image = data['image']
        # grayscale
        input = image[:, np.random.randint(args.band_num), :, :].clone()
        input = input.unsqueeze(1).repeat(1, args.band_num, 1, 1)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)  # [NCHW]
        else:
            input = input.cuda(non_blocking=True)  # [NCHW]

        output = model(input)
        loss = criterion(output, input)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        # Meters update and visualize
        loss_hist.append(loss.item())
        if not args.mp_distributed or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                tools.train_log(
                    ' Epoch: [{} | {}] iters: {:6d} loss: {:.3f} Time: {:.2f}s\r'.format(
                        epoch, args.mepoch, i, loss.item(), time.time() - tic), '\t')
                tic = time.time()  # update time
    if args.vis and (epoch % max(args.mepoch // 10, 1) == 0) and (
            not args.mp_distributed or args.gpu == 0):
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        with torch.no_grad():
            image = transform.unnormalize(image, args.mean, args.std, True)
            input = transform.unnormalize(input, args.mean, args.std, True)
            output = transform.unnormalize(output, args.mean, args.std, True)

            for j, (org, gray, color) in enumerate(zip(image, input, output)):
                org = Image.fromarray(org[:, :, :3], 'RGB')
                gray = Image.fromarray(gray[:, :, :3], 'RGB')
                color = Image.fromarray(color[:, :, :3], 'RGB')

                org.save(vis_dir+'/Org_%d.jpg' % j)
                gray.save(vis_dir+'/Croped_%d.jpg' % j)
                color.save(vis_dir+'/Color_%d.jpg' % j)
                if j > 32:
                    break
    return np.mean(loss_hist)


def gan_train(train_loader, netG, netD, criterion, optimizerG, optimizerD, epoch, args):
    ''' One epoch training use inpainting pretext task. '''
    netD.train()
    netG.train()
    loss_hist = deque(maxlen=len(train_loader))
    tic = time.time()
    vis_dir = VIS_DIR + args.exp_name

    real_label, fake_label = 1., 0.
    for i, data in enumerate(train_loader):

        x_real = data['image']
        bs = x_real.shape[0]
        # grayscale
        if args.gpu is not None:
            x_real = x_real.cuda(args.gpu, non_blocking=True)  # [NCHW]
            z = torch.randn((bs, 100, 1, 1)).cuda(args.gpu, non_blocking=True)
        else:
            x_real = x_real.cuda(non_blocking=True)  # [NCHW]
            z = torch.randn((bs, 100, 1, 1)).cuda(non_blocking=True)
        # y_real = torch.ones(bs).cuda(args.gpu, non_blocking=True)
        # y_fake = torch.zeros(bs).cuda(args.gpu, non_blocking=True)

        ############################
        # Train discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        f_real, p_real = netD(x_real)
        device = p_real.device
        label = torch.full((bs, ),
                           real_label,
                           dtype=torch.float,
                           device=device)
        errD_real = criterion(p_real.view(-1), label)
        errD_real.backward()
        D_x = p_real.mean().item()

        x_fake = netG(z)
        label.fill_(fake_label)
        _, p_fake = netD(x_fake.detach())  # Use detach to keep netG from gradgraph here
        errD_fake = criterion(p_fake.view(-1), label)
        errD_fake.backward()
        D_G_z1 = p_fake.mean().item()
        # Add the gradients from the all-real and all-fake batches
        lossD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # Train generator: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        f_fake, p_fake_ = netD(x_fake)
        G_perceptual_loss = criterion(p_fake_.view(-1), label)
        # f_real must be `detach()`, because the gradgraph has been realse during lossD.backward()
        # G_feature_match_loss = torch.abs(f_real.detach().mean(dim=1)-f_fake.mean(dim=1)).mean()
        G_feature_match_loss = (f_fake - f_real.detach()).pow(2).mean()
        lossG = G_perceptual_loss + G_feature_match_loss
        lossG.backward()
        D_G_z2 = p_fake_.mean().item()
        # Update G
        optimizerG.step()

        # Meters update and visualize
        loss_hist.append(lossD.item() + lossG.item())
        if not args.mp_distributed or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                D_G_z1 = p_fake.mean().item()
                D_G_z2 = p_fake_.mean().item()
                tools.train_log(
                    ' Epoch: [{} | {}] iters: {:6d} lossD: {:.3f} lossG: {:.3f}({:.3}+{:.3}) '.format(
                        epoch, args.mepoch, i, lossD.item(),
                        lossG.item(), G_perceptual_loss.item(), G_feature_match_loss.item()
                    ) + 'D(x): {:.4f} D(G(z)): {:.4f}/{:.4f} Time: {:.2f}s\r'.format(
                        D_x, D_G_z1, D_G_z2, time.time() - tic), '\t')
                tic = time.time()  # update time
    if args.vis and (epoch % max(args.mepoch // 10, 1) == 0) and (
            not args.mp_distributed or args.gpu == 0):
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        with torch.no_grad():
            val_seed = np.random.RandomState(2020)
            fixed_noise = torch.from_numpy(val_seed.randn(bs, 100, 1, 1).astype(np.float32)).cuda(args.gpu)
            x_fake = netG(fixed_noise)
            x_real = transform.unnormalize(x_real, args.mean, args.std, True)
            x_fake = transform.unnormalize(x_fake, args.mean, args.std, True)

            for j, (real, fake) in enumerate(zip(x_real, x_fake)):
                real = Image.fromarray(real[:, :, :3], 'RGB')
                fake = Image.fromarray(fake[:, :, :3], 'RGB')

                real.save(vis_dir+'/Real_%d.jpg' % j)
                fake.save(vis_dir+'/Fake_%d.jpg' % j)

                if j > 32:
                    break
    return np.mean(loss_hist)


def jigsaw_train(train_loader, model, criterion, optimizer, epoch, args):
    ''' One epoch training use inpainting pretext task. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    tic = time.time()
    # all_perm = np.load('data/permutations_hamming_max_1000.npy')
    vis_dir = VIS_DIR + args.exp_name

    if args.preload:
        train_loader = tools.DataPrefetcher(train_loader, args)
    else:
        train_loader = iter(train_loader)

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        input, target = data['image'], data['label']

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)  # [NCHW]
        else:
            input = input.cuda(non_blocking=True)  # [NCHW]
        target = target.cuda(args.gpu, non_blocking=True)
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if not args.mp_distributed or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                predict = output.argmax(1)
                acc = predict.eq(target).float().mean().cpu().numpy()
                tools.train_log(
                    ' Epoch: [{} | {}] iters: {:6d} loss: {:.3f} acc: {:.3f} Time: {:.2f}s\r'.format(
                        epoch, args.mepoch, i, loss.item(), acc, time.time() - tic), '\t')
                tic = time.time()  # update time
    if args.vis and (epoch % max(args.mepoch // 10, 1) == 0) and (
            not args.mp_distributed or args.gpu == 0):
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        with torch.no_grad():
            nImages = data['org_img'].numpy()
            nPatches = data['image_']
            predict = output.argmax(1)
            puzzle_size = len(nPatches)
            for n in range(puzzle_size):
                nPatches[n] = transform.unnormalize(nPatches[n], args.mean, args.std, True)

            for j, (image, pre) in enumerate(zip(nImages, predict)):
                patchs = [nPatches[t][j] for t in range(puzzle_size)]
                order = train_loader.dataset.permutations[pre]
                patchs_pre = [patchs[t] for t in order]
                if puzzle_size == 9:
                    puzzle = np.vstack([
                        np.hstack([patchs[0], patchs[1], patchs[2]]),
                        np.hstack([patchs[3], patchs[4], patchs[5]]),
                        np.hstack([patchs[6], patchs[7], patchs[8]]),
                    ])
                    reorder = np.vstack((
                        np.hstack([patchs_pre[0], patchs_pre[1], patchs_pre[2]]),
                        np.hstack([patchs_pre[3], patchs_pre[4], patchs_pre[5]]),
                        np.hstack([patchs_pre[6], patchs_pre[7], patchs_pre[8]]),
                    ))
                elif puzzle_size == 4:
                    puzzle = np.vstack([
                        np.hstack([patchs[0], patchs[1]]),
                        np.hstack([patchs[2], patchs[3]])
                    ])
                    reorder = np.vstack([
                        np.hstack([patchs_pre[0], patchs_pre[1]]),
                        np.hstack([patchs_pre[2], patchs_pre[3]])
                    ])

                img1 = Image.fromarray(image[:, :, :3], 'RGB').resize(reorder.shape[:2])
                img2 = Image.fromarray(puzzle[:, :, :3], 'RGB')
                img3 = Image.fromarray(reorder[:, :, :3], 'RGB')

                img1.save(vis_dir+'/Org_%d.jpg' % j)
                img2.save(vis_dir+'/Puzzle_%d.jpg' % j)
                img3.save(vis_dir+'/Pre_%d.jpg' % j)

                if j > 32:
                    break
    return np.mean(loss_hist)


def SimCLR_train(train_loader, model, criterion, optimizer, epoch, args):
    ''' One epoch training use SimCLR. '''
    model.train()
    loss_hist = deque(maxlen=len(train_loader))
    if args.mode in [912, 913]:
        cls_num = args.num_classes  # True class_num
    elif args.mode in [914, 915]:
        cls_num = args.moco_km[0]  # num of cluster

    data_loader = tools.package_loader(train_loader, args)

    tic = time.time()
    for i in range(len(train_loader)):
        data = data_loader.next()
        if data is None:
            break

        optimizer.zero_grad()
        input_1, input_2 = data['image']
        if args.gpu is not None:
            input_1 = input_1.cuda(args.gpu, non_blocking=True)  # [NCHW]
            input_2 = input_2.cuda(args.gpu, non_blocking=True)  # [NCHW]
        else:
            input_1 = input_1.cuda(non_blocking=True)  # [NCHW]
            input_2 = input_2.cuda(non_blocking=True)  # [NCHW]

        target = None
        if args.mode in [912, 913, 914, 915]:
            target = data['label']
            # target = torch.nn.functional.one_hot(target, cls_num).type(torch.float32)
            target = torch.zeros(input_1.shape[0], cls_num).scatter_(
                1, target.view(-1, 1), 9)
            target = target.cuda(args.gpu, non_blocking=True)

        _, out_1 = model(input_1)  # out_1: z_i
        _, out_2 = model(input_2)  # out_2: z_j

        loss = criterion({'zi': out_1, 'zj': out_2}, target)
        loss.backward()
        optimizer.step()

        # Meters update and visualize
        loss_hist.append(loss.item())
        if not args.mp_distributed or args.gpu == 0:
            if i % (len(train_loader) // args.print_freq + 1) == 0:
                tools.train_log(
                    ' Epoch: [{} | {}] iters: {:6d} loss: {:.3f} ({:.3f}) Time: {:.2f}s\r'.format(
                        epoch, args.mepoch, i, loss.item(), np.mean(loss_hist), time.time() - tic), '\t')
                tic = time.time()  # update time
                # wandb.log({"Train/Train_Loss": loss})

    return np.mean(loss_hist)


def log_data_for_ss(input_1, input_2, epoch, args, vis_dir=None, local=False):
    if local:
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
    with torch.no_grad():
        input_1 = transform.unnormalize(input_1, args.mean, args.std, True)
        input_2 = transform.unnormalize(input_2, args.mean, args.std, True)
        for j, (img1, img2) in enumerate(zip(input_1, input_2)):
            img1 = Image.fromarray(img1[:, :, :3], 'RGB')
            img2 = Image.fromarray(img2[:, :, :3], 'RGB')
            if local:
                img1.save(vis_dir+'/branch1_%d.jpg' % j)
                img2.save(vis_dir+'/branch2_%d.jpg' % j)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    # main()
    pass
