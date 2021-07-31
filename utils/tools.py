'''
Pytorch basic tools for main.

'''
import math
import sys
import datetime

# import cv2
import numpy as np
import torch


def quick_val_c(model, dataloader, args, print_result=False):
    ''' 验证集Accuracy(Patch-level classification) '''
    model.eval()
    runscore = ClassifyScore(args.num_classes, args.category.names)
    with torch.no_grad():
        for ii, data in enumerate(dataloader):
            input, target = data['image'], data['label']
            if isinstance(target, torch.Tensor):
                target = target.numpy()
            target = target.astype(np.int64)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            else:
                input = input.cuda(non_blocking=True)
            predict = model(input).argmax(1)
            runscore.update(target, predict.cpu().numpy().astype(target.dtype))
        score = runscore.get_scores()

    if print_result:
        runscore.print_score(score)
    return score


def fast_hist(y_true, y_pred, n_class):
    """ Computational confusion matrix.
    -------------------------------------------
    |          | p_cls_1 | p_cls_2 |   ....   |
    -------------------------------------------
    | gt_cls_1 |         |         |          |
    -------------------------------------------
    | gt_cls_2 |         |         |          |
    -------------------------------------------
    |   ....   |         |         |          |
    -------------------------------------------
    """
    # mask = (y_true >= 0) & (y_true < n_class)
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    hist = np.bincount(
        n_class * y_true.astype(int) + y_pred,
        minlength=n_class ** 2, ).reshape(n_class, n_class)
    return hist


class runingScore(object):
    """ Evaluation class.

    Args: (Specify one of the following two parameters is Ok)
        n_classes: (int) Number of categories.
        target_names: A string list of category names.
    """
    def __init__(self, n_classes=2, target_names=None):
        if target_names is None:
            self.n_classes = n_classes
            self.target_names = [str(c) for c in range(n_classes)]
        else:
            self.target_names = target_names
            self.n_classes = len(target_names)

        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def reset(self):
        """ Reset confusion_matrix. """
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update_all(self, y_trues, y_preds):
        """ Add new pairs of predicted label and GT label to update the confusion_matrix.
        Note: Only suitable for segmentation
        """
        for lt, lp in zip(y_trues, y_preds):
            self.confusion_matrix += fast_hist(lt, lp, self.n_classes)

    def print_hist(self, hist=None):
        """ Print the confusion matrix in markdown table style.

        Args:
            class_table: Dict of class_name and num, {'BG': 0, 'Road': 1, ...}.
            hist: Confusion_matrix array.
        """
        hist = self.confusion_matrix if hist is None else hist

        form = '|  | '
        # Form title row and second row
        for name in self.target_names:
            form += ' %s |' % name
        second_row = '| -: |' + ' -: |' * self.n_classes + '\n'
        form += '\n' + second_row

        # Rest rows
        for i in range(self.n_classes):
            form += '| %s |' % self.target_names[i]
            for j in range(self.n_classes):
                form += ' %d |' % hist[i, j]
            form += '\n'

        print(form)

    def print_score_in_md(self, score, cls_header, ave_header, digits=4):
        """ Print the score dict in markdown style. """

        form = ''
        # 1. Class-wise evaluation scores
        ind_num = len(cls_header)  # num of indicators (inducding cls names)
        title_fmt = '|  |' + ' {} |' * ind_num + ' {cat} |\n'  # Title row
        form += title_fmt.format(*cls_header, cat='category')
        form += '| -: |' + ' -: |' * (ind_num + 1) + '\n'  # Second row

        row_fmt = '| {} |' + ' {:.{digits}f}|' * ind_num + ' {} |\n'
        rows_conttents = [score[key] for key in cls_header]
        indexs = range(self.n_classes)
        rows_conttents.insert(0, indexs)  # first col is indexs
        rows_conttents.append(self.target_names)  # last col is category names
        for i, row in enumerate(zip(*rows_conttents)):
            form += row_fmt.format(*row, digits=digits)

        # Overall evaluation
        for head in ave_header:
            form += '| {} | {:.{digits}f} |\n'.format(head, score[head], digits=digits)
        print(form)

        return form


class ClassifyScore(runingScore):
    """ Accuracy evaluation for classification(multi-class)"""
    def update(self, y_true, y_pred, step_score=False):
        """Evaluate a new pair of predicted label and GT label,
        and update the confusion_matrix."""
        hist = fast_hist(y_true, y_pred, self.n_classes)
        self.confusion_matrix += hist
        if step_score:
            return self.get_scores(hist)

    def get_scores(self, hist=None):
        """Returns accuracy score evaluation result.
            'hist': Computational confusion matrix.

            'precision': precision of per (User accuracy)

            'recall': recall of per category  (Producer accuracy)

            'f1-score': f1-score of per category

            'OA': Overall accuracy (micro average in sklearn, averaging the
            total true positives, false negatives and false positives)

            'AA': Average accuracy (macro average in sklearn, averaging the
            unweighted mean per label)

            'Kappa': Cohen’s kappa, a statistic that measures inter-annotator agreement
        """
        hist = self.confusion_matrix if hist is None else hist

        # Class-wise evaluation
        TP = np.diag(hist)  # class-wise TP
        TPFP = hist.sum(axis=0)  # class-wise TP + FP; (row)
        TPFN = hist.sum(axis=1)  # class-wise TP + FN; (col)
        precision = TP / (TPFP + 1e-8)  # TP / (TP + FP)
        recall = TP / (TPFN + 1e-8)  # TP / (TP + FN)
        f1_score = 2 * TP / (TPFN + TPFP + 1e-8)  # 2TP / (2TP + FP + FN)

        # Overall evaluation
        n = hist.sum()
        p0 = TP.sum()

        micro_precision = p0 / (n+1e-8)  # Overall accuracy
        macro_precision = np.nanmean(precision)  # Average accuracy
        kappa = (n*p0-np.inner(TPFP, TPFN)) / (n*n - np.inner(TPFP, TPFN) + 1e-8)  # Kappa

        return ({
            "hist": hist,  # confusion matrix

            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,

            "OA": micro_precision,
            "AA": macro_precision,
            "Kappa": kappa,
        })  # Return as a dictionary

    def print_score(self, score, digits=4):
        """ Print the scores in Markdown style.

        Args:
            score: A dict of all evaluation scores.
            digits: Number of digits for formatting output floating point values.
        """
        cls_header = ['precision', 'recall', 'f1-score']
        avg_header = ['OA', 'AA', 'Kappa']
        p_score = runingScore.print_score_in_md(self, score, cls_header, avg_header, digits)
        return p_score


# **************************************************
# ***************** Training ***********************
# **************************************************
def train_log(X, prefix='', f=None):
    """ Print with time. To console or a file(f) """
    time_stamp = datetime.datetime.now().strftime("[%d %H:%M:%S]")
    if f is not None:
        if type(f) == str:
            with open(f, 'a') as f:
                f.write(time_stamp + " " + X)
        else:
            f.write(time_stamp + " " + X)

    sys.stdout.write(prefix + time_stamp + " " + X)
    sys.stdout.flush()


def set_optimizer(opt, model):
    if opt.hos == 1:
        params = [{'params': model.classifier.parameters()},
                  {'params': model.feature.parameters(), 'lr': opt.lr/5}, ]
    else:
        params = model.parameters()
    params = list(filter(lambda p: p.requires_grad, params))
    print('Optimize params num: ', len(params))

    # Setup optimizer
    if opt.optimizer == 'sgd':
        moptimizer = torch.optim.SGD(
            params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        moptimizer = torch.optim.Adam(
            params, lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.9, 0.99))

    return moptimizer


def place_model(model, args):
    if args.distributed:
        if args.mp_distributed:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model = model.cuda(args.gpu)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.ngpus_per_node)])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model.cuda()

    return model


class CudaDevices():
    """ A simple class to know about your cuda devices

    Args:
        param1:  param1 specification

    Attributes:
        self.gpu_num (int): Total GPU num
        self.avail_num (int): Available GPU num (memory occupied < 100M)
        self.avail_ids (list of int): Available GPU Id.

    """
    def __init__(self, ):
        self.gpu_num = torch.cuda.device_count()  # Total GPU num
        self.avail_ids = []
        print("%d device(s) found:" % self.gpu_num)
        for i in range(self.gpu_num):
            memory_occupied = torch.cuda.memory_allocated(i) / 1024 / 1024
            print('\t%s (Id %d): current GPU memory occupied = %.2f M' % (
                torch.cuda.get_device_name(i), i, memory_occupied))
            if memory_occupied < 100:
                self.avail_ids.append(i)

        self.avail_num = len(self.avail_ids)

        print('Total GPU num = %d, availiable GPU num = %d' % (
            self.gpu_num, self.avail_num))

    def total_gpu(self):
        """返回 cuda 设备的总数量"""
        return torch.cuda.device_count()

    def total_ids_str(self):
        '''以 a string 返回所有设备ids'''
        num = torch.cuda.device_count()
        if num > 0:
            ids = ''
            for i in range(num):
                ids += str(i) + ','
            return ids[:-1]
        else:
            return None

    def total_ids_list(self):
        '''以 list of int 形式返回所有设备ids'''
        num = torch.cuda.device_count()
        if num > 0:
            return [i for i in range(num)]
        else:
            return None

    def avail_ids_str(self):
        '''以 a string 返回可用设备ids'''
        if len(self.avail_ids) > 0:
            ids = ''
            for i in self.avail_ids:
                ids += str(i) + ','
            return ids[:-1]
        else:
            return None

    def avail_ids_list(self):
        '''以 list of int 形式返回可用设备ids'''
        if self.avail_num > 0:
            return [i for i in range(self.avail_num)]
        else:
            return None

    def devices(self, vis=None):
        """获取所有可用的设备的名称"""
        self.gpu_num = torch.cuda.device_count()
        self.avail_ids = []
        log_str = ''
        log_str += "%d device(s) found:\n" % self.gpu_num
        for i in range(self.gpu_num):
            memory_occupied = torch.cuda.memory_allocated(i) / 1024 / 1024
            log_str += '\t%s (Id %d): current GPU memory occupied = %.2f M\n' % (
                torch.cuda.get_device_name(i), i, memory_occupied)
            if memory_occupied < 100:
                self.avail_ids.append(i)

        self.avail_num = len(self.avail_ids)
        log_str += 'Total GPU num = %d, availiable GPU num = %d\n' % (
            self.gpu_num, self.avail_num)

        if vis is not None:
            vis.log('\n' + log_str, if_print=True)
        else:
            print(log_str)


class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        # with torch.cuda.stream(self.stream):
        #     for k in self.next_batch:
        #         if k != 'meta':
        #             self.next_batch[k] = self.next_batch[k].to(device=self.opt.gpu, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch = self.next_batch
        self.preload()
        return next_batch


def package_loader(dataloader, args):
    if args.preload:
        dataloader = DataPrefetcher(dataloader, args)
    else:
        dataloader = iter(dataloader)
    return dataloader


def adjust_learning_rate(optimizer, cepoch, args, lr=None):
    """Decay the learning rate based on schedule"""
    lr = args.lr if lr is None else lr

    def _cal_coefficient(epoch):
        if epoch < args.warmup:
            coefficient = (epoch + 1) / args.warmup  # warmup
        else:
            if args.lr_policy == 'cosine':  # cosine lr schedule
                coefficient = 0.5 * (1. + math.cos(math.pi * epoch / args.mepoch))
            elif args.lr_policy == 'linear':
                coefficient = 1 - (epoch-args.warmup) / (args.mepoch-args.warmup)
            elif args.lr_policy == 'step':  # stepwise lr schedule
                for milestone in args.milestones:
                    coefficient = args.lr_decay[1] if epoch >= milestone else 1.
            elif args.lr_policy == 'polyline':  # stepwise lr schedule
                if epoch > args.milestones[0]:
                    # start decay
                    coefficient = (args.mepoch - epoch)/(args.mepoch - args.milestones[0])
        return coefficient

    last_epoch = max(1, cepoch-1)
    last_coeff = _cal_coefficient(last_epoch)
    cur_coeff = _cal_coefficient(cepoch)
    for param_group in optimizer.param_groups:
        base_lr = param_group['lr'] / last_coeff
        new_lr = base_lr * cur_coeff
        param_group['lr'] = new_lr

    return lr * cur_coeff  # main base lr (no hos)
