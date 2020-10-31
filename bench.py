# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
from collections import namedtuple
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models

from preactresnet import preactresnet101

Measure = namedtuple("Measure", ["dataloader", "model", "workers", "sequence", "seed",
                                 "epoch", "batch_size", "batches_cnt", "epoch_time",
                                 "batch_time", "data_time", "train_time", "data_path"])

models.__dict__["preactresnet101"] = preactresnet101


def build_parser():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    model_names.append("preactresnet101")
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('version', metavar='VER', help='Data loader version')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batches', default=None, type=int, metavar='N',
                        help='number of total batches to run in an epoch')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--out', metavar='DIR', default='',
                        help='output directory')
    parser.add_argument('--checkpoint', default='checkpoint.pth.tar', type=str,
                        metavar='FILENAME', help='checkpoint file name')
    parser.add_argument('--resume', action='store_true',
                        help='resume from latest checkpoint')
    parser.add_argument('--load-data', action='store_true',
                        help='data loading only')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. (default: 0)')
    parser.add_argument('--sequence', action='store_true',
                        help='load images in sequences')
    return parser


def main_worker(make_dataloaders, dataloader_name, args):
    measures_path = os.path.join(args.out,
                                 "{}_measures.csv".format(dataloader_name))
    dataloader_name = "{} - {}".format(dataloader_name, args.version)
    checkpoint_path = os.path.join(args.out, args.checkpoint)

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(checkpoint_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    is_header_match = False
    header = ','.join(Measure._fields) + '\n'

    if os.path.exists(measures_path):
        with open(measures_path, 'r') as measures_file:
            for line in measures_file:
                if line == header:
                    is_header_match = True
                    break

    if not is_header_match:
        with open(measures_path, 'a') as measures_file:
            measures_file.write(header)

    train_loader, val_loader = make_dataloaders(**vars(args))

    epoch_time = AverageMeter()
    acc1, acc5 = 0, 0
    best_acc1 = 0
    best_epoch = 0
    begin = time.time()
    end = begin
    for epoch in range(args.start_epoch, args.epochs):
        if args.load_data:
            batch_time, data_time, train_time = \
                load_data(train_loader, model, criterion, optimizer, epoch,
                          args)

            torch.cuda.synchronize()
            epoch_time.update(time.time() - end)
            print('Epoch time: {time:.3f}'.format(time=epoch_time.avg))

        else:
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            batch_time, data_time, train_time = \
                train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            # acc1, acc5 = validate(val_loader, model, criterion, args)

            torch.cuda.synchronize()
            epoch_time.update(time.time() - end)
            print('Epoch time: {time:.3f}'.format(time=epoch_time.avg))

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'dataloader': dataloader_name
            }, False, checkpoint_path)

        with open(measures_path, 'a') as measures_file:
            measures_file.write(','.join([str(field) for field in
                Measure(dataloader=dataloader_name, model=args.arch,
                        workers=args.workers, sequence=args.sequence,
                        seed=args.seed, epoch=epoch, batch_size=args.batch_size,
                        batches_cnt=batch_time.count, epoch_time=epoch_time.avg,
                        batch_time=batch_time.avg, data_time=data_time.avg,
                        train_time=train_time.avg, data_path=args.data)]) + \
                                   '\n')

        end = time.time()

    return acc1, acc5


def load_data(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_time = AverageMeter()

    len_loader = len(train_loader)

    batches_cnt = args.batches if args.batches else None
    iter_loader = iter(train_loader)

    begin = time.time()
    end = begin
    for i, (images, target) in enumerate(iter_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        train_time.update(batch_time.val - data_time.val)

        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Waiting for data {data_time.val:.3f} '
                  '({data_time.avg:.3f})\t'
                  'Training time {train_time.val:.3f} '
                  '({train_time.avg:.3f})\t'
                  .format(epoch, i, len_loader, batch_time=batch_time,
                          data_time=data_time, train_time=train_time),
                  flush=True)

        if batches_cnt is not None and i > batches_cnt:
            break

    return batch_time, data_time, train_time

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_time = AverageMeter()

    # switch to train mode
    model.train()

    len_loader = len(train_loader)

    batches_cnt = args.batches if args.batches else None
    iter_loader = iter(train_loader)

    begin = time.time()
    end = begin
    for i, (images, target) in enumerate(iter_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        train_time.update(batch_time.val - data_time.val)

        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Waiting for data {data_time.val:.3f} '
                  '({data_time.avg:.3f})\t'
                  'Training time {train_time.val:.3f} '
                  '({train_time.avg:.3f})\t'
                  .format(epoch, i, len_loader, batch_time=batch_time,
                          data_time=data_time, train_time=train_time),
                  flush=True)

        if batches_cnt is not None and i > batches_cnt:
            break

    return batch_time, data_time, train_time


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    len_loader = len(val_loader)
    iter_loader = iter(val_loader)

    losses_cache = []
    top1_cache = []
    top5_cache = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(iter_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses_cache.append({"val": loss.detach(), "n": images.size(0)})
            top1_cache.append({"val": acc1.detach(), "n": images.size(0)})
            top5_cache.append({"val": acc5.detach(), "n": images.size(0)})

            if i % args.print_freq == 0:
                for l, t1, t5 in zip(losses_cache, top1_cache, top5_cache):
                    losses.update(l["val"].item(), l["n"])
                    top1.update(t1["val"].item(), t1["n"])
                    top5.update(t5["val"].item(), t5["n"])
                losses_cache[:] = []
                top1_cache[:] = []
                top5_cache[:] = []

                print('Val: [{0}/{1}]\t'
                      'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {losses.val:.4e} ({losses.avg:.4e})\t'
                      'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                      'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'
                      .format(i, len_loader, batch_time=batch_time,
                              losses=losses, top1=top1, top5=top5),
                      flush=True)

        for l, t1, t5 in zip(losses_cache, top1_cache, top5_cache):
            losses.update(l["val"].item(), l["n"])
            top1.update(t1["val"].item(), t1["n"])
            top5.update(t5["val"].item(), t5["n"])

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].reshape(-1).to(torch.float32,
                                                   non_blocking=True) \
                .sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
