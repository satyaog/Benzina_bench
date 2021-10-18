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

Measure = namedtuple("Measure", ["data_loader", "model", "workers", "dl_only",
                                 "sequence", "seed", "epoch", "batch_size",
                                 "batches_cnt", "chan_cnt", "epoch_time",
                                 "batch_time", "data_time", "train_time",
                                 "data_path"])


def build_parser():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
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
    parser.add_argument('--batches', default=100, type=int, metavar='N',
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
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. (default: 0)')
    parser.add_argument('--dl-only', action='store_true',
                        help='Mesure dataloader only')
    parser.add_argument('--sequence', action='store_true',
                        help='Load images in sequences')
    return parser


def main_worker(make_dataloader, dataloader_name, args):
    dataloader_name = "{} - {}".format(dataloader_name, args.version)

    if args.seed is None:
        args.seed = random.randint(1, 9999)

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

    is_header_match = False
    header = ','.join(Measure._fields) + '\n'

    if os.path.exists("measures.csv"):
        with open("measures.csv", 'r') as measures_file:
            for line in measures_file:
                if line == header:
                    is_header_match = True
                    break

    if not is_header_match:
        with open("measures.csv", 'a') as measures_file:
            measures_file.write(header)

    train_loader = make_dataloader(args)

    try:
        chan_cnt = args.chan_cnt
    except AttributeError:
        chan_cnt = "N/A"

    for epoch in range(args.epochs):
        # train for one epoch
        epoch_time, batch_time, data_time, train_time = \
            train(train_loader, model, criterion, optimizer, epoch, args)

        with open("measures.csv", 'a') as measures_file:
            measures_file.write(','.join([str(field) for field in
                Measure(data_loader=dataloader_name, model=args.arch,
                        workers=args.workers, dl_only=args.dl_only,
                        sequence=args.sequence, seed=args.seed, epoch=epoch,
                        batch_size=args.batch_size,
                        batches_cnt=batch_time.count, chan_cnt=chan_cnt,
                        epoch_time=epoch_time.avg, batch_time=batch_time.avg,
                        data_time=data_time.avg, train_time=train_time.avg,
                        data_path=args.data)]) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    epoch_time = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_time = AverageMeter()

    # switch to train mode
    model.train()

    begin = time.time()
    batch_iterator = iter(train_loader)
    print(f"Create loader iterator: {time.time() - begin}")
    try:
        len_loader = len(train_loader)
    except TypeError:
        len_loader = "NA"

    batches_cnt = args.batches if args.batches else None

    # warm-up
    begin = time.time()
    for i in range(10):
        next(batch_iterator)
    print(f"Warm-up: {time.time() - begin}")

    begin = time.time()
    end = begin
    for i, (images, target) in enumerate(batch_iterator):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        target = target.view((-1,))

        if not args.dl_only:
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
                          data_time=data_time,
                          train_time=train_time))

        if batches_cnt is not None and i >= batches_cnt:
            break

    torch.cuda.synchronize()
    epoch_time.update(time.time() - begin)
    print('Epoch time: {time:.3f}'.format(time=epoch_time.avg))

    return epoch_time, batch_time, data_time, train_time


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
