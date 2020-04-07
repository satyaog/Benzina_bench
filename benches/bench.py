import argparse
from collections import namedtuple
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models

Measure = namedtuple("Measure", ["data_loader", "workers", "seed", "epoch",
                                 "batch_size", "batches_cnt", "batch_time",
                                 "waiting_for_data_time", "train_time", "data_path"])

def build_parser():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
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
    parser.add_argument('--batch-sync', default=False, action='store_true',
                        help='Synchronize steps in batches and give the details '
                             'about the batch timing. Otherwise, only the epoch '
                             'time will be gathered. (default: False)')
    return parser

def main_worker(make_dataloaders, loader_method, args):
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

    with open("measures.csv", 'a') as measures_file:
        measures_file.write(','.join(Measure._fields) + "\n")

    train_loader, valid_loader = make_dataloaders(args)
    train_func = train_sync if args.batch_sync else train_async

    for epoch in range(args.epochs):
        # train for one epoch
        batch_time, waiting_for_data_time, train_time = \
            train_func(train_loader, model, criterion, optimizer, epoch, args)

        with open("measures.csv", 'a') as measures_file:
            measures_file.write(','.join([str(field) for field in
                Measure(data_loader=loader_method, workers=args.workers, seed=args.seed,
                        epoch=epoch, batch_size=args.batch_size, batches_cnt=batch_time.count,
                        batch_time=batch_time.avg, waiting_for_data_time=waiting_for_data_time.avg,
                        train_time=train_time.avg, data_path=args.data)]) + '\n')

def train_sync(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    waiting_for_data_time = AverageMeter()
    train_time = AverageMeter()

    # switch to train mode
    model.train()

    batch_generator = iter(train_loader)
    batches_cnt = args.batches if args.batches else len(train_loader) - 10

    # warm-up
    for i in range(10):
        input, target = next(batch_generator)

    begin = time.time()
    end = begin
    input, target = None, None
    for i in range(batches_cnt):
        try:
            input, target = next(batch_generator)
        except StopIteration:
            break

        input = input.cuda(args.gpu)
        target = target.cuda(args.gpu)

        torch.cuda.synchronize()

        # measure waiting for data time
        waiting_for_data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        train_time.update(batch_time.val - waiting_for_data_time.val)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Waiting for data {waiting_for_data_time.val:.3f} '
                  '({waiting_for_data_time.avg:.3f})\t'
                  'Training time {train_time.val:.3f} '
                  '({train_time.avg:.3f})\t'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          waiting_for_data_time=waiting_for_data_time,
                          train_time=train_time))

        end = time.time()

    # torch.cuda.synchronize()
    # end = time.time()
    # print('Epoch time: {time:.3f}'.format(time=end-begin))

    return batch_time, waiting_for_data_time, train_time

def train_async(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    waiting_for_data_time = AverageMeter()
    train_time = AverageMeter()

    # switch to train mode
    model.train()

    batch_generator = iter(train_loader)
    batches_cnt = args.batches if args.batches else len(train_loader) - 10

    # warm-up
    # for i in range(10):
    #     next(batch_generator)

    begin = time.time()
    end = begin
    input, target = None, None
    for i in range(batches_cnt):
        try:
            input, target = next(batch_generator)
        except StopIteration:
            batch_time.count = i
            break
    # batches_cnt = 0
    # for input, target in train_loader:
    #     # input = input.cuda(args.gpu, non_blocking=True)
    #     # target = target.cuda(args.gpu, non_blocking=True)
        input = input.cuda(args.gpu)
        # target = target.cuda(args.gpu)
        # batches_cnt += 1

        # # compute output
        # output = model(input)
        # loss = criterion(output, target)
        #
        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    else:
        batch_time.count = batches_cnt

    # torch.cuda.synchronize()

    end = time.time()
    print('Epoch time: {time:.3f}'.format(time=end-begin))

    batch_time.sum = end-begin
    batch_time.avg = batch_time.sum / batch_time.count if batch_time.count else 0
    batch_time.val = batch_time.avg

    return batch_time, waiting_for_data_time, train_time

# def train_async(train_loader, model, criterion, optimizer, epoch, args):
#     print("train_async")
#
#     batch_time = AverageMeter()
#     waiting_for_data_time = AverageMeter()
#     train_time = AverageMeter()
#
#     # switch to train mode
#     model.train()
#
#     batch_generator = iter(train_loader)
#     batches_cnt = args.batches if args.batches else len(train_loader) - 10
#     print("batches_cnt=[{}]".format(batches_cnt))
#
#     # warm-up
#     for i in range(10):
#         next(batch_generator)
#
#     begin = time.time()
#     end = begin
#     input, target = None, None
#     for i in range(batches_cnt):
#         try:
#             input, target = next(batch_generator)
#         except StopIteration:
#             batch_time.count = i
#             break
#         input = input.cuda(args.gpu)
#         target = target.cuda(args.gpu)
#
#         # compute output
#         output = model(input)
#         loss = criterion(output, target)
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     else:
#         batch_time.count = batches_cnt
#
#     torch.cuda.synchronize()
#
#     end = time.time()
#     print('Epoch time: {time:.3f}'.format(time=end-begin))
#
#     batch_time.sum = end-begin
#     batch_time.avg = batch_time.sum / batch_time.count if batch_time.count else 0
#     batch_time.val = batch_time.avg
#
#     return batch_time, waiting_for_data_time, train_time

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
