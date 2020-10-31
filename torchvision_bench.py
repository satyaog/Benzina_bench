import os

import jug
from jug import TaskGenerator
import orion.client
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from bench import main_worker, build_parser


make_dataloaders = {}


def _(data, batch_size, workers, sequence=False, *_args, **_kwargs):
    # Data loading code
    # Dataset
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # Dataloaders
    train_set = datasets.CIFAR10(
        root=data,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    train_sampler = torch.utils.data.SequentialSampler \
        if sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=batch_size,
                              num_workers=workers, pin_memory=True)

    test_set = datasets.CIFAR10(
        root=data,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)

    return train_loader, test_loader


make_dataloaders["cifar10"] = _


def _(data, batch_size, workers, sequence=False, *_args, **_kwargs):
    # Data loading code
    # Dataset
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # Dataloaders
    train_set = datasets.CIFAR100(
        root=data,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    train_sampler = torch.utils.data.SequentialSampler \
        if sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=batch_size,
                              num_workers=workers, pin_memory=True)

    test_set = datasets.CIFAR100(
        root=data,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)

    return train_loader, test_loader


make_dataloaders["cifar100"] = _


def _(data, batch_size, workers, sequence=False, *_args, **_kwargs):
    # Data loading code
    # Dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Dataloaders
    train_set = datasets.ImageNet(
        root=data,
        split="train",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    train_sampler = torch.utils.data.SequentialSampler \
        if sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=batch_size, num_workers=workers,
                              pin_memory=True)

    val_set = datasets.ImageNet(
        root=data,
        split="val",
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)

    return train_loader, val_loader


make_dataloaders["imagenet"] = _


def _(data, batch_size, workers, sequence=False, *_args, **_kwargs):
    # Data loading code
    # Dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Dataloaders
    train_set = datasets.ImageFolder(
        root=os.path.join(data, "train"),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    train_sampler = torch.utils.data.SequentialSampler \
        if sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=batch_size, num_workers=workers,
                              pin_memory=True)

    val_set = datasets.ImageFolder(
        root=os.path.join(data, "val"),
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            normalize
        ]))

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)

    return train_loader, val_loader


make_dataloaders["tinyimagenet"] = _


@TaskGenerator
def main(args):
    data = args.data.rstrip(os.path.sep)
    data = os.path.basename(data).split('_')
    dataset, dataloader = '_'.join(data[:-1]).lower(), data[-1]
    return main_worker(make_dataloaders[dataset], dataloader, args)


parser = build_parser()
acc1, _ = jug.bvalue(main(parser.parse_args()))
orion.client.report_objective(1 - acc1)
