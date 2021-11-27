import csv
import io
import os
import random

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from bcachefs import Bcachefs, Cursor
from bcachefs.bcachefs import DirEnt
from jug import TaskGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from bench import main_worker, build_parser

MAX_FOLDS_CNT = 3
MAX_PATIENTS_PER_FOLDS = 65
MAX_IMGS_PER_PATIENT = 3
MAX_CHAN_CNT = 91


def CachedFunction(f, *args, **kwargs):
    from jug import CachedFunction as _CachedFunction
    if isinstance(f, TaskGenerator):
        return _CachedFunction(f.f, *args, **kwargs)
    else:
        return _CachedFunction(f, *args, **kwargs)


def read_array(cursor: Cursor, inode, fmt):
    if isinstance(inode, DirEnt):
        inode = inode.inode
    (dtype, _, chunk) = fmt
    count = 1
    for d in chunk:
        count *= d
    return np.frombuffer(cursor.read_file(inode),
                         dtype=dtype, count=count).reshape(chunk)


def read_format(cursor: Cursor, inode):
    if isinstance(inode, str):
        inode = cursor.find_dirent(inode).inode
    attrs = {}
    fmt = bytes(cursor.read_file(inode)).split(b'\n')
    fmt = (fmt[0], [int(d) for d in fmt[1].split(b',')],
           [int(d) for d in fmt[2].split(b',')])
    return fmt


def read_attrs(cursor: Cursor):
    attrs = {}
    for dirent in cursor.ls(".attrs"):
        if dirent.is_file:
            continue
        try:
            fmt = read_format(cursor, os.path.join(".attrs", dirent.name, "format"))
            attr_dirent = cursor.find_dirent(os.path.join(".attrs", dirent.name, "arr.rec"))
            attrs[dirent.name] = read_array(cursor, attr_dirent, fmt)
        except ValueError:
            pass
    return attrs


class BcachefsDataset(Dataset):
    def __init__(self, cursor: Cursor, channels=tuple(range(MAX_CHAN_CNT)),
                 transform=None, target_transform=None):
        self._cursor = cursor
        self._channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self._samples = self.find_samples()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        if self._cursor.closed:
            self._cursor.open()

        (samples, (dtype, shape, chunk)), (target, roi_bbs) = self._samples[index]
        sample_idx = random.randint(0, len(samples) - 1)
        sample = samples[sample_idx]
        roi_bb = roi_bbs[sample_idx]
        count = 1
        for d in chunk:
            count *= d
        sample = np.stack([np.frombuffer(self._cursor.read_file(sample[c].inode),
                                         dtype=dtype, count=count).reshape(chunk)
                           for c in self._channels])
        sample = sample[:, roi_bb[0]:roi_bb[0]+roi_bb[2], roi_bb[1]:roi_bb[1]+roi_bb[3]]

        if self.transform is not None:
            sample = torch.tensor(sample, dtype=torch.float)
            sample = self.transform(sample)
            sample = sample.to(torch.float16, non_blocking=True)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def find_samples(self):
        instances = []
        cwd = self._cursor.pwd
        with self._cursor as cursor:
            for root, _, files in sorted(cursor.walk()):
                fmt_dirent = next((f for f in files if f.name == "format"), None)
                if fmt_dirent is None or ".attrs" in root:
                    continue
                fmt = read_format(cursor, fmt_dirent.inode)
                chunks = sorted((_f for _f in files if _f is not fmt_dirent),
                                key=lambda _f: _f.name)
                cursor.cd(root)
                attrs = read_attrs(cursor)
                cursor.cd(cwd)

                num_imgs, num_chan = fmt[1][0:2]
                label = attrs.get("labels", [None])[0]
                roi_bb = attrs["roi_bounding_box"]
                items = []
                for i in range(num_imgs):
                    chunk_i = i*num_chan
                    items.append(chunks[chunk_i:chunk_i+num_chan])
                instances.append(((items, fmt), (label, roi_bb)))

        return instances


def get_bchfs(filename):
    bchfs = Bcachefs(filename)
    bchfs.open()
    bchfs.close()
    return bchfs


def make_dataloader(args):
    # Data loading code
    # Dataset
    channels = random.sample(list(range(MAX_CHAN_CNT)), args.chan_cnt)
    with CachedFunction(get_bchfs, args.data) as bchfs:
        train_set = BcachefsDataset(bchfs.cd(), channels,
                transform=transforms.Compose([
                    transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation([-180., +180.])
                ]))
        val_set = BcachefsDataset(bchfs.cd(), channels)
    train_set = torch.utils.data.Subset(
            train_set, range((MAX_FOLDS_CNT - 1) * MAX_PATIENTS_PER_FOLDS))
    val_set = torch.utils.data.Subset(
            val_set, range(len(train_set),
                           MAX_FOLDS_CNT * MAX_PATIENTS_PER_FOLDS))

    # Dataloaders
    train_sampler = torch.utils.data.SequentialSampler \
        if args.sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    return train_loader, val_loader


@TaskGenerator
def main(args):
    main_worker(make_dataloader, "Bcachefs", args)


parser = build_parser()
parser.add_argument("-c", "--chan-cnt", default=MAX_CHAN_CNT, type=int,
                    metavar="N",
                    help=f"channels count (default: {MAX_CHAN_CNT}), this is "
                          "the number of channels to be loaded from the image "
                          "file")
main(parser.parse_args())
