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
from jug import TaskGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from bench import main_worker, build_parser

MAX_CHAN_CNT = 91


def CachedFunction(f, *args, **kwargs):
    from jug import CachedFunction as _CachedFunction
    if isinstance(f, TaskGenerator):
        return _CachedFunction(f.f, *args, **kwargs)
    else:
        return _CachedFunction(f, *args, **kwargs)


def load_array(cursor: Cursor, inode, fmt):
    (dtype, _, chunk) = fmt
    count = 1
    for d in chunk:
        count *= d
    return np.frombuffer(cursor.read_file(inode),
                         dtype=dtype, count=count).reshape(chunk)


def find_attrs(cursor: Cursor):
    attrs = {}
    for dirent in cursor.ls(".attrs"):
        if dirent.is_file:
            continue
        fmt = bytes(cursor.read_file(os.path.join(".attrs", dirent.name, "format"))).split(b'\n')
        fmt = (fmt[0], [int(d) for d in fmt[1].split(b',')],
               [int(d) for d in fmt[2].split(b',')])
        attr_dirent = cursor.find_dirent(os.path.join(".attrs", dirent.name, "array.rec"))
        attrs[dirent.name] = attr_dirent, fmt
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

        sample, (dtype, shape, chunk), (i, attrs) = self._samples[index]
        count = 1
        for d in chunk:
            count *= d
        sample = np.stack([np.frombuffer(self._cursor.read_file(sample[c].inode),
                                         dtype=dtype, count=count).reshape(chunk)
                           for c in self._channels])
        label_dirent, label_fmt = attrs["labels"]
        target = load_array(self._cursor, label_dirent.inode, label_fmt)[i]
        roi_bb_dirent, roi_bb_fmt = attrs["roi_bounding_box"]
        roi_bb = load_array(self._cursor, roi_bb_dirent.inode, roi_bb_fmt)[i]
        sample = sample[:, roi_bb[0]:roi_bb[0]+roi_bb[2], roi_bb[1]:roi_bb[1]+roi_bb[3]]
        sample = torch.tensor(sample, dtype=torch.float)

        if self.transform is not None:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=torch.float16)

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
                fmt = bytes(cursor.read_file(fmt_dirent.inode)).split(b'\n')
                fmt = (fmt[0], [int(d) for d in fmt[1].split(b',')],
                       [int(d) for d in fmt[2].split(b',')])
                chunks = sorted((_f for _f in files if _f is not fmt_dirent),
                                key=lambda _f: _f.name)
                cursor.cd(root)
                attrs = find_attrs(cursor)
                cursor.cd(cwd)
                num_imgs, num_chan = fmt[1][0:2]
                for i in range(num_imgs):
                    item = chunks[i*num_chan:(i+1)*num_chan], fmt, (i, attrs)
                    instances.append(item)

        return instances


def get_dataset(filename, channels):
    with Bcachefs(filename) as bchfs:
        dataset = BcachefsDataset(bchfs.cd(), channels,
                transform=transforms.Compose([
                    transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation([-180., +180.])
                ]))
    return dataset


def make_dataloader(args):
    # Data loading code
    # Dataset
    channels = random.sample(list(range(MAX_CHAN_CNT)), args.chan_cnt)
    train_set = CachedFunction(get_dataset, args.data, channels)

    # Dataloaders
    train_sampler = torch.utils.data.SequentialSampler \
        if args.sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True)

    return train_loader


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
