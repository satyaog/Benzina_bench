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


class BcachefsDataset(Dataset):
    def __init__(self, bch_cursor: Cursor, labels_csv=None,
                 channels=MAX_CHAN_CNT, transform=None, target_transform=None):
        self._cursor = bch_cursor
        self._channels = channels
        self._labels_csv = labels_csv
        self.transform = transform
        self.target_transform = target_transform
        self._samples = self.find_samples()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        if self._cursor.closed:
            self._cursor.open()

        sample, target, (dtype, shape) = self._samples[index]
        count = 1
        for d in shape:
            count *= d
        sample = np.stack([np.frombuffer(self._cursor.read_file(sample[c].inode),
                                         dtype=dtype, count=count).reshape(shape)
                           for c in self._channels])
        sample = torch.tensor(sample, dtype=torch.float)

        if self.transform is not None:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=torch.float16)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def find_samples(self):
        if self._labels_csv is not None:
            with self._cursor as cursor, \
                    cursor.read_file(self._labels_csv) as csvf:
                csv_reader = csv.reader(bytes(csvf).rstrip(b"\n\x00").decode().split('\n'))
            # Skip header
            next(csv_reader)
            images, classes = zip(*list(csv_reader))
            classes = [int(c) for c in classes]
            available_images = set([de.name for de in self._cursor.ls() if de.is_dir])
            images, classes = zip(*[(i, c) for i, c in zip(images, classes) if i in available_images])
        else:
            images = [de.name for de in self._cursor.ls() if de.is_dir]
            classes = [-1] * len(images)

        instances = []
        with self._cursor as cursor:
            for image, class_idx in zip(images, classes):
                for _, _, files in sorted(self._cursor.walk(image)):
                    format = bytes(cursor.read_file(os.path.join(image, "format"))).split(b'\n')
                    format = (format[0], [int(d) for d in format[1].split(b',')])
                    item = (sorted((_f for _f in files if _f.name != "format"),
                                   key=lambda _f: _f.name),
                            class_idx, format)
                    instances.append(item)

        return instances


def get_dataset(filename, split, channels):
    with Bcachefs(filename) as bchfs:
        dataset = BcachefsDataset(bchfs.cd(split), "/trainLabels.csv", channels,
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
    train_set = CachedFunction(get_dataset, args.data, "train", channels)

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
