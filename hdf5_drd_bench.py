import io
import random

import h5py
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from jug import TaskGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from bench import main_worker, build_parser

MAX_CHAN_CNT = 91


class HDF5Dataset(Dataset):
    def __init__(self, path, channels=tuple(range(MAX_CHAN_CNT)), transform=None,
                 target_transform=None):
        self.fname = path
        self.channels = sorted(channels)
        self.transform = transform
        self.target_transform = target_transform
        self._file = None
        self._datasets = self.find_datasets()

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.fname, 'r')
        return self._file

    def __getitem__(self, index):
        dataset = self.file[self._datasets[index // 3]]
        i = index % 3
        sample = dataset[i, self.channels, ...]
        target = dataset.attrs["labels"][i]
        roi_bb = dataset.attrs["roi_bounding_box"][i]
        sample = sample[:, roi_bb[0]:roi_bb[0]+roi_bb[2], roi_bb[1]:roi_bb[1]+roi_bb[3]]
        sample = torch.tensor(sample, dtype=torch.float)

        if self.transform is not None:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=torch.float16)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self._datasets) * 3

    def find_datasets(self):
        h5f = h5py.File(self.fname, 'r')
        instances = []
        for fold in h5f.values():
            for dataset in fold.values():
                instances.append(dataset.name)
        return instances


def make_dataloader(args):
    # Data loading code
    # Dataset
    channels = random.sample(list(range(MAX_CHAN_CNT)), args.chan_cnt)
    train_set = HDF5Dataset(args.data, channels,
            transform=transforms.Compose([
                transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation([-180., +180.])
            ]))

    # Dataloaders
    train_sampler = torch.utils.data.SequentialSampler \
        if args.sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True)

    return train_loader


@TaskGenerator
def main(args):
    main_worker(make_dataloader, "HDF5", args)


parser = build_parser()
parser.add_argument("-c", "--chan-cnt", default=MAX_CHAN_CNT, type=int,
                    metavar="N",
                    help=f"channels count (default: {MAX_CHAN_CNT}), this is "
                          "the number of channels to be loaded from the image "
                          "file")
main(parser.parse_args())
