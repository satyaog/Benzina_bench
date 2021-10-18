import io
import random

import h5py
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from jug import TaskGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from bench import main_worker, build_parser

MAX_CHAN_CNT = 91


class HDF5Dataset(Dataset):
    def __init__(self, path, channels=MAX_CHAN_CNT, transform=None,
                 target_transform=None):
        self.fname = path
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self._file = None
        self._len = h5py.File(self.fname, 'r')['images'].shape[0]

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.fname, 'r')
        return self._file

    def __getitem__(self, index):
        sample = self.file["images"][index][self.channels, ...]
        try:
            target = self.file["labels"][index]
        except:
            target = -1

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self._len


def make_dataloader(args):
    # Data loading code
    # Dataset
    channels = random.sample(list(range(MAX_CHAN_CNT)), args.chan_cnt)
    train_set = HDF5Dataset(args.data, channels)
    train_set = torch.utils.data.Subset(train_set, range(256))

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
