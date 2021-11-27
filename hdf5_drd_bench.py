import io
import random

import tables as tb
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from jug import TaskGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from bench import main_worker, build_parser

MAX_FOLDS_CNT = 3
MAX_PATIENTS_PER_FOLDS = 65
MAX_IMGS_PER_PATIENT = 3
MAX_CHAN_CNT = 91


class HDF5Dataset(Dataset):
    def __init__(self, path, channels=tuple(range(MAX_CHAN_CNT)), transform=None,
                 target_transform=None):
        self.fname = path
        self.channels = tuple(sorted(channels))
        self.transform = transform
        self.target_transform = target_transform
        self._file = None
        self._samples = self.find_samples()

    @property
    def file(self):
        if self._file is None:
            self._file = tb.File(self.fname, 'r')
        return self._file

    def __getitem__(self, index):
        samples, (target, roi_bbs) = self._samples[index]
        samples = self.file.get_node(samples)
        sample_idx = random.randint(0, samples.shape[0] - 1)
        sample = samples[sample_idx, self.channels]
        roi_bb = roi_bbs[sample_idx]
        sample = sample[:, roi_bb[0]:roi_bb[0]+roi_bb[2], roi_bb[1]:roi_bb[1]+roi_bb[3]]

        if self.transform is not None:
            sample = torch.tensor(sample, dtype=torch.float)
            sample = self.transform(sample)
            sample = sample.to(torch.float16, non_blocking=True)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self._samples)

    def find_samples(self):
        instances = []
        with tb.File(self.fname, 'r') as tbf:
            for array in tbf.walk_nodes("/", "Array"):
                try:
                    label = array._v_attrs.labels[0]
                except AttributeError:
                    label = None
                roi_bb = array._v_attrs.roi_bounding_box
                instances.append((array._v_pathname, (label, roi_bb)))
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
    val_set = HDF5Dataset(args.data, channels)
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
    main_worker(make_dataloader, "HDF5", args)


parser = build_parser()
parser.add_argument("-c", "--chan-cnt", default=MAX_CHAN_CNT, type=int,
                    metavar="N",
                    help=f"channels count (default: {MAX_CHAN_CNT}), this is "
                          "the number of channels to be loaded from the image "
                          "file")
main(parser.parse_args())
