import io

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from bcachefs import BCacheFS, Cursor
from jug import TaskGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from bench import main_worker, build_parser


def pil_loader(b: io.BytesIO):
    return Image.open(b).convert("RGB")


class BCacheFSDataset(Dataset):
    def __init__(self, bch_cursor: Cursor, transform=None,
                 target_transform=None, loader=pil_loader):
        self._cursor = bch_cursor
        self.transform = transform
        self.target_transform = target_transform
        self._loader = loader
        self._file = None
        self._samples = self.find_samples()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        if self._cursor.closed:
            self._cursor.open()

        sample, target = self._samples[index]
        sample = self._loader(self._cursor.open_file(sample.inode))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def find_samples(self):
        classes = [de for de in self._cursor.ls() if de.is_dir]
        class_to_idx = {cls.name: i for i, cls in enumerate(classes)}

        instances = []
        available_classes = set()
        for target_class in classes:
            class_idx = class_to_idx[target_class.name]
            for _, _, files in \
                    sorted(self._cursor.walk(target_class.name)):
                for f in sorted(files, key=lambda _f: _f.name):
                    item = f, class_idx
                    instances.append(item)
                    if target_class.name not in available_classes:
                        available_classes.add(target_class.name)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)

        return instances


def make_dataloader(args):
    # Data loading code
    # Dataset
    if args.dl_only:
        _transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        _transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    # Dataloaders
    with BCacheFS(args.data) as bchfs:
        train_set = BCacheFSDataset(bchfs.cd("train"), _transforms)

    train_sampler = torch.utils.data.SequentialSampler \
        if args.sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True)

    return train_loader


@TaskGenerator
def main(args):
    main_worker(make_dataloader, "BCacheFS", args)


parser = build_parser()
main(parser.parse_args())
