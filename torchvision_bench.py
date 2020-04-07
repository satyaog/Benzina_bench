from jug import TaskGenerator
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from bench import main_worker, build_parser


def make_dataloader(args):
    # Data loading code
    # Dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Dataloaders
    train_set = datasets.ImageNet(
        root=args.data,
        split="train",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.SequentialSampler \
        if args.sequence else torch.utils.data.RandomSampler

    train_loader = DataLoader(train_set, sampler=train_sampler(train_set),
                              batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True)

    return train_loader


@TaskGenerator
def main(args):
    main_worker(make_dataloader, "PyTorch", args)


parser = build_parser()
main(parser.parse_args())
