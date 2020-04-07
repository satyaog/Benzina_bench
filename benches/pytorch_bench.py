import os

import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from bench import main_worker, build_parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    def make_dataloaders(args):
        # Data loading code
        # Dataset
        traindir = os.path.join(args.data, 'train')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Dataloaders
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True)

        return train_loader, None

    main_worker(make_dataloaders, "PyTorch", args)

if __name__ == "__main__":
    main()
