import torch.utils.data

import benzina.torch as bz
import benzina.torch.operations as ops

from bench import main_worker, build_parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.workers = 1

    def make_dataloaders(args):
        # Data loading code
        # Dataset
        dataset = bz.ImageNet(args.data)

        # indices = list(range(len(dataset)))
        # n_valid = 50000
        # n_test = 100000
        # n_train = len(dataset) - n_valid - n_test
        # train_sampler = torch.utils.data.SubsetRandomSampler(indices[:n_train])

        # Dataloaders
        bias = ops.ConstantBiasTransform(bias=(123.675, 116.28, 103.53))
        std = ops.ConstantNormTransform(norm=(58.395, 57.12, 57.375))

        train_loader = bz.DataLoader(dataset, batch_size=args.batch_size,
            seed=args.seed, shape=(224,224), bias_transform=bias,
            norm_transform=std, warp_transform=ops.SimilarityTransform(flip_h=0.5))

        return train_loader, None

    main_worker(make_dataloaders, "Benzina", args)

if __name__ == "__main__":
    main()
