from jug import TaskGenerator
import torch.utils.data

import benzina.torch as bz
import benzina.torch.operations as ops
from benzina.utils import File, Track

from bench import main_worker, build_parser


def make_dataloader(args):
    with Track(args.file, "bzna_input") as input_track, \
         Track(args.file, "bzna_target") as target_track:
        # Data loading code
        # Dataset
        dataset = bz.dataset.ImageNet(input_track, target_track, b"bzna_thumb")

        indices = list(range(len(dataset)))
        n_valid = 50000
        n_test = 100000
        n_train = len(dataset) - n_valid - n_test
        train_set = torch.utils.data.Subset(dataset, indices[:n_train])
        train_sampler = torch.utils.data.SequentialSampler \
            if args.sequence else torch.utils.data.RandomSampler

        # Dataloaders
        bias = ops.ConstantBiasTransform(bias=(0.485 * 255, 0.456 * 255, 0.406 * 255))
        std = ops.ConstantNormTransform(norm=(0.229 * 255, 0.224 * 255, 0.225 * 255))

        train_loader = bz.DataLoader(train_set, args.data,
                                     sampler=train_sampler(train_set), batch_size=args.batch_size,
                                     seed=args.seed, shape=(224, 224),
                                     bias_transform=bias, norm_transform=std,
                                     warp_transform=ops.SimilarityTransform(flip_h=0.5, random_crop=True))

        return train_loader


@TaskGenerator
def main(args):
    with File(args.data) as f:
        args.file = f
        main_worker(make_dataloader, "Benzina", args)


parser = build_parser()
args = parser.parse_args()
args.workers = 1
main(args)
