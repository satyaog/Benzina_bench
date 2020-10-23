from jug import TaskGenerator

import benzina.torch as bz
import benzina.torch.operations as ops
from benzina.utils import File, Track

from bench import main_worker, build_parser


def make_dataloader(args):
    bias = ops.ConstantBiasTransform(bias=(0.485 * 255, 0.456 * 255, 0.406 * 255))
    std = ops.ConstantNormTransform(norm=(0.229 * 255, 0.224 * 255, 0.225 * 255))

    return bz.DataLoader(
        bz.dataset.ImageNet(args.data, split="train"),
        shape=(224, 224), batch_size=args.batch_size, shuffle=True,
        seed=args.seed, bias_transform=bias, norm_transform=std,
        warp_transform=ops.SimilarityTransform(
            scale=(0.08, 1.0), ratio=(3./4., 4./3.), flip_h=0.5,
            random_crop=True))


@TaskGenerator
def main(args):
    with File(args.data) as f:
        args.file = f
        main_worker(make_dataloader, "Benzina", args)


parser = build_parser()
args = parser.parse_args()
args.workers = 1
main(args)
