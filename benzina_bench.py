import glob
import os

import jug
from jug import TaskGenerator
import orion.client

import benzina.torch as bz
import benzina.torch.operations as ops
from benzina.utils import File, Track

from bench import main_worker, build_parser

make_dataloaders = {}


def _(data, batch_size, seed, sequence=False, *_args, **_kwargs):
    bias = ops.ConstantBiasTransform(bias=(0.485 * 255, 0.456 * 255, 0.406 * 255))
    std = ops.ConstantNormTransform(norm=(0.229 * 255, 0.224 * 255, 0.225 * 255))

    train_set = bz.dataset.ImageNet(data, split="train")
    val_set = bz.dataset.ImageNet(data, split="val")

    train_loader = bz.DataLoader(
        train_set, shape=(224, 224), batch_size=batch_size,
        shuffle=not sequence, seed=seed, bias_transform=bias,
        norm_transform=std, warp_transform=ops.SimilarityTransform(
            scale=(0.08, 1.0), ratio=(3./4., 4./3.), flip_h=0.5,
            random_crop=True))

    val_loader = bz.DataLoader(
        val_set, shape=(224, 224), batch_size=batch_size, shuffle=False,
        bias_transform=bias, norm_transform=std,
        warp_transform=ops.CenterResizedCrop(224/256))

    # Init cache
    train_set[0]
    val_set[0]

    return train_loader, val_loader


make_dataloaders["imagenet"] = _


def _(data, batch_size, seed, sequence=False, *_args, **_kwargs):
    bias = ops.ConstantBiasTransform(bias=(0.485 * 255, 0.456 * 255, 0.406 * 255))
    std = ops.ConstantNormTransform(norm=(0.229 * 255, 0.224 * 255, 0.225 * 255))

    train_set = bz.dataset.ImageNet(data, split="train")
    val_set = bz.dataset.ImageNet(data, split="val")

    train_loader = bz.DataLoader(
        train_set, shape=(56, 56), batch_size=batch_size, shuffle=not sequence,
        seed=seed, bias_transform=bias, norm_transform=std,
        warp_transform=ops.SimilarityTransform(
            scale=(0.08, 1.0), ratio=(3./4., 4./3.), flip_h=0.5,
            random_crop=True))

    val_loader = bz.DataLoader(
        val_set, shape=(56, 56), batch_size=batch_size, shuffle=False,
        bias_transform=bias, norm_transform=std,
        warp_transform=ops.CenterResizedCrop(56/64))

    # Init cache
    train_set[0]
    val_set[0]

    return train_loader, val_loader


make_dataloaders["tinyimagenet"] = _


@TaskGenerator
def main(args):
    data = args.data.rstrip(os.path.sep)
    data = os.path.basename(data).split('_')
    dataset, dataloader = '_'.join(data[:-1]).lower(), data[-1]
    dataset_files = glob.glob(os.path.join(args.data, "*.mp4"))
    dataset_files.sort()
    args.data = next(iter(dataset_files), args.data)
    with File(args.data) as f:
        return main_worker(make_dataloaders[dataset], dataloader, args)


parser = build_parser()
args = parser.parse_args()
args.workers = 1
acc1, _ = jug.bvalue(main(args))
orion.client.report_objective(1 - acc1)
