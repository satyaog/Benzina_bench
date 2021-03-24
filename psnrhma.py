import argparse
import copy
import os.path

import jug
import orion.client
import torch.utils.data
import torchvision.transforms as transforms
from jug import CachedFunction, TaskGenerator
from PIL import Image

from pybenzinaconcat import benzinaconcat, create_container, index_metadata
import pybenzinaconcat.test.psnrhma as psnrhma


class PilDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.transform = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = image.convert("RGB")
        img_shape = image.size
        if self.transform is not None:
            image = self.transform(image)
        if max(img_shape) > max(image.shape):
            scale = max(image.shape) / max(img_shape)
            img_shape = (int(img_shape[0] * scale), int(img_shape[1] * scale))
        return image, img_shape


@TaskGenerator
def get_reference_dataset(images, shape):
    return PilDataset(images, transforms.Compose([
        psnrhma.PilResizeMaxSide((shape, shape)),
        psnrhma.PilFill((shape, shape), padding_mode="edge"),
        transforms.ToTensor()
    ]))


@TaskGenerator
def array_split(array, batch_size, max_size=None):
    # split into batches
    splits = [array[:max_size]]
    while splits[0]:
        splits.append(splits[0][:batch_size])
        del splits[0][:batch_size]
    splits.pop(0)
    return splits


@TaskGenerator
def array_flatten(array):
    flatten = array.pop() if len(array) == 1 else []
    for subarr in array:
        flatten.extend(subarr)
    return flatten


@TaskGenerator
def psnrhma_test(reference, corrupted, threshold, batch_size, eps):
    size = min(CachedFunction(len, reference),
               CachedFunction(len, corrupted))

    kwargs = dict(shape=512, eps=eps, start=0)

    if size and batch_size:
        batch_size = min(size, batch_size)
        batches_kwargs = []
        for batch_start in range(0, size, batch_size):
            batch_kwargs = copy.deepcopy(kwargs)
            batch_kwargs["start"] = batch_start
            batch_kwargs["size"] = batch_size
            batches_kwargs.append(batch_kwargs)
    else:
        batches_kwargs = [kwargs]

    results = []
    for batch_kwargs in batches_kwargs:
        batch_scores = psnrhma.psnrhma_batch(reference, corrupted,
                                             **batch_kwargs)
        results.append(psnrhma.psnrhma_validate(batch_scores, threshold))
    return psnrhma.array_flatten(results)


def read_csv(filename):
    with open(filename, 'r') as f:
        csv = [line.split(',') for line in f.readlines()]
    return csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--psnrhma-scores", type=argparse.FileType('r'),
                   help="a csv containing the list of images indices followed "
                        "by their psnrhma score to extract to build the "
                        "subset")
    p.add_argument("--aux-indices", type=argparse.FileType('r'),
                   help="a csv containing the list of images indices followed "
                        "by custom data")
    p.add_argument("--reference", metavar="PATH",
                   help="ImageNet HDF5 dataset path")
    p.add_argument("--subset", metavar="PATH",
                   help="directory containing the images subset to use")
    p.add_argument("--corrupted", metavar="PATH", help="transcoded Benzina mp4")
    p.add_argument("--subset-size", default=None, type=int,
                   help="subset size to use. If bigger then `aux_indices`, "
                        "images with the worst psnrhma score will be selected.")
    p.add_argument("--max-subset-size", default=None, type=int,
                   help="max subset size to extract.")
    p.add_argument("--crf", default="10", type=int,
                   help="constant rate factor to use for the transcoded image")
    p.add_argument("--batch-size", default=1024, metavar="NUM", type=int,
                   help="the batch size for a single job")
    p.add_argument("--shape", metavar="SIZE", default=512, type=int,
                   help="images sides size")
    p.add_argument("--threshold", default=45., type=float,
                   help="similarity threshold")
    p.add_argument("--eps", metavar="SIZE", default=1e-30, type=float)
    args = p.parse_args()

    args.psnrhma_scores.close()
    args.aux_indices.close()
    psnrhma_scores = CachedFunction(read_csv, args.psnrhma_scores.name)
    psnrhma_scores.sort(key=lambda v: float(v[1]))
    aux_indices = CachedFunction(read_csv, args.aux_indices.name)

    args.corrupted, ext = os.path.splitext(args.corrupted)
    args.corrupted = "{}_{}{}".format(args.corrupted, args.subset_size, ext)

    indices = [int(v[0]) for v in aux_indices]
    indices_set = set(indices)
    for i in (int(v[0]) for v in psnrhma_scores):
        if i not in indices_set:
            indices.append(i)

    indices = indices[:args.max_subset_size]

    # extract
    extracted_imgs = benzinaconcat.extract(
        args.reference, args.subset, "imagenet", "hdf5", indices)
    extracted_imgs = array_split(array_flatten(extracted_imgs), args.batch_size,
                                 args.max_subset_size)

    jug.barrier()

    # transcode
    transcoded_imgs = [benzinaconcat.transcode(
        split, os.path.dirname(args.corrupted), mp4=True, crf=args.crf)
        for split in jug.value(extracted_imgs)[:args.subset_size//args.batch_size + 1]]

    # concat
    CachedFunction(create_container, args.corrupted)
    concatenation_batches = benzinaconcat.concat(transcoded_imgs,
                                                 args.corrupted)
    transcoded_imgs_flat = array_flatten(transcoded_imgs)

    jug.barrier()

    transcoded_imgs_set = CachedFunction(set, jug.value(transcoded_imgs_flat)[:args.subset_size])
    total_concats = 0
    concatenated_files_set = set()
    for batch in concatenation_batches:
        total_concats, concatenated_files = jug.value(batch)
        assert set(concatenated_files) not in concatenated_files_set
        concatenated_files_set.update(concatenated_files)

    assert total_concats == args.subset_size
    assert len(concatenated_files_set) == args.subset_size
    assert concatenated_files_set == transcoded_imgs_set

    CachedFunction(index_metadata, args.corrupted)

    # psnrhma
    scores = psnrhma_test(
        get_reference_dataset(array_flatten(extracted_imgs), args.shape),
        psnrhma.get_benzina_dataset(args.corrupted), args.threshold,
        args.batch_size, args.eps)

    jug.barrier()

    scores = jug.value(scores)
    ratio = len([v for v in scores if v[2]]) / len(scores)
    orion.client.report_objective(ratio)
    return ratio


main()
