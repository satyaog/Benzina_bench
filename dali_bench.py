import os

import jug
from jug import TaskGenerator
import orion.client
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types

from bench import main_worker, build_parser


class DALIDataLoader:
    def __init__(self, pipeline, output_map, reader_name, *args, **kwargs):
        self._pipeline = pipeline
        self._output_map = output_map
        self._len = pipeline.epoch_size(reader_name)
        self._iter = DALIGenericIterator(pipeline, output_map,
                                         reader_name=reader_name,
                                         *args, **kwargs)

    def __getitem__(self, _item):
        raise NotImplementedError

    def __iter__(self):
        for batches in self._iter:
            for batch in batches:
                yield [batch[out] for out in self._output_map]

    def __len__(self):
        return self._len


make_dataloaders = {}


def _(data, batch_size, workers, gpu, seed, sequence=False, *_args, **_kwargs):
    # Data loading code
    class TrainPipeline(Pipeline):
        def __init__(self, root, shape, num_threads=-1, device_id=-1, random_shuffle=False):
            super(TrainPipeline, self).__init__(batch_size, num_threads, device_id,
                                                seed if seed is not None else -1)
            self.input = ops.FileReader(file_root=root, random_shuffle=random_shuffle)
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            self.rrc = ops.RandomResizedCrop(device="gpu", size=shape)
            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.coin = ops.CoinFlip(probability=0.5)
            self.target_cast = ops.Cast(dtype=types.INT64)

        def define_graph(self):
            jpegs, target = self.input(name="Reader")
            images = self.decode(jpegs)
            images = self.rrc(images)
            images = self.cmn(images, mirror=self.coin())
            target = self.target_cast(target)
            return images, target

    class ValPipeline(Pipeline):
        def __init__(self, root, resize, shape, num_threads=-1, device_id=-1):
            super(ValPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.FileReader(file_root=root, random_shuffle=False)
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            self.resize = ops.Resize(device="gpu", resize_shorter=resize)
            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               crop=shape,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.target_cast = ops.Cast(dtype=types.INT64)

        def define_graph(self):
            jpegs, target = self.input(name="Reader")
            images = self.decode(jpegs)
            images = self.resize(images)
            images = self.cmn(images)
            target = self.target_cast(target)
            return images, target

    # Dataloaders
    train_pipeline = TrainPipeline(
        os.path.join(data, "train"), shape=(224, 224), num_threads=workers,
        device_id=gpu, random_shuffle=not sequence)
    train_pipeline.build()

    val_pipeline = ValPipeline(
        os.path.join(data, "val"), resize=256, shape=(224, 224),
        num_threads=workers, device_id=gpu)
    val_pipeline.build()

    return (DALIDataLoader(train_pipeline, ["images", "target"],
                           reader_name="Reader"),
            DALIDataLoader(val_pipeline, ["images", "target"],
                           reader_name="Reader"))


make_dataloaders["imagenet"] = _


def _(data, batch_size, workers, gpu, seed, sequence=False, *_args, **_kwargs):
    # Data loading code
    class TrainPipeline(Pipeline):
        def __init__(self, root, shape, num_threads=-1, device_id=-1, random_shuffle=False):
            super(TrainPipeline, self).__init__(batch_size, num_threads, device_id,
                                                seed if seed is not None else -1)
            self.input = ops.FileReader(file_root=root, random_shuffle=random_shuffle)
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            self.rrc = ops.RandomResizedCrop(device="gpu", crop=shape)
            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.coin = ops.CoinFlip(probability=0.5)
            self.target_cast = ops.Cast(dtype=types.INT64)

        def define_graph(self):
            jpegs, target = self.input(name="Reader")
            images = self.decode(jpegs)
            images = self.rrc(images)
            images = self.cmn(images, mirror=self.coin())
            target = self.target_cast(target)
            return images, target

    class ValPipeline(Pipeline):
        def __init__(self, root, resize, center_crop, num_threads=-1, device_id=-1):
            super(ValPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.FileReader(file_root=root, random_shuffle=False)
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            self.resize = ops.Resize(device="gpu", resize_shorter=resize)
            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               crop=center_crop,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.target_cast = ops.Cast(dtype=types.INT64)

        def define_graph(self):
            jpegs, target = self.input(name="Reader")
            images = self.decode(jpegs)
            images = self.resize(images)
            images = self.cmn(images)
            target = self.target_cast(target)
            return images, target

    # Dataloaders
    train_pipeline = TrainPipeline(
        os.path.join(data, "train"), shape=(56, 56), num_threads=workers,
        device_id=gpu, random_shuffle=not sequence)
    train_pipeline.build()

    val_pipeline = ValPipeline(
        os.path.join(data, "val"), resize=64, center_crop=56,
        num_threads=workers, device_id=gpu)
    val_pipeline.build()

    return (DALIDataLoader(train_pipeline, ["images", "target"],
                           reader_name="Reader"),
            DALIDataLoader(val_pipeline, ["images", "target"],
                           reader_name="Reader"))


make_dataloaders["tinyimagenet"] = _


@TaskGenerator
def main(args):
    data = args.data.rstrip(os.path.sep)
    data = os.path.basename(data).split('_')
    dataset, dataloader = '_'.join(data[:-1]).lower(), data[-1]
    return main_worker(make_dataloaders[dataset], dataloader, args)


parser = build_parser()
acc1, _ = jug.bvalue(main(parser.parse_args()))
orion.client.report_objective(1 - acc1)
