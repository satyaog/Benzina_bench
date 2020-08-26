from jug import TaskGenerator
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types

from bench import main_worker, build_parser


class HybridPipeline(Pipeline):
    def __init__(self, root, batch_size, shape, num_threads=-1, device_id=-1,
                 seed=-1, random_shuffle=False):
        super(HybridPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.input = ops.FileReader(file_root=root, random_shuffle=random_shuffle)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_shorter=max(*shape))
        self.cmn = ops.CropMirrorNormalize(device="gpu",
                                           dtype=types.FLOAT,
                                           crop=shape,
                                           mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                           std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.coin = ops.CoinFlip(probability=0.5)
        self.target_cast = ops.Cast(dtype=types.INT64)

    def define_graph(self):
        jpegs, target = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.resize(images)
        images = self.cmn(images, mirror=self.coin(),
                          crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        target = self.target_cast(target)
        return images, target


def make_dataloader(args):
    # Data loading code
    # Dataloaders
    pipeline = HybridPipeline(args.data, batch_size=args.batch_size, shape=(224, 224),
                              num_threads=args.workers, device_id=args.gpu,
                              seed=args.seed if args.seed is not None else -1,
                              random_shuffle=not args.sequence)
    pipeline.build()
    return ((batch[0]["images"], batch[0]["target"])
            for batch in DALIGenericIterator(pipeline,
                                             ["images", "target"],
                                             reader_name="Reader"))


@TaskGenerator
def main(args):
    main_worker(make_dataloader, "DALI", args)


parser = build_parser()
main(parser.parse_args())
