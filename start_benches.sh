#!/bin/bash

MACHINE=eos2

imagenet_jpeg=/network/data1/ImageNet2012_jpeg
imagenet_jpeg_256=/network/data1/ImageNet2012_256x256
imagenet_h264=/network/data1/ImageNet2012_h264

proj_path="$(cd "$(dirname "$0")"; pwd -P)"

echo imagenet_jpeg:$imagenet_jpeg
echo imagenet_jpeg_256:$imagenet_jpeg_256
echo imagenet_h264:$imagenet_h264

cd scripts

##sbatch --gres=gpu -c 1 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 128
##sbatch --gres=gpu -c 4 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 128
##sbatch --gres=gpu -c 1 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 256
#-#sbatch --gres=gpu -c 4 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 256
#sbatch --gres=gpu -c 1 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 128
#sbatch --gres=gpu -c 4 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 128
#sbatch --gres=gpu -c 1 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 256
#sbatch --gres=gpu -c 4 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 256

#sbatch --gres=gpu -c 1 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_benzina.sh $proj_path $imagenet_h264 resnet18 128
#sbatch --gres=gpu -c 1 --qos=unkillable -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_benzina.sh $proj_path $imagenet_h264 resnet18 256

##sbatch --gres=gpu -c 8 --qos=low -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 128
##sbatch --gres=gpu -c 8 --qos=low -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 256
##sbatch --gres=gpu -c 8 --qos=low -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 128
##sbatch --gres=gpu -c 8 --qos=low -w kepler5 --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 256


# sbatch --gres=gpu -c 8 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 128
# sbatch --gres=gpu -c 8 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 128
# sbatch --gres=gpu -c 8 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 256
# sbatch --gres=gpu -c 8 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 256

sbatch --gres=gpu -c 1 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_benzina.sh $proj_path $imagenet_h264 resnet18 128 1234
sbatch --gres=gpu -c 6 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 128 1234
sbatch --gres=gpu -c 6 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 128 1234

sbatch --gres=gpu -c 1 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_benzina.sh $proj_path $imagenet_h264 resnet18 256 2345
sbatch --gres=gpu -c 6 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 256 2345
sbatch --gres=gpu -c 6 --qos=low --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 256 2345

sbatch --gres=gpu -c 4 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 128 3456
sbatch --gres=gpu -c 4 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 128 3456

sbatch --gres=gpu -c 4 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 256 4567
sbatch --gres=gpu -c 4 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 256 4567

sbatch --gres=gpu -c 1 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 128 5678
sbatch --gres=gpu -c 1 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 128 5678

sbatch --gres=gpu -c 1 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg resnet18 256 6789
sbatch --gres=gpu -c 1 --qos=unkillable --reservation=test --wait --mail-type=ALL --mail-user=satya.ortiz-gagne@mila.quebec bench_pytorch.sh $proj_path $imagenet_jpeg_256 resnet18 256 6789
