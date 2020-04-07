#!/bin/bash

export PATH="/network/home/ortizgas/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

proj_path=$1

data_ori=$2
data_dir=/Tmp/$USER/data
data=$data_dir/$(basename $data_ori)
arch=$3
batch_size=$4
seed=$5

mkdir -p $proj_path/results/pytorch_bench/$SLURM_JOB_ID
cd $proj_path/results/pytorch_bench/$SLURM_JOB_ID
echo data:$data > arguments
echo arch:$arch >> arguments
echo batch_size:$batch_size >> arguments
echo command:rsync --ignore-existing -r $data_ori $data_dir >> arguments
echo command:python $proj_path/benches/pytorch_bench.py --arch=$arch --workers=$SLURM_CPUS_PER_TASK --epochs=1 --batch-size=$batch_size --batches=0 --seed=$seed --gpu=0 --batch-sync $data >> arguments

mkdir -p $data
cd $data
cat $proj_path/scripts/$(basename $data_ori)_folder_list | xargs mkdir -p

cd $proj_path/results/pytorch_bench/$SLURM_JOB_ID

mkdir -p $data_dir
rsync --ignore-existing -r $data_ori $data_dir

# Prevent error:
# OSError: cannot identify image file <_io.BufferedReader name='/Tmp/ortizgas/data/ImageNet2012_256x256/train/n01744401/n01744401_80593.jpeg'>
rm $data_dir/ImageNet2012_256x256/train/n01744401/n01744401_80593.jpeg

pyenv activate pytorch_bench
python $proj_path/benches/pytorch_bench.py --arch=$arch --workers=$SLURM_CPUS_PER_TASK --epochs=1 --batch-size=$batch_size --batches=0 --seed=$seed --gpu=0 --batch-sync $data
