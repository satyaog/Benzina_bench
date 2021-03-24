#!/bin/bash
set -o errexit -o pipefail -o noclobber

_CODE_DIR=$(pwd -P)
_RESULTS_DIR=/miniscratch/${USER}/psnrhma_results/
_DS_UTILS_COMMIT=2b1bf2471df00c6a0f1d9e1c8ab47eace67a8c4d

module load python/3.6 cuda/10.1

mkdir -p /miniscratch/"${USER}"/psnrhma_results/

cd "${_RESULTS_DIR}"

mkdir -p "${SLURM_TMPDIR}"/localscratch/

# Get dataset_utils
pushd "$(mktemp -d -p "${SLURM_TMPDIR}"/localscratch/)"
wget -O ds_utils.zip https://github.com/satyaog/datasets_utils/archive/${_DS_UTILS_COMMIT}.zip
unzip ds_utils.zip
mv datasets_utils-${_DS_UTILS_COMMIT}/ datasets_utils/
./datasets_utils/jug/utils.sh tmp_jug_exec --tmp "${_RESULTS_DIR}"/tmp/ -- --jugdir "${_RESULTS_DIR}"/jugdir_setup/ -- \
        cp -ra ./datasets_utils/ "${_RESULTS_DIR}"/datasets_utils/
popd

# Install jug
./datasets_utils/jug/utils.sh tmp_jug_exec --tmp tmp/ -- --jugdir jugdir_setup/ -- \
        ./datasets_utils/utils.sh init_venv --name 3.6/jug --tmp ./
source datasets_utils/utils.sh init_venv --name 3.6/jug --tmp ./
./datasets_utils/jug/utils.sh tmp_jug_exec --tmp tmp/ -- --jugdir jugdir_setup/ -- \
        "$(which python3)" -m pip install jug

# Get ffmpeg and add it to PATH
mkdir -p bin/
./datasets_utils/jug/utils.sh jug_exec --jugdir jugdir_setup/ -- \
        wget -O bin/md5sums https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.2.1-amd64-static.tar.xz.md5
./datasets_utils/jug/utils.sh jug_exec --jugdir jugdir_setup/ -- \
        wget -O bin/ffmpeg-4.2.1-amd64-static.tar.xz \
        https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.2.1-amd64-static.tar.xz
./datasets_utils/jug/utils.sh jug_exec --jugdir jugdir_setup/ -- \
        tar -C bin/ -xf bin/ffmpeg-4.2.1-amd64-static.tar.xz
pushd bin/
md5sum -c md5sums
ln -sf ffmpeg-4.2.1-amd64-static/ffmpeg .
popd

# Copy code
./datasets_utils/jug/utils.sh jug_exec --jugdir jugdir_setup/ -- \
        cp -ra "${_CODE_DIR}" CODE/

# Setup venv
./datasets_utils/jug/utils.sh jug_exec --jugdir jugdir_setup/ -- \
        ./datasets_utils/utils.sh init_venv --name 3.6/bzna --tmp ./
source datasets_utils/utils.sh init_venv --name 3.6/bzna --tmp ./
# Add ffmpeg to PATH
pushd bin/
PATH=$(pwd -P):${PATH}
export PATH
popd
./datasets_utils/jug/utils.sh tmp_jug_exec --tmp tmp/ -- --jugdir jugdir_setup/ -- \
        "$(which python3)" -m pip install "meson==0.54" "ninja>=1.8.2"
./datasets_utils/jug/utils.sh tmp_jug_exec --tmp tmp/ -- --jugdir jugdir_setup/ -- \
        "$(which python3)" -m pip install \
        "pybenzinaconcat[psnrhma,h5py] @ git+https://github.com/satyaog/pybenzinaconcat.git@feature/psnrhma#egg=pybenzinaconcat-0.0.0" \
        "orion==0.1.8"

mkdir -p upload/
mkdir -p queue/
# Run orion
exec orion hunt -n benzina_imagenet_psnrhma \
        --config CODE/scripts/orion_config.yml \
        --working-dir "$(pwd -P)" \
        jug execute --jugdir jugdir/ CODE/psnrhma.py -- \
        --psnrhma-scores CODE/psnrhma_scores.csv \
        --aux-indices CODE/low_occurences_pixformats.csv \
        --reference /network/datasets/imagenet.var/imagenet_hdf5/ilsvrc2012.hdf5 \
        --subset "{exp.working_dir}/subset/" \
        --corrupted "{exp.working_dir}/ilsvrc2012_subset.mp4" \
        --subset-size~"fidelity(low=60000,high=200000,base=4)" \
        --max-subset-size"choices([200000])" \
        --crf~"choices([20,16,10,8,4,2,1])" \
        --batch-size 1024 --shape 512 --threshold 45.0
