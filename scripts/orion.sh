#!/bin/bash

if [[ -z ${SLURM_CPUS_PER_TASK} ]]
then
	MAX_WORKERS=${SLURM_JOB_CPUS_PER_NODE}
else
	MAX_WORKERS=${SLURM_CPUS_PER_TASK}
fi

if [[ -z ${OUT} ]]
then
	OUT=results
fi

while [[ $# -gt 0 ]]
do
	arg="$1"; shift
	case "${arg}" in
		--ds-dir)
		DS_DIR=$1
		shift
		echo "ds-dir = [${DS_DIR}]"
		if [[ ! -d ${DS_DIR} ]]
		then
			>&2 echo "--ds-dir DIR option must be an existing directory"
			unset DS_DIR
		fi
		;;
		--out) OUT=$1; shift
		echo "out = [${OUT}]"
		;;
		-h | --help | *)
		if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
		then
			>&2 echo "Unknown option [${arg}]"
		fi
		>&2 echo "Options for $(basename "$0") are:"
		>&2 echo "--ds-dir DIR directory where to look for the datasets"
		exit 1
		;;
	esac
done

if [[ -z ${DS_DIR} ]]
then
	>&2 echo "--ds-dir DIR option must be an existing directory"
	>&2 echo "Missing --ds-dir option"
	exit 1
fi

if [[ -z ${MAX_WORKERS} ]]
then
	MAX_WORKERS=1
fi


mkdir -p results/
mkdir -p ${OUT}
OUT=$(cd ${OUT}; pwd)

for dataloader in "benzina" "torchvision"; do
for arch in "vgg19" "preactresnet101"; do
for dataset in "cifar10" "cifar100" "imagenet" "tinyimagenet"
do
	if [[ ${dataset} = "cifar"* ]] && [[ ${dataloader} = "benzina" ]]
	then
		continue
	fi
	if [[ ${dataloader} = "benzina" ]]
	then
		VER="$(git -C Benzina/ rev-parse HEAD)"
	else
		VER="$(python -m pip freeze | grep "torch")"
		VER="$(echo ${VER//[[:blank:]]/})"
	fi
	orion hunt -n ${dataloader}_${arch}_${dataset} \
		--config scripts/orion_config.yml \
		--working-dir ${OUT} \
		jug execute ${dataloader}_bench.py -- \
		--epochs~"fidelity(low=1, high=120, base=4)" \
		--batch-size~"choices([128])" \
		--lr~"loguniform(1e-4, 1e-1)" \
		--momentum~"uniform(0, 0.9)" \
		--weight-decay~"loguniform(1e-10, 1e-2)" \
		--arch=${arch} \
		--out="{exp.working_dir}/${dataloader}_${dataset}" \
		--checkpoint="checkpoint_${arch}_{trial.hash_params}.pth.tar" \
		--resume \
		--workers=${MAX_WORKERS} \
		--seed=1234 \
		--gpu=0 \
		"${DS_DIR}/${dataset}_${dataloader}" \
		"${VER}"
done
done
done
