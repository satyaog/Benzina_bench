#!/bin/bash

if [[ -z ${SLURM_CPUS_PER_TASK} ]]
then
	MAX_WORKERS=${SLURM_JOB_CPUS_PER_NODE}
else
	MAX_WORKERS=${SLURM_CPUS_PER_TASK}
fi

while [[ $# -gt 0 ]]
do
	arg=$1
	case ${arg} in
		--ds_dir)
		shift
		DS_DIR=$1
		shift
		echo "ds_dir = [${DS_DIR}]"
		if [[ ! -d ${DS_DIR} ]]
		then
			>&2 echo "--ds_dir DIR option must be an existing directory"
			unset DS_DIR
		fi
		;;
		--max_workers)
		shift
		_MAX_WORKERS=$1
		shift
		if [[ "${_MAX_WORKERS}" -lt "${MAX_WORKERS}" ]] || [[ -z ${MAX_WORKERS} ]]
		then
			MAX_WORKERS=${_MAX_WORKERS}
		fi
		echo "max_workers = [${MAX_WORKERS}]"
		unset _MAX_WORKERS
		;;
		--batch_sizes)
		shift
		BATCH_SIZE=$1
		shift
		echo "batch_size = [${BATCH_SIZE}]"
		;;
		--batches)
		shift
		BATCHES=$1
		shift
		echo "batches = [${BATCHES}]"
		;;
		-h | --help | *)
		if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
		then
			>&2 echo "Unknown option [${arg}]"
		fi
		>&2 echo "Options for $(basename "$0") are:"
		>&2 echo "--ds_dir DIR directory where to look for the datasets"
		>&2 echo "[--max_workers] INT maximum number of workers to use (optional)"
		>&2 echo "[--batch_sizes \"INT ...\"] batch sizes to use (optional)"
		>&2 echo "[--batches \"INT ...\"] number of batches to train (optional)"
		exit 1
		;;
	esac
done

if [[ -z ${DS_DIR} ]]
then
	>&2 echo "--ds_dir DIR option must be an existing directory"
	>&2 echo "Missing --ds_dir option"
	exit 1
fi

if [[ -z ${MAX_WORKERS} ]]
then
	MAX_WORKERS=1
fi

if [[ -z ${BATCH_SIZE} ]]
then
	BATCH_SIZE="128 256"
fi

if [[ -z ${BATCHES} ]]
then
	BATCHES="1000 0"
fi

MACHINE_NAME=$(hostname | grep -o "[^\.]*" | head -1)

mkdir -p results/${MACHINE_NAME}
touch results/${MACHINE_NAME}/benzina_measures.csv
touch results/${MACHINE_NAME}/torchvision_measures.csv
touch results/${MACHINE_NAME}/dali_measures.csv

echo "# BENCH COMMIT HASH:" $(git rev-parse HEAD) > results/${MACHINE_NAME}/env
echo "# BENZINA COMMIT HASH:" $(git -C Benzina/ rev-parse HEAD) >> results/${MACHINE_NAME}/env
# Trim the trailing ".server.mila.quebec"
echo "# MACHINE NAME:" ${MACHINE_NAME} >> results/${MACHINE_NAME}/env
echo "# CPU:" $(cat /proc/cpuinfo | grep "model name" | sort | uniq) >> results/${MACHINE_NAME}/env
echo "# GPU:" $(nvidia-smi -q | grep "Product Name") >> results/${MACHINE_NAME}/env
pip freeze >> results/${MACHINE_NAME}/env


for cmd in "status" "execute"; do
for batches in ${BATCHES}; do
for workers in 1 2 4 6 8 16; do
	if [[ "${workers}" -gt "${MAX_WORKERS}" ]]
	then
		continue
	fi
	if [[ "${batches}" -eq 0 ]] && { [[ "${workers}" -eq 1 ]] || [[ "${workers}" -eq "${MAX_WORKERS}" ]]; }
	then
		continue
	fi
	for batch_size in ${BATCH_SIZE}; do
	for arch in "resnet18" "resnet50"; do
	for sequence in "" "--sequence"; do
		if [[ "${batches}" -eq 0 ]] && [[ ! -z "${sequence}" ]]
		then
			continue
		fi
		if [[ "${workers}" -gt 1 ]] && [[ ! -z "${sequence}" ]]
		then
			continue
		fi
		ln -sf results/${MACHINE_NAME}/benzina_measures.csv measures.csv
		jug ${cmd} -- benzina_bench.py --arch=${arch} \
			--workers=1 \
			--epochs=1 \
			--batch-size=${batch_size} \
			--batches=${batches} \
			--seed=1234 \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_benzina \
			"$(git -C Benzina/ rev-parse HEAD)" >> benzina_bench.out 2>> benzina_bench.err

		ln -sf results/${MACHINE_NAME}/torchvision_measures.csv measures.csv
		jug ${cmd} -- torchvision_bench.py --arch=${arch} \
			--workers=${workers} \
			--epochs=1 \
			--batch-size=${batch_size} \
			--batches=${batches} \
			--seed=1234 \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_torchvision \
			"$(echo $(grep "torch" results/${MACHINE_NAME}/env))" >> torchvision_bench.out 2>> torchvision_bench.err

		ln -sf results/${MACHINE_NAME}/dali_measures.csv measures.csv
		jug ${cmd} -- dali_bench.py --arch=${arch} \
			--workers=${workers} \
			--epochs=1 \
			--batch-size=${batch_size} \
			--batches=${batches} \
			--seed=1234 \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_torchvision/train \
			"$(echo $(grep "dali" results/${MACHINE_NAME}/env))" >> dali_bench.out 2>> dali_bench.err
	done
	done
	done
done
done
done

rm measures.csv
