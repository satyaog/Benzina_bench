#!/bin/bash

BATCH_SIZE="128 256"

if [ -z "${SLURM_CPUS_PER_TASK}" ]
then
	MAX_WORKERS=${SLURM_JOB_CPUS_PER_NODE}
else
	MAX_WORKERS=${SLURM_CPUS_PER_TASK}
fi

for ((i = 1; i <= ${#@}; i++))
do
	arg=${!i}
	case ${arg} in
		--ds_dir)
		i=$((i+1))
		DS_DIR=${!i}
		echo "ds_dir = [${DS_DIR}]"
		if [ ! -d ${DS_DIR} ]
		then
			>&2 echo "--ds_dir DIR option must be an existing directory"
			unset DS_DIR
		fi
		;;
		--max_workers)
		i=$((i+1))
		_MAX_WORKERS=${!i}
		if [[ "${_MAX_WORKERS}" -lt "${MAX_WORKERS}" ]] || [ -z ${MAX_WORKERS} ]
		then
			MAX_WORKERS=${_MAX_WORKERS}
		fi
		echo "max_workers = [${MAX_WORKERS}]"
		unset _MAX_WORKERS
		;;
		--batch_sizes)
		i=$((i+1))
		BATCH_SIZE=${!i}
		echo "batch_size = [${BATCH_SIZE}]"
		;;
		-h | --help | *)
		>&2 echo "Unknown option [${arg}]. Valid options are:"
		>&2 echo "--ds_dir DIR directory where to look for the datasets"
		>&2 echo "[--max_workers] INT maximum number of workers to use (optional)"
		>&2 echo "[--batch_sizes \"INT ...\"] batch sizes to use (optional)"
		exit 1
		;;
	esac
done

if [ -z "${DS_DIR}" ]
then
	>&2 echo "--ds_dir DIR option must be an existing directory"
	>&2 echo "Missing --ds_dir option"
	exit 1
fi

if [ -z ${MAX_WORKERS} ]
then
	MAX_WORKERS=1
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
for workers in 1 2 4 6 8 16; do
	if [[ "${workers}" -gt "${MAX_WORKERS}" ]]
	then
		break
	fi
	for batch_size in ${BATCH_SIZE}; do
	for arch in "resnet18" "resnet50"; do
	for sequence in "" "--sequence"; do
		ln -sf results/${MACHINE_NAME}/benzina_measures.csv measures.csv
		jug ${cmd} -- benzina_bench.py --arch=${arch}\
			--workers=1 \
			--epochs=1 \
			--batch-size=${batch_size} \
			--batches=100 \
			--seed=1234 \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_benzina/ilsvrc2012.bzna \
			"$(git -C Benzina/ rev-parse HEAD)" >> bench.out 2>> bench.err

		ln -sf results/${MACHINE_NAME}/torchvision_measures.csv measures.csv
		jug ${cmd} -- torchvision_bench.py --arch=${arch} \
			--workers=${workers} \
			--epochs=1 \
			--batch-size=${batch_size} \
			--batches=100 \
			--seed=1234 \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_torchvision \
			"$(echo $(grep "torch" results/${MACHINE_NAME}/env))" >> bench.out 2>> bench.err

		ln -sf results/${MACHINE_NAME}/dali_measures.csv measures.csv
		jug ${cmd} -- dali_bench.py --arch=${arch} \
			--workers=${workers} \
			--epochs=1 \
			--batch-size=${batch_size} \
			--batches=100 \
			--seed=1234 \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_torchvision/train \
			"$(echo $(grep "dali" results/${MACHINE_NAME}/env))" >> bench.out 2>> bench.err
	done
	done
	done
done
done

rm measures.csv
