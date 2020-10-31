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
		--max-workers)
		_MAX_WORKERS=$1
		shift
		if [[ "${_MAX_WORKERS}" -lt "${MAX_WORKERS}" ]] || [[ -z ${MAX_WORKERS} ]]
		then
			MAX_WORKERS=${_MAX_WORKERS}
		fi
		echo "max-workers = [${MAX_WORKERS}]"
		unset _MAX_WORKERS
		;;
		--batch-sizes) BATCH_SIZE=$1; shift
		echo "batch-sizes = [${BATCH_SIZE}]"
		;;
		--batches) BATCHES=$1; shift
		echo "batches = [${BATCHES}]"
		;;
		--) break ;;
		-h | --help | *)
		if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
		then
			>&2 echo "Unknown option [${arg}]"
		fi
		>&2 echo "Options for $(basename "$0") are:"
		>&2 echo "--ds-dir DIR directory where to look for the datasets"
		>&2 echo "[--max-workers] INT maximum number of workers to use (optional)"
		>&2 echo "[--batch-sizes \"INT ...\"] batch sizes to use (optional)"
		>&2 echo "[--batches \"INT ...\"] number of batches to train (optional)"
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

if [[ -z ${BATCH_SIZE} ]]
then
	BATCH_SIZE="128 256"
fi

if [[ -z ${BATCHES} ]]
then
	BATCHES="1000 0"
fi


# Trim the trailing ".server.mila.quebec"
MACHINE_NAME=$(hostname | grep -o "[^\.]*" | head -1)

mkdir -p ${OUT}/${MACHINE_NAME}
OUT=$(cd ${OUT}/${MACHINE_NAME}; pwd)

BENZINA_PATH=$(pip freeze | grep -Ee "^benzina @ ")
BENZINA_PATH=${BENZINA_PATH#*file://}
BENZINA_COMMIT_HASH=$(git -C "${BENZINA_PATH}" rev-parse HEAD)
echo "# BENCH COMMIT HASH:" $(git rev-parse HEAD) > ${OUT}//env
echo "# BENZINA COMMIT HASH:" "${BENZINA_COMMIT_HASH}" >> ${OUT}/env
echo "# MACHINE NAME:" ${MACHINE_NAME} >> ${OUT}/env
echo "# CPU:" $(cat /proc/cpuinfo | grep "model name" | sort | uniq) >> ${OUT}/env
echo "# GPU:" $(nvidia-smi -q | grep "Product Name") >> ${OUT}/env
pip freeze >> ${OUT}/env

for cmd in "status" "execute"; do
for batches in ${BATCHES}; do
for workers in 1 2 4 6 8 16
do
	if [[ "${workers}" -gt "${MAX_WORKERS}" ]]
	then
		continue
	fi
	if [[ "${batches}" -eq 0 ]] && [[ ! "${workers}" -eq "${MAX_WORKERS}" ]]
	then
		continue
	fi
	for batch_size in ${BATCH_SIZE}; do
	for arch in "resnet18" "resnet50"; do
	for sequence in "" "--sequence"
	do
		if [[ "${batches}" -eq 0 ]] && [[ ! -z "${sequence}" ]]
		then
			continue
		fi
		if [[ "${workers}" -gt 1 ]] && [[ ! -z "${sequence}" ]]
		then
			continue
		fi
		
		# benzina
		VER="${BENZINA_COMMIT_HASH}"
		jug ${cmd} -- benzina_bench.py --arch=${arch} \
			--workers=1 \
			--epochs=2 \
			--batch-size=${batch_size} \
			--batches=${batches} \
			--seed=1234 \
			--out="${OUT}/benzina" \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_benzina \
			"${VER}" \
			"$@" >> benzina_bench.out 2>> benzina_bench.err

		# torchvision
		VER="$(python -m pip freeze | grep "torch")"
		VER="$(echo ${VER//[[:blank:]]/})"
		jug ${cmd} -- torchvision_bench.py --arch=${arch} \
			--workers=${workers} \
			--epochs=2 \
			--batch-size=${batch_size} \
			--batches=${batches} \
			--seed=1234 \
			--out="${OUT}/torchvision" \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_torchvision \
			"$(echo $(grep "torch" results/${MACHINE_NAME}/env))" \
			"$@" >> torchvision_bench.out 2>> torchvision_bench.err

		# dali
		VER="$(python -m pip freeze | grep "dali")"
		VER="$(echo ${VER//[[:blank:]]/})"
		jug ${cmd} -- dali_bench.py --arch=${arch} \
			--workers=${workers} \
			--epochs=2 \
			--batch-size=${batch_size} \
			--batches=${batches} \
			--seed=1234 \
 			--out="${OUT}/dali" \
			--gpu=0 \
			${sequence} \
			${DS_DIR}/imagenet_dali \
			"$(echo $(grep "dali" results/${MACHINE_NAME}/env))" \
			"$@" >> dali_bench.out 2>> dali_bench.err
	done
	done
	done
done
done
done
