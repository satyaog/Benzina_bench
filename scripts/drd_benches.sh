#!/bin/bash
set -o errexit -o pipefail -o noclobber

if [[ -z ${SLURM_CPUS_PER_TASK} ]]
then
	_max_workers=${SLURM_JOB_CPUS_PER_NODE}
elif [[ ! -z ${SLURM_CPUS_PER_TASK} ]]
then
	_max_workers=${SLURM_CPUS_PER_TASK}
else
	_max_workers=32
fi

while [[ $# -gt 0 ]]
do
	_arg="$1"; shift
	case "${_arg}" in
		--ds-dir)
			_ds_dir="$1"; shift
			echo "ds-dir = [${_ds_dir}]"
			if [[ ! -d ${_ds_dir} ]]
			then
				>&2 echo "--ds-dir DIR option must be an existing directory"
				unset _ds_dir
			fi
			;;
		--max-workers)
			__max_workers="$1"; shift
			if [[ "${__max_workers}" -lt "${_max_workers}" ]] || [[ -z ${_max_workers} ]]
			then
				_max_workers=${__max_workers}
			fi
			unset __max_workers
			echo "max-workers = [${_max_workers}]"
			;;
		--batch-sizes)
			_batch_sizes="$1"; shift
			echo "batch-sizes = [${_batch_sizes}]"
			;;
		--batches)
			_batches="$1"; shift
			echo "batches = [${_batches}]"
			;;
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

if [[ -z ${_ds_dir} ]]
then
	>&2 echo "--ds-dir DIR option must be an existing directory"
	>&2 echo "Missing --ds-dir option"
	exit 1
fi

if [[ -z ${_max_workers} ]]
then
	_max_workers=1
fi

if [[ -z ${_batch_sizes} ]]
then
	_batch_sizes="128 256"
fi

if [[ -z ${_batches} ]]
then
	_batches="1000 0"
fi

MACHINE_NAME=$(hostname | grep -o "[^\.]*" | head -1)

mkdir -p results/${MACHINE_NAME}
touch results/${MACHINE_NAME}/bcachefs_drd_measures.csv
touch results/${MACHINE_NAME}/hdf5_drd_measures.csv

rm -f results/${MACHINE_NAME}/env
echo "# BENCH COMMIT HASH:" $(git rev-parse HEAD) > results/${MACHINE_NAME}/env
echo "# BCACHEFS COMMIT HASH:" $(python3 -m pip freeze | grep -o "^bcachefs.*") >> results/${MACHINE_NAME}/env
# Trim the trailing ".server.mila.quebec"
echo "# MACHINE NAME:" ${MACHINE_NAME} >> results/${MACHINE_NAME}/env
echo "# CPU:" $(cat /proc/cpuinfo | grep "model name" | sort | uniq) >> results/${MACHINE_NAME}/env
echo "# GPU:" $(nvidia-smi -q | grep "Product Name") >> results/${MACHINE_NAME}/env
pip freeze >> results/${MACHINE_NAME}/env

for cmd in "status" "execute"; do
for batches in ${_batches}; do
for workers in 1 2 4 6 8 16 32; do
	if [[ "${workers}" -gt "${_max_workers}" ]]
	then
		continue
	fi
	for batch_size in ${_batch_sizes}; do
	for dl_only in "" "--dl-only"; do
	for sequence in "" "--sequence"; do
		if [[ -z "${dl_only}" ]] && [[ "${batches}" -eq 0 ]] && { [[ "${workers}" -eq 1 ]] || [[ ! "${workers}" -eq "${_max_workers}" ]]; }
		then
			continue
		fi
		if [[ "${workers}" -gt 1 ]] && [[ ! -z "${sequence}" ]]
		then
			continue
		fi

		ln -sf results/${MACHINE_NAME}/bcachefs_drd_measures.csv measures.csv
		{ time jug ${cmd} -- bcachefs_drd_bench.py \
			--workers=${workers} \
			--epochs=1 \
			--batch-size=${batch_size} \
			--chan-cnt=3 \
			--batches=${batches} \
			--seed=1234 \
			--gpu=0 \
			${dl_only} \
			${sequence} \
			${_ds_dir}/diabetic-retinopathy-detection_bcachefs/*.img \
			"$(echo $(grep -o "^bcachefs.*" results/${MACHINE_NAME}/env))" >> bcachefs_drd_bench.out 2>> bcachefs_drd_bench.err ; } 2>> bcachefs_drd_bench.out \
			|| [[ "${cmd}" == "status" ]]

		ln -sf results/${MACHINE_NAME}/hdf5_drd_measures.csv measures.csv
		{ time jug ${cmd} -- hdf5_drd_bench.py \
			--workers=${workers} \
			--epochs=1 \
			--batch-size=${batch_size} \
			--chan-cnt=3 \
			--batches=${batches} \
			--seed=1234 \
			--gpu=0 \
			${dl_only} \
			${sequence} \
			${_ds_dir}/diabetic-retinopathy-detection_hdf5_single_img/*.h5 \
			"$(echo $(grep -o "^h5py.*" results/${MACHINE_NAME}/env))" >> hdf5_drd_bench.out 2>> hdf5_drd_bench.err ; } 2>> hdf5_drd_bench.out \
			|| [[ "${cmd}" == "status" ]]
	done
	done
	done
done
done
done

rm measures.csv
