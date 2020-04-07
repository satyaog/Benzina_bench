#!/bin/bash

function exit_on_error_code {
	ERR=$?
	if [ $ERR -ne 0 ]
	then
		>&2 echo "$(tput setaf 1)ERROR$(tput sgr0): $1: $ERR"
		exit $ERR
	fi
}

function load_git_annex {
	module load miniconda/3

	if [ ! -d "${TMPDIR}/env/git_annex/" ]
	then
		conda create --prefix ${TMPDIR}/env/git_annex/ \
			--yes --no-default-packages \
			-c conda-forge --use-local --no-channel-priority \
			python=$(python -V 2>&1 | grep -Eo "[0-9]+\.[0-9]+\.[0-9]+") \
			git-annex=7.20190819
	fi

	conda activate ${TMPDIR}/env/git_annex/
	exit_on_error_code "Failed to activate git-annex conda env"

	module unload miniconda/3
}

function copy_datasets {
	# Copy datasets
	mkdir -p ${DS_TMPDIR}

	datalad install -s ${SUPER_DS}/imagenet.var/imagenet_benzina \
		${DS_TMPDIR}/imagenet_benzina

	git -C ${DS_TMPDIR}/imagenet_benzina config remote.cache-0fea6a.url \
		${SUPER_DS}/.annex-cache
	git -C ${DS_TMPDIR}/imagenet_benzina config remote.cache-0fea6a.fetch \
		+refs/heads/empty_branch:refs/remotes/cache-0fea6a/empty_branch
	git -C ${DS_TMPDIR}/imagenet_benzina config remote.cache-0fea6a.annex-speculate-present true
	git -C ${DS_TMPDIR}/imagenet_benzina config remote.cache-0fea6a.annex-pull false
	git -C ${DS_TMPDIR}/imagenet_benzina config remote.cache-0fea6a.annex-push false

	(cd ${DS_TMPDIR}/imagenet_benzina && git-annex get --fast --from cache-0fea6a)
	exit_on_error_code "Failed to copy dataset imagenet_benzina"

	mkdir -p ${DS_TMPDIR}/imagenet_torchvision
	cp -aut ${DS_TMPDIR}/imagenet_torchvision ${SUPER_DS}/imagenet.var/imagenet_torchvision/*
	exit_on_error_code "Failed to copy dataset imagenet_torchvision"
}

function module_load_bench {
	module load python/3.7 \
		cuda/10.1 \
		python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.4.1
}

function setup_bench_env {
	if [ ! -d "${TMPDIR}/venv/bzna/" ]
	then
		mkdir -p ${TMPDIR}/venv/
		virtualenv --no-download ${TMPDIR}/venv/bzna/
	fi

	source ${TMPDIR}/venv/bzna/bin/activate

	pip install \
		-r scripts/requirements_benches.txt
	exit_on_error_code "Failed to install requirements: pip install"

	git clone https://github.com/satyaog/Benzina.git
	(cd Benzina/ && \
	 git fetch && \
	 git checkout ${TREE_ISH} && \
	 git tag --force _bench)
	exit_on_error_code "Failed to clone Benzina"
	rm -rf ${TMPDIR}/Benzina/
	git clone --branch _bench --depth 1 -- file://$(cd Benzina; pwd)/.git/ ${TMPDIR}/Benzina/
	pip uninstall --yes pybenzinaparse
	(cd ${TMPDIR}/Benzina/ && \
	 pip install meson==0.51.1 pytest==6.0.1 && \
	 pip install .)
	exit_on_error_code "Failed to install Benzina"

	# Install Datalad
	pip install datalad==0.11.8
}

PYTHONNOUSERSITE=true

for ((i = 1; i <= ${#@}; i++))
do
	arg=${!i}
	case ${arg} in
		--tree_ish)
		i=$((i+1))
		TREE_ISH=${!i}
		echo "tree_ish = [${TREE_ISH}]"
		;;
		--super_ds)
		i=$((i+1))
		SUPER_DS=${!i}
		echo "super_ds = [${SUPER_DS}]"
		if [ ! -d ${SUPER_DS} ]
		then
			>&2 echo "--super_ds DIR option must be an existing directory"
			unset SUPER_DS
		fi
		;;
		--tmpdir)
		i=$((i+1))
		TMPDIR=${!i}
		echo "tmpdir = [${TMPDIR}]"
		if [ ! -d ${TMPDIR} ]
		then
			>&2 echo "--tmpdir DIR option must be an existing directory"
			unset TMPDIR
		fi
		;;
		-h | --help | *)
		>&2 echo "Unknown option [${arg}]. Valid options are:"
		>&2 echo "--tree_ish TREE-ISH Benzina git tree-ish to use"
		>&2 echo "--super_ds DIR super dataset path"
		>&2 echo "[--tmpdir DIR] tmp dir to hold conda, virtualenv prefixes and datasets (optional)"
		exit 1
		;;
	esac
done

if [ -z ${TREE_ISH} ] || [ -z ${SUPER_DS} ]
then
	>&2 echo "--tree_ish TREE-ISH Benzina git tree-ish to use"
	>&2 echo "--super_ds DIR option must be an existing directory"
	>&2 echo "Missing --tree_ish and/or --super_ds options"
	exit 1
fi

if [ -z ${TMPDIR} ]
then
	TMPDIR=tmp
fi

DS_TMPDIR=${TMPDIR}/datasets

which git-annex || load_git_annex; exit_on_error_code
module_load_bench; exit_on_error_code
setup_bench_env; exit_on_error_code
copy_datasets; exit_on_error_code

datalad run scripts/run_benches.sh --ds_dir ${DS_TMPDIR}
