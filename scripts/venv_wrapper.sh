#!/bin/bash

source scripts/utils.sh

function exit_on_error_code {
	ERR=$?
	if [[ $ERR -ne 0 ]]
	then
		>&2 echo "$(tput setaf 1)ERROR$(tput sgr0): $1: $ERR"
		exit $ERR
	fi
}

function _load_gitannex {
	which conda || module load miniconda/3
	# Configure conda for bash shell
	eval "$(conda shell.bash hook)"
	load_gitannex $@
	# Unload to prevent conflict with other python module loads
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

while [[ $# -gt 0 ]]
do
	arg=$1
	case ${arg} in
		--tree_ish)
		shift
		TREE_ISH=$1
		shift
		echo "tree_ish = [${TREE_ISH}]"
		;;
		--super_ds)
		shift
		SUPER_DS=$1
		shift
		echo "super_ds = [${SUPER_DS}]"
		if [[ ! -d ${SUPER_DS} ]]
		then
			>&2 echo "--super_ds DIR option must be an existing directory"
			unset SUPER_DS
		fi
		;;
		--tmpdir)
		shift
		TMPDIR=$1
		shift
		echo "tmpdir = [${TMPDIR}]"
		if [[ ! -d ${TMPDIR} ]]
		then
			>&2 echo "--tmpdir DIR option must be an existing directory"
			unset TMPDIR
		fi
		;;
		-h | --help)
		>&2 echo "Options for $(basename "$0") are:"
		>&2 echo "--tree_ish TREE-ISH Benzina git tree-ish to use"
		>&2 echo "--super_ds DIR super dataset path"
		>&2 echo "[--tmpdir DIR] tmp dir to hold conda, virtualenv prefixes and datasets (optional)"
		exit 1
		;;
		*)
		break
		;;
	esac
done

if [[ -z ${TREE_ISH} ]] || [[ -z ${SUPER_DS} ]]
then
	>&2 echo "--tree_ish TREE-ISH Benzina git tree-ish to use"
	>&2 echo "--super_ds DIR option must be an existing directory"
	>&2 echo "Missing --tree_ish and/or --super_ds options"
	exit 1
fi

if [[ -z "${TMPDIR}" ]]
then
	TMPDIR=tmp
fi

DS_TMPDIR="${TMPDIR}/datasets/"

which git-annex || _load_gitannex --tmpdir "${TMPDIR}/"; exit_on_error_code
module_load_bench; exit_on_error_code
setup_bench_env --tree_ish ${TREE_ISH} --tmpdir "${TMPDIR}/"; exit_on_error_code
if [[ ! -z "$(git diff --name-only && git diff --cached --name-only && datalad diff)" ]]
then
	$(exit 1); exit_on_error_code "Environment is dirty, Datalad will not run"
fi
exit_on_error_code "Failed to check if environment is clean"
copy_datasets; exit_on_error_code

datalad run scripts/run_benches.sh --ds_dir ${DS_TMPDIR} "$@"
