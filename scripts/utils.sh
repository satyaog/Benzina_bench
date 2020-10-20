#!/bin/bash

function exit_on_error_code {
	ERR=$?
	if [[ $ERR -ne 0 ]]
	then
		>&2 echo "$(tput setaf 1)ERROR$(tput sgr0): $1: $ERR"
		exit $ERR
	fi
}

function load_gitannex {
	PYTHONNOUSERSITE=true

	while [[ $# -gt 0 ]]
	do
		arg=$1
		case ${arg} in
			--tmpdir)
			shift
			TMPDIR=$1
			shift
			echo "tmpdir = [${TMPDIR}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "[--tmpdir DIR] tmp dir to hold conda prefix"
			exit 1
			;;
		esac
	done

	if [[ -z "${TMPDIR}" ]]
	then
		TMPDIR=tmp
	fi
	
	# Configure conda for bash shell
	eval "$(conda shell.bash hook)"

	if [[ ! -d "${TMPDIR}/env/gitannex/" ]]
	then
		conda create --prefix "${TMPDIR}/env/gitannex/" --yes --no-default-packages
		exit_on_error_code "Failed to create git-annex conda env"
	fi

	conda activate "${TMPDIR}/env/gitannex/" && \
		conda install --strict-channel-priority \
			--use-local -c defaults -c conda-forge \
			git-annex=7.20190819
	exit_on_error_code "Failed to activate git-annex conda env"
}

function module_load_bench {
	module load python/3.7 \
		cuda/10.1 \
		python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.4.1
	exit_on_error_code "Failed to load bench modules"
}

function load_bench_env {
	PYTHONNOUSERSITE=true

	while [[ $# -gt 0 ]]
	do
		arg=$1
		case ${arg} in
			--tmpdir)
			shift
			TMPDIR=$1
			shift
			echo "tmpdir = [${TMPDIR}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "[--tmpdir DIR] tmp dir to hold conda, virtualenv prefixes and datasets (optional)"
			exit 1
			;;
		esac
	done

	if [[ -z "${TMPDIR}" ]]
	then
		TMPDIR=tmp
	fi

	if [[ ! -d "${TMPDIR}/venv/bzna/" ]]
	then
		mkdir -p "${TMPDIR}/venv/bzna/" && \
			virtualenv --no-download "${TMPDIR}/venv/bzna/"
		exit_on_error_code "Failed to create bench venv"
	fi

	source "${TMPDIR}/venv/bzna/"
	exit_on_error_code "Failed to activate bench venv"
	python -m pip install --no-index --upgrade pip
}

function setup_bench_env {
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
			--tmpdir)
			shift
			TMPDIR=$1
			shift
			echo "tmpdir = [${TMPDIR}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--tree_ish TREE-ISH Benzina git tree-ish to use"
			>&2 echo "[--tmpdir DIR] tmp dir to hold conda, virtualenv prefixes and datasets (optional)"
			exit 1
			;;
		esac
	done

	if [[ -z "${TMPDIR}" ]]
	then
		TMPDIR=tmp
	fi

	load_bench_env --tmpdir "${TMPDIR}/"
	exit_on_error_code "Failed to load bench env"

	# Install Datalad
	python -m pip install datalad==0.11.8
	exit_on_error_code "Failed to install datalad"

	python -m pip install -r scripts/requirements_benches.txt
	exit_on_error_code "Failed to install requirements: python -m pip install"

	git clone https://github.com/satyaog/Benzina.git
	(cd Benzina/ && \
	 git fetch && \
	 git checkout ${TREE_ISH} && \
	 git tag --force _bench)
	exit_on_error_code "Failed to clone/checkout Benzina"
	rm -rf "${TMPDIR}/Benzina/"
	git clone --branch _bench --depth 1 -- file://$(cd Benzina/; pwd)/.git/ "${TMPDIR}/Benzina/"
	python -m pip uninstall --yes pybenzinaparse
	(cd "${TMPDIR}/Benzina/" && \
	 python -m pip install meson==0.54 && \
	 python -m pip install .)
	exit_on_error_code "Failed to install Benzina"
}

$@