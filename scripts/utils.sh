#!/bin/bash

PYTHONNOUSERSITE=true

function exit_on_error_code {
	ERR=$?
	if [[ ${ERR} -ne 0 ]]
	then
		>&2 echo "$(tput setaf 1)ERROR$(tput sgr0): $1: ${ERR}"
		exit ${ERR}
	fi
}

function slurm_wrapper {
	mkdir -p "${SLURM_TMPDIR}/localscratch/"

	script=$1
	shift
	${script} --tmp "${SLURM_TMPDIR}/localscratch" --super-ds "/network/datasets" "$@"
}

function jug_exec {
	JUG_ARGV=()
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--) break ;;
			*) JUG_ARGV+=("${arg}") ;;
		esac
	done
	# Remove trailing '/' in argv before sending to jug
	scripts/jug_exec.py "${JUG_ARGV[@]%/}" -- "${@%/}"
	jug sleep-until "${JUG_ARGV[@]%/}" scripts/jug_exec.py -- "${@%/}"
}

function tmp_jug_exec {
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--tmp) TMPDIR="$1"; shift # past value
			echo "tmp = [${TMPDIR}]"
			;;
			-h | --help)
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--tmp DIR tmp dir to hold conda, virtualenv prefixes and datasets"
			exit 1
			;;
			--) break ;;
			*) >&2 echo "Unknown argument [${arg}]"; exit 3 ;;
		esac
	done
	which python3 || module load python/3.6
	which virtualenv || module load python/3.6
	_tmpjug=`mktemp -d -p ${TMPDIR}`
	trap "rm -rf ${_tmpjug}" EXIT
	init_venv --name jug --tmp "${_tmpjug}"
	python -m pip install -r scripts/requirements_jug.txt
	jug_exec "$@"
}

function init_full_env {
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--tree-ish) TREE_ISH="$1"; shift # past value
			echo "tree-ish = [${TREE_ISH}]"
			;;
			--tmp) TMPDIR="$1"; shift # past value
			echo "tmp = [${TMPDIR}]"
			;;
			-h | --help)
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--tree-ish TREE-ISH Benzina git tree-ish to use"
			>&2 echo "--tmp DIR tmp dir to hold conda, virtualenv prefixes and datasets"
			exit 1
			;;
			--) break ;;
			*) >&2 echo "Unknown argument [${arg}]"; exit 3 ;;
		esac
	done

	which conda || module load miniconda/3 || \
		{ module unload python && module load miniconda/3; }
	install_gitannex --tmp "${TMPDIR}"
	module unload conda miniconda

	load_modules
	init_venv --name bzna --tmp "${TMPDIR}"
	python -m pip install -r scripts/requirements_jug.txt || \
	exit_on_error_code "Failed to install jug"
	install_requirements --tree-ish ${TREE_ISH} --tmp "${TMPDIR}"
}

function install_gitannex {
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--tmp) TMPDIR="$1"; shift # past value
			echo "tmp = [${TMPDIR}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--tmp DIR tmp dir to hold the conda prefix"
			exit 1
			;;
		esac
	done

	# Configure conda for bash shell
	eval "$(conda shell.bash hook)"

	if [[ ! -d "${TMPDIR}/env/gitannex/" ]]
	then
		conda create --prefix "${TMPDIR}/env/gitannex/" --yes --no-default-packages || \
		exit_on_error_code "Failed to create git-annex conda env"
	fi

	conda activate "${TMPDIR}/env/gitannex/" && \
	conda install --yes --strict-channel-priority \
		--use-local -c defaults -c conda-forge \
		git-annex=7.20190819 || \
	exit_on_error_code "Failed to install git-annex"
}

function load_modules {
	module load python/3.7 \
		cuda/10.1 \
		python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.4.1 || \
	exit_on_error_code "Failed to load dl bench modules"
}

function init_venv {
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--name) NAME="$1"; shift # past value
			echo "name = [${NAME}]"
			;;
			--tmp) TMPDIR="$1"; shift # past value
			echo "tmp = [${TMPDIR}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--name NAME venv prefix name"
			>&2 echo "--tmp DIR tmp dir to hold the virtualenv prefix"
			exit 1
			;;
		esac
	done

	if [[ -z "${NAME}" ]]
	then
		>&2 echo "--name NAME venv prefix name"
		>&2 echo "--tmp DIR tmp dir to hold the virtualenv prefix"
		>&2 echo "Missing --name and/or --tmp options"
		exit 1
	fi

	if [[ ! -d "${TMPDIR}/venv/${NAME}/" ]]
	then
		mkdir -p "${TMPDIR}/venv/${NAME}/" && \
		virtualenv --no-download "${TMPDIR}/venv/${NAME}/" || \
		exit_on_error_code "Failed to create ${NAME} venv"
	fi

	source "${TMPDIR}/venv/${NAME}/bin/activate" || \
	exit_on_error_code "Failed to activate ${NAME} venv"
	python -m pip install --no-index --upgrade pip
}

function install_requirements {
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--tmp) TMPDIR="$1"; shift # past value
			echo "tmp = [${TMPDIR}]"
			;;
			--tree-ish) TREE_ISH="$1"; shift # past value
			echo "tree-ish = [${TREE_ISH}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--tree-ish TREE-ISH Benzina git tree-ish to use"
			>&2 echo "--tmp DIR tmp dir to hold the virtualenv prefix"
			exit 1
			;;
		esac
	done

	python -m pip install -r scripts/requirements.txt || \
	exit_on_error_code "Failed to install requirements: python -m pip install"

	! git clone https://github.com/satyaog/Benzina.git
	git -C Benzina/ fetch
	TREE_ISH_HASH=`git -C Benzina/ rev-parse --revs-only --short ${TREE_ISH} --`
	git -C Benzina/ tag --force "_bench_${TREE_ISH_HASH}" ${TREE_ISH}
	exit_on_error_code "Failed to clone/checkout Benzina"

	! git clone --branch "_bench_${TREE_ISH_HASH}" --depth 1 -- \
		"file://$(cd Benzina/; pwd)/.git/" "${TMPDIR}/Benzina_${TREE_ISH_HASH}/"
	python -m pip uninstall --yes pybenzinaparse
	(cd "${TMPDIR}/Benzina_${TREE_ISH_HASH}/" && \
	 python -m pip install meson==0.54 && \
	 python -m pip install .) || \
	exit_on_error_code "Failed to install Benzina"
}

function link_cache_0fea6a {
	git config remote.cache-0fea6a.url $1
	git config remote.cache-0fea6a.fetch \
		+refs/heads/empty_branch:refs/remotes/cache-0fea6a/empty_branch
	git config remote.cache-0fea6a.annex-speculate-present true
	git config remote.cache-0fea6a.annex-pull false
	git config remote.cache-0fea6a.annex-push false
}

function copy_datalad_dataset {
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--src) SRC="$1"; shift # past value
			echo "src = [${SRC}]"
			;;
			--dest) DEST="$1"; shift # past value
			echo "dest = [${DEST}]"
			;;
			--super-ds) SUPER_DS="$1"; shift # past value
			echo "super-ds = [${SUPER_DS}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--src {DIR|DATASET} source dataset directory or name"
			>&2 echo "--dest DIR ddestination directory"
			>&2 echo "--super-ds DIR super dataset directory"
			exit 1
			;;
		esac
	done

	if [[ ! -d "${SRC}" ]]
	then
		SRC=${SUPER_DS}/${SRC}
	fi

	mkdir -p ${DEST}

	! datalad install -s ${SRC}/ ${DEST}/
	(cd ${DEST}/ && \
	 link_cache_0fea6a ${SUPER_DS}/.annex-cache && \
	 git-annex get --fast --from cache-0fea6a || \
	 git-annex get --fast --from origin || \
	 git-annex get --fast) || \
	exit_on_error_code "Failed to copy dataset ${SRC}"
}

function copy_tinyimagenet_torchvision {
	while [[ $# -gt 0 ]]
	do
		arg="$1"; shift # past argument
		case "${arg}" in
			--src) SRC="$1"; shift # past value
			echo "src = [${SRC}]"
			;;
			--dest) DEST="$1"; shift # past value
			echo "dest = [${DEST}]"
			;;
			-h | --help | *)
			if [[ "${arg}" != "-h" ]] && [[ "${arg}" != "--help" ]]
			then
				>&2 echo "Unknown option [${arg}]"
			fi
			>&2 echo "Options for $(basename "$0") are:"
			>&2 echo "--src DIR source dataset directory"
			>&2 echo "--dest DIR ddestination directory"
			exit 1
			;;
		esac
	done

	DEST=$(realpath ${DEST})
	mkdir -p ${DEST}

	rm -r ${DEST}/
	unzip ${SRC}/tiny-imagenet-200.zip -d ${DEST}.tmp/ || \
	exit_on_error_code "Failed to unpack tinyimagenet"

	mv ${DEST}.tmp/tiny-imagenet-200/ ${DEST}/
	rm -r ${DEST}.tmp/
	for dir in ${DEST}/test ${DEST}/train/* ${DEST}/val
	do
		mv ${dir}/ ${dir}_/ && \
		mv ${dir}_/images/ ${dir}/ && \
		( [[ -z "$(ls ${dir}_/*)" ]] || mv -t ${dir}/../ ${dir}_/* ) && \
		rm -r ${dir}_/ || \
		exit_on_error_code "Failed to set tinyimagenet's hiearchy to ImageFolder"
	done
	while IFS= read -r line
	do
		line=(${line})
		mkdir -p ${DEST}/val/${line[1]}/ && \
		mv ${DEST}/val/${line[0]} ${DEST}/val/${line[1]}/ || \
		exit_on_error_code "Failed to move tinyimagenet's val images into labeled subfolders"
	done < ${DEST}/val_annotations.txt
}

"$@"
