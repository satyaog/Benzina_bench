#!/bin/bash
# https://stackoverflow.com/a/29754866
# Saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber

PYTHONNOUSERSITE=true

! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]
then
	>&2 echo "`getopt --test` failed indicating that enhanced getopt not available in this environment."
	exit 1
fi

OPTS=h
LONGOPTS=tree-ish:,super-ds:,tmp:,help

! PARSED=$(getopt --options=${OPTS} --longoptions=${LONGOPTS} --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]
then
	exit 2
fi
eval set -- "${PARSED}"

while [[ $# -gt 0 ]]
do
	arg="$1"
	shift # past argument
	case "${arg}" in
		--tree-ish) TREE_ISH="$1"; shift # past value
		echo "tree-ish = [${TREE_ISH}]"
		;;
		--super-ds) SUPER_DS="$1"; shift # past value
		echo "super-ds = [${SUPER_DS}]"
		;;
		--tmp) TMPDIR="$1"; shift # past value
		echo "tmp = [${TMPDIR}]"
		;;
		-h | --help)
		>&2 echo "Options for $(basename "$0") are:"
		>&2 echo "--super-ds DIR super dataset directory"
		>&2 echo "[--tree-ish] TREE-ISH Benzina git tree-ish to use (optional)"
		>&2 echo "[--tmp DIR] tmp dir to hold conda, virtualenv prefixes and datasets (optional)"
		exit 1
		;;
		--) break ;;
		*) >&2 echo "Unknown argument [${arg}]"; exit 3 ;;
	esac
done

if [[ ! -d ${SUPER_DS} ]]
then
	>&2 echo "--super-ds DIR option must be an existing directory"
	unset SUPER_DS
fi

if [[ -z "${SUPER_DS}" ]]
then
	>&2 echo "--super-ds DIR option must be an existing directory"
	>&2 echo "Missing --super-ds option"
	exit 1
fi					

if [[ -z ${TREE_ISH} ]]
then
	TREE_ISH='HEAD'
fi

if [[ -z "${TMPDIR}" ]]
then
	TMPDIR="tmp"
fi

DS_TMPDIR="${TMPDIR}/datasets"
JUG_TMPDIR="${TMPDIR}/jugdir"

###

source scripts/utils.sh echo -n

function _load_gitannex {
	which conda || module load miniconda/3
	# Configure conda for bash shell
	eval "$(conda shell.bash hook)"
	conda activate "${TMPDIR}/env/gitannex/" || \
	exit_on_error_code "Failed to activate git-annex conda env"
	# Unload to prevent conflict with other python module loads
	module unload conda miniconda
}

function _copy_datasets {
	mkdir -p "${DS_TMPDIR}"

	# imagenet_benzina
	jug_exec --jugdir ${JUG_TMPDIR} -- \
		scripts/utils.sh copy_datalad_dataset \
		--src imagenet.var/imagenet_benzina --dest ${DS_TMPDIR}/imagenet_benzina --super-ds ${SUPER_DS} &

	# tinyimagenet_benzina
	jug_exec --jugdir ${JUG_TMPDIR} -- \
		scripts/utils.sh copy_datalad_dataset \
		--src tinyimagenet.var/imagenet_benzina --dest ${DS_TMPDIR}/tinyimagenet_benzina --super-ds ${SUPER_DS} &

	# cifar10_torchvision
	jug_exec --jugdir ${JUG_TMPDIR} -- \
		scripts/utils.sh copy_datalad_dataset \
		--src cifar10 --dest ${DS_TMPDIR}/cifar10_torchvision --super-ds ${SUPER_DS} &

	# cifar100_torchvision
	jug_exec --jugdir ${JUG_TMPDIR} -- \
		scripts/utils.sh copy_datalad_dataset \
		--src cifar100 --dest ${DS_TMPDIR}/cifar100_torchvision --super-ds ${SUPER_DS} &

	# imagenet_torchvision
	jug_exec --jugdir ${JUG_TMPDIR} -- \
		scripts/utils.sh copy_datalad_dataset \
		--src imagenet --dest ${DS_TMPDIR}/imagenet_torchvision --super-ds ${SUPER_DS} &

	# tinyimagenet_torchvision
	jug_exec --jugdir ${JUG_TMPDIR} -- \
		scripts/utils.sh copy_tinyimagenet_torchvision \
		--src tinyimagenet --dest ${DS_TMPDIR}/tinyimagenet_torchvision &

	wait
}

# Use a tmp jug to handle concurrent attempts to setup env
(tmp_jug_exec --tmp "${TMPDIR}" -- \
 	--jugdir "${JUG_TMPDIR}" -- \
 	scripts/utils.sh init_full_env --tree-ish ${TREE_ISH} --tmp "${TMPDIR}")

which git-annex || _load_gitannex
load_modules
source "${TMPDIR}/venv/bzna/bin/activate" || \
exit_on_error_code "Failed to activate bzna venv"
 
if [[ ! -z "$(git diff --name-only && git diff --cached --name-only && datalad diff)" ]]
then
	(exit 1) || \
	exit_on_error_code "Environment is dirty, Datalad will not run"
fi
exit_on_error_code "Failed to check if environment is clean"
_copy_datasets
datalad run scripts/orion.sh --ds-dir ${DS_TMPDIR} "$@"
