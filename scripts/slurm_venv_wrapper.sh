#!/bin/bash

if [[ ! -z "${SLURM_TMPDIR}" ]]
then
	mkdir -p ${SLURM_TMPDIR}/localscratch
fi
scripts/venv_wrapper.sh --tmpdir ${SLURM_TMPDIR}/localscratch --super_ds /network/datasets $@
