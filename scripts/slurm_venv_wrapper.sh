#!/bin/bash

scripts/venv_wrapper.sh --tmpdir $SLURM_TMPDIR/localscratch --super_ds /network/datasets $@
