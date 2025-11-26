#!/bin/bash

start=`date +%s`

echo "syncing SubpopBench repo in $SLURM_TMPDIR"
rsync -av ../SubpopBench $SLURM_TMPDIR --exclude output --exclude .venv --exclude .git --exclude slurm_logs
cd $SLURM_TMPDIR/SubpopBench

bash scripts/create_venv_slurm.sh
echo "activating virtual environment"
source .venv/bin/activate
echo "running experiment"

end=`date +%s`
echo "Intractive session installed in $((end - start)) seconds"