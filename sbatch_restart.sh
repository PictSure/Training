#!/bin/bash
#SBATCH --output=netscratch/schiesser/output_%j.log
#SBATCH --partition=H100-SEE
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=220G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@90

set -euo pipefail

# Default values, overridden by --export
CONFIG_PATH=${CONFIG_PATH:-NewVisAllStart-S.yaml}
JOB_NAME=${JOB_NAME:-pictsure}

# Dynamically set the job name
scontrol update JobId=$SLURM_JOB_ID Name=$JOB_NAME

function _requeue_on_timeout {
    echo "Caught SIGTERM, requeuing job at $(date)..."
    scontrol requeue $SLURM_JOB_ID
}
trap _requeue_on_timeout SIGTERM
SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
CMD="python3 trainer.py --config ./configs/models/$CONFIG_PATH"
if [ "$SLURM_RESTART_COUNT" -eq 0 ]; then
    CMD="$CMD -n"
fi

echo "$CMD"

srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"$(pwd)":"$(pwd)",/ds:/ds \
--container-workdir="$(pwd)" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
install.sh $CMD

echo "Job finished at $(date)"

