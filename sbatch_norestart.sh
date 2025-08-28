#!/bin/bash
#SBATCH --output=netscratch/schiesser/output_%j.log
#SBATCH --partition=H100-SEE,H100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=220G

set -euo pipefail

# Default values, overridden by --export
CONFIG_PATH=${CONFIG_PATH:-NewVisAllStart-S.yaml}
JOB_NAME=${JOB_NAME:-pictsure}

# Dynamically set the job name
scontrol update JobId=$SLURM_JOB_ID Name=$JOB_NAME


CMD="python3 trainer.py --config ./configs/models/$CONFIG_PATH -n"

echo "$CMD"

srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"$(pwd)":"$(pwd)",/ds:/ds \
--container-workdir="$(pwd)" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh \
install.sh $CMD

echo "Job finished at $(date)"

