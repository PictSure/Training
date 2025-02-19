srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`",/ds:/ds \
--container-workdir="`pwd`" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
--task-prolog="`pwd`/install.sh" \
--gpus=1 --partition=H100-SEE --mem=120000 --job-name=icltest \
python3 train.py --config ./configs/slurm.yaml