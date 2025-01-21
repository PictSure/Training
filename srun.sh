srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" \
--container-workdir="`pwd`" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
--task-prolog="`pwd`/install.sh" \
--gpus=1 --partition=A100-40GB --mem=30000 \
python3 train.py