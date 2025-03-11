# test
srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`",/ds:/ds \
--container-workdir="`pwd`" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
--task-prolog="`pwd`/install.sh" \
--gpus=1 --cpus-per-gpu=4 --partition=H100-SEE --mem=190000 --job-name=icltest \
python3 train.py --config ./configs/slurm.yaml

# S-Model
srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`",/ds:/ds \
--container-workdir="`pwd`" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
--task-prolog="`pwd`/install.sh" \
--gpus=1 --cpus-per-gpu=16 --partition=H100-SEE --mem=205000 --job-name=pictsures \
python3 train.py --config ./configs/models/PictSureS.yaml

# M-Model
srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`",/ds:/ds \
--container-workdir="`pwd`" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
--task-prolog="`pwd`/install.sh" \
--gpus=1 --cpus-per-gpu=16 --partition=H100-SEE --mem-per-cpu=20GB --job-name=pictsurem  \
python3 train.py --config ./configs/models/PictSureM.yaml

# L-Model
srun \
--container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`",/ds:/ds \
--container-workdir="`pwd`" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
--task-prolog="`pwd`/install.sh" \
--gpus=1 --cpus-per-gpu=16 --partition=A100-80GB --mem-per-cpu=25G --job-name=pictsurel --time=3-00:00 \
python3 train.py --config ./configs/models/PictSureL.yaml
