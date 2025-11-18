# PictSure: Training In-Context Learning Image Classifiers

> Reference implementation for the training of the PictSure In-Context Learning Image classifiers presented in **“PictSure: Pretraining Embeddings Matters for In-Context Learning Image Classifiers”** ([arXiv:2506.14842](https://arxiv.org/abs/2506.14842)).

PictSure rethinks few-shot image classification (FSIC) by centering the **visual embedding model** within an in-context learning (ICL) pipeline. The codebase lets you reproduce the paper’s results, explore alternative encoders (ResNet, ViT, DINOv2/v3, CLIP), and run large-scale experiments on Slurm clusters or small-scale ablations on laptops.

## Why PictSure?

- **Embedding-first methodology** – isolates how pretraining objectives, architectures, and fine-tuning strategies affect downstream ICL performance.
- **Transformer reasoning head** – `CustomTransformerModel` consumes support/query sets (image + label tokens) and predicts the query class in one forward pass.
- **Episode-driven dataloaders** – custom CIFAR/ImageNet/Datadings loaders that form balanced few-shot tasks on the fly, mirroring FSIC evaluation.
- **Cluster-ready tooling** – shipping Slurm scripts, container hooks, resumable training, and long-horizon LR schedules for 1k+ epoch runs.
- **Batteries-included experimentation** – configs for each encoder, utilities to pretrain ViTs with triplet loss, and tests to sanity-check embeddings.

## Repository tour

| Path | Purpose |
| --- | --- |
| `trainer.py`, `train.py` | Main training entry points (class-based trainer vs. script). |
| `model/` | Transformer head (`model_PictSure.py`), wrapper modules for ResNet/ViT/DINOv2/v3/CLIP, and ViT backbone code. |
| `utils/` | Dataloaders (CIFAR/ImageNet/Datadings), scheduler, logging (`SummaryWriter`), dataset/model factories. |
| `configs/` | Ready-to-run YAML configs (`local.yaml`, `cifar.yaml`, cluster presets in `configs/models/`). |
| `generate_dd.py` | Converts ImageNet / ImageNet-21k folders into MsgPack shards for fast cluster sampling. |
| `pretrain_imagenet.py` | Optional ViT triplet-pretraining routine used in the paper. |
| `tests/` | Lightweight sanity checks (e.g., `wrapper_test.py` verifies embedding wrappers and preprocessing). |
| `sbatch_*.sh`, `srun.sh`, `install.sh` | Slurm launchers + container bootstrap for large-scale runs. |

## Getting started

### Prerequisites

- Python 3.10+ and a recent PyTorch build with CUDA, ROCm, or Apple Silicon (Metal) acceleration.
- GPU with ≥24 GB memory recommended for Dinov3/CLIP runs; CPU-only mode is supported for smoke tests.
- (Optional) `HF_TOKEN` env var with access to `facebook/dinov3-vith16plus-pretrain-lvd1689m` on Hugging Face.
- ImageNet / ImageNet-21k data if you plan to mirror paper-scale experiments; CIFAR-10 downloads automatically.

### Installation

```bash
git clone https://github.com/PictSure/embed-then-classify.git
cd embed-then-classify
python -m venv .venv && source .venv/bin/activate  # or use conda/mamba
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Cluster note** – `install.sh` is already wired into the Slurm scripts to perform the same pip install on the first task per node.

### Quick health check

Run the embedding-wrapper sanity test (downloads CIFAR-10 if missing) to ensure preprocessing + encoders work on your device:

```bash
pytest tests/wrapper_test.py -k DINOV2Wrapper --maxfail=1 --disable-warnings
```

## Dataset preparation

### CIFAR-10 episodes

Nothing to do—`utils/data_loader_cifar10.py` downloads data automatically and resizes samples to 224×224 when `resize_to_224=True` (see `configs/cifar.yaml`).

### ImageNet / ImageNet-21k episodes (MsgPack via Datadings)

The cluster loaders expect MsgPack shards. Use `generate_dd.py` to convert your datasets once:

```bash
# ImageNet train split (exclude held-out test classes)
python generate_dd.py \
  --input /path/to/imagenet \
  --output data/imagenet-train.msgpack \
  --dataset inet

# ImageNet validation split used as FSIC test episodes
python generate_dd.py \
  --input /path/to/imagenet \
  --output data/imagenet-test.msgpack \
  --dataset inet --test

# ImageNet-21k
python generate_dd.py \
  --input /path/to/imagenet21k \
  --output data/imagenet21k-train.msgpack \
  --dataset inet21k
```

Tips:

- `utils/cluster_dataloader.py` can cache per-class tensors; set `paths.class_index` in your config to reuse a precomputed pickle file.
- The paper’s held-out evaluation classes for ImageNet-21k are encoded in `generate_dd.py`; reusing them reproduces the benchmark splits.

## Configuring experiments

Every run is driven by a YAML file in `configs/` or `configs/models/`. Key sections:

| Field | Meaning |
| --- | --- |
| `training_loc` | Chooses which dataloader to instantiate: `cluster` (MsgPack/ImageNet-21k), `imagenet`, or `cifar`. |
| `encoder` / `resnet` / `dinov2` etc. | Selects the visual backbone. Set `encoder: dinov3` (preferred) or the legacy individual flags (`resnet: 50`, `clip: true`, …). |
| `paths` | Locations for datasets, MsgPack shards, output directories, optional ViT checkpoint (`visnet_weights`), and cached `class_index`. |
| `model` | Transformer hyperparameters (`nheads`, `nlayers`, `embed_dim`). |
| `optimizer` | Training length, learning rates for encoder vs. transformer, label smoothing (`epsilon`), gradient accumulation, and LR scheduler endpoints. |
| `dataloader` | Episode shape (number of classes/images), batch size, worker count, class resample cadence, etc. |
| `resample` | (Cluster only) how frequently to rebuild the sampled class cache. |

Use `configs/local.yaml` as a starting point for workstation experiments and switch to the presets in `configs/models/` to reproduce paper-scale runs (e.g., `PictSureDinov3.yaml`, `PictSureCLIP.yaml`).

## Running training

### Local / single-GPU workflow

```bash
# fresh run
python trainer.py --config configs/local.yaml -n

# resume from most recent checkpoint written to paths.output/name_YYYYMMDD_HHMMSS
python trainer.py --config configs/local.yaml
```

`trainer.py` handles checkpointing, learning-rate schedules, and logging via `SummaryWriter`. `train.py` is a simpler script that exposes the same functionality without the OO wrapper; both respect the same configs.

### Cluster / Slurm workflow

- **One-off run with srun** (see `srun.sh` for templates) – mounts your workspace into the NVIDIA PyTorch container and calls `train.py`.
- **Managed job with sbatch** – `sbatch_norestart.sh` (single shot) and `sbatch_restart_clean.sh` (auto requeue on preemption) run `trainer.py` inside the container. Override configs via environment variables:

```bash
sbatch --export=CONFIG_PATH=PictSureDinov3.yaml,JOB_NAME=pictsuredv3 sbatch_restart_clean.sh
```

Both scripts automatically run `install.sh`, which prepares Python + dependencies once per node, and pass along `HF_TOKEN` when provided.

## Monitoring, checkpoints, and outputs

- **Logging** – every run writes `hyperparameters.json`, `batch_metrics.csv`, and `epoch_metrics.csv` under `paths.output/<run_name_timestamp>/`.
- **Checkpoints** – `checkpoint.pt` plus `best_loss_model.pt`, `best_acc_model.pt`, and `final_model.pt` are saved in the same directory.
- **Metric visualization** – load the CSV files into your plotting tool of choice or tail them live with `watch csvlook ...`. Notebooks such as `dataset.ipynb` show how to analyze them.

## Pretraining embeddings (optional but paper-aligned)

The paper reports that stronger embedding pretraining dramatically boosts FSIC. Use `pretrain_imagenet.py` to run a triplet-augmented ViT pretraining pass on ImageNet with stratified sampling:

```bash
python pretrain_imagenet.py --data_path /path/to/imagenet --device cuda
```

Point `paths.visnet_weights` to the produced checkpoint so `VitNetWrapper` can load it.

## Reproducing paper results

1. Convert ImageNet-21k data to MsgPack (see above) and copy the paper’s configs from `configs/models/`.
2. Export `HF_TOKEN=<your_token>` if you plan to use DINOv3.
3. Launch the relevant Slurm script:

```bash
sbatch --export=CONFIG_PATH=PictSureDinov3.yaml,JOB_NAME=pictsure_dv3 sbatch_restart_clean.sh
```

4. After training, compare `epoch_metrics.csv` across encoders; Figure 3 of the paper corresponds to the Dinov3 vs. CLIP configs logged in `accuracy.pdf`.

## Testing & linting

- **Embedding wrappers** – `pytest tests/wrapper_test.py --config configs/local.yaml --wrapper DINOV2Wrapper`.
- **Style / lint** – the project relies on PyTorch + standard library, so `ruff`/`black` are optional. If you add them, remember to document the commands here.

## Troubleshooting

- **HF downloads fail** – ensure `HF_TOKEN` is exported when running on clusters without anonymous access to `facebook/dinov3-vith16plus-pretrain-lvd1689m`.
- **pin_memory warnings on MPS** – set `dataloader.pin_memory: false` in configs when running on Apple Silicon.
- **“not enough images” errors** – decrease `dataloader.num_images` or increase `resample` frequency so the MsgPack cache fits into GPU memory.
- **Slow datadings decode** – precompute `paths.class_index` once (see `cluster_dataloader.py`) and point configs to the resulting pickle.

## Citation

If you use this work, please cite the paper:

```
@article{schiesser2025pictsure,
	title   = {PictSure: Pretraining Embeddings Matters for In-Context Learning Image Classifiers},
	author  = {Lukas Schiesser and Cornelius Wolff and Sophie Haas and Simon Pukrop},
	journal = {arXiv preprint arXiv:2506.14842},
	year    = {2025}
}
```

## Contributing & support

Issues and pull requests are welcome—especially new encoder wrappers, dataloaders for medical imagery, or experiment configs that extend the paper. For questions, open a GitHub issue referencing the config or script you are using.

## Acknowledgements

Built on top of PyTorch, torchvision, Hugging Face Transformers, and Datadings. The Dinov3 weights are © Meta AI; please abide by their licensing terms when downloading or reusing them.
