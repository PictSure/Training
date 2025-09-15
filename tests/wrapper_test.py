#!/usr/bin/env python3
"""
tests/test_embedding_wrapper.py

Quick sanity test for embedding wrappers.

- Loads config YAML (default: ./configs/local.yaml)
- Builds training/test dataloaders via DatasetFactory
- Grabs one training batch and a "pred_image" sample from your loader
- Normalizes & resizes samples for the chosen model family (default: dinov2)
- Instantiates an embedding wrapper (default: DINOV2Wrapper) on the chosen device
- Runs a forward pass to produce embeddings
- Prints basic stats (shapes, dtypes, mins/maxes) and a simple similarity check
"""

import argparse
import sys
import yaml
import torch
import math
from pathlib import Path
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Project imports (must exist in your repo) ---
from utils.dataset_factory import DatasetFactory
from utils.data_loader_imagenet import normalize_samples
import model.wrapper as wrapper_mod  # we'll getattr the class from here


def _resolve_device(dev_arg: str) -> torch.device:
    dev_arg = dev_arg.lower()
    if dev_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dev_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_embed(model, x):
    """
    Try common wrapper call patterns without knowing the exact API.
    Priority:
      1) model.embed(x)
      2) model(x)
      3) model.forward(x)
    """
    if hasattr(model, "embed") and callable(getattr(model, "embed")):
        return model.embed(x)
    if callable(model):
        return model(x)
    if hasattr(model, "forward") and callable(getattr(model, "forward")):
        return model.forward(x)
    raise AttributeError(
        "Wrapper does not expose an embedding method I recognize "
        "(tried .embed(x), __call__(x), and .forward(x))."
    )


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float()
    b = b.float()
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a * b).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Test an embedding wrapper.")
    parser.add_argument("--config", default="./configs/local.yaml", type=str,
                        help="Path to YAML config used by DatasetFactory.")
    parser.add_argument("--wrapper", default="DINOV2Wrapper", type=str,
                        help="Wrapper class name from model.wrapper (e.g., DINOV2Wrapper).")
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"],
                        help="Device to use.")
    parser.add_argument("--resize", nargs=2, type=int, default=[224, 224],
                        metavar=("H", "W"), help="Resize applied in normalize_samples.")
    parser.add_argument("--model-family", default="dinov2", type=str,
                        help='Normalization preset to use in normalize_samples (e.g., "dinov2", "clip", ...).')
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override workers in config (optional).")
    parser.add_argument("--pin-memory", type=str, default=None, choices=["true", "false"],
                        help="Override pin_memory in config (optional).")
    parser.add_argument("--batch-index", type=int, default=0,
                        help="Which training batch index to sample (default: 0).")
    args = parser.parse_args()

    dev = _resolve_device(args.device)
    print(f"[INFO] Using device: {dev}")
    

    # --- Load config YAML ---
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Optionally override a couple of common DataLoader flags at runtime.
    # (This avoids the MPS pin_memory warning you saw.)
    if args.num_workers is not None:
        try:
            config["dataloader"]["num_workers"] = int(args.num_workers)
        except Exception:
            pass
    if args.pin_memory is not None:
        try:
            config["dataloader"]["pin_memory"] = (args.pin_memory.lower() == "true")
        except Exception:
            pass

    # --- Build dataloaders ---
    print("[INFO] Building dataloaders via DatasetFactory...")
    training, test = DatasetFactory(config, None, 0).get_dataloaders()

    # Warn user if pin_memory might be noisy on MPS
    pin_memory = None
    try:
        pin_memory = config.get("dataloader", {}).get("pin_memory", None)
    except Exception:
        pass
    if str(dev) == "mps" and pin_memory:
        print("[WARN] pin_memory=True is not supported on MPS; PyTorch will warn and ignore it.")

    # --- Get a batch (and the pred sample) ---
    print(f"[INFO] Grabbing batch index {args.batch_index} from training loader...")
    it = iter(training)
    batch = None
    for i in range(args.batch_index + 1):
        batch = next(it)
    images, labels, pred_image, pred_label = batch

    # --- Move to device ---
    images = images.to(dev)
    pred_image = pred_image.to(dev)

    # --- Normalize / resize ---
    print("[INFO] Normalizing samples...")
    images, pred_image = normalize_samples(
        images,
        pred_image,
        resize=tuple(args.resize),
        model=args.model_family,
    )

    # Print range checks similar to your REPL
    print(f"[STATS] images.min={images.min().item():.4f}  images.max={images.max().item():.4f}")
    print(f"[STATS] pred_image.min={pred_image.min().item():.4f}  pred_image.max={pred_image.max().item():.4f}")

    # --- Instantiate wrapper ---
    print(f"[INFO] Instantiating wrapper: {args.wrapper}...")
    try:
        WrapperClass = getattr(wrapper_mod, args.wrapper)
    except AttributeError:
        print(f"[ERROR] '{args.wrapper}' not found in model.wrapper", file=sys.stderr)
        sys.exit(2)

    # Most wrappers in this repo seem to accept a device string in ctor, matching your REPL.
    model = WrapperClass(str(dev))

    # --- Forward pass / embedding test ---
    model_device = next((p.device for p in getattr(model, "parameters", lambda: [])() or []), dev)
    print(f"[INFO] Wrapper parameters device (first param if any): {model_device}")

    model_was_training = getattr(model, "training", False)
    model.eval() if hasattr(model, "eval") else None

    with torch.no_grad():
        # Make a small test set: batch embeddings + single pred image embedding
        batch_emb = _maybe_embed(model, images)
        pred_emb = _maybe_embed(model, pred_image.unsqueeze(0))

    # --- Stats on embeddings ---
    def _shape_str(t):
        return "None" if t is None else f"{tuple(t.shape)} {str(t.dtype)} on {t.device}"

    print(f"[EMBED] batch_emb: {_shape_str(batch_emb)}")
    print(f"[EMBED] pred_emb:  {_shape_str(pred_emb)}")

    # Restore train mode if needed
    if model_was_training and hasattr(model, "train"):
        model.train()


if __name__ == "__main__":
    main()
