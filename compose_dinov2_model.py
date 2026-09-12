import argparse
import json
import os
import sys
from typing import Tuple

import torch
import yaml
from torchvision import models
from huggingface_hub import HfApi, create_repo
from safetensors.torch import load_file, save_file
from transformers import AutoModel, CLIPModel

from model.model_PictSure import CustomTransformerModel
from model.wrapper import (
    DINOV2Wrapper,
    DINOV2_LargeWrapper,
    DINOV3Wrapper,
    CLIPWrapper,
    ResNetWrapper,
    VitNetWrapper,
)
from utils.summary_writer import find_latest_run_directory


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def _resolve_checkpoint(config: dict, explicit: str | None) -> str:
    if explicit:
        return explicit

    base_output = config["paths"]["output"]
    run_name = config["name"]
    latest = find_latest_run_directory(base_output, run_name)
    if latest is None:
        raise FileNotFoundError("No run directory found; pass --checkpoint explicitly")

    candidates = [
        os.path.join(base_output, latest, "best_loss_model.pt"),
        os.path.join(base_output, latest, "final_model.pt"),
        os.path.join(base_output, latest, "checkpoint.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("No checkpoint file found in latest run; pass --checkpoint explicitly")


def _build_encoder(encoder_name: str, config: dict, device: str):
    name = encoder_name.lower()
    if name == "dinov2":
        encoder = DINOV2Wrapper(device=device).to(device)
        latent_dim = encoder.latent_dim
    elif name == "dinov2-large":
        encoder = DINOV2_LargeWrapper(device=device).to(device)
        latent_dim = encoder.latent_dim
    elif name == "dinov3":
        encoder = DINOV3Wrapper(device=device).to(device)
        latent_dim = encoder.latent_dim
    elif name == "clip":
        encoder = CLIPWrapper(device=device).to(device)
        latent_dim = encoder.latent_dim
    elif name in {"resnet18", "resnet34", "resnet50"}:
        depth = int(name.replace("resnet", ""))
        classifier = (
            models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if depth == 18
            else models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            if depth == 34
            else models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        )
        encoder = ResNetWrapper(classifier).to(device)
        latent_dim = encoder.latent_dim
    elif name == "vit":
        vit_path = config.get("paths", {}).get("visnet_weights") if config.get("pretrained", False) else None
        encoder = VitNetWrapper(path=vit_path, device=device).to(device)
        latent_dim = encoder.latent_dim
    else:
        raise ValueError(f"Unsupported encoder '{encoder_name}'. Choices: dinov2, dinov2-large, dinov3, clip, resnet18/34/50, vit")

    # Keep encoder frozen by default when composing
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder, latent_dim


def _build_model(config: dict, device: str, encoder_name: str) -> Tuple[CustomTransformerModel, int]:
    encoder, latent_dim = _build_encoder(encoder_name, config, device)
    model = CustomTransformerModel(
        embedding_layer=encoder,
        num_classes=config["dataloader"]["num_classes"],
        nheads=config["model"]["nheads"],
        nlayer=config["model"]["nlayers"],
        embed_dim=config["model"]["embed_dim"],
        device=device,
    ).to(device)
    return model, latent_dim


def _load_transformer_weights(model: CustomTransformerModel, checkpoint_path: str, device: str) -> dict:
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload.get("model_state", payload)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading (expected for embedding layer): {missing}")
    if unexpected:
        print(f"Unexpected keys when loading: {unexpected}")

    return state_dict


def _load_embedding_weights(embedding_layer, encoder_name: str, device: str, state_dict: dict) -> None:
    prefix = "embedding."
    embedding_state = {
        key[len(prefix):]: tensor.to(device)
        for key, tensor in state_dict.items()
        if key.startswith(prefix)
    }

    if embedding_state:
        missing, unexpected = embedding_layer.load_state_dict(embedding_state, strict=False)
        if missing:
            print(f"Missing keys when loading {encoder_name} embedding: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading {encoder_name} embedding: {unexpected}")
        return

    print("Embedding weights missing from checkpoint; falling back to pretrained HuggingFace weights")
    _reload_embedding_from_hf(embedding_layer, encoder_name, device)


def _reload_embedding_from_hf(embedding_layer, encoder_name: str, device: str) -> None:
    name = encoder_name.lower()
    hf_token = os.getenv("HF_TOKEN")
    loader = None

    if name == "dinov2":
        loader = lambda: AutoModel.from_pretrained("facebook/dinov2-base")
    elif name == "dinov2-large":
        loader = lambda: AutoModel.from_pretrained("facebook/dinov2-large")
    elif name == "dinov3":
        loader = lambda: AutoModel.from_pretrained(
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            token=hf_token,
        )
    elif name == "clip":
        loader = lambda: CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    if loader is None:
        print(f"No HuggingFace fallback available for '{encoder_name}'; keeping current weights.")
        return

    model_attr = getattr(embedding_layer, "model", None)
    if model_attr is None:
        print(f"Encoder wrapper for '{encoder_name}' does not expose a nested model; cannot reload.")
        return

    try:
        pretrained = loader().to(device)
        missing, unexpected = model_attr.load_state_dict(pretrained.state_dict(), strict=False)
        if missing:
            print(f"Missing keys when reloading pretrained {encoder_name}: {missing}")
        if unexpected:
            print(f"Unexpected keys when reloading pretrained {encoder_name}: {unexpected}")
        print(f"Reloaded {encoder_name} weights from HuggingFace")
    except Exception as exc:
        print(f"Failed to reload {encoder_name} weights from HuggingFace: {exc}")


def _smoke_test(model: CustomTransformerModel, device: str, num_classes: int, num_images: int, batch_size: int) -> None:
    model.eval()
    with torch.no_grad():
        images = torch.rand(batch_size, num_images, 3, 224, 224, device=device)
        labels = torch.randint(0, num_classes, (batch_size, num_images), device=device)
        pred_image = torch.rand(batch_size, 3, 224, 224, device=device)
        logits = model(images, labels, pred_image, embedd=True)
    print(f"Smoke test logits shape: {tuple(logits.shape)}")


def _push_to_huggingface(model_path: str, repo_id: str, token: str, encoder_name: str, config: dict, private: bool = False) -> None:
    """Upload the model to HuggingFace Hub."""
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
        print(f"Repository '{repo_id}' ready on HuggingFace ({'private' if private else 'public'})")
    except Exception as e:
        print(f"Warning: Could not create/verify repo: {e}")
    
    # Create a simple README for the model
    readme_content = f"""---
license: mit
tags:
- image-classification
- transformer
- {encoder_name}
---

# Combined {encoder_name.upper()} + Transformer Model

This model combines a {encoder_name} encoder with a custom transformer classifier.

## Model Details

- Encoder: {encoder_name}
- Transformer heads: {config['model']['nheads']}
- Transformer layers: {config['model']['nlayers']}
- Embedding dimension: {config['model']['embed_dim']}
- Number of classes: {config['dataloader']['num_classes']}

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Download the model
model_path = hf_hub_download(repo_id="{repo_id}", filename="model.safetensors")

# Load the model state
state_dict = load_file(model_path)
# ... initialize your model architecture and load state_dict
```
"""
    
    readme_path = os.path.join(os.path.dirname(model_path), "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    # Create config.json
    config_json = {
        "device": "cpu",
        "embedding": encoder_name,
        "nheads": config["model"]["nheads"],
        "nlayer": config["model"]["nlayers"],
        "embedd_dim": config["model"]["embed_dim"],
        "num_classes": config["dataloader"]["num_classes"],
        "pretrained": True
    }
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)
    
    # Upload model file
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ Model uploaded to https://huggingface.co/{repo_id}")
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ README uploaded")
        
        # Upload config.json
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ config.json uploaded")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose encoder (default DINOv2) with trained transformer and export")
    parser.add_argument("--config", default="configs/local_duckdb.yaml", help="Config file used for transformer training")
    parser.add_argument("--checkpoint", default=None, help="Path to transformer checkpoint; defaults to latest run")
    parser.add_argument("--output", default=None, help="Where to save the composed model state_dict")
    parser.add_argument(
        "--encoder",
        default="dinov2",
        choices=["dinov2", "dinov2-large", "dinov3", "clip", "resnet18", "resnet34", "resnet50", "vit"],
        help="Embedding backbone to attach (default: dinov2)",
    )
    parser.add_argument("--smoke-batch", type=int, default=2, help="Batch size for quick forward check (0 to skip)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token for uploading model")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo ID (e.g., 'username/model-name')")
    parser.add_argument("--hf-private", action="store_true", help="Make the HuggingFace repo private")
    args = parser.parse_args()

    config = _load_config(args.config)
    device = _device()
    print(f"Using device: {device}")

    checkpoint_path = _resolve_checkpoint(config, args.checkpoint)
    print(f"Loading transformer weights from: {checkpoint_path}")

    model, latent_dim = _build_model(config, device, args.encoder)

    print(f"Encoder '{args.encoder}' latent dim: {latent_dim}")

    state_dict = _load_transformer_weights(model, checkpoint_path, device)

    _load_embedding_weights(model.embedding, args.encoder, device, state_dict)

    if args.smoke_batch > 0:
        _smoke_test(
            model,
            device=device,
            num_classes=config["dataloader"]["num_classes"],
            num_images=config["dataloader"].get("num_images", 5),
            batch_size=args.smoke_batch,
        )

    output_path = args.output
    if output_path is None:
        output_encoder = args.encoder.replace("-", "_")
        output_path = os.path.join(os.path.dirname(checkpoint_path), f"{output_encoder}_transformer_combined.safetensors")

    save_file(model.state_dict(), output_path)
    print(f"Composed model saved to: {output_path}")

    # Push to HuggingFace if token and repo are provided
    if args.hf_token and args.hf_repo:
        print(f"\nUploading to HuggingFace repo: {args.hf_repo}")
        _push_to_huggingface(output_path, args.hf_repo, args.hf_token, args.encoder, config, args.hf_private)
    elif args.hf_token or args.hf_repo:
        print("Warning: Both --hf-token and --hf-repo are required for HuggingFace upload")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # Ensure clear exit in CLI usage
        print(f"Error: {exc}")
        sys.exit(1)