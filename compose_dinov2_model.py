import argparse
import os
import sys
from typing import Tuple

import torch
import yaml
from torchvision import models

from model.model_PictSure import CustomTransformerModel
from model.wrapper import (
    DINOV2Wrapper,
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
        raise ValueError(f"Unsupported encoder '{encoder_name}'. Choices: dinov2, dinov3, clip, resnet18/34/50, vit")

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


def _load_transformer_weights(model: CustomTransformerModel, checkpoint_path: str, device: str) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading (expected for embedding layer): {missing}")
    if unexpected:
        print(f"Unexpected keys when loading: {unexpected}")


def _smoke_test(model: CustomTransformerModel, device: str, num_classes: int, num_images: int, batch_size: int) -> None:
    model.eval()
    with torch.no_grad():
        images = torch.rand(batch_size, num_images, 3, 224, 224, device=device)
        labels = torch.randint(0, num_classes, (batch_size, num_images), device=device)
        pred_image = torch.rand(batch_size, 3, 224, 224, device=device)
        logits = model(images, labels, pred_image, embedd=True)
    print(f"Smoke test logits shape: {tuple(logits.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose encoder (default DINOv2) with trained transformer and export")
    parser.add_argument("--config", default="configs/local_duckdb.yaml", help="Config file used for transformer training")
    parser.add_argument("--checkpoint", default=None, help="Path to transformer checkpoint; defaults to latest run")
    parser.add_argument("--output", default=None, help="Where to save the composed model state_dict")
    parser.add_argument(
        "--encoder",
        default="dinov2",
        choices=["dinov2", "dinov3", "clip", "resnet18", "resnet34", "resnet50", "vit"],
        help="Embedding backbone to attach (default: dinov2)",
    )
    parser.add_argument("--smoke-batch", type=int, default=2, help="Batch size for quick forward check (0 to skip)")
    args = parser.parse_args()

    config = _load_config(args.config)
    device = _device()
    print(f"Using device: {device}")

    checkpoint_path = _resolve_checkpoint(config, args.checkpoint)
    print(f"Loading transformer weights from: {checkpoint_path}")

    model, latent_dim = _build_model(config, device, args.encoder)
    print(f"Encoder '{args.encoder}' latent dim: {latent_dim}")

    _load_transformer_weights(model, checkpoint_path, device)

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
        output_path = os.path.join(os.path.dirname(checkpoint_path), "dinov2_transformer_combined.pt")

    torch.save({"model_state": model.state_dict()}, output_path)
    print(f"Composed model saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # Ensure clear exit in CLI usage
        print(f"Error: {exc}")
        sys.exit(1)