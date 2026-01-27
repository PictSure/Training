#!/usr/bin/env python3
"""
Test the few-shot learning model on CIFAR-10 dataset.
Uses normalize_dinov2 for image preprocessing.
"""
import argparse
import os
import sys
import torch
import yaml
import numpy as np
from collections import defaultdict
from torchvision import datasets, transforms
from tqdm import tqdm

# Try importing from the local modules
try:
    from model.model_PictSure import CustomTransformerModel
    from model.wrapper import DINOV2Wrapper
    from utils.data_loader_imagenet import normalize_dinov2
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def _load_model(checkpoint_path: str, device: str, config_path: str = None) -> CustomTransformerModel:
    """Load the composed model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state", payload)
    
    # Load config if provided to get model architecture params
    nheads = 8
    nlayers = 4
    embed_dim = 512
    
    if config_path and os.path.exists(config_path):
        config = _load_config(config_path)
        nheads = config.get("model", {}).get("nheads", nheads)
        nlayers = config.get("model", {}).get("nlayers", nlayers)
        embed_dim = config.get("model", {}).get("embed_dim", embed_dim)
        print(f"Loaded model config: nheads={nheads}, nlayers={nlayers}, embed_dim={embed_dim}")
    
    # Build model with DINOv2 encoder
    encoder = DINOV2Wrapper(device=device).to(device)
    model = CustomTransformerModel(
        embedding_layer=encoder,
        num_classes=10,
        nheads=nheads,
        nlayer=nlayers,
        embed_dim=embed_dim,
        device=device,
    ).to(device)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading: {unexpected}")
    
    return model


def _load_cifar10(root: str = "./data", train: bool = True):
    """Load CIFAR-10 dataset and build class index."""
    dataset = datasets.CIFAR10(root=root, train=train, download=True)
    
    # Build class index
    class_to_indices = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        class_to_indices[target].append(idx)
    
    # Add transform to convert PIL to tensor
    dataset.transform = transforms.ToTensor()
    
    return dataset, class_to_indices


def _sample_few_shot_batch(
    dataset, 
    class_to_indices, 
    num_classes: int = 5,
    num_shots: int = 5,
    num_queries: int = 1
):
    """
    Sample a few-shot learning batch from CIFAR-10.
    
    Returns:
        (support_images, support_labels, query_images, query_labels)
    """
    chosen_classes = np.random.choice(10, num_classes, replace=False)
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    for label_idx, class_id in enumerate(chosen_classes):
        indices = class_to_indices[class_id]
        selected = np.random.choice(indices, num_shots + num_queries, replace=False)
        
        # Support images
        for idx in selected[:num_shots]:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(label_idx)
        
        # Query images
        for idx in selected[num_shots:]:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(label_idx)
    
    support_images = torch.stack(support_images)  # (N*K, 3, 32, 32)
    support_labels = torch.tensor(support_labels, dtype=torch.long)
    query_images = torch.stack(query_images)  # (N*Q, 3, 32, 32)
    query_labels = torch.tensor(query_labels, dtype=torch.long)
    
    return support_images, support_labels, query_images, query_labels


def _evaluate(
    model,
    dataset,
    class_to_indices,
    device: str,
    num_episodes: int = 100,
    num_classes: int = 5,
    num_shots: int = 5,
    num_queries: int = 1
) -> float:
    """Evaluate the model on few-shot learning tasks."""
    model.eval()
    correct = 0
    total = 0
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            support_images, support_labels, query_images, query_labels = _sample_few_shot_batch(
                dataset, class_to_indices, num_classes=num_classes, 
                num_shots=num_shots, num_queries=num_queries
            )
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # Add batch dimension and reshape for model input
            N = num_classes
            K = num_shots
            
            # For normalize_dinov2:
            # - sampled_images expects (batch, num_images, C, H, W)
            # - pred_image expects (batch, C, H, W) - single image per batch
            
            # We process each query image separately
            for query_idx in range(query_images.shape[0]):
                # Reshape support: (N*K, 3, 32, 32) -> (1, N*K, 3, 32, 32)
                support_batch = support_images.unsqueeze(0)
                # Get single query: (3, 32, 32) -> (1, 3, 32, 32)
                query_batch = query_images[query_idx].unsqueeze(0)
                
                # Normalize
                support_norm, query_norm = normalize_dinov2(
                    support_batch, query_batch, gaussian=False, sharpness=False, resize=None
                )
                
                # Reshape for model: (N*K, 3, 224, 224) -> (1, N*K, 3, 224, 224)
                support_norm = support_norm.view(1, -1, 3, 224, 224)
                query_norm = query_norm.view(1, 1, 3, 224, 224)
                
                # Add batch dimension to labels: (N*K,) -> (1, N*K)
                support_labels_batch = support_labels.unsqueeze(0)
                
                # Get logits from model
                logits_single = model(support_norm, support_labels_batch, query_norm, embedd=True)
                
                # Get prediction for this query
                pred = logits_single.squeeze(0).argmax(dim=0 if logits_single.squeeze(0).dim() == 1 else -1)
                
                # Compute accuracy
                correct += (pred == query_labels[query_idx]).sum().item()
                total += 1
    
    accuracy = correct / total
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Test few-shot learning model on CIFAR-10")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None, help="Config file used for training (to load model architecture)")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--num-shots", type=int, default=5, help="Number of support samples per class")
    parser.add_argument("--num-queries", type=int, default=1, help="Number of query samples per class")
    parser.add_argument("--data-root", default="./data", help="Path to CIFAR-10 dataset")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split to use")
    args = parser.parse_args()
    
    device = _device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    model = _load_model(args.checkpoint, device, args.config)
    print("Model loaded successfully")
    
    # Load CIFAR-10
    print(f"Loading CIFAR-10 {args.split} split...")
    dataset, class_to_indices = _load_cifar10(args.data_root, train=(args.split == "train"))
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Evaluate
    print(f"\nEvaluating with {args.num_classes}-way {args.num_shots}-shot learning...")
    accuracy = _evaluate(
        model,
        dataset,
        class_to_indices,
        device,
        num_episodes=args.num_episodes,
        num_classes=args.num_classes,
        num_shots=args.num_shots,
        num_queries=args.num_queries
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Evaluated on {args.num_episodes} episodes")
    print(f"Task: {args.num_classes}-way {args.num_shots}-shot learning")
    print(f"{'='*60}\n")
    
    return 0 if accuracy > 0.5 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
