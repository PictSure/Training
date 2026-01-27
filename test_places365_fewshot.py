#!/usr/bin/env python3
"""
Test the few-shot learning model on Places365 dataset.
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
from PIL import Image

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


def _load_places365(root: str = "/home/rechenschieber/Documents/GitHub/dataset/data/places365_standard", split: str = "val"):
    """Load Places365 dataset and build class index."""
    split_dir = os.path.join(root, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Places365 {split} split not found at {split_dir}")
    
    # Get all class folders
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Build index of all images
    class_to_images = defaultdict(list)
    for class_name in classes:
        class_dir = os.path.join(split_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            img_path = os.path.join(class_dir, img)
            class_to_images[class_to_idx[class_name]].append(img_path)
    
    print(f"Loaded {len(classes)} classes from Places365 {split} split")
    return split_dir, class_to_idx, class_to_images


def _sample_few_shot_batch(
    class_to_images,
    class_to_idx,
    num_classes: int = 5,
    num_shots: int = 5,
    num_queries: int = 1,
    device: str = "cpu"
):
    """
    Sample a few-shot learning batch from Places365.
    
    Returns:
        (support_images, support_labels, query_images, query_labels)
    """
    all_classes = list(class_to_idx.values())
    chosen_class_ids = np.random.choice(all_classes, num_classes, replace=False)
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    for label_idx, class_id in enumerate(chosen_class_ids):
        available_images = class_to_images[class_id]
        
        if len(available_images) < num_shots + num_queries:
            # If not enough images, sample with replacement
            selected_indices = np.random.choice(len(available_images), num_shots + num_queries, replace=True)
        else:
            selected_indices = np.random.choice(len(available_images), num_shots + num_queries, replace=False)
        
        selected_images = [available_images[i] for i in selected_indices]
        
        # Support images
        for img_path in selected_images[:num_shots]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transforms.ToTensor()(img)
                support_images.append(img_tensor)
                support_labels.append(label_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        # Query images
        for img_path in selected_images[num_shots:num_shots + num_queries]:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transforms.ToTensor()(img)
                query_images.append(img_tensor)
                query_labels.append(label_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
    
    if not support_images or not query_images:
        raise RuntimeError("Failed to load images from Places365")
    
    support_images = torch.stack(support_images).to(device)  # (N*K, 3, H, W)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_images = torch.stack(query_images).to(device)  # (N*Q, 3, H, W)
    query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)
    
    return support_images, support_labels, query_images, query_labels


def _evaluate(
    model,
    class_to_images,
    class_to_idx,
    device: str,
    num_episodes: int = 100,
    num_classes: int = 5,
    num_shots: int = 5,
    num_queries: int = 1,
    batch_size: int = 10
) -> float:
    """Evaluate the model on few-shot learning tasks with Places365."""
    model.eval()
    correct = 0
    total = 0
    
    print(f"\nStarting evaluation with batch size {batch_size}...")
    with torch.no_grad():
        # Process episodes in batches
        num_batches = (num_episodes + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            # Determine how many episodes in this batch
            start_ep = batch_idx * batch_size
            end_ep = min(start_ep + batch_size, num_episodes)
            current_batch_size = end_ep - start_ep
            
            batch_support_images = []
            batch_support_labels = []
            batch_query_images = []
            batch_query_labels = []
            
            # Sample multiple episodes
            for _ in range(current_batch_size):
                try:
                    support_images, support_labels, query_images, query_labels = _sample_few_shot_batch(
                        class_to_images, class_to_idx,
                        num_classes=num_classes, 
                        num_shots=num_shots, 
                        num_queries=num_queries,
                        device=device
                    )
                    batch_support_images.append(support_images)
                    batch_support_labels.append(support_labels)
                    batch_query_images.append(query_images)
                    batch_query_labels.append(query_labels)
                except Exception as e:
                    print(f"Error sampling batch for episode: {e}")
                    continue
            
            if not batch_support_images:
                continue
            
            # Process each episode in the batch
            for support_images, support_labels, query_images, query_labels in zip(
                batch_support_images, batch_support_labels, batch_query_images, batch_query_labels
            ):
                # Reshape support images for normalization: (N*K, 3, H, W) -> (N, K, 3, H, W)
                N = num_classes
                K = num_shots
                actual_support_count = support_images.shape[0]
                K = actual_support_count // N
                
                support_images = support_images.view(N, K, 3, support_images.shape[2], support_images.shape[3])
                
                # Normalize images using normalize_dinov2
                support_images, query_images = normalize_dinov2(
                    support_images, query_images, gaussian=False, sharpness=False, resize=224
                )
                
                # Reshape back: normalize_dinov2 returns (N*K, 3, H, W) and (Q, 3, H, W)
                # We need (N, K, 3, H, W) for the model
                support_images = support_images.view(N, K, 3, 224, 224)
                # query_images is (N*Q, 3, H, W) already
                
                # Get logits from model
                logits = model(support_images, support_labels, query_images, embedd=True)
                
                # Get predictions
                preds = logits.argmax(dim=1)
                
                # Compute accuracy
                correct += (preds == query_labels).sum().item()
                total += query_labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Test few-shot learning model on Places365 dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None, help="Config file used for training (to load model architecture)")
    parser.add_argument("--data-root", default="/home/rechenschieber/Documents/GitHub/dataset/data/places365_standard", 
                       help="Path to Places365 dataset root")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--num-shots", type=int, default=5, help="Number of support samples per class")
    parser.add_argument("--num-queries", type=int, default=1, help="Number of query samples per class")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of episodes to process in parallel")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split to use")
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
    
    # Load Places365
    print(f"Loading Places365 {args.split} split from {args.data_root}...")
    split_dir, class_to_idx, class_to_images = _load_places365(args.data_root, split=args.split)
    total_images = sum(len(imgs) for imgs in class_to_images.values())
    print(f"Dataset loaded: {total_images} images from {len(class_to_idx)} classes")
    
    # Evaluate
    print(f"\nEvaluating with {args.num_classes}-way {args.num_shots}-shot learning...")
    accuracy = _evaluate(
        model,
        class_to_images,
        class_to_idx,
        device,
        num_episodes=args.num_episodes,
        num_classes=args.num_classes,
        num_shots=args.num_shots,
        batch_size=args.batch_size,
        num_queries=args.num_queries
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results (Places365 {args.split}):")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Evaluated on {args.num_episodes} episodes")
    print(f"Task: {args.num_classes}-way {args.num_shots}-shot learning")
    print(f"Dataset: Places365 ({args.split} split)")
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
