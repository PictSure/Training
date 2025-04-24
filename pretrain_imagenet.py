import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import random
import argparse
from torch.utils.data import Sampler
from model.model_ViT import VisionTransformer
from tqdm import tqdm


class ClassStratifiedSampler(Sampler):
    """
    Samples each batch such that it contains exactly (batch_size // 4) classes,
    and from each class picks 4 examples -> batch_size total.

    If a chosen class has fewer than 4 examples left, we'll recycle or skip
    that class to ensure full batch size. The simplest approach is to re-shuffle
    and pick a different class as needed.
    """

    def __init__(self, dataset, batch_size, num_classes_per_batch=None):
        """
        dataset: a Dataset object, assumed to have `dataset.targets` or `dataset.labels`
        batch_size: how many samples in a batch
        num_classes_per_batch: how many distinct classes per batch.
                              Defaults to batch_size//4 if None.
        """
        # We assume dataset.targets is a list/array of integer labels.
        # If using a custom dataset, adapt accordingly.
        self.dataset = dataset
        self.targets = dataset.targets  # or dataset.labels
        self.batch_size = batch_size
        if num_classes_per_batch is None:
            num_classes_per_batch = batch_size // 4
        self.num_classes_per_batch = num_classes_per_batch

        # Build a mapping: class_label -> list of sample indices
        self.class_to_indices = {}
        for idx, label in enumerate(self.targets):
            self.class_to_indices.setdefault(label, []).append(idx)

        # Convert class_to_indices to a list of (class_label, indices_list) for convenience
        self.class_to_indices = list(self.class_to_indices.items())

        # We'll compute how many total batches we can draw in one epoch
        # in a naive way. If some classes are tiny, you might have fewer valid batches.
        # Here we aim for a full "epoch" to be enough draws to cover all data, but
        # details can vary by application.
        self.num_samples = len(self.targets)
        self.num_batches = self.num_samples // self.batch_size

    def __iter__(self):
        """
        Yields one batch of indices at a time.
        """
        for _ in range(self.num_batches):
            # Step 1: sample distinct classes
            chosen_classes = random.sample(self.class_to_indices, self.num_classes_per_batch)

            batch_indices = []

            # Step 2: from each chosen class, pick 4 random indices
            for cls_label, indices_list in chosen_classes:
                # if a class doesn't have enough to pick 4, you can handle it differently
                # e.g. repeat or skip. Here we just random sample with replacement
                selected = random.choices(indices_list, k=4)
                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self):
        return self.num_batches

def build_stratified_imagenet_loaders(
    data_path="/path/to/imagenet",
    batch_size=64,
    num_workers=8
):
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    train_set = torchvision.datasets.ImageNet(
        root=data_path, split="train", transform=train_transform
    )
    val_set = torchvision.datasets.ImageNet(
        root=data_path, split="val", transform=val_transform
    )

    # Create Sampler
    train_sampler = ClassStratifiedSampler(
        dataset=train_set, batch_size=batch_size, num_classes_per_batch=batch_size//4
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,  # Instead of `batch_size=` or `shuffle=`, we use a sampler
        num_workers=num_workers
    )

    # For validation, we can just do a regular sequential sampler
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader

def sample_triplets(embeddings, labels):
    """
    embeddings: [B, D]
    labels: [B] (integer class labels)

    Returns anchor, pos, neg: each [M, D]
    """
    # This is a toy example. In practice, you might do more advanced mining.
    # We pick triplets at random from the batch, ensuring we find distinct
    # anchor/pos for the same label and anchor/neg for a different label.
    device = embeddings.device
    B = embeddings.size(0)

    anchors = []
    positives = []
    negatives = []

    # a dictionary from label -> list of indices having that label
    label_dict = {}
    for i, lab in enumerate(labels):
        label_dict.setdefault(lab.item(), []).append(i)

    for i in range(B):
        anchor_label = labels[i].item()
        # skip if there's only one example of that class in the batch
        if len(label_dict[anchor_label]) < 2:
            continue

        # anchor
        anchor_embed = embeddings[i]

        # pick pos from same label, but different index
        pos_index = i
        while pos_index == i:
            pos_index = random.choice(label_dict[anchor_label])
        pos_embed = embeddings[pos_index]

        # pick neg from any different label
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(list(label_dict.keys()))
        neg_index = random.choice(label_dict[neg_label])
        neg_embed = embeddings[neg_index]

        anchors.append(anchor_embed.unsqueeze(0))
        positives.append(pos_embed.unsqueeze(0))
        negatives.append(neg_embed.unsqueeze(0))

    if len(anchors) == 0:
        # no valid triplets found
        return None, None, None

    anchors = torch.cat(anchors, dim=0).to(device)
    positives = torch.cat(positives, dim=0).to(device)
    negatives = torch.cat(negatives, dim=0).to(device)
    return anchors, positives, negatives

def combined_loss(logits, cls_embeddings, labels, margin=1.0, lambda_triplet=1.0):
    # Cross Entropy
    ce_loss = F.cross_entropy(logits, labels)

    # Triplet
    anchors, positives, negatives = sample_triplets(cls_embeddings, labels)
    if anchors is None:
        # if no valid triplet was found, fallback to just CE
        return ce_loss

    # standard TripletMarginLoss
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2.0)
    t_loss = triplet_loss_fn(anchors, positives, negatives)
    return ce_loss + lambda_triplet * t_loss

def train_vit_triplet_stratified(
    data_path="/path/to/imagenet",
    batch_size=64,
    epochs=20,
    lr=1e-4,
    device="cuda"
):
    from torch.utils.data import DataLoader

    # Build your custom train/val loaders
    train_loader, val_loader = build_stratified_imagenet_loaders(
        data_path=data_path, batch_size=batch_size
    )

    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=1000,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss, total_correct, total_count = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, embeddings = model(images)
            loss = combined_loss(logits, embeddings, labels,
                                 margin=1.0, lambda_triplet=1.0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = logits.max(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_count += images.size(0)

        train_acc = 100.0 * total_correct / total_count
        avg_loss = total_loss / total_count

        # Evaluate on val set
        val_acc = validate(model, val_loader, device=device)

        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"Train Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Acc={val_acc:.2f}%")

    return model


def validate(model, loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer with Triplet Loss")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the ImageNet dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    data_path = args.data_path
    device = args.device
    train_vit_triplet_stratified(data_path=data_path, device=device)