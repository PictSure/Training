from torchvision import datasets, transforms
from torch.utils.data import Dataset
import random
import numpy as np
import torch
import torch.nn.functional as F

class CIFAR10RandomDataset(Dataset):
    """
    A dataset that, on each __getitem__ call, samples images from
    random classes of the CIFAR-10 (or CIFAR-100) dataset.
    """

    def __init__(
        self,
        root="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        train=True,
        data_type="Cifar10"
    ):
        """
        Initialize the dataset by loading CIFAR (10 or 100).

        Args:
            root (str): Path to download or locate the dataset.
            num_images (int): Number of images to sample per class.
            num_samples (int): Total number of samples (length of the dataset).
            num_classes (int): How many distinct classes to randomly choose for each sample.
            random_classes (list or None): If provided, use these classes instead of sampling them randomly.
            train (bool): Whether to load the train or test split.
            data_type (str): "Cifar10" or "Cifar100".
        """
        super().__init__()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Load the dataset
        if data_type == "Cifar10":
            self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
            self.num_total_classes = 10
        elif data_type == "Cifar100":
            self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
            self.num_total_classes = 100
        else:
            raise ValueError("Invalid data_type. Must be either 'Cifar10' or 'Cifar100'.")

        # CIFAR data is stored as a NumPy array in dataset.data
        self.data = self.dataset.data
        self.targets = np.array(self.dataset.targets)

        # Parameters for random sampling
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes

        # If random_classes is provided, we will not choose them randomly each time;
        # instead, we'll always use the classes in this list (make sure it has size num_classes).
        self.fixed_classes = random_classes  # e.g. [0, 1] to always use classes 0 and 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        On each call, randomly sample `self.num_classes` classes (unless fixed_classes is not None),
        then pick `self.num_images` images from each class, and also pick one 'prediction' image
        from one of those classes.
        """

        # Choose classes (either fixed or random)
        if self.fixed_classes is not None:
            chosen_classes = self.fixed_classes
        else:
            chosen_classes = random.sample(range(self.num_total_classes), self.num_classes)

        # Gather indices in the dataset for each chosen class
        class_indices = {
            cls: np.where(self.targets == cls)[0] for cls in chosen_classes
        }

        sampled_images = []
        sampled_labels = []

        # For each chosen class, randomly select `num_images` images
        class_to_label = {}
        for label_idx, cls in enumerate(chosen_classes):
            available_indices = class_indices[cls]
            chosen_indices = np.random.choice(available_indices, self.num_images, replace=False)
            images_from_class = self.data[chosen_indices]  # Shape: (num_images, H, W, C)

            # Append images and a label array corresponding to this class
            sampled_images.append(images_from_class)
            sampled_labels.append(np.full(self.num_images, label_idx))
            class_to_label[cls] = label_idx

        # Randomly pick which class to draw the "prediction" image from
        pred_class = random.choice(chosen_classes)
        pred_indices = class_indices[pred_class]
        pred_index = np.random.choice(pred_indices, 1, replace=False)[0]
        pred_image_np = self.data[pred_index]  # (H, W, C)
        pred_label = class_to_label[pred_class]

        # Concatenate the sampled images/labels for all chosen classes
        sampled_images_np = np.concatenate(sampled_images, axis=0)  # (num_classes * num_images, H, W, C)
        sampled_labels_np = np.concatenate(sampled_labels, axis=0)  # (num_classes * num_images,)

        # Convert NumPy arrays to tensors and apply transforms
        # For the 'sampled_images', shape is (B, H, W, C) -> (B, C, H, W)
        sampled_images_torch = torch.tensor(sampled_images_np, dtype=torch.float32).permute(0, 3, 1, 2)
        sampled_labels_torch = torch.tensor(sampled_labels_np, dtype=torch.long)

        # For the 'pred_image', shape is (H, W, C) -> (C, H, W)
        pred_image_torch = torch.tensor(pred_image_np, dtype=torch.float32).permute(2, 0, 1)
        pred_label_torch = torch.tensor(pred_label, dtype=torch.long)

        # Optional: If you need to apply transforms (like normalization), you can do so here:
        # Example: pred_image_torch = self.transform(pred_image_torch)  (if your transform expects PIL or Tensor)
        # For the stacked images, you may need a different approach or apply transform in a loop.

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, pred_label_torch


def normalize_samples(sampled_images, pred_image, resize=None):
    """
    Normalize the input and prediction images to the range [0, 1].
    
    Args:
        sampled_images (torch.Tensor): Batch of sampled images with shape (N, B, C, H, W).
        pred_image (torch.Tensor): Single prediction image with shape (B, C, H, W).
        
    Returns:
        normalized_sampled_images (torch.Tensor): Normalized sampled images.
        normalized_pred_image (torch.Tensor): Normalized prediction image.
    """
    # Define mean and std for normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=sampled_images.device).view(1, -1, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=sampled_images.device).view(1, -1, 1, 1)
    
    # Get shapes
    N, B, C, H, W = sampled_images.size()  # sampled_images shape: (N, B, C, H, W)
    
    # Reshape sampled_images to (N*B, C, H, W)
    sampled_images = sampled_images.view(N * B, C, H, W)

    # Normalize between [0, 1]
    sampled_images = torch.clamp(sampled_images, 0, 255) / 255.0
    
    # Normalize sampled_images using mean and std
    sampled_images = (sampled_images - mean) / std
    
    # Normalize pred_image, which has shape (N, C, H, W)
    pred_image = (pred_image - mean) / std

    # Resize if necessary
    if resize is not None:
        # Resize sampled_images (reshaped as (N*B, C, H, W))
        sampled_images = F.interpolate(sampled_images, size=resize, mode="bilinear", align_corners=False)
        
        # Resize pred_image, handling (N, C, H, W)
        pred_image = F.interpolate(pred_image, size=resize, mode="bilinear", align_corners=False)
    
    # Reshape sampled_images back to (N, B, C, H, W)
    sampled_images = sampled_images.view(N, B, C, resize[0], resize[1]) if resize else sampled_images.view(N, B, C, H, W)
    
    return sampled_images, pred_image

def get_cifar10_random_loader(batch_size=16, num_images=10, num_samples=10000, num_classes=2, train=True, data_type="Cifar10"):
    """
    Get DataLoader for the CIFAR-10 Random Dataset.
    
    Args:
        batch_size (int): Number of samples per batch.
        num_images (int): Number of images to sample per class.
        num_samples (int): Number of samples to precompute.
        train (bool): Whether to use the training set.
        
    Returns:
        cifar10_loader (DataLoader): DataLoader for CIFAR-10 Random Dataset.
    """
    dataset = CIFAR10RandomDataset(num_images=num_images, num_samples=num_samples, train=train, num_classes=num_classes, data_type=data_type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader