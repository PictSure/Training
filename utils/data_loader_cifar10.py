from torchvision import datasets, transforms
from torch.utils.data import Dataset
import random
import numpy as np
import torch

class CIFAR10RandomDataset(Dataset):
    """
    A dataset that samples random images from 2 random classes of the CIFAR-10 dataset.
    """

    def __init__(self, root="./data", num_images=10, num_samples=10000, num_classes=2, random_classes=None, train=True, data_type="Cifar10"):
        """
        Initialize the dataset by loading CIFAR-10 and precomputing samples.

        Args:
            root (str): Path to download or locate the CIFAR-10 dataset.
            num_images (int): Number of images to sample per class.
            random_classes (list or None): Optionally specify classes to sample from.
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if data_type == "Cifar10":
            self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=self.transform)
        elif data_type == "Cifar100":
            self.dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=self.transform)
        else:
            raise ValueError("Invalid data type. Please specify either Cifar10 or Cifar100.")
        self.data = self.dataset.data  # CIFAR-10 data is stored as numpy arrays
        self.targets = np.array(self.dataset.targets)

        # Set parameters
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Precompute samples
        self.samples = []
        self.precompute_samples()

    def precompute_samples(self):
        """
        Precompute samples for the dataset.
        """
        self.sampled_images_list = []
        self.sampled_labels_list = []
        self.pred_images_list = []
        self.pred_labels_list = []

        for _ in range(self.num_samples):  # Precompute a fixed number of samples
            sampled_images = []
            sampled_labels = []

            # Randomly choose 2 classes
            random_classes = random.sample(range(10), self.num_classes)

            # Get indices for each class
            class_indices = {
                cls: np.where(self.targets == cls)[0] for cls in random_classes
            }
            index_dict = {}
            for i, cls in enumerate(random_classes):
                indices = class_indices[cls]
                index_dict[cls] = i
                sampled_indices = np.random.choice(indices, self.num_images, replace=False)
                sampled_images.append(self.data[sampled_indices])
                sampled_labels.append(np.full(self.num_images, index_dict[cls]))

            # Randomly select a prediction image from one of the random classes
            pred_class = np.random.choice(random_classes, 1, replace=False)[0]
            pred_indices = class_indices[pred_class]
            pred_index = np.random.choice(pred_indices, 1, replace=False)[0]

            pred_image = self.data[pred_index]
            pred_label = index_dict[pred_class]

            # Combine and store sampled images and labels
            sampled_images = np.concatenate(sampled_images, axis=0)
            sampled_labels = np.concatenate(sampled_labels, axis=0)

            self.sampled_images_list.append(sampled_images)
            self.sampled_labels_list.append(sampled_labels)
            self.pred_images_list.append(pred_image)
            self.pred_labels_list.append(pred_label)

    def __len__(self):
        return len(self.sampled_images_list)

    def __getitem__(self, idx):
        """
        Return precomputed samples.
        """
        sampled_images = self.sampled_images_list[idx]
        sampled_labels = self.sampled_labels_list[idx]
        pred_image = self.pred_images_list[idx]
        pred_label = self.pred_labels_list[idx]

        # Convert to tensors
        sampled_images = torch.tensor(sampled_images, dtype=torch.float32).permute(0, 3, 1, 2)  # (B, C, H, W)
        sampled_labels = torch.tensor(sampled_labels, dtype=torch.long)
        pred_image = torch.tensor(pred_image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        pred_label = torch.tensor(pred_label, dtype=torch.long)

        return sampled_images, sampled_labels, pred_image, pred_label


def normalize_samples(sampled_images, pred_image):
    """
    Normalize the input images to the range [0, 1].
    
    Args:
        sampled_images (torch.Tensor): Batch of sampled images with shape (B, C, H, W).
        pred_image (torch.Tensor): Single prediction image with shape (C, H, W).
        
    Returns:
        normalized_sampled_images (torch.Tensor): Normalized sampled images.
        normalized_pred_image (torch.Tensor): Normalized prediction image.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    # Normalize sampled images
    normalized_sampled_images = (sampled_images - mean) / std
    normalized_pred_image = (pred_image - mean) / std

    return normalized_sampled_images, normalized_pred_image

def get_cifar10_random_loader(batch_size=16, num_images=10, num_samples=10000, num_classes=2, train=True):
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
    dataset = CIFAR10RandomDataset(num_images=num_images, num_samples=num_samples, train=train, num_classes=num_classes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader