import numpy as np
import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import Dataset, DataLoader

class MNISTRandomDataset(Dataset):
    """
    A dataset that samples random images from 2 random classes of the MNIST dataset.
    """

    def __init__(self, root="./data", num_images=10, random_classes=None):
        """
        Initialize the dataset by loading MNIST and selecting random classes.

        Args:
            root (str): Path to download or locate the MNIST dataset.
            num_images (int): Number of images to sample per class.
            random_classes (list or None): Optionally specify classes to sample from.
        """
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.data = self.dataset.data.numpy()
        self.targets = self.dataset.targets.numpy()

        # Set the random classes
        self.random_classes = random_classes or random.sample(range(10), 2)
        self.num_images = num_images

        # Precompute indices for the random classes
        self.class_indices = {
            cls: np.where(self.targets == cls)[0] for cls in self.random_classes
        }

    def __len__(self):
        # Arbitrary length for random sampling
        return 100000

    def __getitem__(self, idx):
        """
        Generate a single sample of data.
        """
        sampled_images = []
        sampled_labels = []

        index_dict = {}
        for i, cls in enumerate(self.random_classes):
            indices = self.class_indices[cls]

            # Sample `num_images` indices without replacement
            sampled_indices = np.random.choice(indices, self.num_images, replace=False)
            sampled_images.append(self.data[sampled_indices])
            sampled_labels.append(np.full(self.num_images, i))
            index_dict[cls] = i

        # Select a prediction image from one of the classes
        pred_class = np.random.choice(self.random_classes, 1, replace=False)
        pred_indices = self.class_indices[pred_class[0]]
        pred_index = np.random.choice(pred_indices, 1, replace=False)[0]

        pred_image = self.data[pred_index]
        pred_label = index_dict[pred_class[0]]

        # Prepare tensors
        sampled_images = np.concatenate(sampled_images, axis=0)
        sampled_labels = np.concatenate(sampled_labels, axis=0)

        sampled_images = torch.tensor(sampled_images, dtype=torch.float32).unsqueeze(1)
        sampled_labels = torch.tensor(sampled_labels, dtype=torch.long)

        pred_image = torch.tensor(pred_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred_label = torch.tensor(pred_label, dtype=torch.long)

        return sampled_images, sampled_labels, pred_image, pred_label


# Usage
def get_mnist_random_loader(batch_size, num_workers=4, prefetch_factor=2):
    """
    Create a DataLoader for the MNISTRandomDataset.

    Args:
        batch_size (int): Number of batches.
        num_workers (int): Number of worker threads for data loading.
        prefetch_factor (int): Number of batches to prefetch.

    Returns:
        DataLoader: A DataLoader instance.
    """
    dataset = MNISTRandomDataset()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        pin_memory=True,
    )
    return data_loader