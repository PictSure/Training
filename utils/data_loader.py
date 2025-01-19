import numpy as np
import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import Dataset, DataLoader

class MNISTRandomDataset(Dataset):
    """
    A dataset that samples random images from 2 random classes of the MNIST dataset.
    """

    def __init__(self, root="./data", num_images=10, num_samples=10000, random_classes=None, train=True):
        """
        Initialize the dataset by loading MNIST and precomputing samples.

        Args:
            root (str): Path to download or locate the MNIST dataset.
            num_images (int): Number of images to sample per class.
            random_classes (list or None): Optionally specify classes to sample from.
        """
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST(root=root, train=train, download=True, transform=self.transform)
        self.data = self.dataset.data.numpy()
        self.targets = self.dataset.targets.numpy()

        # Set the random classes
        self.num_images = num_images
        self.num_samples = num_samples

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

            random_classes = random.sample(range(10), 2)

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

            pred_class = np.random.choice(random_classes, 1, replace=False)[0]
            pred_indices = class_indices[pred_class]
            pred_index = np.random.choice(pred_indices, 1, replace=False)[0]

            pred_image = self.data[pred_index]
            pred_label = index_dict[pred_class]

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

        sampled_images = torch.tensor(sampled_images, dtype=torch.float32).unsqueeze(1)
        sampled_labels = torch.tensor(sampled_labels, dtype=torch.long)
        pred_image = torch.tensor(pred_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred_label = torch.tensor(pred_label, dtype=torch.long)

        return sampled_images, sampled_labels, pred_image, pred_label


# Usage
def get_mnist_random_loader(batch_size, num_workers=4, prefetch_factor=2, num_samples=10000, train=True):
    """
    Create a DataLoader for the MNISTRandomDataset.

    Args:
        batch_size (int): Number of batches.
        num_workers (int): Number of worker threads for data loading.
        prefetch_factor (int): Number of batches to prefetch.
        train (bool): Whether to use the training set.

    Returns:
        DataLoader: A DataLoader instance.
    """
    dataset = MNISTRandomDataset(train=train, num_samples=num_samples)
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