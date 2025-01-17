import numpy as np
import torch
from torchvision import datasets, transforms
import random

class MNISTRandomSampler:
    """
    A class to sample random images from 2 random classes of the MNIST dataset.
    """

    def __init__(self, root="./data"):
        """
        Initialize the sampler by loading the MNIST dataset.

        Args:
            root (str): Path to download or locate the MNIST dataset.
        """
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.data = self.dataset.data.numpy()
        self.targets = self.dataset.targets.numpy()

    def sample(self, num_images, batch_size):
        """
        Sample a random set of images for 2 random classes, organized into batches.

        Args:
            num_images (int): Number of images to sample per class.
            batch_size (int): Number of batches.

        Returns:
            tuple: (images, labels, pred_image, pred_label)
            - images: Tensor of shape (batch_size, 2 * num_images, 1, 28, 28)
            - labels: Tensor of shape (batch_size, 2 * num_images)
            - pred_image: Tensor of shape (batch_size, 1, 28, 28)
            - pred_label: Tensor of shape (batch_size, 1)
        """
        random_classes = random.sample(range(10), 2)

        all_batches_images = []
        all_batches_labels = []

        for _ in range(batch_size):
            sampled_images = []
            sampled_labels = []

            for i, cls in enumerate(random_classes):
                indices = np.where(self.targets == cls)[0]

                sampled_indices = np.random.choice(indices, num_images, replace=False)

                sampled_images.append(self.data[sampled_indices])
                sampled_labels.append(np.full(num_images, i))

            pred_class = np.random.choice(random_classes, 1, replace=False)
            pred_indices = np.where(self.targets == pred_class)[0]
            pred_index = np.random.choice(pred_indices, 1, replace=False)
            pred_image = sampled_images[pred_index]
            pred_label = sampled_labels[pred_index]

            sampled_images = np.concatenate(sampled_images, axis=0)
            sampled_labels = np.concatenate(sampled_labels, axis=0)

            sampled_labels = torch.tensor(sampled_labels, dtype=torch.long)

            all_batches_images.append(sampled_images)
            all_batches_labels.append(sampled_labels)

        # Stack batches into a single tensor
        all_batches_images = torch.stack(all_batches_images)
        all_batches_labels = torch.stack(all_batches_labels)

        return all_batches_images, all_batches_labels, pred_image, pred_label