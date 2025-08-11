import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict
from PIL import Image


class CIFAR10RandomDataset(Dataset):
    """
    A dataset that, on each __getitem__ call, samples images from
    random classes of the CIFAR-10 dataset.
    """

    def __init__(
        self,
        root="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        train=True,
        excluded_classes=None,
        included_classes=None,
        resize_to_224=False
    ):
        """
        Initialize the dataset by loading CIFAR-10.

        Args:
            root (str): Path to CIFAR-10 dataset.
            num_images (int): Number of images to sample per class.
            num_samples (int): Total number of samples (length of the dataset).
            num_classes (int): How many distinct classes to randomly choose for each sample.
            random_classes (list or None): If provided, use these classes instead of sampling them randomly.
            train (bool): Whether to load the train or test split.
            excluded_classes (list or None): Classes to exclude when sampling.
            included_classes (list or None): Only include these classes when sampling.
            resize_to_224 (bool): Resize images to 224x224 if True.
        """
        super().__init__()

        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.data = self.dataset.data
        self.targets = np.array(self.dataset.targets)

        self.excluded_classes = excluded_classes
        self.included_classes = included_classes

        self.num_total_classes = 10
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.dataset.classes)}

        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.fixed_classes = random_classes

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)) if resize_to_224 else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])

        self.class_index = self._build_class_index()

    def _build_class_index(self):
        class_to_images = defaultdict(list)
        for i, label in enumerate(self.targets):
            class_to_images[label].append(i)
        return class_to_images

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Select classes (either fixed or random)
        if self.fixed_classes is not None:
            chosen_classes = self.fixed_classes
        else:
            if self.included_classes is not None:
                chosen_classes = random.sample(self.included_classes, self.num_classes)
            elif self.excluded_classes is not None:
                candidates = [cls for cls in range(self.num_total_classes) if cls not in self.excluded_classes]
                chosen_classes = random.sample(candidates, self.num_classes)
            else:
                chosen_classes = random.sample(range(self.num_total_classes), self.num_classes)

        sampled_images = []
        sampled_labels = []

        class_to_label = {}
        for label_idx, cls in enumerate(chosen_classes):
            available_indices = self.class_index[cls]
            chosen_indices = np.random.choice(available_indices, self.num_images, replace=False)

            for idx in chosen_indices:
                image = Image.fromarray(self.data[idx])
                sampled_images.append(self.transform(image))
                sampled_labels.append(label_idx)

            class_to_label[cls] = label_idx

        pred_class = random.choice(chosen_classes)
        pred_index = np.random.choice(self.class_index[pred_class], 1, replace=False)[0]
        pred_image = Image.fromarray(self.data[pred_index])
        pred_image_torch = self.transform(pred_image)
        pred_label = class_to_label[pred_class]

        sampled_images_torch = torch.stack(sampled_images)
        sampled_labels_torch = torch.tensor(sampled_labels, dtype=torch.long)

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)


def get_cifar10_random_loader(
    root="./data",
    num_images=10,
    num_samples=10000,
    num_classes=2,
    random_classes=None,
    train=True,
    batch_size=32,
    num_workers=4,
    exclude_classes=None,
    include_classes=None,
    resize_to_224=False
):
    """
    Returns a DataLoader for the CIFAR10RandomDataset.

    Args:
        root (str): Path to CIFAR-10 dataset.
        num_images (int): Number of images to sample per class.
        num_samples (int): Total number of samples (length of the dataset).
        num_classes (int): How many distinct classes to randomly choose for each sample.
        random_classes (list or None): If provided, use these classes instead of sampling them randomly.
        train (bool): Whether to load the train or test split.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the DataLoader.
        exclude_classes (list or None): List of class indices to exclude.
        include_classes (list or None): List of class indices to include.
        resize_to_224 (bool): Resize images to 224x224 if True.
    """
    dataset = CIFAR10RandomDataset(
        root=root,
        num_images=num_images,
        num_samples=num_samples,
        num_classes=num_classes,
        random_classes=random_classes,
        train=train,
        excluded_classes=exclude_classes,
        included_classes=include_classes,
        resize_to_224=resize_to_224
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
