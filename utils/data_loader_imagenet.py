from torch.utils.data import Dataset
import torch
import random
import numpy as np
from torchvision import transforms, datasets
import webdataset as wds

class ImageNetRandomDataset(Dataset):
    """
    A dataset that, on each __getitem__ call, samples images from
    random classes of the ImageNet dataset.
    """

    def __init__(
        self,
        root="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        train=True,
    ):
        """
        Initialize the dataset by loading ImageNet.

        Args:
            root (str): Path to ImageNet dataset.
            num_images (int): Number of images to sample per class.
            num_samples (int): Total number of samples (length of the dataset).
            num_classes (int): How many distinct classes to randomly choose for each sample.
            random_classes (list or None): If provided, use these classes instead of sampling them randomly.
            train (bool): Whether to load the train or val split.
        """
        super().__init__()

        # Load ImageNet dataset
        split = "train" if train else "val"
        self.dataset = datasets.ImageNet(root=root, split=split)

        # Class and target mapping
        self.targets = np.array([label for _, label in self.dataset])  # Targets as numpy array
        self.num_total_classes = len(self.dataset.classes)  # Total number of classes
        self.class_to_idx = self.dataset.class_to_idx

        # Parameters for sampling
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.fixed_classes = random_classes  # Fixed classes if provided

        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        On each call, randomly sample `self.num_classes` classes (unless fixed_classes is not None),
        then pick `self.num_images` images from each class, and also pick one 'prediction' image
        from one of those classes.
        """
        # Select classes (either fixed or random)
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

        # For each chosen class, sample images
        class_to_label = {}
        for label_idx, cls in enumerate(chosen_classes):
            available_indices = class_indices[cls]
            chosen_indices = np.random.choice(available_indices, self.num_images, replace=False)

            for idx in chosen_indices:
                image, _ = self.dataset[idx]
                sampled_images.append(self.transform(image))
                sampled_labels.append(label_idx)

            class_to_label[cls] = label_idx

        # Randomly pick a prediction image from one of the chosen classes
        pred_class = random.choice(chosen_classes)
        pred_indices = class_indices[pred_class]
        pred_index = np.random.choice(pred_indices, 1, replace=False)[0]
        pred_image, _ = self.dataset[pred_index]
        pred_image_torch = self.transform(pred_image)
        pred_label = class_to_label[pred_class]

        # Stack sampled images and labels
        sampled_images_torch = torch.stack(sampled_images)  # Shape: (num_classes * num_images, C, H, W)
        sampled_labels_torch = torch.tensor(sampled_labels, dtype=torch.long)  # Shape: (num_classes * num_images,)

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)
    
class ImageNetRandomWebDataset(Dataset):
    def __init__(
            self,
            shard_pattern="/imagenet-train/imagenet-train-{00000..00125}.tar",
            num_images=10,
            num_samples=10000,
            num_classes=2,
            random_classes=None):
        super().__init__()

        self.shard_pattern = shard_pattern
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.fixed_classes = random_classes  # Fixed classes if provided

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Load dataset and cache class-to-index mapping
        self.dataset = wds.WebDataset(self.shard_pattern).decode(
            "pil").to_tuple("jpg", "cls")
        self.class_to_samples = self._build_class_index()

        # Total classes in dataset
        self.num_total_classes = len(self.class_to_samples.keys())

    def _build_class_index(self):
        """
        Build an index mapping each class to its list of images.
        """
        class_to_samples = {}
        for img, cls in self.dataset:
            cls_name = cls.decode('utf-8')  # Convert class label to string
            if cls_name not in class_to_samples:
                class_to_samples[cls_name] = []
            class_to_samples[cls_name].append(img)
        return class_to_samples
    
    def __iter__(self):
        for _ in range(self.num_samples):
            # Select classes (fixed or random)
            if self.fixed_classes is not None:
                chosen_classes = self.fixed_classes
            else:
                chosen_classes = random.sample(
                    list(self.class_to_samples.keys()), self.num_classes)

            sampled_images = []
            sampled_labels = []
            class_to_label = {}

            # Sample images from chosen classes
            for label_idx, cls in enumerate(chosen_classes):
                available_images = self.class_to_samples[cls]
                chosen_images = random.sample(
                    available_images, self.num_images)

                for img in chosen_images:
                    img_transformed = self.transform(img)
                    sampled_images.append(img_transformed)
                    sampled_labels.append(label_idx)

                class_to_label[cls] = label_idx

            # Randomly pick a prediction image from one of the chosen classes
            pred_class = random.choice(chosen_classes)
            pred_image = random.choice(self.class_to_samples[pred_class])
            pred_image_torch = self.transform(pred_image)
            pred_label = class_to_label[pred_class]

            # Stack sampled images and labels
            # Shape: (num_classes * num_images, C, H, W)
            sampled_images_torch = torch.stack(sampled_images)
            # Shape: (num_classes * num_images,)
            sampled_labels_torch = torch.tensor(
                sampled_labels, dtype=torch.long)

            yield sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)

    def __len__(self):
        return self.num_samples
