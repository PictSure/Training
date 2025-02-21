from torch.utils.data import Dataset
import torch
import random
import numpy as np
from torchvision import transforms, datasets
import torch.nn.functional as F
from utils.cluster_dataloader import ImageNetDataDingsSet, ImageNetDataDingsSet2
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF


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
        excluded_classes=None,
        included_classes=None,
        mini=False
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

        self.excluded_classes = excluded_classes
        self.included_classes = included_classes

        # Class and target mapping
        self.class_index = self._build_class_index()
        self.num_total_classes = len(self.dataset.classes)  # Total number of classes
        self.class_to_idx = self.dataset.class_to_idx

        # Parameters for sampling
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.fixed_classes = random_classes  # Fixed classes if provided

        if mini:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            # Transformation pipeline
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def _build_class_index(self):
        class_to_images = defaultdict(list)
        for img_path, label in self.dataset.samples:
            class_to_images[label].append(img_path)
        return class_to_images
    
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
            if self.included_classes is not None:
                chosen_classes = random.sample(self.included_classes, self.num_classes)
            elif self.excluded_classes is not None:
                chosen_classes = [cls for cls in range(self.num_total_classes) if cls not in self.excluded_classes]
                chosen_classes = random.sample(chosen_classes, self.num_classes)
            else:
                chosen_classes = random.sample(range(self.num_total_classes), self.num_classes)

        sampled_images = []
        sampled_labels = []

        # For each chosen class, sample images
        class_to_label = {}
        for label_idx, cls in enumerate(chosen_classes):
            available_images = self.class_index[cls]
            chosen_images = np.random.choice(available_images, self.num_images, replace=False)

            for img_path in chosen_images:
                image = Image.open(img_path).convert("RGB")
                sampled_images.append(self.transform(image))
                sampled_labels.append(label_idx)


            class_to_label[cls] = label_idx

        # Randomly pick a prediction image from one of the chosen classes
        pred_class = random.choice(chosen_classes)
        pred_images = self.class_index[pred_class]
        pred_img_path = np.random.choice(pred_images, 1, replace=False)[0]
        pred_image = Image.open(pred_img_path).convert("RGB")
        pred_image_torch = self.transform(pred_image)
        pred_label = class_to_label[pred_class]

        # Stack sampled images and labels
        sampled_images_torch = torch.stack(sampled_images)  # Shape: (num_classes * num_images, C, H, W)
        sampled_labels_torch = torch.tensor(sampled_labels, dtype=torch.long)  # Shape: (num_classes * num_images,)

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)


def normalize_samples(sampled_images, pred_image, gaussian=False, sharpness=False, resize=None):
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
    mean = torch.tensor([0.4914, 0.4822, 0.4465],
                        device=sampled_images.device).view(1, -1, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010],
                       device=sampled_images.device).view(1, -1, 1, 1)

    # Get shapes
    N, B, C, H, W = sampled_images.size()  # sampled_images shape: (N, B, C, H, W)

    # Reshape sampled_images to (N*B, C, H, W)
    sampled_images = sampled_images.view(N * B, C, H, W)

    # Normalize between [0, 1]
    # sampled_images = torch.clamp(sampled_images, 0, 255) / 255.0

    if gaussian:
        # Implement the equivalent to transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        kernel_size = 5
        sigma = random.uniform(0.1, 2.0)
        sampled_images = TF.gaussian_blur(sampled_images, kernel_size=kernel_size, sigma=sigma)
        pred_image = TF.gaussian_blur(pred_image, kernel_size=kernel_size, sigma=sigma)

    if sharpness:
        # Implement the equivalent to transforms.RandomAdjustSharpness(0.5, 0.5)
        sharpness_factor = random.uniform(0.5, 1.5)
        sampled_images = TF.adjust_sharpness(sampled_images, sharpness_factor=sharpness_factor)
        pred_image = TF.adjust_sharpness(pred_image, sharpness_factor=sharpness_factor)

    # Normalize sampled_images using mean and std
    sampled_images = (sampled_images - mean) / std

    # Normalize pred_image, which has shape (N, C, H, W)
    pred_image = (pred_image - mean) / std

    # Resize if necessary
    if resize is not None:
        # Resize sampled_images (reshaped as (N*B, C, H, W))
        sampled_images = F.interpolate(
            sampled_images, size=resize, mode="bilinear", align_corners=False)

        # Resize pred_image, handling (N, C, H, W)
        pred_image = F.interpolate(
            pred_image, size=resize, mode="bilinear", align_corners=False)

    # Reshape sampled_images back to (N, B, C, H, W)
    sampled_images = sampled_images.view(
        N, B, C, resize[0], resize[1]) if resize else sampled_images.view(N, B, C, H, W)

    return sampled_images, pred_image


def get_imagenet_random_loader(
    root="./data",
    num_images=10,
    num_samples=10000,
    num_classes=2,
    random_classes=None,
    train=True,
    batch_size=32,
    num_workers=16,
    exclude_images=None,
    include_images=None,
    mini=False
):
    """
    Returns a DataLoader for the ImageNetRandomDataset.

    Args:
        root (str): Path to ImageNet dataset.
        num_images (int): Number of images to sample per class.
        num_samples (int): Total number of samples (length of the dataset).
        num_classes (int): How many distinct classes to randomly choose for each sample.
        random_classes (list or None): If provided, use these classes instead of sampling them randomly.
        train (bool): Whether to load the train or val split.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the DataLoader.
    """
    dataset = ImageNetRandomDataset(
        root=root,
        num_images=num_images,
        num_samples=num_samples,
        num_classes=num_classes,
        random_classes=random_classes,
        train=train,
        excluded_classes=exclude_images,
        included_classes=include_images,
        mini=mini
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def get_cluster_random_loader(
    root="./data",
    num_images=10,
    num_samples=10000,
    num_classes=2,
    random_classes=None,
    batch_size=32,
    num_workers=4,
    mini=False,
    ratio=0.1
):
    """
    Returns a DataLoader for the ImageNetRandomDataset.

    Args:
        root (str): Path to ImageNet dataset.
        num_images (int): Number of images to sample per class.
        num_samples (int): Total number of samples (length of the dataset).
        num_classes (int): How many distinct classes to randomly choose for each sample.
        random_classes (list or None): If provided, use these classes instead of sampling them randomly.
        train (bool): Whether to load the train or val split.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the DataLoader.
    """
    if num_workers > 0:
        dataset = ImageNetDataDingsSet2(
            data_path=root,
            num_images=num_images,
            num_samples=num_samples,
            num_classes=num_classes,
            random_classes=random_classes,
            mini=mini,
            ratio=ratio
        )
    else:
        dataset = ImageNetDataDingsSet(
            data_path=root,
            num_images=num_images,
            num_samples=num_samples,
            num_classes=num_classes,
            random_classes=random_classes,
            mini=mini,
            ratio=ratio
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
