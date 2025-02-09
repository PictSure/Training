from torch.utils.data import Dataset
from torchvision import transforms
from datadings.reader import MsgpackReader
from collections import defaultdict
from datadings.torch import CompressedToPIL
import random
import numpy as np
import torch

class ImageNetDataDingsSet(Dataset):
    def __init__(
        self,
        data_path="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        train=True,
        excluded_classes=None,
        included_classes=None,
        mini=False
    ):
        super().__init__()
        self.data_path = data_path
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.train = train
        self.excluded_classes = excluded_classes
        self.included_classes = included_classes
        dataset = MsgpackReader(self.data_path)
        self.class_index = self._build_class_index()
        self.fixed_classes = random_classes
        self.num_total_classes = len(self.class_index.keys())
        if mini:
            self.transform = transforms.Compose([
                CompressedToPIL(),
                transforms.Resize((64, 64)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            # Transformation pipeline
            self.transform = transforms.Compose([
                CompressedToPIL(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(0.2, 0.2)
            ])
    
    def _build_class_index(self):
        data_dict = defaultdict(list)
        for i in range(len(dataset)):
            sample = dataset[i]
            data_dict[sample["label"]].append(i)
        return data_dict
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.fixed_classes is not None:
            chosen_classes = self.fixed_classes
        else:
            if self.included_classes is not None:
                chosen_classes = random.sample(
                    self.included_classes, self.num_classes)
            elif self.excluded_classes is not None:
                chosen_classes = [cls for cls in range(
                    self.num_total_classes) if cls not in self.excluded_classes]
                chosen_classes = random.sample(
                    chosen_classes, self.num_classes)
            else:
                chosen_classes = random.sample(
                    range(self.num_total_classes), self.num_classes)
        sampled_images = []
        sampled_labels = []

        class_to_label={}

        for label_idx, cls in enumerate(chosen_classes):
            available_indices = self.class_index[cls]
            chosen_indices = np.random.choice(available_indices, self.num_images, replace=False)

            for idx in chosen_indices:
                sample = dataset[idx]
                sampled_images.append(self.transform(sample["image"]))
                sampled_labels.append(label_idx)
            class_to_label[cls] = label_idx
        
        pred_class = random.choice(chosen_classes)
        pred_indices = self.class_index[pred_class]
        pred_index = np.random.choice(pred_indices, 1, replace=False)[0]
        pred_sample = dataset[pred_index]
        pred_image_torch = self.transform(pred_sample["image"])
        pred_label = class_to_label[pred_class]
        # Shape: (num_classes * num_images, C, H, W)
        sampled_images_torch = torch.stack(sampled_images)
        # Shape: (num_classes * num_images,)
        sampled_labels_torch = torch.tensor(sampled_labels, dtype=torch.long)

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)


class ImageNetDataDingsSet2(Dataset):
    def __init__(
        self,
        data_path="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        train=True,
        excluded_classes=None,
        included_classes=None,
        mini=False
    ):
        super().__init__()
        self.data_path = data_path
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.train = train
        self.excluded_classes = excluded_classes
        self.included_classes = included_classes
        with MsgpackReader(self.data_path) as dataset:
            self.class_index = self._build_class_index(dataset)
        self.fixed_classes = random_classes
        self.num_total_classes = len(self.class_index.keys())
        if mini:
            self.transform = transforms.Compose([
                CompressedToPIL(),
                transforms.Resize((64, 64)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            # Transformation pipeline
            self.transform = transforms.Compose([
                CompressedToPIL(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(0.2, 0.2)
            ])

    def _build_class_index(self, dataset):
        data_dict = defaultdict(list)
        for i in range(len(dataset)):
            sample = dataset[i]
            data_dict[sample["label"]].append(i)
        return data_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with MsgpackReader(self.data_path) as dataset:
            if self.fixed_classes is not None:
                chosen_classes = self.fixed_classes
            else:
                if self.included_classes is not None:
                    chosen_classes = random.sample(
                        self.included_classes, self.num_classes)
                elif self.excluded_classes is not None:
                    chosen_classes = [cls for cls in range(
                        self.num_total_classes) if cls not in self.excluded_classes]
                    chosen_classes = random.sample(
                        chosen_classes, self.num_classes)
                else:
                    chosen_classes = random.sample(
                        range(self.num_total_classes), self.num_classes)
            sampled_images = []
            sampled_labels = []

            class_to_label = {}

            for label_idx, cls in enumerate(chosen_classes):
                available_indices = self.class_index[cls]
                chosen_indices = np.random.choice(
                    available_indices, self.num_images, replace=False)

                for idx in chosen_indices:
                    sample = dataset[idx]
                    sampled_images.append(self.transform(sample["image"]))
                    sampled_labels.append(label_idx)
                class_to_label[cls] = label_idx

            pred_class = random.choice(chosen_classes)
            pred_indices = self.class_index[pred_class]
            pred_index = np.random.choice(pred_indices, 1, replace=False)[0]
            pred_sample = dataset[pred_index]
            pred_image_torch = self.transform(pred_sample["image"])
            pred_label = class_to_label[pred_class]
        # Shape: (num_classes * num_images, C, H, W)
        sampled_images_torch = torch.stack(sampled_images)
        # Shape: (num_classes * num_images,)
        sampled_labels_torch = torch.tensor(sampled_labels, dtype=torch.long)

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)
