from torch.utils.data import Dataset
from torchvision import transforms
from datadings.reader import MsgpackReader
from collections import defaultdict
from datadings.torch import CompressedToPIL
import random
import numpy as np
import torch
from tqdm import trange
import gc

class ImageNetDataDingsSet(Dataset):
    def __init__(
        self,
        data_path="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        mini=False,
        ratio=0.25
    ):
        super().__init__()
        self.data_path = data_path
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dataset = MsgpackReader(self.data_path)
        self.data = None
        self.classes = []
        self.class_index = self._build_class_index()
        self.fixed_classes = random_classes
        self.num_total_classes = len(self.class_index.keys())
        self.ratio = ratio
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
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
        #self.build_image_index(ratio=ratio)

    def clear_cache(self):
        """Clears stored dataset and GPU cache."""
        del self.data  # Remove reference
        self.data = None  # Reset the variable
        gc.collect()  # Force garbage collection
        if self.device == "cuda":
            torch.cuda.empty_cache()  # Clear GPU memory if applicable
        elif self.device == "mps":
            torch.mps.empty_cache()
        
    def _build_class_index(self):
        data_dict = defaultdict(list)
        progressbar = trange(len(self.dataset))
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            data_dict[sample["label"]].append(i)
            progressbar.update()
        progressbar.close()
        return data_dict
    
    def build_image_index(self):
        while True:
            self.clear_cache()
            chosen_classes = random.sample(list(self.class_index.keys()), k=int(self.ratio*self.num_total_classes))
            data_dict = defaultdict(list)
            progessbar = trange(len(chosen_classes), leave=False)
            for i in chosen_classes:
                samples = self.class_index[i]
                for sample_idx in samples:
                    sample = self.dataset[sample_idx]
                    img = self.transform(sample["image"])
                    data_dict[i].append(img)
                progessbar.update()
            progessbar.close()
            self.classes = chosen_classes
            self.data = data_dict
            if not self._check_bad_image_index():
                break
        
    def _check_bad_image_index(self):
        return any(len(v) < self.num_classes * self.num_images for v in self.data.values()) if self.data else True

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        chosen_classes = random.sample(self.classes, self.num_classes)
        sampled_images = []
        sampled_labels = []

        class_to_label={}

        for label_idx, cls in enumerate(chosen_classes):
            available_images = self.data[cls]
            chosen_images = random.sample(available_images, self.num_images)

            for img in chosen_images:
                # sampled_images.append(self.transform(img))
                sampled_images.append(img)
                sampled_labels.append(label_idx)
            class_to_label[cls] = label_idx
        
        pred_class = random.choice(chosen_classes)
        pred_images = self.data[pred_class]
        pred_img = random.sample(pred_images, 1)[0]
        # pred_image_torch = self.transform(pred_img)
        pred_image_torch = pred_img
        pred_label = class_to_label[pred_class]
        # Shape: (num_classes * num_images, C, H, W)
        sampled_images_torch = torch.stack(sampled_images)
        # Shape: (num_classes * num_images,)
        sampled_labels_torch = torch.tensor(sampled_labels, dtype=torch.long)

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)


class RandomClassDataDingsSet(Dataset):
    def __init__(
        self,
        data_path="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        mini=False,
        ratio=0.25
    ):
        super().__init__()
        self.data_path = data_path
        self.num_images = num_images
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dataset = MsgpackReader(self.data_path)
        self.data = None
        self.classes = []
        self.class_index = self._build_class_index()
        self.fixed_classes = random_classes
        self.num_total_classes = len(self.class_index.keys())
        self.ratio = ratio
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
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
        #self.build_image_index(ratio=ratio)

    def clear_cache(self):
        """Clears stored dataset and GPU cache."""
        del self.data  # Remove reference
        self.data = None  # Reset the variable
        gc.collect()  # Force garbage collection
        if self.device == "cuda":
            torch.cuda.empty_cache()  # Clear GPU memory if applicable
        elif self.device == "mps":
            torch.mps.empty_cache()
        
    def _build_class_index(self):
        data_dict = defaultdict(list)
        progressbar = trange(len(self.dataset))
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            data_dict[sample["label"]].append(i)
            progressbar.update()
        progressbar.close()
        return data_dict
    
    def build_image_index(self):
        while True:
            self.clear_cache()
            chosen_classes = random.sample(list(self.class_index.keys()), k=int(self.ratio*self.num_total_classes))
            data_dict = defaultdict(list)
            progessbar = trange(len(chosen_classes), leave=False)
            for i in chosen_classes:
                samples = self.class_index[i]
                for sample_idx in samples:
                    sample = self.dataset[sample_idx]
                    img = self.transform(sample["image"])
                    data_dict[i].append(img)
                progessbar.update()
            progessbar.close()
            self.classes = chosen_classes
            self.data = data_dict
            if not self._check_bad_image_index():
                break
        
    def _check_bad_image_index(self):
        return any(len(v) < self.num_classes * self.num_images for v in self.data.values()) if self.data else True

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        num_classes = random.randint(2, self.num_classes)
        total_images = self.num_classes * self.num_images
        num_images = total_images // num_classes
        chosen_classes = random.sample(self.classes, num_classes)
        sampled_images = []
        sampled_labels = []

        class_to_label={}

        for label_idx, cls in enumerate(chosen_classes):
            available_images = self.data[cls]
            chosen_images = random.sample(available_images, num_images)

            for img in chosen_images:
                # sampled_images.append(self.transform(img))
                sampled_images.append(img)
                sampled_labels.append(label_idx)
            class_to_label[cls] = label_idx
        
        while len(sampled_images) < total_images:
            cls = random.choice(chosen_classes)
            available_images = self.data[cls]
            chosen_image = random.choice(available_images)
            sampled_images.append(chosen_image)
            sampled_labels.append(class_to_label[cls])
        
        pred_class = random.choice(chosen_classes)
        pred_images = self.data[pred_class]
        pred_img = random.sample(pred_images, 1)[0]
        # pred_image_torch = self.transform(pred_img)
        pred_image_torch = pred_img
        pred_label = class_to_label[pred_class]
        # Shape: (num_classes * num_images, C, H, W)
        sampled_images_torch = torch.stack(sampled_images)
        # Shape: (num_classes * num_images,)
        sampled_labels_torch = torch.tensor(sampled_labels, dtype=torch.long)

        return sampled_images_torch, sampled_labels_torch, pred_image_torch, torch.tensor(pred_label, dtype=torch.long)
