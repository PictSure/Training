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
import json

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

class SemanticSimilarityDataDingsSet(Dataset):
    def __init__(
        self,
        hierarchy_path,
        class_index_path,
        data_path="./data",
        num_images=10,
        num_samples=10000,
        num_classes=2,
        random_classes=None,
        mini=False,
        ratio=0.01
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
        self.resample_counter = 0

        with open(class_index_path, "r") as f:
            self.label_to_idx = json.loads(f)
        
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
        
        hierarchy_data = torch.load(hierarchy_path, weights_only=False)
        self.class_tree_list = hierarchy_data["class_tree_list"]
        class_labels = list(hierarchy_data["class_list"])

        num_classes = len(self.class_tree_list)
        class_depth = torch.zeros(num_classes)
        for i in range(num_classes):
            class_depth[i] = len(self.class_tree_list[i]) - 1

        self.depth_to_indices = {}
        self.index_to_label = {i: class_labels[i] for i in range(
            num_classes)}

        for i in range(num_classes):
            level = int(class_depth[i].item())
            if level not in self.depth_to_indices:
                self.depth_to_indices[level] = []
            self.depth_to_indices[level].append(i)

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

    def build_image_index(self, chosen_classes):
        while True:
            self.clear_cache()
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

    def _get_random_class_children(self):
        target_level = random.randint(1, 11)
        while target_level not in self.depth_to_indices:
            target_level = random.randint(1, 11)
        random_index = random.choice(self.depth_to_indices[target_level])
        random_class_label = self.index_to_label[random_index]
        children = []
        next_level = target_level + 1
        if next_level in self.depth_to_indices:
            for child_index in self.depth_to_indices[next_level]:
                if random_index in self.class_tree_list[child_index]:
                    children.append(self.index_to_label[child_index])
        return children

    def resample(self):
        self.resample_counter -= 1
        if self.resample_counter <= 0:
            self.build_random_index()

    def build_random_index(self):
        children = []
        counter = 0
        condition_met = False
        while counter <= 0:
            if len(children) < 2:
                children = self._get_random_class_children()
                children = [self.label_to_idx[child] for child in children]
            else:
                condition_met = True
                break
            counter += 1
        if not condition_met and counter > 5:
            children = random.sample(list(self.class_index.keys()), k=int(self.ratio*self.num_total_classes))

        self.build_image_index(chosen_classes=children)

        if len(children) < self.num_images:
            self.resample_counter = 5
        elif len(children) == self.num_images:
            self.resample_counter = 15
        else:
            self.resample_counter = 30

    def __getitem__(self, idx):
        total_images = self.num_images * self.num_classes
        if len(self.classes) > self.num_classes:
            chosen_classes = random.sample(self.classes, self.num_classes)
            num_images = self.num_images
            num_classes = self.num_classes
        elif len(self.classes) == self.num_classes:
            chosen_classes = self.classes
            num_images = self.num_images
            num_classes = self.num_classes
        else:
            chosen_classes = self.classes
            num_classes = len(self.classes)
            num_images = total_images // num_classes

        sampled_images = []
        sampled_labels = []
        class_to_label = {}

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
