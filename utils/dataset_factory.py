import os
from utils.data_loader_imagenet import get_cluster_random_loader, get_imagenet_random_loader


class DatasetFactory:
    def __init__(self, config, args, start_epoch):
        self.config = config
        self.args = args
        self.start_epoch = start_epoch
        self.test_classes = [87, 155, 178, 181, 199, 217, 284, 321, 452, 
                           469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
    
    def get_dataloaders(self):
        if self.config.get("training_loc") == "cluster":
            return self._get_cluster_loaders()
        return self._get_imagenet_loaders()

    def _get_cluster_loaders(self):
        training_loader = get_cluster_random_loader(
            root=os.path.join(
                self.config["paths"]["dataset"], self.config["paths"]["train"]),
            class_index_path=self.config["paths"]["class_index"],
            batch_size=self.config["dataloader"]["batch_size"],
            num_classes=self.config["dataloader"]["num_classes"],
            num_samples=self.config["dataloader"]["num_samples"],
            num_images=self.config["dataloader"]["num_images"],
            mini=False,
            num_workers=self.config["dataloader"]["num_workers"],
            ratio=self.config["dataloader"]["train_ratio"]
        )
        test_loader = get_cluster_random_loader(
            root=os.path.join(
                self.config["paths"]["dataset"], self.config["paths"]["test"]),
            batch_size=self.config["dataloader"]["batch_size"],
            num_classes=5,
            num_samples=500,
            num_images=5,
            mini=True,
            num_workers=self.config["dataloader"]["num_workers"],
            ratio=self.config["dataloader"]["test_ratio"]
        )
        test_loader.dataset.build_image_index()
        if not self.args.new and self.start_epoch > 0 and self.start_epoch % self.config.get("resample", 30) != 0:
                training_loader.dataset.build_image_index()
        return training_loader, test_loader
    
    def _get_imagenet_loaders(self):
        training_loader = get_imagenet_random_loader(
            root=self.config["paths"]["dataset"],
            batch_size=self.config["dataloader"]["batch_size"],
            num_classes=self.config["dataloader"]["num_classes"],
            num_samples=10000,
            num_images=self.config["dataloader"]["num_images"],
            train=True,
            exclude_images=self.test_classes,
            mini=False,
            num_workers=self.config["dataloader"]["num_workers"]
        )
        test_loader = get_imagenet_random_loader(
            root=self.config["paths"]["dataset"],
            batch_size=self.config["dataloader"]["batch_size"],
            num_classes=self.config["dataloader"]["num_classes"],
            num_samples=10000,
            num_images=self.config["dataloader"]["num_images"],
            train=True,
            include_images=self.test_classes,
            mini=True,
            num_workers=self.config["dataloader"]["num_workers"]
        )
        return training_loader, test_loader

