import os
from torchvision.datasets import ImageNet
import webdataset as wds
from utils.data_loader_imagenet import get_imagenet_random_loader
from tqdm import trange
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()
    sink = wds.TarWriter(args.input)
    test_classes = [87, 155, 178, 181, 199, 217, 284, 321,
                    452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
    batch_size = 16
    num_classes = 5
    num_images = 5
    print("Create DataLoader")
    # dataset = get_imagenet_random_loader(
    #     batch_size=batch_size, num_classes=num_classes, num_samples=10000, num_images=num_images, train=True, exclude_images=test_classes, mini=False, num_workers=4)
    dataset = get_imagenet_random_loader(batch_size=batch_size, num_classes=num_classes,
                                             num_samples=500, num_images=num_images, train=True, include_images=test_classes, mini=True, num_workers=4)
    print("DataLoader created")
    progress = trange(len(dataset))
    for index, (images, labels, pred_image, pred_label) in enumerate(dataset):
        sink.write({
            "__key__": "sample%06d" % index,
            "sampled_images.pyd": images,
            "sampled_labels.pyd": labels,
            "pred_image.pyd": pred_image,
            "pred_label.pyd": pred_label,
        })
        progress.update()
    progress.close()
    sink.close()