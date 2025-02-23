from datadings.writer import FileWriter
from torch.utils.data import Subset
from torchvision.datasets import ImageNet, ImageFolder
import numpy as np
from simplejpeg import encode_jpeg
from tqdm import trange
import argparse
from PIL import Image


def encode_img(img, quality=85, short_side=375, long_side=500, colorsubsampling='422'):
    target_size = 3 * short_side * long_side
    # bio = io.BytesIO(data)
    im = np.array(img)
    colorspace = 'RGB'  # converted to RGB guaranteed
    compress = True  # force compression since simplejpeg failed
    # if images are CMYK or
    if colorspace == 'CMYK' or compress:
        # for CMYK or non-JPEG images,
        # quality might not be given, so assume 99
        if quality is None:
            quality = 98
        # default to subsampling 422
        # use full color resolution for small images
        # or if compression is disabled,
        # i.e. for CMYK images or if simplejpeg failed to decode
        if not compress or im.size <= 0.5*target_size:
            colorsubsampling = '444'
        # downscale large images
        if im.size > target_size*1.5:
            h, w = im.shape[:2]
            s = max(h, w)
            r = long_side/s
            h, w = int(round(r*h)), int(round(r*w))
            pil = Image.fromarray(im, 'RGB')
            im = np.array(pil.resize((w, h), resample=Image.LANCZOS))
        return encode_jpeg(im, quality=quality, colorsubsampling=colorsubsampling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="./data")
    parser.add_argument('--test', "-t", action="store_true")
    parser.add_argument('--output', '-o')
    parser.add_argument('--dataset', '-d', default="inet")
    args = parser.parse_args()

    if args.dataset == "inet":
        dataset = ImageNet(root=args.input, split="train")
        if args.test:
            include_classes = [87, 155, 178, 181, 199, 217, 284, 321,
                            452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
            exclude_classes = None
        else:
            exclude_classes = [87, 155, 178, 181, 199, 217, 284, 321,
                            452, 469, 483, 541, 574, 753, 777, 788, 826, 927, 946]
            include_classes = None
    elif args.dataset == "inet21k":
        print("Creating Dataset")
        dataset = ImageFolder(root=args.input)
        print("Dataset created")
        if args.test:
            include_classes = [895, 1387, 1415, 1420, 1445, 1472, 1572, 1776, 2872, 3071, 3193, 3809, 4259, 5598, 5855, 5987, 6299, 7282, 7523]
            exclude_classes = None
        else:
            exclude_classes = [895, 1387, 1415, 1420, 1445, 1472, 1572, 1776, 2872, 3071, 3193, 3809, 4259, 5598, 5855, 5987, 6299, 7282, 7523]
            include_classes = None

    indices = []
    for idx, (_, target) in enumerate(dataset.samples):
        if include_classes is not None and target not in include_classes:
            continue
        if exclude_classes is not None and target in exclude_classes:
            continue
        indices.append(idx)

    ds = Subset(dataset, indices)


    with FileWriter(args.output) as writer:
        progressbar = trange(len(ds))
        for i, (img, label) in enumerate(ds):
            if args.dataset == "inet21k":
                nimg = encode_img(img, short_side=224, long_side=224)
            else:
                nimg = encode_img(img)
            sample = {'key': str(i), 'image': nimg, 'label': label}
            writer.write(sample)
            progressbar.update()
        progressbar.close()


