import os

import pytorch_lightning as pl
from torchvision import transforms
import torch


def data_loader(fn):

    def func_wrapper(self):
        try:  # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except:  # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


def create_transforms(img_size, horizontal_flip, mean=None, std=None, crop=None):
    # We don't need any square padding because CAER-S uncropped
    # images are all the same size
    transform = transforms.Compose([
        # transforms.CenterCrop(148),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        # ToTensor converts images to [0, 1]
        transforms.ToTensor(),
        # Normalize has to happen after ToTensor because it expects tensors
    ])
    if std and mean:
        transform.transforms.insert(2, transforms.Normalize(mean=mean, std=std))
    if horizontal_flip:
        transform.transforms.insert(1, transforms.RandomHorizontalFlip())
    if crop:
        transform.transforms.insert(0, transforms.CenterCrop(crop))
    return transform

