import zipfile
import os
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
import numpy
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.4, .4, .4),
    numpy.array,
    iaa.Sequential([
        iaa.Affine(
            rotate=(-45, 45),
            shear=(-25, 25)
        ),
        iaa.PiecewiseAffine(scale=0.045)
    ], random_order=True).augment_image,
    numpy.array,
    transforms.ToTensor(),
    normalize
])

val_data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])