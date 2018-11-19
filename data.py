import zipfile
import os
import torchvision.transforms as transforms
import transforms as transformsp
from imgaug import augmenters as iaa
import numpy
import torch

toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_data_transforms = transforms.Compose([
    transforms.Resize(480),
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
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])

val_data_transforms = transforms.Compose([
    lambda img: [img],
    transformsp.ResizeAug([480]),
    transformsp.FlipAug(),
    lambda imgs: [toTensor(img) for img in imgs],
    lambda imgs: [normalize(img) for img in imgs]
])